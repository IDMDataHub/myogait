"""Generate OpenSim setup files from myogait data.

Provides functions to create XML configuration files for OpenSim's
Scale Tool, Inverse Kinematics Tool, and a MocoTrack template.

Functions
---------
export_opensim_scale_setup
    Generate an XML Scale Tool setup file.
export_ik_setup
    Generate an XML InverseKinematicsTool setup file.
export_moco_setup
    Generate a MocoTrack XML template with comments.
get_opensim_marker_names
    Return the myogait-to-OpenSim marker name mapping.
"""

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from xml.dom.minidom import parseString

from .constants import OPENSIM_MARKER_MAP

logger = logging.getLogger(__name__)


def _prettify(root: ET.Element) -> str:
    """Return a pretty-printed XML string for an ElementTree Element."""
    rough = ET.tostring(root, encoding="unicode")
    dom = parseString(rough)
    pretty = dom.toprettyxml(indent="  ")
    # Remove the XML declaration line that minidom adds
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    # Remove trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def _get_subject_info(data: dict) -> dict:
    """Extract subject metadata from data, checking both locations."""
    # Check data["meta"]["subject"] first (as per spec), then data["subject"]
    subject = {}
    meta_subject = data.get("meta", {}).get("subject", {})
    if meta_subject and isinstance(meta_subject, dict):
        subject = meta_subject
    top_subject = data.get("subject")
    if top_subject and isinstance(top_subject, dict):
        # Merge: top-level subject fills in missing keys
        for k, v in top_subject.items():
            if k not in subject:
                subject[k] = v
    return subject


def _compute_segment_length(data: dict, landmark_a: str, landmark_b: str) -> Optional[float]:
    """Compute average Euclidean distance between two landmarks across frames.

    Returns None if landmarks are not found in any frame.
    """
    frames = data.get("frames", [])
    if not frames:
        return None

    distances = []
    for frame in frames:
        lm = frame.get("landmarks", {})
        a = lm.get(landmark_a)
        b = lm.get(landmark_b)
        if a is None or b is None:
            continue
        ax, ay = a.get("x"), a.get("y")
        bx, by = b.get("x"), b.get("y")
        if ax is None or ay is None or bx is None or by is None:
            continue
        if math.isnan(ax) or math.isnan(ay) or math.isnan(bx) or math.isnan(by):
            continue
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        distances.append(dist)

    if not distances:
        return None
    return sum(distances) / len(distances)


def _compute_midpoint_distance(
    data: dict,
    a1: str, a2: str,
    b1: str, b2: str,
) -> Optional[float]:
    """Compute average distance between midpoint(a1, a2) and midpoint(b1, b2)."""
    frames = data.get("frames", [])
    if not frames:
        return None

    distances = []
    for frame in frames:
        lm = frame.get("landmarks", {})
        pa1, pa2 = lm.get(a1), lm.get(a2)
        pb1, pb2 = lm.get(b1), lm.get(b2)
        if pa1 is None or pa2 is None or pb1 is None or pb2 is None:
            continue
        vals = [
            pa1.get("x"), pa1.get("y"),
            pa2.get("x"), pa2.get("y"),
            pb1.get("x"), pb1.get("y"),
            pb2.get("x"), pb2.get("y"),
        ]
        if any(v is None for v in vals):
            continue
        if any(math.isnan(v) for v in vals):
            continue
        mid_a_x = (vals[0] + vals[2]) / 2.0
        mid_a_y = (vals[1] + vals[3]) / 2.0
        mid_b_x = (vals[4] + vals[6]) / 2.0
        mid_b_y = (vals[5] + vals[7]) / 2.0
        dist = math.sqrt((mid_a_x - mid_b_x) ** 2 + (mid_a_y - mid_b_y) ** 2)
        distances.append(dist)

    if not distances:
        return None
    return sum(distances) / len(distances)


def _get_landmark_names(data: dict) -> list:
    """Get all landmark names from the first frame."""
    frames = data.get("frames", [])
    if not frames:
        return []
    return list(frames[0].get("landmarks", {}).keys())


def export_opensim_scale_setup(
    data: dict,
    output_path: str = "scale_setup.xml",
    model_file: str = "gait2392_simbody.osim",
    static_frames: tuple = (0, 30),
    output_model: str = "scaled_model.osim",
) -> str:
    """Generate an XML Scale Tool setup file for OpenSim.

    Creates a valid XML file that configures the OpenSim Scale Tool
    with subject anthropometry, segment measurement definitions,
    and marker placer settings.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with frames and optionally subject metadata.
    output_path : str
        Output XML file path (default ``"scale_setup.xml"``).
    model_file : str
        Path to the generic OpenSim model file.
    static_frames : tuple of (int, int)
        Frame range for the static trial (start, end).
    output_model : str
        Filename for the scaled output model.

    Returns
    -------
    str
        Path to the created XML file.
    """
    subject = _get_subject_info(data)
    subject_name = subject.get("name", "subject")
    mass = subject.get("weight_kg", 75.0)
    height = subject.get("height_m", 1.75)

    fps = data.get("meta", {}).get("fps", 30.0)
    start_time = static_frames[0] / fps
    end_time = static_frames[1] / fps

    # Build a .trc filename from the data source
    source = data.get("meta", {}).get("video_path", "static_trial")
    trc_file = Path(source).stem + "_static.trc"

    # Root element
    root = ET.Element("OpenSimDocument", Version="40000")
    scale_tool = ET.SubElement(root, "ScaleTool", name=subject_name)

    # Mass
    mass_el = ET.SubElement(scale_tool, "mass")
    mass_el.text = str(float(mass))

    # Height
    height_el = ET.SubElement(scale_tool, "height")
    height_el.text = str(float(height))

    # Notes
    notes_el = ET.SubElement(scale_tool, "notes")
    notes_el.text = f"Generated by myogait for subject {subject_name}"

    # GenericModelMaker
    gmm = ET.SubElement(scale_tool, "GenericModelMaker")
    model_file_el = ET.SubElement(gmm, "model_file")
    model_file_el.text = model_file

    # ModelScaler
    model_scaler = ET.SubElement(scale_tool, "ModelScaler")
    apply_el = ET.SubElement(model_scaler, "apply")
    apply_el.text = "true"

    scaling_order = ET.SubElement(model_scaler, "scaling_order")
    scaling_order.text = "measurements"

    # MeasurementSet
    meas_set = ET.SubElement(model_scaler, "MeasurementSet")
    objects = ET.SubElement(meas_set, "objects")

    # Measurement definitions based on segment lengths
    measurements = [
        {
            "name": "femur_length",
            "landmarks": [
                ("LEFT_HIP", "LEFT_KNEE"),
                ("RIGHT_HIP", "RIGHT_KNEE"),
            ],
            "bodies": "femur_l femur_r",
        },
        {
            "name": "tibia_length",
            "landmarks": [
                ("LEFT_KNEE", "LEFT_ANKLE"),
                ("RIGHT_KNEE", "RIGHT_ANKLE"),
            ],
            "bodies": "tibia_l tibia_r",
        },
        {
            "name": "trunk_height",
            "landmarks": [],  # midpoint-based, handled separately
            "bodies": "torso",
        },
    ]

    # Compute actual measurement values for reference
    femur_length = _compute_segment_length(data, "LEFT_HIP", "LEFT_KNEE")
    tibia_length = _compute_segment_length(data, "LEFT_KNEE", "LEFT_ANKLE")
    trunk_height = _compute_midpoint_distance(
        data,
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_HIP", "RIGHT_HIP",
    )

    marker_map = OPENSIM_MARKER_MAP.get("gait2392", {})

    for meas_def in measurements:
        measurement = ET.SubElement(objects, "Measurement", name=meas_def["name"])
        apply_m = ET.SubElement(measurement, "apply")
        apply_m.text = "true"

        # MarkerPairSet
        mp_set = ET.SubElement(measurement, "MarkerPairSet")
        mp_objects = ET.SubElement(mp_set, "objects")

        if meas_def["name"] == "femur_length":
            for side_a, side_b in [("LEFT_HIP", "LEFT_KNEE"), ("RIGHT_HIP", "RIGHT_KNEE")]:
                mp = ET.SubElement(mp_objects, "MarkerPair")
                markers_el = ET.SubElement(mp, "markers")
                osim_a = marker_map.get(side_a, side_a)
                osim_b = marker_map.get(side_b, side_b)
                markers_el.text = f"{osim_a} {osim_b}"
        elif meas_def["name"] == "tibia_length":
            for side_a, side_b in [("LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_KNEE", "RIGHT_ANKLE")]:
                mp = ET.SubElement(mp_objects, "MarkerPair")
                markers_el = ET.SubElement(mp, "markers")
                osim_a = marker_map.get(side_a, side_a)
                osim_b = marker_map.get(side_b, side_b)
                markers_el.text = f"{osim_a} {osim_b}"
        elif meas_def["name"] == "trunk_height":
            # Use shoulder and hip markers to define trunk
            for side_a, side_b in [("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP")]:
                mp = ET.SubElement(mp_objects, "MarkerPair")
                markers_el = ET.SubElement(mp, "markers")
                osim_a = marker_map.get(side_a, side_a)
                osim_b = marker_map.get(side_b, side_b)
                markers_el.text = f"{osim_a} {osim_b}"

        # BodyScaleSet
        bs_set = ET.SubElement(measurement, "BodyScaleSet")
        bs_objects = ET.SubElement(bs_set, "objects")
        for body_name in meas_def["bodies"].split():
            body_scale = ET.SubElement(bs_objects, "BodyScale", name=body_name)
            axes_el = ET.SubElement(body_scale, "axes")
            axes_el.text = "X Y Z"

    # Scale file for output
    output_scale_file = ET.SubElement(model_scaler, "output_scale_file")
    output_scale_file.text = subject_name + "_scale_factors.xml"

    # MarkerPlacer
    marker_placer = ET.SubElement(scale_tool, "MarkerPlacer")
    mp_apply = ET.SubElement(marker_placer, "apply")
    mp_apply.text = "true"

    # Static trial marker file
    marker_file_el = ET.SubElement(marker_placer, "marker_file")
    marker_file_el.text = trc_file

    # Time range
    time_range_el = ET.SubElement(marker_placer, "time_range")
    time_range_el.text = f"{start_time:.6f} {end_time:.6f}"

    # Output model
    output_model_el = ET.SubElement(marker_placer, "output_model_file")
    output_model_el.text = output_model

    # Output marker file
    output_marker_el = ET.SubElement(marker_placer, "output_marker_file")
    output_marker_el.text = subject_name + "_markers_adjusted.xml"

    # Output motion file
    output_motion_el = ET.SubElement(marker_placer, "output_motion_file")
    output_motion_el.text = subject_name + "_static_output.mot"

    # Write XML
    xml_str = _prettify(root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        f.write(xml_str)

    logger.info(f"Exported OpenSim Scale setup: {path}")
    return str(path)


def export_ik_setup(
    trc_path: str,
    output_path: str = "ik_setup.xml",
    model_file: str = "scaled_model.osim",
    start_time: float = None,
    end_time: float = None,
    output_motion: str = "ik_results.mot",
) -> str:
    """Generate an XML InverseKinematicsTool setup file for OpenSim.

    Creates a valid XML file for the Inverse Kinematics tool with
    marker tasks for each marker found in the .trc file path.

    Parameters
    ----------
    trc_path : str
        Path to the .trc marker file.
    output_path : str
        Output XML file path (default ``"ik_setup.xml"``).
    model_file : str
        Path to the scaled OpenSim model.
    start_time : float, optional
        Start time in seconds. Defaults to 0.0 if not specified.
    end_time : float, optional
        End time in seconds. Defaults to 1.0 if not specified.
    output_motion : str
        Output .mot file for IK results.

    Returns
    -------
    str
        Path to the created XML file.
    """
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = 1.0

    # Standard marker names for gait2392
    marker_map = OPENSIM_MARKER_MAP.get("gait2392", {})
    marker_names = list(marker_map.values())

    # Root element
    root = ET.Element("OpenSimDocument", Version="40000")
    ik_tool = ET.SubElement(root, "InverseKinematicsTool", name="ik_setup")

    # Model file
    model_el = ET.SubElement(ik_tool, "model_file")
    model_el.text = model_file

    # Constraint weight
    constraint_weight = ET.SubElement(ik_tool, "constraint_weight")
    constraint_weight.text = "Inf"

    # Accuracy
    accuracy_el = ET.SubElement(ik_tool, "accuracy")
    accuracy_el.text = "1e-5"

    # IKTaskSet
    ik_task_set = ET.SubElement(ik_tool, "IKTaskSet")
    objects = ET.SubElement(ik_task_set, "objects")

    for marker_name in marker_names:
        task = ET.SubElement(objects, "IKMarkerTask", name=marker_name)
        apply_el = ET.SubElement(task, "apply")
        apply_el.text = "true"
        weight_el = ET.SubElement(task, "weight")
        weight_el.text = "1.0"

    # Marker file
    marker_file_el = ET.SubElement(ik_tool, "marker_file")
    marker_file_el.text = trc_path

    # Time range
    time_range_el = ET.SubElement(ik_tool, "time_range")
    time_range_el.text = f"{start_time:.6f} {end_time:.6f}"

    # Output motion file
    output_el = ET.SubElement(ik_tool, "output_motion_file")
    output_el.text = output_motion

    # Report errors
    report_errors = ET.SubElement(ik_tool, "report_errors")
    report_errors.text = "true"

    # Report marker locations
    report_markers = ET.SubElement(ik_tool, "report_marker_locations")
    report_markers.text = "false"

    # Write XML
    xml_str = _prettify(root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        f.write(xml_str)

    logger.info(f"Exported OpenSim IK setup: {path}")
    return str(path)


def export_moco_setup(
    mot_path: str,
    output_path: str = "moco_setup.xml",
    model_file: str = "scaled_model.osim",
    start_time: float = None,
    end_time: float = None,
) -> str:
    """Generate a MocoTrack XML template for OpenSim Moco.

    This creates a template/guide file with comments to help the user
    get started with Moco. It is not directly executable by Moco but
    provides the structure and paths needed.

    Parameters
    ----------
    mot_path : str
        Path to the .mot kinematics file from IK.
    output_path : str
        Output XML file path (default ``"moco_setup.xml"``).
    model_file : str
        Path to the scaled OpenSim model.
    start_time : float, optional
        Start time in seconds. Defaults to 0.0 if not specified.
    end_time : float, optional
        End time in seconds. Defaults to 1.0 if not specified.

    Returns
    -------
    str
        Path to the created XML file.
    """
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = 1.0

    # Root element
    root = ET.Element("OpenSimDocument", Version="40000")
    root.append(ET.Comment(
        " MocoTrack template generated by myogait. "
        "This file is a guide to help set up a Moco tracking problem. "
        "Modify as needed for your specific analysis. "
    ))

    moco_study = ET.SubElement(root, "MocoStudy", name="moco_tracking")

    # Problem
    moco_study.append(ET.Comment(
        " The MocoProblem defines the model, time bounds, and goals. "
    ))
    problem = ET.SubElement(moco_study, "MocoProblem")

    # Model
    problem.append(ET.Comment(
        " Specify the scaled OpenSim model to use for tracking. "
    ))
    model_el = ET.SubElement(problem, "model_file")
    model_el.text = model_file

    # Time bounds
    problem.append(ET.Comment(
        " Time range for the tracking simulation (in seconds). "
    ))
    time_initial = ET.SubElement(problem, "time_initial_bounds")
    time_initial.text = str(start_time)
    time_final = ET.SubElement(problem, "time_final_bounds")
    time_final.text = str(end_time)

    # Goals section
    problem.append(ET.Comment(
        " MocoTrack uses a MocoStateTrackingGoal to minimize the difference "
        "between simulated and experimental kinematics. Add tracking goals below. "
    ))
    goals = ET.SubElement(problem, "MocoGoalSet")
    goals_objects = ET.SubElement(goals, "objects")

    # State tracking goal
    tracking_goal = ET.SubElement(goals_objects, "MocoStateTrackingGoal", name="state_tracking")
    tracking_goal.append(ET.Comment(
        " Reference file: the .mot file from Inverse Kinematics. "
    ))
    reference_el = ET.SubElement(tracking_goal, "reference_file")
    reference_el.text = mot_path
    weight_el = ET.SubElement(tracking_goal, "weight")
    weight_el.text = "1.0"

    # Control effort goal
    effort_goal = ET.SubElement(goals_objects, "MocoControlGoal", name="control_effort")
    effort_goal.append(ET.Comment(
        " Regularization term to minimize muscle activations. "
        "Adjust weight to balance tracking vs. effort minimization. "
    ))
    effort_weight = ET.SubElement(effort_goal, "weight")
    effort_weight.text = "0.001"

    # Solver section
    moco_study.append(ET.Comment(
        " Solver settings: MocoCasADiSolver is recommended for MocoTrack. "
        "Adjust num_mesh_intervals and convergence_tolerance as needed. "
    ))
    solver = ET.SubElement(moco_study, "MocoCasADiSolver")
    mesh_el = ET.SubElement(solver, "num_mesh_intervals")
    mesh_el.text = "50"
    tol_el = ET.SubElement(solver, "convergence_tolerance")
    tol_el.text = "1e-4"
    max_iter_el = ET.SubElement(solver, "max_iterations")
    max_iter_el.text = "1000"

    # Write XML
    xml_str = _prettify(root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        f.write(xml_str)

    logger.info(f"Exported Moco setup template: {path}")
    return str(path)


def get_opensim_marker_names(model: str = "gait2392") -> dict:
    """Return the mapping from myogait landmark names to OpenSim marker names.

    Parameters
    ----------
    model : str
        OpenSim model name. Supported: ``"gait2392"``, ``"rajagopal2015"``.

    Returns
    -------
    dict
        Mapping ``{myogait_landmark: opensim_marker_name}``.

    Raises
    ------
    ValueError
        If the model is not supported.
    """
    if model not in OPENSIM_MARKER_MAP:
        supported = ", ".join(sorted(OPENSIM_MARKER_MAP.keys()))
        raise ValueError(
            f"Unknown OpenSim model '{model}'. Supported models: {supported}"
        )
    return dict(OPENSIM_MARKER_MAP[model])
