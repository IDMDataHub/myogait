"""Export gait data to various file formats.

Provides functions to export myogait analysis results to CSV, OpenSim
(.mot and .trc), and Excel workbook formats.

Functions
---------
export_csv
    Export angles, events, cycles, and statistics to CSV files.
export_mot
    Export joint angles to OpenSim .mot (motion) format.
export_trc
    Export landmark positions to OpenSim .trc (marker) format.
export_excel
    Export all data to a multi-tab Excel workbook (requires openpyxl).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import OPENSIM_MARKER_MAP, EXTENDED_FOOT_LANDMARKS

logger = logging.getLogger(__name__)


# ── CSV export ───────────────────────────────────────────────────────


def export_csv(
    data: dict,
    output_dir: str,
    cycles: Optional[dict] = None,
    stats: Optional[dict] = None,
    prefix: str = "",
) -> list:
    """Export gait data to CSV files.

    Creates separate CSV files for angles, events, cycles, and
    statistics in the specified output directory.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with angles and events.
    output_dir : str
        Directory path for output files. Created if it does not exist.
    cycles : dict, optional
        Output of ``segment_cycles()``.
    stats : dict, optional
        Output of ``analyze_gait()``.
    prefix : str, optional
        Filename prefix (e.g. ``"patient01_"``).

    Returns
    -------
    list of str
        Paths to all created CSV files.

    Raises
    ------
    TypeError
        If *data* is not a dict.
    OSError
        If the output directory cannot be created.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    # Auto-detect cycles from pipeline data when not explicitly passed
    if cycles is None:
        cycles = data.get("cycles_data")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = []

    # Angles
    angles = data.get("angles")
    if angles and angles.get("frames"):
        rows = []
        for af in angles["frames"]:
            row = {"frame_idx": af.get("frame_idx")}
            for key in ["hip_L", "hip_R", "knee_L", "knee_R",
                        "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt"]:
                row[key] = af.get(key)
            rows.append(row)
        df = pd.DataFrame(rows)
        path = out / f"{prefix}angles.csv"
        df.to_csv(path, index=False, float_format="%.3f")
        created.append(str(path))

    # Events
    events = data.get("events")
    if events:
        rows = []
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            for ev in events.get(key, []):
                rows.append({
                    "event_type": key,
                    "frame": ev["frame"],
                    "time": ev["time"],
                    "confidence": ev.get("confidence", 1.0),
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("frame")
            path = out / f"{prefix}events.csv"
            df.to_csv(path, index=False, float_format="%.4f")
            created.append(str(path))

    # Cycles
    if cycles and cycles.get("cycles"):
        rows = []
        for c in cycles["cycles"]:
            rows.append({
                "cycle_id": c["cycle_id"],
                "side": c["side"],
                "start_frame": c["start_frame"],
                "end_frame": c["end_frame"],
                "toe_off_frame": c.get("toe_off_frame"),
                "duration": c["duration"],
                "stance_pct": c.get("stance_pct"),
                "swing_pct": c.get("swing_pct"),
            })
        df = pd.DataFrame(rows)
        path = out / f"{prefix}cycles.csv"
        df.to_csv(path, index=False, float_format="%.3f")
        created.append(str(path))

        # Normalized cycle curves
        for c in cycles["cycles"]:
            an = c.get("angles_normalized", {})
            if an:
                first_vals = next(iter(an.values()))
                n_pts = len(first_vals)
                cycle_df = pd.DataFrame({"pct": np.linspace(0, 100, n_pts)})
                for joint, vals in an.items():
                    cycle_df[joint] = vals
                path = out / f"{prefix}cycle_{c['cycle_id']}_{c['side']}.csv"
                cycle_df.to_csv(path, index=False, float_format="%.3f")
                created.append(str(path))

    # Stats
    if stats:
        rows = []
        for section, values in stats.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    rows.append({"section": section, "parameter": k, "value": v})
            elif isinstance(values, list):
                for item in values:
                    rows.append({"section": section, "parameter": "flag", "value": item})
        if rows:
            df = pd.DataFrame(rows)
            path = out / f"{prefix}stats.csv"
            df.to_csv(path, index=False)
            created.append(str(path))

    logger.info(f"Exported {len(created)} CSV files to {output_dir}")
    return created


# ── Height scale helper ──────────────────────────────────────────────


def _compute_height_scale(data: dict) -> Optional[float]:
    """Estimate pixel-to-meter scale factor from subject height_m.

    Computes the vertical distance in pixels from NOSE to the midpoint
    of (LEFT_ANKLE, RIGHT_ANKLE) over the first stable frames, then
    returns ``height_m / height_pixels``.

    Returns ``None`` if height_m is not available or the estimation
    fails.
    """
    height_m = (data.get("meta", {}).get("subject") or {}).get("height_m")
    if height_m is None:
        height_m = (data.get("subject") or {}).get("height_m")
    if height_m is None or height_m <= 0:
        return None

    frames = data.get("frames", [])
    if not frames:
        return None

    h = data.get("meta", {}).get("height", 1080)

    # Sample up to first 30 stable frames (confidence > 0.5)
    pixel_heights = []
    for frame in frames[:60]:
        if frame.get("confidence", 0) < 0.5:
            continue
        lm = frame.get("landmarks", {})
        nose = lm.get("NOSE")
        l_ankle = lm.get("LEFT_ANKLE")
        r_ankle = lm.get("RIGHT_ANKLE")
        if not all([nose, l_ankle, r_ankle]):
            continue
        nose_y = nose.get("y")
        la_y = l_ankle.get("y")
        ra_y = r_ankle.get("y")
        if any(v is None for v in [nose_y, la_y, ra_y]):
            continue
        if any(isinstance(v, float) and np.isnan(v) for v in [nose_y, la_y, ra_y]):
            continue
        mid_ankle_y = (la_y + ra_y) / 2.0
        height_pixels = abs(mid_ankle_y - nose_y) * h
        if height_pixels > 10:  # avoid degenerate cases
            pixel_heights.append(height_pixels)
        if len(pixel_heights) >= 30:
            break

    if not pixel_heights:
        return None

    median_height_px = float(np.median(pixel_heights))
    if median_height_px <= 0:
        return None
    return height_m / median_height_px


# ── OpenSim .mot export ──────────────────────────────────────────────


def export_mot(
    data: dict,
    output_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> str:
    """Export joint angles to OpenSim .mot (motion) format.

    The ``.mot`` file contains time-series of joint angles compatible
    with OpenSim's Inverse Kinematics and MocoTrack tools.

    Pelvis translations (pelvis_tx, pelvis_ty, pelvis_tz) are computed
    from the midpoint of LEFT_HIP and RIGHT_HIP.  If ``height_m`` is
    available in subject metadata, positions are converted to meters.

    Extended angles (shoulder, elbow, head, pelvis sagittal tilt) and
    frontal-plane angles (pelvis_list, hip_adduction) are included
    when present in the data.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` populated.
    output_path : str
        Output ``.mot`` file path.
    start_frame : int, optional
        First frame to export (default: 0).
    end_frame : int, optional
        Last frame to export (default: all).

    Returns
    -------
    str
        Path to the created file.

    Raises
    ------
    ValueError
        If *data* has no computed angles.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    angles = data.get("angles")
    if not angles or not angles.get("frames"):
        raise ValueError("No angles in data. Run compute_angles() first.")

    fps = data.get("meta", {}).get("fps", 30.0)
    aframes = angles["frames"]

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(aframes)
    aframes = aframes[start_frame:end_frame]

    # OpenSim column mapping (myogait name -> OpenSim name)
    col_map = {
        "hip_L": "hip_flexion_l",
        "hip_R": "hip_flexion_r",
        "knee_L": "knee_angle_l",
        "knee_R": "knee_angle_r",
        "ankle_L": "ankle_angle_l",
        "ankle_R": "ankle_angle_r",
        "trunk_angle": "lumbar_extension",
        "pelvis_tilt": "pelvis_tilt",
    }

    # Extended angle mapping (only added when present in data)
    extended_col_map = {
        "shoulder_flex_L": "arm_flex_l",
        "shoulder_flex_R": "arm_flex_r",
        "elbow_flex_L": "elbow_flexion_l",
        "elbow_flex_R": "elbow_flexion_r",
        "head_angle": "head_flexion",
        "pelvis_sagittal_tilt": "pelvis_tilt",  # override if more precise
    }

    # Frontal-plane angle mapping
    frontal_col_map = {
        "pelvis_list": "pelvis_list",
        "hip_adduction_L": "hip_adduction_l",
        "hip_adduction_R": "hip_adduction_r",
    }

    # Determine which extended/frontal columns actually exist in the data
    active_extended = {}
    active_frontal = {}
    if aframes:
        sample = aframes[0]
        for myogait_key, osim_key in extended_col_map.items():
            if sample.get(myogait_key) is not None:
                active_extended[myogait_key] = osim_key
        for myogait_key, osim_key in frontal_col_map.items():
            if sample.get(myogait_key) is not None:
                active_frontal[myogait_key] = osim_key

    # Compute scale factor for pelvis translations
    scale = _compute_height_scale(data)
    w = data.get("meta", {}).get("width", 1920)
    h = data.get("meta", {}).get("height", 1080)

    # Get frame-level landmark data for pelvis position
    all_frames = data.get("frames", [])

    # Build data rows
    rows = []
    for af in aframes:
        fidx = af.get("frame_idx", 0)
        row = {"time": fidx / fps}
        for myogait_key, osim_key in col_map.items():
            val = af.get(myogait_key)
            row[osim_key] = val if val is not None else 0.0

        # Extended angles
        for myogait_key, osim_key in active_extended.items():
            val = af.get(myogait_key)
            row[osim_key] = val if val is not None else 0.0

        # Frontal angles
        for myogait_key, osim_key in active_frontal.items():
            val = af.get(myogait_key)
            row[osim_key] = val if val is not None else 0.0

        # Pelvis translations from landmark data
        pelvis_tx = 0.0
        pelvis_ty = 0.0
        pelvis_tz = 0.0
        if fidx < len(all_frames):
            frame_lm = all_frames[fidx].get("landmarks", {})
            lhip = frame_lm.get("LEFT_HIP", {})
            rhip = frame_lm.get("RIGHT_HIP", {})
            lhx = lhip.get("x")
            lhy = lhip.get("y")
            rhx = rhip.get("x")
            rhy = rhip.get("y")
            if all(v is not None for v in [lhx, lhy, rhx, rhy]):
                mid_x = (lhx + rhx) / 2.0
                mid_y = (lhy + rhy) / 2.0
                if scale is not None:
                    pelvis_tx = mid_x * w * scale
                    pelvis_ty = mid_y * h * scale
                else:
                    pelvis_tx = mid_x
                    pelvis_ty = mid_y

        row["pelvis_tx"] = pelvis_tx
        row["pelvis_ty"] = pelvis_ty
        row["pelvis_tz"] = pelvis_tz
        rows.append(row)

    df = pd.DataFrame(rows)

    # Build column order
    columns = ["time"] + list(col_map.values())
    for osim_key in active_extended.values():
        if osim_key not in columns:
            columns.append(osim_key)
    for osim_key in active_frontal.values():
        if osim_key not in columns:
            columns.append(osim_key)
    columns.extend(["pelvis_tx", "pelvis_ty", "pelvis_tz"])

    # Write .mot header
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"{path.stem}\n")
        f.write("version=1\n")
        f.write(f"nRows={len(df)}\n")
        f.write(f"nColumns={len(columns)}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("\t".join(columns) + "\n")
        for _, row in df.iterrows():
            vals = [f"{row[c]:.6f}" for c in columns]
            f.write("\t".join(vals) + "\n")

    logger.info(f"Exported .mot: {path} ({len(df)} frames)")
    return str(path)


# ── OpenSim .trc export ──────────────────────────────────────────────


def export_trc(
    data: dict,
    output_path: str,
    marker_names: Optional[list] = None,
    units: str = "m",
    use_depth: bool = False,
    depth_scale: float = 1.0,
    opensim_model: Optional[str] = None,
) -> str:
    """Export landmark positions to OpenSim .trc (marker) format.

    The ``.trc`` file contains 3D marker positions (x, y, z) over time.
    Since myogait operates in 2D, the z coordinate is set to 0 unless
    depth data is available and ``use_depth=True``.

    Unit conversion is based on the subject's ``height_m`` when
    available in ``data["meta"]["subject"]`` or ``data["subject"]``.
    A pixel-to-meter scale factor is estimated from the vertical
    distance NOSE to midpoint(LEFT_ANKLE, RIGHT_ANKLE).  If
    ``height_m`` is not available, coordinates remain normalised and
    the header ``Units`` field is set to ``"normalized"``.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    output_path : str
        Output ``.trc`` file path.
    marker_names : list of str, optional
        Landmark names to export. Defaults to major joints.
    units : {'m', 'mm', 'normalized'}
        Desired coordinate units (default ``'m'``).  Overridden to
        ``'normalized'`` when no ``height_m`` is provided.
    use_depth : bool, optional
        If ``True``, look for ``landmark_depths`` in each frame and
        use depth values as the Z coordinate (default ``False``).
    depth_scale : float, optional
        Multiplier applied to raw depth values before unit conversion
        (default ``1.0``).
    opensim_model : str, optional
        If provided (e.g. ``'gait2392'``), rename markers in the TRC
        header according to ``OPENSIM_MARKER_MAP[opensim_model]``.

    Returns
    -------
    str
        Path to the created file.

    Raises
    ------
    ValueError
        If *data* has no frames, or *opensim_model* is unknown.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    frames = data.get("frames")
    if not frames:
        raise ValueError("No frames in data. Run extract() first.")

    if opensim_model is not None and opensim_model not in OPENSIM_MARKER_MAP:
        raise ValueError(
            f"Unknown opensim_model {opensim_model!r}. "
            f"Available: {list(OPENSIM_MARKER_MAP.keys())}"
        )

    fps = data.get("meta", {}).get("fps", 30.0)

    if marker_names is None:
        marker_names = [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        ]
        # Add extended foot landmarks (big_toe, small_toe) when available
        # in any frame.  These come from Sapiens/RTMW auxiliary data.
        _has_extended = any(
            "LEFT_BIG_TOE" in f.get("landmarks", {})
            for f in frames
        )
        if _has_extended:
            marker_names.extend(EXTENDED_FOOT_LANDMARKS)

    # Build display names (with optional OpenSim renaming)
    if opensim_model is not None:
        rename_map = OPENSIM_MARKER_MAP[opensim_model]
        display_names = [rename_map.get(n, n) for n in marker_names]
    else:
        display_names = list(marker_names)

    n_markers = len(marker_names)
    n_frames = len(frames)

    w = data.get("meta", {}).get("width", 1920)
    h = data.get("meta", {}).get("height", 1080)

    # Determine scale factor from height_m
    scale = _compute_height_scale(data)
    if scale is None:
        # No height_m available: use normalized coordinates
        units = "normalized"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Header line 1
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(path.name))
        # Header line 2
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\t{units}\t{fps:.2f}\t1\t{n_frames}\n")

        # Column headers with incremental X/Y/Z indices
        header1 = ["Frame#", "Time"]
        header2 = ["", ""]
        for i, dname in enumerate(display_names):
            marker_idx = i + 1
            header1.extend([dname, "", ""])
            header2.extend([f"X{marker_idx}", f"Y{marker_idx}", f"Z{marker_idx}"])
        f.write("\t".join(header1) + "\n")
        f.write("\t".join(header2) + "\n")
        f.write("\n")

        # Data rows
        for frame in frames:
            idx = frame["frame_idx"]
            time = idx / fps
            vals = [str(idx + 1), f"{time:.6f}"]
            lm = frame.get("landmarks", {})
            depths = frame.get("landmark_depths", {}) if use_depth else {}

            for name in marker_names:
                pt = lm.get(name, {})
                x_norm = pt.get("x") if pt else None
                y_norm = pt.get("y") if pt else None

                # Detect missing markers (None or NaN)
                x_missing = (
                    x_norm is None
                    or (isinstance(x_norm, float) and np.isnan(x_norm))
                )
                y_missing = (
                    y_norm is None
                    or (isinstance(y_norm, float) and np.isnan(y_norm))
                )

                if x_missing or y_missing:
                    # TRC standard: empty strings for missing markers
                    vals.extend(["", "", ""])
                    continue

                # Depth (Z coordinate)
                z = 0.0
                if use_depth and name in depths:
                    raw_depth = depths[name]
                    if raw_depth is not None:
                        z = raw_depth * depth_scale

                # Convert coordinates
                if scale is not None:
                    x_pixel = x_norm * w
                    y_pixel = y_norm * h
                    x = x_pixel * scale
                    y = y_pixel * scale
                    z = z * scale if use_depth and name in depths else z
                else:
                    # Normalized mode
                    x = x_norm
                    y = y_norm

                vals.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            f.write("\t".join(vals) + "\n")

    logger.info(f"Exported .trc: {path} ({n_frames} frames, {n_markers} markers)")
    return str(path)


# ── Excel export ─────────────────────────────────────────────────────


def export_excel(
    data: dict,
    output_path: str,
    cycles: Optional[dict] = None,
    stats: Optional[dict] = None,
) -> str:
    """Export gait data to a multi-tab Excel workbook.

    Creates sheets: Angles, Events, Cycles, Summary, Stats.
    Requires the ``openpyxl`` package.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with angles and events.
    output_path : str
        Output ``.xlsx`` file path.
    cycles : dict, optional
        Output of ``segment_cycles()``.
    stats : dict, optional
        Output of ``analyze_gait()``.

    Returns
    -------
    str
        Path to the created file.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    # Auto-detect cycles from pipeline data when not explicitly passed
    if cycles is None:
        cycles = data.get("cycles_data")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel export: pip install openpyxl"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Angles sheet
        angles = data.get("angles")
        if angles and angles.get("frames"):
            rows = []
            for af in angles["frames"]:
                row = {"frame_idx": af.get("frame_idx")}
                fps = data.get("meta", {}).get("fps", 30.0)
                row["time_s"] = round(af.get("frame_idx", 0) / fps, 4)
                for key in ["hip_L", "hip_R", "knee_L", "knee_R",
                            "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt"]:
                    row[key] = af.get(key)
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name="Angles", index=False)

        # Events sheet
        events = data.get("events")
        if events:
            rows = []
            for key in ["left_hs", "right_hs", "left_to", "right_to"]:
                for ev in events.get(key, []):
                    rows.append({
                        "type": key, "frame": ev["frame"],
                        "time_s": ev["time"], "confidence": ev.get("confidence"),
                    })
            if rows:
                pd.DataFrame(rows).sort_values("frame").to_excel(
                    writer, sheet_name="Events", index=False)

        # Cycles sheet
        if cycles and cycles.get("cycles"):
            rows = [{
                "id": c["cycle_id"], "side": c["side"],
                "start": c["start_frame"], "end": c["end_frame"],
                "to_frame": c.get("toe_off_frame"),
                "duration_s": c["duration"],
                "stance_%": c.get("stance_pct"), "swing_%": c.get("swing_pct"),
            } for c in cycles["cycles"]]
            pd.DataFrame(rows).to_excel(writer, sheet_name="Cycles", index=False)

            # Summary sheet (mean curves)
            for side in ("left", "right"):
                summary = cycles.get("summary", {}).get(side)
                if not summary:
                    continue
                summary_df = pd.DataFrame({"pct": np.linspace(0, 100, 101)})
                for joint in ["hip", "knee", "ankle", "trunk"]:
                    mean = summary.get(f"{joint}_mean")
                    std = summary.get(f"{joint}_std")
                    if mean:
                        summary_df[f"{joint}_mean"] = mean
                        summary_df[f"{joint}_std"] = std
                summary_df.to_excel(writer, sheet_name=f"Summary_{side}", index=False)

        # Stats sheet
        if stats:
            rows = []
            for section, values in stats.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        rows.append({"section": section, "parameter": k, "value": v})
                elif isinstance(values, list):
                    for item in values:
                        rows.append({"section": section, "parameter": "flag", "value": item})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name="Stats", index=False)

    logger.info(f"Exported Excel: {path}")
    return str(path)


# ── C3D export ────────────────────────────────────────────────────────


def export_c3d(data: dict, path: str) -> str:
    """Export landmark data to C3D format.

    Requires optional dependency 'c3d'. Install with:
        pip install c3d

    Parameters
    ----------
    data : dict
        Pivot JSON dict with frames and landmarks.
    path : str
        Output file path (.c3d).

    Returns
    -------
    str
        Path to the created C3D file.

    Raises
    ------
    ImportError
        If c3d package is not installed.
    ValueError
        If *data* has no frames.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    try:
        import c3d
    except ImportError:
        raise ImportError(
            "c3d is required for C3D export. Install with: pip install c3d"
        )

    frames = data.get("frames")
    if not frames:
        raise ValueError("No frames in data. Run extract() first.")

    fps = data.get("meta", {}).get("fps", 30.0)

    # Determine point labels from the first frame's landmarks
    first_landmarks = frames[0].get("landmarks", {})
    point_labels = sorted(first_landmarks.keys())
    n_points = len(point_labels)

    # Create a C3D writer
    writer = c3d.Writer(point_rate=float(fps))

    # Set point labels in the writer's parameter group
    writer.set_point_labels(point_labels)

    # Write each frame
    for frame in frames:
        lm = frame.get("landmarks", {})
        points = np.zeros((n_points, 5), dtype=np.float32)
        for i, label in enumerate(point_labels):
            pt = lm.get(label, {})
            x = pt.get("x", 0.0) if pt else 0.0
            y = pt.get("y", 0.0) if pt else 0.0
            z = 0.0  # 2D data, z is always 0
            # Handle NaN values
            if x is None or (isinstance(x, float) and np.isnan(x)):
                x = 0.0
            if y is None or (isinstance(y, float) and np.isnan(y)):
                y = 0.0
            points[i, 0] = float(x)
            points[i, 1] = float(y)
            points[i, 2] = float(z)
            points[i, 3] = 0.0  # residual
            points[i, 4] = 0.0  # camera mask
        writer.add_frames([(points, np.array([]))])

    # Save the file
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)

    logger.info(f"Exported C3D: {out_path} ({len(frames)} frames, {n_points} points)")
    return str(out_path)


# ── DataFrame conversion ─────────────────────────────────────────────


def to_dataframe(data: dict, what: str = "angles") -> "pd.DataFrame | dict":
    """Convert gait data to pandas DataFrame(s).

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles``, ``frames``, and ``events``.
    what : str, optional
        What to convert:
        - ``"angles"`` : joint angles per frame.
        - ``"landmarks"`` : landmark positions in wide format.
        - ``"events"`` : gait events table.
        - ``"all"`` : dict of all three DataFrames.

    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        A single DataFrame when *what* is ``"angles"``, ``"landmarks"``,
        or ``"events"``; a dict ``{"angles": df, "landmarks": df,
        "events": df}`` when *what* is ``"all"``.

    Raises
    ------
    ValueError
        If *what* is not one of the recognized values.
    """
    valid_whats = ("angles", "landmarks", "events", "all")
    if what not in valid_whats:
        raise ValueError(f"what must be one of {valid_whats}, got {what!r}")

    def _angles_df():
        angles = data.get("angles", {})
        aframes = angles.get("frames", [])
        rows = []
        for af in aframes:
            row = {"frame_idx": af.get("frame_idx")}
            for key in ["hip_L", "hip_R", "knee_L", "knee_R",
                        "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt"]:
                row[key] = af.get(key)
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["frame_idx", "hip_L", "hip_R", "knee_L", "knee_R",
                     "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt"])

    def _landmarks_df():
        frames = data.get("frames", [])
        rows = []
        for f in frames:
            row = {"frame_idx": f.get("frame_idx")}
            for name, coords in f.get("landmarks", {}).items():
                row[f"{name}_x"] = coords.get("x")
                row[f"{name}_y"] = coords.get("y")
                row[f"{name}_vis"] = coords.get("visibility")
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["frame_idx"])

    def _events_df():
        events = data.get("events", {})
        rows = []
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            for ev in events.get(key, []):
                rows.append({
                    "event_type": key,
                    "frame": ev.get("frame"),
                    "time": ev.get("time"),
                    "confidence": ev.get("confidence", 1.0),
                })
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["event_type", "frame", "time", "confidence"])
        if not df.empty:
            df = df.sort_values("frame").reset_index(drop=True)
        return df

    if what == "angles":
        return _angles_df()
    elif what == "landmarks":
        return _landmarks_df()
    elif what == "events":
        return _events_df()
    else:  # "all"
        return {
            "angles": _angles_df(),
            "landmarks": _landmarks_df(),
            "events": _events_df(),
        }


# ── Summary JSON export ──────────────────────────────────────────────


def export_summary_json(
    data: dict,
    cycles: dict,
    stats: dict,
    output_path: str,
) -> str:
    """Export a compact JSON summary of key gait metrics.

    Extracts the most clinically relevant metrics from *stats*, cycle
    counts from *cycles*, and optionally computes a GPS-2D score.
    Writes a JSON file with metadata (version, date, subject info).

    Parameters
    ----------
    data : dict
        Pivot JSON dict.
    cycles : dict
        Output of ``segment_cycles()``.
    stats : dict
        Output of ``analyze_gait()``.
    output_path : str
        Output JSON file path.

    Returns
    -------
    str
        Path to the created JSON file.
    """
    spatiotemporal = stats.get("spatiotemporal", {})
    symmetry = stats.get("symmetry", {})
    ws = stats.get("walking_speed", {})

    cycle_list = cycles.get("cycles", [])
    n_left = len([c for c in cycle_list if c["side"] == "left"])
    n_right = len([c for c in cycle_list if c["side"] == "right"])

    summary = {
        "metadata": {
            "version": data.get("version", "unknown"),
            "date": datetime.now().isoformat(),
            "source": data.get("meta", {}).get("source", ""),
            "subject": data.get("subject", {}),
        },
        "cycles": {
            "n_left": n_left,
            "n_right": n_right,
            "n_total": n_left + n_right,
        },
        "spatiotemporal": {
            "cadence_steps_per_min": spatiotemporal.get("cadence_steps_per_min"),
            "stride_time_mean_s": spatiotemporal.get("stride_time_mean_s"),
            "stance_pct_left": spatiotemporal.get("stance_pct_left"),
            "stance_pct_right": spatiotemporal.get("stance_pct_right"),
        },
        "symmetry": {
            "overall_si": symmetry.get("overall_si"),
        },
        "walking_speed": {
            "speed_mean": ws.get("speed_mean"),
            "unit": ws.get("unit"),
        },
    }

    # Attempt GPS-2D if available
    try:
        from .scores import gait_profile_score_2d
        gps = gait_profile_score_2d(cycles)
        summary["gps_2d"] = {
            "gps_left": gps.get("gps_2d_left"),
            "gps_right": gps.get("gps_2d_right"),
            "gps_overall": gps.get("gps_2d_overall"),
        }
    except Exception as exc:
        logger.warning("Could not compute GPS-2D summary: %s", exc)
        summary["gps_2d"] = None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Exported summary JSON: {path}")
    return str(path)


# ── OpenPose JSON export ─────────────────────────────────────────────


def export_openpose_json(
    data: dict,
    output_dir: str,
    model: str = "COCO",  # "COCO" | "BODY_25" | "HALPE_26"
    prefix: str = "",
) -> list:
    """Export landmark data to per-frame OpenPose-format JSON files.

    For each frame in *data*, writes a JSON file following the OpenPose
    output convention. Landmarks are mapped from MediaPipe names to the
    target skeleton model using the inverse mapping tables in
    :mod:`myogait.constants`.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated (each frame must
        contain ``landmarks`` keyed by MediaPipe name).
    output_dir : str
        Directory for output JSON files. Created if it does not exist.
    model : {'COCO', 'BODY_25', 'HALPE_26'}
        Target skeleton model (default ``'COCO'``).
    prefix : str, optional
        Filename prefix prepended to the standard OpenPose naming
        pattern ``{prefix}{frame_idx:012d}_keypoints.json``.

    Returns
    -------
    list of str
        Paths to all created JSON files.

    Raises
    ------
    TypeError
        If *data* is not a dict.
    ValueError
        If *data* has no frames or *model* is unknown.
    """
    from .constants import MP_TO_COCO_17, MP_TO_BODY25, MP_TO_HALPE26

    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    frames = data.get("frames")
    if not frames:
        raise ValueError("No frames in data. Run extract() first.")

    model_upper = model.upper()
    if model_upper == "COCO":
        mapping = MP_TO_COCO_17
        n_keypoints = 17
    elif model_upper == "BODY_25":
        mapping = MP_TO_BODY25
        n_keypoints = 25
    elif model_upper == "HALPE_26":
        mapping = MP_TO_HALPE26
        n_keypoints = 26
    else:
        raise ValueError(
            f"Unknown model {model!r}. Choose from 'COCO', 'BODY_25', 'HALPE_26'."
        )

    width = data.get("meta", {}).get("width", 1920)
    height = data.get("meta", {}).get("height", 1080)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = []

    for frame in frames:
        idx = frame.get("frame_idx", 0)
        lm = frame.get("landmarks", {})
        kps = [0.0] * (n_keypoints * 3)

        # Fill direct mappings
        for mp_name, target_idx in mapping.items():
            pt = lm.get(mp_name)
            if pt is not None:
                x = pt.get("x", 0.0)
                y = pt.get("y", 0.0)
                vis = pt.get("visibility", 0.0)
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    continue
                if y is None or (isinstance(y, float) and np.isnan(y)):
                    continue
                kps[target_idx * 3] = x * width
                kps[target_idx * 3 + 1] = y * height
                kps[target_idx * 3 + 2] = vis

        # Composite keypoints
        def _midpoint(name_a, name_b):
            """Return (x_pixel, y_pixel, confidence) midpoint or None."""
            pa = lm.get(name_a)
            pb = lm.get(name_b)
            if pa is None or pb is None:
                return None
            xa, ya, va = pa.get("x"), pa.get("y"), pa.get("visibility", 0.0)
            xb, yb, vb = pb.get("x"), pb.get("y"), pb.get("visibility", 0.0)
            if xa is None or ya is None or xb is None or yb is None:
                return None
            if isinstance(xa, float) and np.isnan(xa):
                return None
            if isinstance(xb, float) and np.isnan(xb):
                return None
            return (
                (xa + xb) / 2.0 * width,
                (ya + yb) / 2.0 * height,
                min(va, vb),
            )

        if model_upper == "BODY_25":
            # Neck (idx 1) = midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
            mid = _midpoint("LEFT_SHOULDER", "RIGHT_SHOULDER")
            if mid:
                kps[1 * 3], kps[1 * 3 + 1], kps[1 * 3 + 2] = mid
            # MidHip (idx 8) = midpoint(LEFT_HIP, RIGHT_HIP)
            mid = _midpoint("LEFT_HIP", "RIGHT_HIP")
            if mid:
                kps[8 * 3], kps[8 * 3 + 1], kps[8 * 3 + 2] = mid

        elif model_upper == "HALPE_26":
            # Head (idx 17) = NOSE
            nose = lm.get("NOSE")
            if nose and nose.get("x") is not None:
                nx = nose.get("x", 0.0)
                ny = nose.get("y", 0.0)
                nv = nose.get("visibility", 0.0)
                if not (isinstance(nx, float) and np.isnan(nx)):
                    kps[17 * 3] = nx * width
                    kps[17 * 3 + 1] = ny * height
                    kps[17 * 3 + 2] = nv
            # Neck (idx 18) = midpoint shoulders
            mid = _midpoint("LEFT_SHOULDER", "RIGHT_SHOULDER")
            if mid:
                kps[18 * 3], kps[18 * 3 + 1], kps[18 * 3 + 2] = mid
            # Hip (idx 19) = midpoint hips
            mid = _midpoint("LEFT_HIP", "RIGHT_HIP")
            if mid:
                kps[19 * 3], kps[19 * 3 + 1], kps[19 * 3 + 2] = mid

        openpose_dict = {
            "version": 1.1,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": kps,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                }
            ],
        }

        fname = f"{prefix}{idx:012d}_keypoints.json"
        fpath = out / fname
        with open(fpath, "w") as f:
            json.dump(openpose_dict, f, indent=2)
        created.append(str(fpath))

    logger.info(
        f"Exported {len(created)} OpenPose JSON files ({model}) to {output_dir}"
    )
    return created
