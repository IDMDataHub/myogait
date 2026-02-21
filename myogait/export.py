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

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

    # OpenSim column mapping (myogait name → OpenSim name)
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

    # Build data rows
    rows = []
    for af in aframes:
        row = {"time": af.get("frame_idx", 0) / fps}
        for myogait_key, osim_key in col_map.items():
            val = af.get(myogait_key)
            row[osim_key] = val if val is not None else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    columns = ["time"] + list(col_map.values())

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
) -> str:
    """Export landmark positions to OpenSim .trc (marker) format.

    The ``.trc`` file contains 3D marker positions (x, y, z) over time.
    Since myogait operates in 2D, the z coordinate is set to 0.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    output_path : str
        Output ``.trc`` file path.
    marker_names : list of str, optional
        Landmark names to export. Defaults to major joints.
    units : {'m', 'mm'}
        Coordinate units (default ``'m'``).

    Returns
    -------
    str
        Path to the created file.

    Raises
    ------
    ValueError
        If *data* has no frames.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    frames = data.get("frames")
    if not frames:
        raise ValueError("No frames in data. Run extract() first.")

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

    n_markers = len(marker_names)
    n_frames = len(frames)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Header line 1
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(path.name))
        # Header line 2
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\t{units}\t{fps:.2f}\t1\t{n_frames}\n")

        # Column headers
        header1 = ["Frame#", "Time"]
        header2 = ["", ""]
        for name in marker_names:
            header1.extend([name, "", ""])
            header2.extend(["X1", "Y1", "Z1"])
        f.write("\t".join(header1) + "\n")
        f.write("\t".join(header2) + "\n")
        f.write("\n")

        # Data rows
        for frame in frames:
            idx = frame["frame_idx"]
            time = idx / fps
            vals = [str(idx + 1), f"{time:.6f}"]
            lm = frame.get("landmarks", {})
            for name in marker_names:
                pt = lm.get(name, {})
                x = pt.get("x", 0.0) if pt else 0.0
                y = pt.get("y", 0.0) if pt else 0.0
                z = 0.0
                # Convert from normalized to meters if needed
                if units == "m" and x is not None and 0 <= x <= 1:
                    w = data.get("meta", {}).get("width", 1920)
                    h = data.get("meta", {}).get("height", 1080)
                    x = x * w / 1000.0  # approximate mm → m
                    y = y * h / 1000.0
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
