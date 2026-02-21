"""Gait visualization with matplotlib.

Produces publication-quality figures for joint angles, gait cycles,
events, phase planes, and summary panels. All functions return
``matplotlib.figure.Figure`` objects for saving or display.

Functions
---------
plot_angles
    Plot joint angle time series.
plot_cycles
    Plot normalized gait cycles (mean +/- SD).
plot_events
    Plot gait event timeline.
plot_summary
    Multi-panel summary figure.
plot_phase_plane
    Joint angle vs angular velocity phase diagram.
plot_normative_comparison
    Plot patient cycles overlaid on normative mean +/- SD bands.
plot_gvs_profile
    Horizontal barplot of GVS per joint (MAP visualization).
plot_quality_dashboard
    Multi-panel data quality dashboard.
plot_longitudinal
    Plot a metric across multiple sessions over time.
plot_arm_swing
    Plot shoulder flexion L/R over gait cycle and amplitude per cycle.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
if matplotlib.get_backend() == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Color scheme
_COLORS = {
    "left": "#2171b5",       # blue
    "right": "#cb181d",      # red
    "left_light": "#6baed6",
    "right_light": "#fc9272",
    "hs": "#1a9850",         # green for HS
    "to": "#d73027",         # red-orange for TO
}

_JOINT_LABELS = {
    "hip_L": "Hip L", "hip_R": "Hip R",
    "knee_L": "Knee L", "knee_R": "Knee R",
    "ankle_L": "Ankle L", "ankle_R": "Ankle R",
    "trunk_angle": "Trunk",
    "hip": "Hip", "knee": "Knee", "ankle": "Ankle", "trunk": "Trunk",
}


def plot_angles(
    data: dict,
    joints: Optional[List[str]] = None,
    events: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot joint angle time series.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` populated.
    joints : list of str, optional
        Joint keys to plot (default: hip, knee, ankle L+R).
    events : bool, optional
        Overlay detected events as vertical lines (default True).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *data* has no computed angles.
    """
    angles = data.get("angles")
    if angles is None:
        raise ValueError("No angles in data. Run compute_angles() first.")

    angle_frames = angles["frames"]
    fps = data.get("meta", {}).get("fps", 30.0)

    if joints is None:
        joints = ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]

    # Group by joint pair (L/R on same subplot)
    pairs = {}
    for j in joints:
        base = j.replace("_L", "").replace("_R", "")
        if base not in pairs:
            pairs[base] = []
        pairs[base].append(j)

    n_plots = len(pairs)
    if figsize is None:
        figsize = (12, 3 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    time = np.array([af["frame_idx"] / fps for af in angle_frames])

    for ax, (base, keys) in zip(axes, pairs.items()):
        for key in keys:
            values = [af.get(key) for af in angle_frames]
            values = [v if v is not None else np.nan for v in values]
            side = "left" if key.endswith("_L") else "right" if key.endswith("_R") else "left"
            color = _COLORS[side]
            label = _JOINT_LABELS.get(key, key)
            ax.plot(time, values, color=color, linewidth=1, label=label)

        # Overlay events
        if events and data.get("events"):
            ev = data["events"]
            for hs in ev.get("left_hs", []) + ev.get("right_hs", []):
                ax.axvline(hs["time"], color=_COLORS["hs"], alpha=0.3, linewidth=0.5)
            for to in ev.get("left_to", []) + ev.get("right_to", []):
                ax.axvline(to["time"], color=_COLORS["to"], alpha=0.3, linewidth=0.5, linestyle="--")

        ax.set_ylabel("Angle (deg)")
        ax.set_title(_JOINT_LABELS.get(base, base))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_cycles(
    cycles: dict,
    side: str = "left",
    joints: Optional[List[str]] = None,
    mode: str = "mean_sd",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot normalized gait cycles (0--100%%).

    Parameters
    ----------
    cycles : dict
        Output of ``segment_cycles()``.
    side : {'left', 'right'}
        Side to plot (default ``'left'``).
    joints : list of str, optional
        Joint names to plot (default: hip, knee, ankle).
    mode : {'mean_sd', 'all'}
        ``'mean_sd'`` for mean + SD band;
        ``'all'`` for individual cycles + mean overlay.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = cycles.get("summary", {}).get(side)
    if summary is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No cycles for {side} side", ha="center", va="center", transform=ax.transAxes)
        return fig

    if joints is None:
        joints = ["hip", "knee", "ankle"]

    n_plots = len(joints)
    if figsize is None:
        figsize = (10, 3 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    x = np.linspace(0, 100, 101)
    color = _COLORS[side]
    color_light = _COLORS[f"{side}_light"]
    side_cycles = [c for c in cycles.get("cycles", []) if c["side"] == side]

    for ax, joint in zip(axes, joints):
        mean = summary.get(f"{joint}_mean")
        std = summary.get(f"{joint}_std")

        if mean is None:
            ax.set_title(f"{_JOINT_LABELS.get(joint, joint)} — no data")
            continue

        mean = np.array(mean)
        std = np.array(std)

        if mode == "all":
            # Individual cycles
            for c in side_cycles:
                vals = c.get("angles_normalized", {}).get(joint)
                if vals:
                    ax.plot(x, vals, color=color_light, linewidth=0.8, alpha=0.4)
            # Mean on top
            ax.plot(x, mean, color=color, linewidth=2.5, label=f"Mean (n={summary['n_cycles']})")
        else:
            # Mean ± SD
            ax.plot(x, mean, color=color, linewidth=2, label=f"Mean (n={summary['n_cycles']})")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, label="SD")

        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"{_JOINT_LABELS.get(joint, joint)} — {side}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("% Gait Cycle")
    fig.tight_layout()
    return fig


def plot_events(
    data: dict,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot gait event timeline.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events`` populated.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *data* has no events.
    """
    events = data.get("events")
    if events is None:
        raise ValueError("No events in data. Run detect_events() first.")

    if figsize is None:
        figsize = (12, 3)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = {"left_hs": 1.0, "right_hs": 0.6, "left_to": -0.6, "right_to": -1.0}
    markers = {"left_hs": "^", "right_hs": "^", "left_to": "v", "right_to": "v"}
    colors = {
        "left_hs": _COLORS["left"], "right_hs": _COLORS["right"],
        "left_to": _COLORS["left_light"], "right_to": _COLORS["right_light"],
    }
    labels = {"left_hs": "HS Left", "right_hs": "HS Right", "left_to": "TO Left", "right_to": "TO Right"}

    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        evts = events.get(key, [])
        times = [e["time"] for e in evts]
        y = [y_positions[key]] * len(times)
        ax.scatter(
            times, y,
            marker=markers[key],
            c=colors[key],
            s=80,
            label=f"{labels[key]} ({len(evts)})",
            zorder=3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_yticks([1.0, 0.6, -0.6, -1.0])
    ax.set_yticklabels(["HS L", "HS R", "TO L", "TO R"])
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Gait Events")
    fig.tight_layout()
    return fig


def plot_summary(
    data: dict,
    cycles: dict,
    stats: Optional[dict] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot comprehensive summary with angles, events, and cycles.

    Multi-panel figure with raw angles (row 1), event timeline and
    ankle angles (row 2), normalized cycles per side (row 3), and
    statistics text (row 4).

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and ``events``.
    cycles : dict
        Output of ``segment_cycles()``.
    stats : dict, optional
        Output of ``analyze_gait()``. Adds a statistics text panel.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (14, 16)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)
    time = np.array([af["frame_idx"] / fps for af in angle_frames])

    # Row 1: Raw angles (hip, knee, ankle)
    for col, joint in enumerate(["hip", "knee"]):
        ax = fig.add_subplot(gs[0, col])
        for side_suffix, side_name in [("_L", "left"), ("_R", "right")]:
            key = f"{joint}{side_suffix}"
            values = [af.get(key) for af in angle_frames]
            values = [v if v is not None else np.nan for v in values]
            ax.plot(time, values, color=_COLORS[side_name], linewidth=0.8, label=f"{side_name.capitalize()}")
        # Events overlay
        if data.get("events"):
            for hs in data["events"].get("left_hs", []) + data["events"].get("right_hs", []):
                ax.axvline(hs["time"], color=_COLORS["hs"], alpha=0.2, linewidth=0.5)
        ax.set_title(f"{_JOINT_LABELS.get(joint, joint)} Angle")
        ax.set_ylabel("deg")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1 continued: ankle
    ax = fig.add_subplot(gs[1, 0])
    for side_suffix, side_name in [("_L", "left"), ("_R", "right")]:
        values = [af.get(f"ankle{side_suffix}") for af in angle_frames]
        values = [v if v is not None else np.nan for v in values]
        ax.plot(time, values, color=_COLORS[side_name], linewidth=0.8, label=f"{side_name.capitalize()}")
    if data.get("events"):
        for hs in data["events"].get("left_hs", []) + data["events"].get("right_hs", []):
            ax.axvline(hs["time"], color=_COLORS["hs"], alpha=0.2, linewidth=0.5)
    ax.set_title("Ankle Angle")
    ax.set_ylabel("deg")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Events timeline
    ax = fig.add_subplot(gs[1, 1])
    if data.get("events"):
        events = data["events"]
        y_pos = {"left_hs": 1.0, "right_hs": 0.6, "left_to": -0.6, "right_to": -1.0}
        mkr = {"left_hs": "^", "right_hs": "^", "left_to": "v", "right_to": "v"}
        clr = {"left_hs": _COLORS["left"], "right_hs": _COLORS["right"],
               "left_to": _COLORS["left_light"], "right_to": _COLORS["right_light"]}
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            evts = events.get(key, [])
            ax.scatter([e["time"] for e in evts], [y_pos[key]] * len(evts),
                       marker=mkr[key], c=clr[key], s=50)
        ax.set_yticks([1.0, 0.6, -0.6, -1.0])
        ax.set_yticklabels(["HS L", "HS R", "TO L", "TO R"], fontsize=8)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Time (s)")
        ax.set_title("Gait Events")
        ax.grid(True, axis="x", alpha=0.3)

    # Row 3: Normalized cycles (left and right)
    x_pct = np.linspace(0, 100, 101)
    for col, side in enumerate(["left", "right"]):
        ax = fig.add_subplot(gs[2, col])
        summary = cycles.get("summary", {}).get(side)
        if summary is None:
            ax.text(0.5, 0.5, f"No {side} cycles", ha="center", va="center", transform=ax.transAxes)
            continue
        color = _COLORS[side]
        for joint in ["hip", "knee", "ankle"]:
            mean = summary.get(f"{joint}_mean")
            std = summary.get(f"{joint}_std")
            if mean is None:
                continue
            mean = np.array(mean)
            std = np.array(std)
            ax.plot(x_pct, mean, linewidth=1.5, label=_JOINT_LABELS.get(joint, joint))
            ax.fill_between(x_pct, mean - std, mean + std, alpha=0.1)
        ax.set_title(f"Normalized Cycles — {side.capitalize()} (n={summary.get('n_cycles', 0)})")
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("deg")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 4: Stats text
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    if stats:
        st = stats.get("spatiotemporal", {})
        sym = stats.get("symmetry", {})
        var = stats.get("variability", {})
        flags = stats.get("pathology_flags", [])

        lines = [
            f"Cadence: {st.get('cadence_steps_per_min', 'N/A')} steps/min    "
            f"Stride: {st.get('stride_time_mean_s', 'N/A')} +/- {st.get('stride_time_std_s', 'N/A')} s    "
            f"Step: {st.get('step_time_mean_s', 'N/A')} +/- {st.get('step_time_std_s', 'N/A')} s",

            f"Stance L: {st.get('stance_pct_left', 'N/A')}%    "
            f"Stance R: {st.get('stance_pct_right', 'N/A')}%    "
            f"Double support: {st.get('double_support_pct', 'N/A')}%",

            f"Symmetry — Hip ROM: {sym.get('hip_rom_si', 'N/A')}%    "
            f"Knee ROM: {sym.get('knee_rom_si', 'N/A')}%    "
            f"Ankle ROM: {sym.get('ankle_rom_si', 'N/A')}%    "
            f"Overall: {sym.get('overall_si', 'N/A')}%",

            f"Variability — Cycle CV: {var.get('cycle_duration_cv', 'N/A')}%    "
            f"Stance CV: {var.get('stance_pct_cv', 'N/A')}%",
        ]

        if flags:
            lines.append("Flags: " + " | ".join(flags))

        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    else:
        ax.text(0.5, 0.5, "No statistics (run analyze_gait())", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")

    fig.suptitle(
        data.get("meta", {}).get("video_path", "Gait Analysis"),
        fontsize=12, fontweight="bold", y=0.99,
    )
    return fig


def plot_phase_plane(
    data: dict,
    joint: str = "knee_L",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot phase plane diagram (angle vs angular velocity).

    Visualizes the relationship between joint angle and its rate
    of change, useful for identifying gait pattern stability and
    abnormalities. Points are color-coded by time.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` populated.
    joint : str, optional
        Joint key (default ``"knee_L"``). Examples: ``"hip_R"``,
        ``"ankle_L"``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *data* has no computed angles.
    """
    angles = data.get("angles")
    if angles is None:
        raise ValueError("No angles in data. Run compute_angles() first.")

    fps = data.get("meta", {}).get("fps", 30.0)
    angle_frames = angles["frames"]

    values = [af.get(joint) for af in angle_frames]
    values = [v if v is not None else np.nan for v in values]
    arr = np.array(values, dtype=float)

    # Compute angular velocity
    velocity = np.gradient(arr, 1.0 / fps)
    velocity[np.isnan(arr)] = np.nan

    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Color by time
    valid = ~np.isnan(arr) & ~np.isnan(velocity)
    time_idx = np.arange(len(arr))

    scatter = ax.scatter(
        arr[valid], velocity[valid],
        c=time_idx[valid], cmap="viridis",
        s=8, alpha=0.6,
    )
    ax.plot(arr[valid], velocity[valid], color="gray", linewidth=0.3, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Frame")
    ax.set_xlabel(f"{_JOINT_LABELS.get(joint, joint)} Angle (deg)")
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.set_title(f"Phase Plane — {_JOINT_LABELS.get(joint, joint)}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_normative_comparison(
    data: dict,
    cycles: dict,
    stratum: str = "adult",
    joints: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot patient normalized gait cycles overlaid on normative bands.

    One subplot per joint shows the patient's mean curve (blue) against
    the normative mean (black dashed) with +/-1 SD (light gray) and
    +/-2 SD (lighter gray) bands.  X-axis: 0--100%% gait cycle.

    Parameters
    ----------
    data : dict
        Pivot JSON dict (used for metadata).
    cycles : dict
        Output of ``segment_cycles()`` with ``summary`` populated.
    stratum : str
        Normative stratum (default ``'adult'``).
    joints : list of str, optional
        Joint names to plot (default: hip, knee, ankle, trunk).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .normative import get_normative_band

    if joints is None:
        joints = ["hip", "knee", "ankle", "trunk"]

    n_plots = len(joints)
    if figsize is None:
        figsize = (12, 3 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    x = np.linspace(0, 100, 101)
    summary = cycles.get("summary", {})

    for ax, joint in zip(axes, joints):
        # Draw normative bands
        band_2sd = get_normative_band(joint, stratum, n_sd=2.0)
        band_1sd = get_normative_band(joint, stratum, n_sd=1.0)
        norm_mean = np.array(band_1sd["mean"])

        ax.fill_between(
            x,
            band_2sd["lower"], band_2sd["upper"],
            color="#e0e0e0", alpha=0.5, label="+/-2 SD",
        )
        ax.fill_between(
            x,
            band_1sd["lower"], band_1sd["upper"],
            color="#c0c0c0", alpha=0.5, label="+/-1 SD",
        )
        ax.plot(x, norm_mean, color="black", linestyle="--", linewidth=1.5,
                label="Normative mean")

        # Overlay patient curves for each side
        for side in ("left", "right"):
            side_summary = summary.get(side)
            if side_summary is None:
                continue
            patient_mean = side_summary.get(f"{joint}_mean")
            if patient_mean is None:
                continue
            patient_mean = np.array(patient_mean)
            color = _COLORS["left"] if side == "left" else _COLORS["right"]
            label_side = side.capitalize()
            ax.plot(x, patient_mean, color=color, linewidth=2,
                    label=f"Patient ({label_side})")

        ax.set_ylabel("Angle (deg)")
        ax.set_title(
            f"{_JOINT_LABELS.get(joint, joint.capitalize())} "
            f"— vs Normative ({stratum})"
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("% Gait Cycle")
    fig.tight_layout()
    return fig


def plot_gvs_profile(
    cycles: dict,
    stratum: str = "adult",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Horizontal barplot of GVS per joint (Movement Analysis Profile).

    Left-side bars extend to the left, right-side bars extend to
    the right.  Bars are colored green (<5 deg), orange (5--10 deg),
    or red (>10 deg).  A vertical line marks the overall GPS-2D.

    Parameters
    ----------
    cycles : dict
        Output of ``segment_cycles()`` with ``summary`` populated.
    stratum : str
        Normative stratum (default ``'adult'``).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .scores import movement_analysis_profile

    map_data = movement_analysis_profile(cycles, stratum)
    joints = map_data["joints"]
    left_vals = map_data["left"]
    right_vals = map_data["right"]
    gps_2d = map_data["gps_2d"]

    n_joints = len(joints)
    if figsize is None:
        figsize = (10, max(4, n_joints * 1.2))

    fig, ax = plt.subplots(figsize=figsize)

    def _bar_color(val):
        """Return color based on GVS magnitude."""
        if val is None:
            return "#999999"
        if val < 5:
            return "#2ca02c"   # green
        elif val <= 10:
            return "#ff7f0e"   # orange
        else:
            return "#d62728"   # red

    y_positions = np.arange(n_joints)

    # Left side: bars extend to the left (negative direction)
    for i, (joint, val) in enumerate(zip(joints, left_vals)):
        bar_val = -(val if val is not None else 0)
        ax.barh(
            y_positions[i] + 0.15, bar_val, height=0.3,
            color=_bar_color(val), edgecolor="white", linewidth=0.5,
        )
        if val is not None:
            ax.text(bar_val - 0.3, y_positions[i] + 0.15, f"{val:.1f}",
                    ha="right", va="center", fontsize=8)

    # Right side: bars extend to the right (positive direction)
    for i, (joint, val) in enumerate(zip(joints, right_vals)):
        bar_val = val if val is not None else 0
        ax.barh(
            y_positions[i] - 0.15, bar_val, height=0.3,
            color=_bar_color(val), edgecolor="white", linewidth=0.5,
        )
        if val is not None:
            ax.text(bar_val + 0.3, y_positions[i] - 0.15, f"{val:.1f}",
                    ha="left", va="center", fontsize=8)

    # GPS-2D vertical line
    if gps_2d is not None:
        ax.axvline(gps_2d, color="black", linestyle="-", linewidth=1.5,
                   label=f"GPS-2D = {gps_2d:.1f} deg")
        ax.axvline(-gps_2d, color="black", linestyle="-", linewidth=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_JOINT_LABELS.get(j, j.capitalize()) for j in joints])
    ax.set_xlabel("GVS (deg)")
    ax.set_title("Movement Analysis Profile (MAP)")
    ax.axvline(0, color="gray", linewidth=0.5)

    # Legend for sides
    ax.text(0.02, 0.98, "Left", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            color=_COLORS["left"])
    ax.text(0.98, 0.98, "Right", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right",
            color=_COLORS["right"])

    if gps_2d is not None:
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_quality_dashboard(
    data: dict,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Multi-panel data quality dashboard.

    Four panels show:
      1. Detection rate per landmark over time (heatmap-like).
      2. Per-frame confidence score (line plot).
      3. Gap locations (scatter of missing frames).
      4. Overall quality score (text display).

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .normalize import data_quality_score

    quality = data_quality_score(data)
    frames = data.get("frames", [])
    n_frames = len(frames)

    if figsize is None:
        figsize = (14, 10)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Collect landmark names from first frame with landmarks
    landmark_names = []
    for f in frames:
        lm = f.get("landmarks", {})
        if lm:
            landmark_names = sorted(lm.keys())
            break

    # Panel 1: Detection rate per landmark over time (heatmap-like)
    if landmark_names and n_frames > 0:
        n_bins = min(50, n_frames)
        bin_size = max(1, n_frames // n_bins)
        # Select key landmarks for readability
        key_landmarks = [
            name for name in landmark_names
            if any(part in name for part in
                   ["HIP", "KNEE", "ANKLE", "SHOULDER", "HEEL"])
        ]
        if not key_landmarks:
            key_landmarks = landmark_names[:10]

        detection_matrix = np.zeros((len(key_landmarks), n_bins))
        for bi in range(n_bins):
            start = bi * bin_size
            end = min(start + bin_size, n_frames)
            for li, lm_name in enumerate(key_landmarks):
                detected = 0
                total = 0
                for fi in range(start, end):
                    lm = frames[fi].get("landmarks", {}).get(lm_name, {})
                    xv = lm.get("x")
                    yv = lm.get("y")
                    total += 1
                    if (xv is not None and yv is not None
                            and not np.isnan(xv) and not np.isnan(yv)):
                        detected += 1
                detection_matrix[li, bi] = (
                    detected / total if total > 0 else 0
                )

        im = ax1.imshow(
            detection_matrix, aspect="auto", cmap="RdYlGn",
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax1.set_yticks(range(len(key_landmarks)))
        ax1.set_yticklabels(
            [n.replace("_", " ") for n in key_landmarks], fontsize=6,
        )
        ax1.set_xlabel("Time bin")
        ax1.set_title("Detection Rate per Landmark")
        plt.colorbar(im, ax=ax1, label="Detection rate", shrink=0.8)
    else:
        ax1.text(
            0.5, 0.5, "No frame data",
            ha="center", va="center", transform=ax1.transAxes,
        )
        ax1.set_title("Detection Rate per Landmark")

    # Panel 2: Confidence score per frame (line plot)
    if n_frames > 0:
        confidences = [f.get("confidence", 0.0) or 0.0 for f in frames]
        frame_indices = list(range(n_frames))
        ax2.plot(
            frame_indices, confidences,
            color=_COLORS["left"], linewidth=0.8, alpha=0.8,
        )
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Per-frame Confidence")
        ax2.grid(True, alpha=0.3)
        mean_conf = float(np.mean(confidences))
        ax2.axhline(
            mean_conf, color=_COLORS["right"], linestyle="--",
            linewidth=1, label=f"Mean = {mean_conf:.2f}",
        )
        ax2.legend(loc="lower right", fontsize=8)
    else:
        ax2.text(
            0.5, 0.5, "No frame data",
            ha="center", va="center", transform=ax2.transAxes,
        )
        ax2.set_title("Per-frame Confidence")

    # Panel 3: Gap locations (scatter)
    gap_frames_list = []
    if n_frames > 0:
        for fi, f in enumerate(frames):
            lm = f.get("landmarks", {})
            all_nan = True
            for name, coords in lm.items():
                xv = coords.get("x")
                yv = coords.get("y")
                if (xv is not None and yv is not None
                        and not np.isnan(xv) and not np.isnan(yv)):
                    all_nan = False
                    break
            if all_nan and len(lm) > 0:
                gap_frames_list.append(fi)

        if gap_frames_list:
            ax3.scatter(
                gap_frames_list, [1] * len(gap_frames_list),
                marker="|", color=_COLORS["to"], s=50, alpha=0.8,
            )
            ax3.set_ylim(0, 2)
        else:
            ax3.text(
                0.5, 0.5, "No gaps detected",
                ha="center", va="center", transform=ax3.transAxes,
                fontsize=10, color="green",
            )
        ax3.set_xlabel("Frame")
        ax3.set_title(f"Gap Locations ({len(gap_frames_list)} gap frames)")
        ax3.set_yticks([])
        ax3.grid(True, axis="x", alpha=0.3)
    else:
        ax3.text(
            0.5, 0.5, "No frame data",
            ha="center", va="center", transform=ax3.transAxes,
        )
        ax3.set_title("Gap Locations")

    # Panel 4: Overall quality score (gauge/text)
    ax4.axis("off")
    overall = quality.get("overall_score", 0.0)

    if overall >= 80:
        score_color = "#2ca02c"
        score_label = "GOOD"
    elif overall >= 50:
        score_color = "#ff7f0e"
        score_label = "FAIR"
    else:
        score_color = "#d62728"
        score_label = "POOR"

    ax4.text(
        0.5, 0.65, f"{overall:.0f}",
        ha="center", va="center", transform=ax4.transAxes,
        fontsize=48, fontweight="bold", color=score_color,
    )
    ax4.text(
        0.5, 0.45, f"Quality: {score_label}",
        ha="center", va="center", transform=ax4.transAxes,
        fontsize=16, color=score_color,
    )

    details = (
        f"Detection rate: {quality.get('detection_rate', 0):.1%}\n"
        f"Mean confidence: {quality.get('mean_confidence', 0):.2f}\n"
        f"Gap percentage: {quality.get('gap_pct', 0):.1%}\n"
        f"Jitter score: {quality.get('jitter_score', 0):.4f}"
    )
    ax4.text(
        0.5, 0.15, details,
        ha="center", va="center", transform=ax4.transAxes,
        fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax4.set_title("Overall Quality Score")

    fig.suptitle(
        "Data Quality Dashboard",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.tight_layout()
    return fig


def plot_longitudinal(
    sessions: List[Dict],
    metric: str = "gps_2d_overall",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot a gait metric across multiple sessions over time.

    Parameters
    ----------
    sessions : list of dict
        Each dict must have ``'date'`` (str, e.g. ``'2024-01-15'``)
        and ``'stats'`` (output of ``analyze_gait()``).
        Supported metrics:
        - ``'gps_2d_overall'``, ``'gps_2d_left'``, ``'gps_2d_right'``
        - ``'cadence'`` (from ``spatiotemporal.cadence_steps_per_min``)
        - ``'symmetry'`` (from ``symmetry.overall_si``)
    metric : str
        Metric key (default ``'gps_2d_overall'``).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (10, 5)

    fig, ax = plt.subplots(figsize=figsize)

    dates = []
    values = []
    errors = []

    _metric_extractors = {
        "gps_2d_overall": lambda s: s.get("gps_2d_overall"),
        "gps_2d_left": lambda s: s.get("gps_2d_left"),
        "gps_2d_right": lambda s: s.get("gps_2d_right"),
        "cadence": lambda s: s.get(
            "spatiotemporal", {}
        ).get("cadence_steps_per_min"),
        "symmetry": lambda s: s.get("symmetry", {}).get("overall_si"),
    }

    extractor = _metric_extractors.get(metric)
    if extractor is None:
        extractor = lambda s: s.get(metric)  # noqa: E731

    for session in sessions:
        date = session.get("date", "")
        stats = session.get("stats", {})
        val = extractor(stats)
        err = session.get("error")

        dates.append(date)
        values.append(val)
        errors.append(err)

    # Filter out None values for plotting
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if not valid_idx:
        ax.text(
            0.5, 0.5, f"No data for metric '{metric}'",
            ha="center", va="center", transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    plot_dates = [dates[i] for i in valid_idx]
    plot_values = [values[i] for i in valid_idx]
    plot_errors = [errors[i] for i in valid_idx]

    has_errors = any(e is not None for e in plot_errors)
    if has_errors:
        yerr = [e if e is not None else 0 for e in plot_errors]
        ax.errorbar(
            range(len(plot_dates)), plot_values, yerr=yerr,
            marker="o", color=_COLORS["left"], linewidth=2,
            markersize=8, capsize=4, capthick=1.5,
        )
    else:
        ax.plot(
            range(len(plot_dates)), plot_values,
            marker="o", color=_COLORS["left"], linewidth=2,
            markersize=8,
        )

    ax.set_xticks(range(len(plot_dates)))
    ax.set_xticklabels(plot_dates, rotation=45, ha="right", fontsize=9)

    _metric_labels = {
        "gps_2d_overall": "GPS-2D Overall (deg)",
        "gps_2d_left": "GPS-2D Left (deg)",
        "gps_2d_right": "GPS-2D Right (deg)",
        "cadence": "Cadence (steps/min)",
        "symmetry": "Symmetry Index (%)",
    }
    ylabel = _metric_labels.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Session")
    ax.set_title(f"Longitudinal Trend: {ylabel}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_arm_swing(
    data: dict,
    cycles: dict,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot arm swing kinematics over gait cycles.

    Top subplot: left and right shoulder flexion (or wrist x as
    fallback) over the normalized gait cycle (mean +/- SD).
    Bottom subplot: per-cycle amplitude bar chart comparing L vs R.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and ``frames`` populated.
    cycles : dict
        Output of ``segment_cycles()``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (12, 8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])
    frames = data.get("frames", [])
    cycle_list = cycles.get("cycles", [])

    # Determine data source: shoulder angles or wrist x fallback
    has_shoulder = False
    if angle_frames:
        sample_keys = set()
        for af in angle_frames[:5]:
            sample_keys.update(af.keys())
        has_shoulder = (
            "shoulder_flex_L" in sample_keys
            and "shoulder_flex_R" in sample_keys
        )

    x_pct = np.linspace(0, 100, 101)

    if has_shoulder:
        data_label = "Shoulder Flexion"
        left_curves = []
        right_curves = []
        for c in cycle_list:
            start_f = c["start_frame"]
            end_f = c["end_frame"]
            cycle_afs = [
                af for af in angle_frames
                if start_f <= af.get("frame_idx", -1) <= end_f
            ]
            if len(cycle_afs) < 5:
                continue
            for key_sfx, clist in [("_L", left_curves), ("_R", right_curves)]:
                vals = [
                    af.get(f"shoulder_flex{key_sfx}") for af in cycle_afs
                ]
                vals = [v if v is not None else np.nan for v in vals]
                if all(np.isnan(v) for v in vals):
                    continue
                arr = np.array(vals, dtype=float)
                valid = ~np.isnan(arr)
                if valid.sum() < 3:
                    continue
                x_old = np.linspace(0, 100, len(arr))
                arr_interp = np.interp(x_pct, x_old[valid], arr[valid])
                clist.append(arr_interp)
    else:
        # Fallback to wrist x displacement
        data_label = "Wrist X Position (norm.)"
        left_curves = []
        right_curves = []
        for c in cycle_list:
            start_f = c["start_frame"]
            end_f = c["end_frame"]
            for wrist_name, clist in [
                ("LEFT_WRIST", left_curves),
                ("RIGHT_WRIST", right_curves),
            ]:
                vals = []
                for fi in range(start_f, min(end_f + 1, len(frames))):
                    lm = frames[fi].get("landmarks", {}).get(wrist_name, {})
                    xv = lm.get("x")
                    vals.append(xv if xv is not None else np.nan)
                if len(vals) < 5:
                    continue
                arr = np.array(vals, dtype=float)
                valid = ~np.isnan(arr)
                if valid.sum() < 3:
                    continue
                arr = arr - np.nanmean(arr)
                x_old = np.linspace(0, 100, len(arr))
                arr_interp = np.interp(x_pct, x_old[valid], arr[valid])
                clist.append(arr_interp * 100)

    # Subplot 1: Mean +/- SD curves for L and R
    for curves, side, color in [
        (left_curves, "Left", _COLORS["left"]),
        (right_curves, "Right", _COLORS["right"]),
    ]:
        if not curves:
            continue
        curves_arr = np.array(curves)
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        ax1.plot(
            x_pct, mean_curve, color=color, linewidth=2,
            label=f"{side} (n={len(curves)})",
        )
        ax1.fill_between(
            x_pct, mean_curve - std_curve, mean_curve + std_curve,
            color=color, alpha=0.15,
        )

    if not left_curves and not right_curves:
        ax1.text(
            0.5, 0.5, "No arm swing data available",
            ha="center", va="center", transform=ax1.transAxes,
        )

    unit_label = " (deg)" if has_shoulder else ""
    ax1.set_xlabel("% Gait Cycle")
    ax1.set_ylabel(data_label + unit_label)
    ax1.set_title(f"Arm Swing — {data_label}")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Per-cycle amplitude bar chart
    left_amps = [float(np.ptp(c)) for c in left_curves] if left_curves else []
    right_amps = (
        [float(np.ptp(c)) for c in right_curves] if right_curves else []
    )

    n_bars = max(len(left_amps), len(right_amps))
    if n_bars > 0:
        bar_x = np.arange(n_bars)
        bar_width = 0.35
        if left_amps:
            left_padded = left_amps + [0] * (n_bars - len(left_amps))
            ax2.bar(
                bar_x - bar_width / 2, left_padded, bar_width,
                color=_COLORS["left"], alpha=0.7, label="Left",
            )
        if right_amps:
            right_padded = right_amps + [0] * (n_bars - len(right_amps))
            ax2.bar(
                bar_x + bar_width / 2, right_padded, bar_width,
                color=_COLORS["right"], alpha=0.7, label="Right",
            )
        ax2.set_xlabel("Cycle")
        amp_unit = " (deg)" if has_shoulder else " (norm.)"
        ax2.set_ylabel(f"Amplitude{amp_unit}")
        ax2.set_xticks(bar_x)
        ax2.set_xticklabels([f"C{i+1}" for i in range(n_bars)], fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(
            0.5, 0.5, "No amplitude data",
            ha="center", va="center", transform=ax2.transAxes,
        )

    ax2.set_title("Arm Swing Amplitude per Cycle")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
