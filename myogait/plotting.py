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
