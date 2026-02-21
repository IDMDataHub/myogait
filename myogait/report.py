"""PDF report generation for gait analysis.

Generates a multi-page clinical PDF report with:

- Page 1: Overview -- angle time series (hip, knee, ankle) with events.
- Page 2: Bilateral comparison -- mean +/- SD overlaid L vs R.
- Page 3: Clinical statistics -- symmetry and variability bar charts.
- Page 4: Trunk and pelvis analysis with pathology annotations.
- Pages 5-6: Normalized cycles per side (all cycles + mean +/- SD).
- Page 7: Detailed text summary of all metrics.

Functions
---------
generate_report
    Generate the full multi-page PDF report.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
if matplotlib.get_backend() == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

_FIG_SIZE = (11.69, 8.27)  # A4 landscape
_DPI = 150


# ── Small page functions ────────────────────────────────────────────


def _page_overview(pdf, data: dict, cycles: dict):
    """Page 1: angle time series (hip, knee, ankle) with event markers."""
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)
    events = data.get("events", {})

    time = np.array([af["frame_idx"] / fps for af in angle_frames])

    fig, axes = plt.subplots(3, 2, figsize=_FIG_SIZE)
    fig.suptitle("Vue d'ensemble — Angles articulaires", fontsize=14, fontweight="bold", y=0.99)

    joint_pairs = [
        ("hip", "Hanche", "Flexion (+) / Extension (-)"),
        ("knee", "Genou", "Flexion (+)"),
        ("ankle", "Cheville", "Dorsiflexion (+) / Plantarflexion (-)"),
    ]

    for row, (joint, label, ylabel) in enumerate(joint_pairs):
        for col, (side, suffix, color) in enumerate([("Gauche", "_L", "#2171b5"), ("Droite", "_R", "#cb181d")]):
            ax = axes[row, col]
            key = f"{joint}{suffix}"
            vals = [af.get(key) for af in angle_frames]
            vals = [v if v is not None else np.nan for v in vals]
            ax.plot(time, vals, color=color, linewidth=1, label=side)

            # Event markers
            for hs in events.get(f"{'left' if col == 0 else 'right'}_hs", []):
                ax.axvline(hs["time"], color="green", alpha=0.3, linewidth=0.5)
            for to in events.get(f"{'left' if col == 0 else 'right'}_to", []):
                ax.axvline(to["time"], color="orange", alpha=0.3, linewidth=0.5, linestyle="--")

            # ROM annotation
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                rom = max(valid) - min(valid)
                ax.set_title(f"{label} {side}  (ROM: {rom:.1f}°)", fontsize=10, fontweight="bold")
            else:
                ax.set_title(f"{label} {side}", fontsize=10)

            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

    axes[2, 0].set_xlabel("Temps (s)")
    axes[2, 1].set_xlabel("Temps (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_bilateral(pdf, cycles: dict, data: dict):
    """Page 2: bilateral comparison — mean ± SD overlaid L vs R."""
    summary = cycles.get("summary", {})
    left = summary.get("left")
    right = summary.get("right")

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE)
    fig.suptitle("Comparaison bilatérale", fontsize=14, fontweight="bold", y=0.99)

    x = np.linspace(0, 100, 101)
    joints = [("hip", "Hanche", axes[0, 0]), ("knee", "Genou", axes[0, 1]), ("ankle", "Cheville", axes[1, 0])]

    for joint, label, ax in joints:
        if left and f"{joint}_mean" in left:
            m = np.array(left[f"{joint}_mean"])
            s = np.array(left[f"{joint}_std"])
            ax.plot(x, m, color="#2171b5", linewidth=2, label="Gauche")
            ax.fill_between(x, m - s, m + s, color="#2171b5", alpha=0.15)

        if right and f"{joint}_mean" in right:
            m = np.array(right[f"{joint}_mean"])
            s = np.array(right[f"{joint}_std"])
            ax.plot(x, m, color="#cb181d", linewidth=2, label="Droite")
            ax.fill_between(x, m - s, m + s, color="#cb181d", alpha=0.15)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("% cycle de marche")
        ax.set_ylabel("Angle (°)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

    # Info box
    ax_info = axes[1, 1]
    ax_info.axis("off")
    meta = data.get("meta", {})
    n_left = left.get("n_cycles", 0) if left else 0
    n_right = right.get("n_cycles", 0) if right else 0
    info = (
        f"Source: {Path(meta.get('video_path', '?')).name}\n"
        f"FPS: {meta.get('fps', '?'):.1f}\n"
        f"Cycles G: {n_left}  D: {n_right}\n"
        f"Modèle: {data.get('extraction', {}).get('model', '?')}\n"
        f"Correction: {data.get('angles', {}).get('correction_factor', '?')}\n"
        f"Méthode angles: {data.get('angles', {}).get('method', '?')}\n"
        f"Détection events: {data.get('events', {}).get('method', '?')}"
    )
    ax_info.text(0.1, 0.9, info, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_statistics(pdf, stats: dict):
    """Page 3: clinical statistics — bar charts + text."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.suptitle("Statistiques cliniques", fontsize=14, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    sym = stats.get("symmetry", {})
    var = stats.get("variability", {})
    st = stats.get("spatiotemporal", {})
    flags = stats.get("pathology_flags", [])

    # 1: Symmetry bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    si_keys = [("hip_rom_si", "Hanche ROM"), ("knee_rom_si", "Genou ROM"),
               ("ankle_rom_si", "Cheville ROM"), ("step_time_si", "Temps de pas"),
               ("stance_time_si", "Temps stance")]
    si_vals = [sym.get(k, 0) for k, _ in si_keys]
    si_labels = [l for _, l in si_keys]
    colors = ["green" if v < 10 else "orange" if v < 20 else "red" for v in si_vals]
    y_pos = np.arange(len(si_labels))
    ax1.barh(y_pos, si_vals, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(si_labels, fontsize=8)
    ax1.set_xlabel("SI (%)")
    ax1.set_title("Indices de symétrie", fontsize=10, fontweight="bold")
    ax1.axvline(10, color="orange", linestyle="--", alpha=0.5)
    ax1.axvline(20, color="red", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="x")

    # 2: Variability bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    cv_keys = [("cycle_duration_cv", "Durée cycle"), ("stance_pct_cv", "Stance %"),
               ("left_hip_rom_cv", "Hanche G ROM"), ("left_knee_rom_cv", "Genou G ROM")]
    cv_vals = [var.get(k, 0) for k, _ in cv_keys]
    cv_labels = [l for _, l in cv_keys]
    colors_cv = ["green" if v < 10 else "orange" if v < 20 else "red" for v in cv_vals]
    y_pos2 = np.arange(len(cv_labels))
    ax2.barh(y_pos2, cv_vals, color=colors_cv, alpha=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(cv_labels, fontsize=8)
    ax2.set_xlabel("CV (%)")
    ax2.set_title("Variabilité (CV)", fontsize=10, fontweight="bold")
    ax2.axvline(10, color="orange", linestyle="--", alpha=0.5)
    ax2.axvline(20, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # 3: Temporal parameters — use a table-like layout
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.set_title("Parametres temporels", fontsize=10, fontweight="bold", loc="left")
    params = [
        ("Cadence", f"{st.get('cadence_steps_per_min', 'N/A')} pas/min"),
        ("Stride (moy)", f"{st.get('stride_time_mean_s', 'N/A')} s"),
        ("Stride (SD)", f"{st.get('stride_time_std_s', 'N/A')} s"),
        ("Step (moy)", f"{st.get('step_time_mean_s', 'N/A')} s"),
        ("Stance G", f"{st.get('stance_pct_left', 'N/A')}%"),
        ("Stance D", f"{st.get('stance_pct_right', 'N/A')}%"),
        ("Double support", f"{st.get('double_support_pct', 'N/A')}%"),
        ("Cycles", f"{st.get('n_cycles_total', 0)}"),
    ]
    table = ax3.table(
        cellText=params, colLabels=["Parametre", "Valeur"],
        loc="center", cellLoc="left",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#D9E2F3" if key[0] % 2 == 0 else "white")

    # 4: Pathology flags
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    if flags:
        flag_lines = ["ALERTES", ""]
        for f in flags:
            flag_lines.append(f"  * {f}")
        flag_text = "\n".join(flag_lines)
        bg_color = "lightcoral"
    else:
        flag_text = "ALERTES\n\nAucune anomalie.\nParametres normaux."
        bg_color = "lightgreen"
    ax4.text(0.05, 0.95, flag_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.3))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_trunk_pelvis(pdf, data: dict):
    """Page 4: trunk and pelvis time series with pathology detection."""
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)
    events = data.get("events", {})

    time = np.array([af["frame_idx"] / fps for af in angle_frames])
    trunk = [af.get("trunk_angle") for af in angle_frames]
    trunk = [v if v is not None else np.nan for v in trunk]
    pelvis = [af.get("pelvis_tilt") for af in angle_frames]
    pelvis = [v if v is not None else np.nan for v in pelvis]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(_FIG_SIZE[0], 10))
    fig.suptitle("Analyse du tronc et du bassin", fontsize=14, fontweight="bold", y=0.99)

    # Trunk
    trunk_arr = np.array(trunk)
    valid_trunk = trunk_arr[~np.isnan(trunk_arr)]
    ax1.plot(time, trunk, color="#2171b5", linewidth=1.5, label="Trunk angle")
    if len(valid_trunk) > 0:
        trunk_mean = np.mean(valid_trunk)
        trunk_std = np.std(valid_trunk)
        trunk_rom = np.ptp(valid_trunk)
        ax1.axhline(trunk_mean, color="red", linestyle="--", linewidth=1.5,
                     label=f"Moy: {trunk_mean:.1f}°")
        # Pathology annotation
        if trunk_mean > 10:
            ax1.text(0.02, 0.95, "PATHOLOGIE: Tronc penche en avant (>10°)",
                     transform=ax1.transAxes, fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="red", alpha=0.4),
                     verticalalignment="top")
        elif trunk_mean > 5:
            ax1.text(0.02, 0.95, "Attention: Inclinaison avant moderee",
                     transform=ax1.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="orange", alpha=0.4),
                     verticalalignment="top")
        else:
            ax1.text(0.02, 0.95, "Posture normale",
                     transform=ax1.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4),
                     verticalalignment="top")
        ax1.set_title(f"Inclinaison du tronc — ROM: {trunk_rom:.1f}°  Moy: {trunk_mean:.1f}° ± {trunk_std:.1f}°",
                       fontsize=10, fontweight="bold")
    # Events
    for hs in events.get("left_hs", []) + events.get("right_hs", []):
        ax1.axvline(hs["time"], color="green", alpha=0.2, linewidth=0.5)
    ax1.set_ylabel("Angle (°)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Pelvis
    pelvis_arr = np.array(pelvis)
    valid_pelvis = pelvis_arr[~np.isnan(pelvis_arr)]
    ax2.plot(time, pelvis, color="purple", linewidth=1.5, label="Pelvis tilt")
    if len(valid_pelvis) > 0:
        pelvis_mean = np.mean(valid_pelvis)
        ax2.axhline(pelvis_mean, color="red", linestyle="--", linewidth=1.5,
                     label=f"Moy: {pelvis_mean:.1f}°")
        ax2.set_title(f"Inclinaison du bassin — Moy: {pelvis_mean:.1f}°  "
                       "(NaN en vue laterale si hanches superposees)",
                       fontsize=10, fontweight="bold")
    else:
        ax2.set_title("Pelvis tilt — Non disponible (vue laterale)", fontsize=10)
    for hs in events.get("left_hs", []) + events.get("right_hs", []):
        ax2.axvline(hs["time"], color="green", alpha=0.2, linewidth=0.5)
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Angle (°)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_normalized_cycles(pdf, cycles: dict, side: str, data: dict):
    """Page 5/6: detailed normalized cycles for one side."""
    summary = cycles.get("summary", {}).get(side)
    side_cycles = [c for c in cycles.get("cycles", []) if c["side"] == side]
    side_label = "Gauche" if side == "left" else "Droite"

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE)
    fig.suptitle(f"Cycles normalisés — {side_label} (n={len(side_cycles)})",
                 fontsize=14, fontweight="bold", y=0.99)

    x = np.linspace(0, 100, 101)
    joints = [("hip", "Hanche", axes[0, 0]), ("knee", "Genou", axes[0, 1]),
              ("ankle", "Cheville", axes[1, 0]), ("trunk", "Tronc", axes[1, 1])]

    for joint, label, ax in joints:
        if summary is None or f"{joint}_mean" not in summary:
            ax.text(0.5, 0.5, "Pas de donnees", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Individual cycles (light)
        for c in side_cycles:
            vals = c.get("angles_normalized", {}).get(joint)
            if vals:
                ax.plot(x, vals, color="gray", linewidth=0.6, alpha=0.3)

        # Mean ± SD
        m = np.array(summary[f"{joint}_mean"])
        s = np.array(summary[f"{joint}_std"])
        color = "#2171b5" if side == "left" else "#cb181d"
        ax.plot(x, m, color=color, linewidth=2.5, label="Moyenne")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, label="±1 SD")

        # ROM
        rom = float(np.ptp(m))
        ax.set_title(f"{label} — ROM moy: {rom:.1f}°", fontsize=10, fontweight="bold")
        ax.set_xlabel("% cycle")
        ax.set_ylabel("Angle (°)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # TO marker if available
        stance_pcts = [c["stance_pct"] for c in side_cycles if c["stance_pct"] is not None]
        if stance_pcts:
            avg_to = np.mean(stance_pcts)
            ax.axvline(avg_to, color="orange", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=f"TO ~{avg_to:.0f}%")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_detailed_text(pdf, data: dict, cycles: dict, stats: dict):
    """Last page: full text summary of all metrics."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.suptitle("Rapport detaille", fontsize=14, fontweight="bold", y=0.99)
    ax = fig.add_subplot(111)
    ax.axis("off")

    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    var = stats.get("variability", {})
    flags = stats.get("pathology_flags", [])
    meta = data.get("meta", {})
    summary = cycles.get("summary", {})

    lines = []
    lines.append("RAPPORT COMPLET — ANALYSE DE MARCHE")
    lines.append("─" * 50)
    lines.append("")
    lines.append(f"Source: {meta.get('video_path', '?')}")
    lines.append(f"FPS: {meta.get('fps', '?')}  |  Resolution: {meta.get('width', '?')}x{meta.get('height', '?')}")
    lines.append(f"Modele: {data.get('extraction', {}).get('model', '?')}  |  Correction: {data.get('angles', {}).get('correction_factor', '?')}")
    lines.append("")

    lines.append("PARAMETRES SPATIO-TEMPORELS")
    lines.append("─" * 35)
    lines.append(f"  Cadence:         {st.get('cadence_steps_per_min', 'N/A')} pas/min  (normal: 100-120)")
    lines.append(f"  Stride (moy):    {st.get('stride_time_mean_s', 'N/A')} +/- {st.get('stride_time_std_s', 'N/A')} s")
    lines.append(f"  Step (moy):      {st.get('step_time_mean_s', 'N/A')} +/- {st.get('step_time_std_s', 'N/A')} s")
    lines.append(f"  Stance G:        {st.get('stance_pct_left', 'N/A')}%    D: {st.get('stance_pct_right', 'N/A')}%  (normal: ~60%)")
    lines.append(f"  Double support:  {st.get('double_support_pct', 'N/A')}%  (normal: ~20%)")
    lines.append("")

    lines.append("AMPLITUDES ARTICULAIRES (ROM)")
    lines.append("─" * 35)
    for side_name, side_key in [("GAUCHE", "left"), ("DROITE", "right")]:
        s = summary.get(side_key)
        if s:
            lines.append(f"  {side_name}:")
            for joint in ["hip", "knee", "ankle"]:
                m = s.get(f"{joint}_mean")
                if m:
                    arr = np.array(m)
                    lines.append(f"    {joint.capitalize():8s}  {np.min(arr):6.1f} a {np.max(arr):6.1f}  (ROM: {np.ptp(arr):.1f})")
    lines.append("")

    lines.append("SYMETRIE")
    lines.append("─" * 35)
    for k in ["hip_rom_si", "knee_rom_si", "ankle_rom_si", "step_time_si", "overall_si"]:
        v = sym.get(k, "N/A")
        lines.append(f"  {k:20s}  {v}%")
    lines.append("  (SI < 10% = excellent, 10-20% = bon, >20% = asymetrie)")
    lines.append("")

    lines.append("VARIABILITE")
    lines.append("─" * 35)
    lines.append(f"  Cycle duration CV:  {var.get('cycle_duration_cv', 'N/A')}%")
    lines.append(f"  Stance % CV:        {var.get('stance_pct_cv', 'N/A')}%")
    lines.append("")

    if flags:
        lines.append("ALERTES")
        lines.append("─" * 35)
        for f in flags:
            lines.append(f"  * {f}")
    else:
        lines.append("Aucune alerte — parametres normaux.")

    lines.append("")
    lines.append("Valeurs normales de reference:")
    lines.append("  Hanche ROM: 40-50 deg | Genou ROM: 60-70 deg | Cheville ROM: 30-40 deg")
    lines.append("")
    lines.append(f"Genere par myogait v{data.get('myogait_version', '?')}")

    text = "\n".join(lines)
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


# ── Public API ──────────────────────────────────────────────────────


def generate_report(
    data: dict,
    cycles: dict,
    stats: dict,
    output_path: str,
) -> str:
    """Generate a multi-page PDF report.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and ``events``.
    cycles : dict
        Output of ``segment_cycles()``.
    stats : dict
        Output of ``analyze_gait()``.
    output_path : str
        Path for the output PDF file.

    Returns
    -------
    str
        Path to the generated PDF file.

    Raises
    ------
    ValueError
        If *data* has no angles.
    TypeError
        If *data*, *cycles*, or *stats* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if not isinstance(cycles, dict):
        raise TypeError("cycles must be a dict")
    if not isinstance(stats, dict):
        raise TypeError("stats must be a dict")
    if not data.get("angles"):
        raise ValueError("No angles in data. Run compute_angles() first.")
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating PDF report: {output_path}")

    with PdfPages(output_path) as pdf:
        _page_overview(pdf, data, cycles)
        _page_bilateral(pdf, cycles, data)
        _page_statistics(pdf, stats)
        _page_trunk_pelvis(pdf, data)
        _page_normalized_cycles(pdf, cycles, "left", data)
        _page_normalized_cycles(pdf, cycles, "right", data)
        _page_detailed_text(pdf, data, cycles, stats)

        d = pdf.infodict()
        d["Title"] = "Analyse cinematique — Rapport myogait"
        d["Author"] = "myogait"
        d["Subject"] = "Gait Analysis"

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"PDF generated: {output_path} ({size_kb:.0f} KB, 7 pages)")

    return output_path
