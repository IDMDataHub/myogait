"""PDF report generation for gait analysis.

Generates a multi-page clinical PDF report with:

- Page 1: Overview -- angle time series (hip, knee, ankle) with events.
- Page 2: Bilateral comparison -- mean +/- SD overlaid L vs R.
- Page 3: Clinical statistics -- symmetry and variability bar charts.
- Page 4: Trunk and pelvis analysis with pathology annotations.
- Pages 5-6: Normalized cycles per side (all cycles + mean +/- SD).
- Page 7: Normative comparison.
- Page 8: GVS/MAP profile.
- Page 9: Quality dashboard.
- Page 10: Detailed text summary of all metrics.

Functions
---------
generate_report
    Generate the full multi-page PDF report.
generate_longitudinal_report
    Generate a multi-session comparison PDF report.
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


# ── Internationalisation strings ──────────────────────────────────

_STRINGS = {
    "fr": {
        "title": "Rapport d'Analyse de la Marche",
        "patient": "Patient",
        "date": "Date",
        "overview_title": "Vue d'ensemble \u2014 Angles articulaires",
        "hip": "Hanche",
        "knee": "Genou",
        "ankle": "Cheville",
        "trunk": "Tronc",
        "pelvis": "Bassin",
        "left": "Gauche",
        "right": "Droite",
        "time_s": "Temps (s)",
        "angle_deg": "Angle (\u00b0)",
        "pct_cycle": "% cycle de marche",
        "flexion_ext": "Flexion (+) / Extension (-)",
        "flexion": "Flexion (+)",
        "dorsi_plantar": "Dorsiflexion (+) / Plantarflexion (-)",
        "bilateral_title": "Comparaison bilat\u00e9rale",
        "stats_title": "Statistiques cliniques",
        "symmetry_title": "Indices de sym\u00e9trie",
        "variability_title": "Variabilit\u00e9 (CV)",
        "temporal_title": "Parametres temporels",
        "parameter": "Parametre",
        "value": "Valeur",
        "cadence": "Cadence",
        "stride_mean": "Stride (moy)",
        "stride_sd": "Stride (SD)",
        "step_mean": "Step (moy)",
        "stance_left": "Stance G",
        "stance_right": "Stance D",
        "double_support": "Double support",
        "cycles": "Cycles",
        "alerts": "ALERTES",
        "no_anomaly": "Aucune anomalie.\nParametres normaux.",
        "trunk_pelvis_title": "Analyse du tronc et du bassin",
        "trunk_forward_pathology": "PATHOLOGIE: Tronc penche en avant (>10\u00b0)",
        "trunk_moderate": "Attention: Inclinaison avant moderee",
        "trunk_normal": "Posture normale",
        "trunk_tilt": "Inclinaison du tronc",
        "pelvis_tilt": "Inclinaison du bassin",
        "pelvis_unavailable": "Pelvis tilt \u2014 Non disponible (vue laterale)",
        "normalized_title": "Cycles normalis\u00e9s",
        "no_data": "Pas de donnees",
        "mean": "Moyenne",
        "sd_band": "\u00b11 SD",
        "detailed_title": "Rapport detaille",
        "full_report_header": "RAPPORT COMPLET \u2014 ANALYSE DE MARCHE",
        "source": "Source",
        "fps": "FPS",
        "resolution": "Resolution",
        "model": "Modele",
        "correction": "Correction",
        "spatiotemporal_header": "PARAMETRES SPATIO-TEMPORELS",
        "rom_header": "AMPLITUDES ARTICULAIRES (ROM)",
        "symmetry_header": "SYMETRIE",
        "variability_header": "VARIABILITE",
        "normal_ref": "Valeurs normales de reference:",
        "ref_values": "Hanche ROM: 40-50 deg | Genou ROM: 60-70 deg | Cheville ROM: 30-40 deg",
        "generated_by": "Genere par myogait",
        "no_alert": "Aucune alerte \u2014 parametres normaux.",
        "si_excellent": "(SI < 10% = excellent, 10-20% = bon, >20% = asymetrie)",
        "normative_title": "Comparaison normative",
        "gvs_title": "Profil GVS / MAP",
        "quality_title": "Tableau de bord qualit\u00e9",
        "steps_per_min": "pas/min",
        "normal_cadence": "normal: 100-120",
        "normal_stance": "normal: ~60%",
        "normal_double_support": "normal: ~20%",
        "hip_rom_si": "Hanche ROM",
        "knee_rom_si": "Genou ROM",
        "ankle_rom_si": "Cheville ROM",
        "step_time_si": "Temps de pas",
        "stance_time_si": "Temps stance",
        "cycle_duration_cv": "Dur\u00e9e cycle",
        "stance_pct_cv": "Stance %",
        "left_hip_rom_cv": "Hanche G ROM",
        "left_knee_rom_cv": "Genou G ROM",
        "longitudinal_title": "Rapport longitudinal \u2014 Comparaison multi-sessions",
        "session": "Session",
        "frontal_title": "Comparaison plan frontal",
    },
    "en": {
        "title": "Gait Analysis Report",
        "patient": "Patient",
        "date": "Date",
        "overview_title": "Overview \u2014 Joint Angles",
        "hip": "Hip",
        "knee": "Knee",
        "ankle": "Ankle",
        "trunk": "Trunk",
        "pelvis": "Pelvis",
        "left": "Left",
        "right": "Right",
        "time_s": "Time (s)",
        "angle_deg": "Angle (\u00b0)",
        "pct_cycle": "% Gait Cycle",
        "flexion_ext": "Flexion (+) / Extension (-)",
        "flexion": "Flexion (+)",
        "dorsi_plantar": "Dorsiflexion (+) / Plantarflexion (-)",
        "bilateral_title": "Bilateral Comparison",
        "stats_title": "Clinical Statistics",
        "symmetry_title": "Symmetry Indices",
        "variability_title": "Variability (CV)",
        "temporal_title": "Temporal Parameters",
        "parameter": "Parameter",
        "value": "Value",
        "cadence": "Cadence",
        "stride_mean": "Stride (mean)",
        "stride_sd": "Stride (SD)",
        "step_mean": "Step (mean)",
        "stance_left": "Stance L",
        "stance_right": "Stance R",
        "double_support": "Double support",
        "cycles": "Cycles",
        "alerts": "ALERTS",
        "no_anomaly": "No anomaly detected.\nParameters within normal range.",
        "trunk_pelvis_title": "Trunk and Pelvis Analysis",
        "trunk_forward_pathology": "PATHOLOGY: Forward trunk lean (>10\u00b0)",
        "trunk_moderate": "Warning: Moderate forward lean",
        "trunk_normal": "Normal posture",
        "trunk_tilt": "Trunk Inclination",
        "pelvis_tilt": "Pelvis Tilt",
        "pelvis_unavailable": "Pelvis tilt \u2014 Not available (lateral view)",
        "normalized_title": "Normalized Cycles",
        "no_data": "No data",
        "mean": "Mean",
        "sd_band": "\u00b11 SD",
        "detailed_title": "Detailed Report",
        "full_report_header": "FULL REPORT \u2014 GAIT ANALYSIS",
        "source": "Source",
        "fps": "FPS",
        "resolution": "Resolution",
        "model": "Model",
        "correction": "Correction",
        "spatiotemporal_header": "SPATIO-TEMPORAL PARAMETERS",
        "rom_header": "JOINT RANGES OF MOTION (ROM)",
        "symmetry_header": "SYMMETRY",
        "variability_header": "VARIABILITY",
        "normal_ref": "Normal reference values:",
        "ref_values": "Hip ROM: 40-50 deg | Knee ROM: 60-70 deg | Ankle ROM: 30-40 deg",
        "generated_by": "Generated by myogait",
        "no_alert": "No alerts \u2014 parameters within normal range.",
        "si_excellent": "(SI < 10% = excellent, 10-20% = good, >20% = asymmetry)",
        "normative_title": "Normative Comparison",
        "gvs_title": "GVS / MAP Profile",
        "quality_title": "Quality Dashboard",
        "steps_per_min": "steps/min",
        "normal_cadence": "normal: 100-120",
        "normal_stance": "normal: ~60%",
        "normal_double_support": "normal: ~20%",
        "hip_rom_si": "Hip ROM",
        "knee_rom_si": "Knee ROM",
        "ankle_rom_si": "Ankle ROM",
        "step_time_si": "Step time",
        "stance_time_si": "Stance time",
        "cycle_duration_cv": "Cycle duration",
        "stance_pct_cv": "Stance %",
        "left_hip_rom_cv": "Hip L ROM",
        "left_knee_rom_cv": "Knee L ROM",
        "longitudinal_title": "Longitudinal Report \u2014 Multi-session Comparison",
        "session": "Session",
        "frontal_title": "Frontal Plane Comparison",
    },
}


# ── Small page functions ────────────────────────────────────────────


def _page_overview(pdf, data: dict, cycles: dict, s: dict):
    """Page 1: angle time series (hip, knee, ankle) with event markers."""
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)
    events = data.get("events", {})

    time = np.array([af["frame_idx"] / fps for af in angle_frames])

    fig, axes = plt.subplots(3, 2, figsize=_FIG_SIZE)
    fig.suptitle(s["overview_title"], fontsize=14, fontweight="bold", y=0.99)

    joint_pairs = [
        ("hip", s["hip"], s["flexion_ext"]),
        ("knee", s["knee"], s["flexion"]),
        ("ankle", s["ankle"], s["dorsi_plantar"]),
    ]

    for row, (joint, label, ylabel) in enumerate(joint_pairs):
        for col, (side_label, suffix, color) in enumerate(
            [(s["left"], "_L", "#2171b5"), (s["right"], "_R", "#cb181d")]
        ):
            ax = axes[row, col]
            key = f"{joint}{suffix}"
            vals = [af.get(key) for af in angle_frames]
            vals = [v if v is not None else np.nan for v in vals]
            ax.plot(time, vals, color=color, linewidth=1, label=side_label)

            # Event markers
            for hs in events.get(f"{'left' if col == 0 else 'right'}_hs", []):
                ax.axvline(hs["time"], color="green", alpha=0.3, linewidth=0.5)
            for to in events.get(f"{'left' if col == 0 else 'right'}_to", []):
                ax.axvline(to["time"], color="orange", alpha=0.3, linewidth=0.5, linestyle="--")

            # ROM annotation
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                rom = max(valid) - min(valid)
                ax.set_title(f"{label} {side_label}  (ROM: {rom:.1f}\u00b0)", fontsize=10, fontweight="bold")
            else:
                ax.set_title(f"{label} {side_label}", fontsize=10)

            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

    axes[2, 0].set_xlabel(s["time_s"])
    axes[2, 1].set_xlabel(s["time_s"])
    # This page includes a matplotlib table that can conflict with tight_layout.
    # Use explicit spacing to avoid layout warnings in automated runs.
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.97, hspace=0.35, wspace=0.25)
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_bilateral(pdf, cycles: dict, data: dict, s: dict):
    """Page 2: bilateral comparison -- mean +/- SD overlaid L vs R."""
    summary = cycles.get("summary", {})
    left = summary.get("left")
    right = summary.get("right")

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE)
    fig.suptitle(s["bilateral_title"], fontsize=14, fontweight="bold", y=0.99)

    x = np.linspace(0, 100, 101)
    joints = [("hip", s["hip"], axes[0, 0]), ("knee", s["knee"], axes[0, 1]), ("ankle", s["ankle"], axes[1, 0])]

    for joint, label, ax in joints:
        if left and f"{joint}_mean" in left:
            m = np.array(left[f"{joint}_mean"])
            sd = np.array(left[f"{joint}_std"])
            ax.plot(x, m, color="#2171b5", linewidth=2, label=s["left"])
            ax.fill_between(x, m - sd, m + sd, color="#2171b5", alpha=0.15)

        if right and f"{joint}_mean" in right:
            m = np.array(right[f"{joint}_mean"])
            sd = np.array(right[f"{joint}_std"])
            ax.plot(x, m, color="#cb181d", linewidth=2, label=s["right"])
            ax.fill_between(x, m - sd, m + sd, color="#cb181d", alpha=0.15)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(s["pct_cycle"])
        ax.set_ylabel(s["angle_deg"])
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
        f"{s['source']}: {Path(meta.get('video_path', '?')).name}\n"
        f"{s['fps']}: {meta.get('fps', '?'):.1f}\n"
        f"{s['cycles']} {s['left'][0]}: {n_left}  {s['right'][0]}: {n_right}\n"
        f"{s['model']}: {data.get('extraction', {}).get('model', '?')}\n"
        f"{s['correction']}: {data.get('angles', {}).get('correction_factor', '?')}"
    )
    ax_info.text(0.1, 0.9, info, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    # This page includes a table that is not always tight_layout-compatible.
    fig.subplots_adjust(
        top=0.90, bottom=0.08, left=0.06, right=0.97, hspace=0.35, wspace=0.25
    )
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_statistics(pdf, stats: dict, s: dict):
    """Page 3: clinical statistics -- bar charts + text."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.suptitle(s["stats_title"], fontsize=14, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    sym = stats.get("symmetry", {})
    var = stats.get("variability", {})
    st = stats.get("spatiotemporal", {})
    flags = stats.get("pathology_flags", [])

    # 1: Symmetry bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    si_keys = [("hip_rom_si", s["hip_rom_si"]), ("knee_rom_si", s["knee_rom_si"]),
               ("ankle_rom_si", s["ankle_rom_si"]), ("step_time_si", s["step_time_si"]),
               ("stance_time_si", s["stance_time_si"])]
    si_vals = [sym.get(k, 0) for k, _ in si_keys]
    si_labels = [lbl for _, lbl in si_keys]
    colors = ["green" if v < 10 else "orange" if v < 20 else "red" for v in si_vals]
    y_pos = np.arange(len(si_labels))
    ax1.barh(y_pos, si_vals, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(si_labels, fontsize=8)
    ax1.set_xlabel("SI (%)")
    ax1.set_title(s["symmetry_title"], fontsize=10, fontweight="bold")
    ax1.axvline(10, color="orange", linestyle="--", alpha=0.5)
    ax1.axvline(20, color="red", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="x")

    # 2: Variability bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    cv_keys = [("cycle_duration_cv", s["cycle_duration_cv"]), ("stance_pct_cv", s["stance_pct_cv"]),
               ("left_hip_rom_cv", s["left_hip_rom_cv"]), ("left_knee_rom_cv", s["left_knee_rom_cv"])]
    cv_vals = [var.get(k, 0) for k, _ in cv_keys]
    cv_labels = [lbl for _, lbl in cv_keys]
    colors_cv = ["green" if v < 10 else "orange" if v < 20 else "red" for v in cv_vals]
    y_pos2 = np.arange(len(cv_labels))
    ax2.barh(y_pos2, cv_vals, color=colors_cv, alpha=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(cv_labels, fontsize=8)
    ax2.set_xlabel("CV (%)")
    ax2.set_title(s["variability_title"], fontsize=10, fontweight="bold")
    ax2.axvline(10, color="orange", linestyle="--", alpha=0.5)
    ax2.axvline(20, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # 3: Temporal parameters -- use a table-like layout
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.set_title(s["temporal_title"], fontsize=10, fontweight="bold", loc="left")
    params = [
        (s["cadence"], f"{st.get('cadence_steps_per_min', 'N/A')} {s['steps_per_min']}"),
        (s["stride_mean"], f"{st.get('stride_time_mean_s', 'N/A')} s"),
        (s["stride_sd"], f"{st.get('stride_time_std_s', 'N/A')} s"),
        (s["step_mean"], f"{st.get('step_time_mean_s', 'N/A')} s"),
        (s["stance_left"], f"{st.get('stance_pct_left', 'N/A')}%"),
        (s["stance_right"], f"{st.get('stance_pct_right', 'N/A')}%"),
        (s["double_support"], f"{st.get('double_support_pct', 'N/A')}%"),
        (s["cycles"], f"{st.get('n_cycles_total', 0)}"),
    ]
    table = ax3.table(
        cellText=params, colLabels=[s["parameter"], s["value"]],
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
        flag_lines = [s["alerts"], ""]
        for f in flags:
            flag_lines.append(f"  * {f}")
        flag_text = "\n".join(flag_lines)
        bg_color = "lightcoral"
    else:
        flag_text = s["alerts"] + "\n\n" + s["no_anomaly"]
        bg_color = "lightgreen"
    ax4.text(
        0.05, 0.95, flag_text, transform=ax4.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.3),
    )

    # This page includes a table that is not always tight_layout-compatible.
    fig.subplots_adjust(
        top=0.90, bottom=0.08, left=0.06, right=0.97, hspace=0.35, wspace=0.25
    )
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_trunk_pelvis(pdf, data: dict, s: dict):
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
    fig.suptitle(s["trunk_pelvis_title"], fontsize=14, fontweight="bold", y=0.99)

    # Trunk
    trunk_arr = np.array(trunk)
    valid_trunk = trunk_arr[~np.isnan(trunk_arr)]
    ax1.plot(time, trunk, color="#2171b5", linewidth=1.5, label="Trunk angle")
    if len(valid_trunk) > 0:
        trunk_mean = np.mean(valid_trunk)
        trunk_std = np.std(valid_trunk)
        trunk_rom = np.ptp(valid_trunk)
        ax1.axhline(trunk_mean, color="red", linestyle="--", linewidth=1.5,
                     label=f"{s['mean']}: {trunk_mean:.1f}\u00b0")
        # Pathology annotation
        if trunk_mean > 10:
            ax1.text(0.02, 0.95, s["trunk_forward_pathology"],
                     transform=ax1.transAxes, fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="red", alpha=0.4),
                     verticalalignment="top")
        elif trunk_mean > 5:
            ax1.text(0.02, 0.95, s["trunk_moderate"],
                     transform=ax1.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="orange", alpha=0.4),
                     verticalalignment="top")
        else:
            ax1.text(0.02, 0.95, s["trunk_normal"],
                     transform=ax1.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4),
                     verticalalignment="top")
        ax1.set_title(f"{s['trunk_tilt']} \u2014 ROM: {trunk_rom:.1f}\u00b0  {s['mean']}: {trunk_mean:.1f}\u00b0 \u00b1 {trunk_std:.1f}\u00b0",
                       fontsize=10, fontweight="bold")
    # Events
    for hs in events.get("left_hs", []) + events.get("right_hs", []):
        ax1.axvline(hs["time"], color="green", alpha=0.2, linewidth=0.5)
    ax1.set_ylabel(s["angle_deg"])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Pelvis
    pelvis_arr = np.array(pelvis)
    valid_pelvis = pelvis_arr[~np.isnan(pelvis_arr)]
    ax2.plot(time, pelvis, color="purple", linewidth=1.5, label=s["pelvis_tilt"])
    if len(valid_pelvis) > 0:
        pelvis_mean = np.mean(valid_pelvis)
        ax2.axhline(pelvis_mean, color="red", linestyle="--", linewidth=1.5,
                     label=f"{s['mean']}: {pelvis_mean:.1f}\u00b0")
        ax2.set_title(f"{s['pelvis_tilt']} \u2014 {s['mean']}: {pelvis_mean:.1f}\u00b0",
                       fontsize=10, fontweight="bold")
    else:
        ax2.set_title(s["pelvis_unavailable"], fontsize=10)
    for hs in events.get("left_hs", []) + events.get("right_hs", []):
        ax2.axvline(hs["time"], color="green", alpha=0.2, linewidth=0.5)
    ax2.set_xlabel(s["time_s"])
    ax2.set_ylabel(s["angle_deg"])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _page_normalized_cycles(pdf, cycles: dict, side: str, data: dict, s: dict):
    """Page 5/6: detailed normalized cycles for one side."""
    summary = cycles.get("summary", {}).get(side)
    side_cycles = [c for c in cycles.get("cycles", []) if c["side"] == side]
    side_label = s["left"] if side == "left" else s["right"]

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE)
    fig.suptitle(f"{s['normalized_title']} \u2014 {side_label} (n={len(side_cycles)})",
                 fontsize=14, fontweight="bold", y=0.99)

    x = np.linspace(0, 100, 101)
    joints = [("hip", s["hip"], axes[0, 0]), ("knee", s["knee"], axes[0, 1]),
              ("ankle", s["ankle"], axes[1, 0]), ("trunk", s["trunk"], axes[1, 1])]

    for joint, label, ax in joints:
        if summary is None or f"{joint}_mean" not in summary:
            ax.text(0.5, 0.5, s["no_data"], ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Individual cycles (light)
        for c in side_cycles:
            vals = c.get("angles_normalized", {}).get(joint)
            if vals:
                ax.plot(x, vals, color="gray", linewidth=0.6, alpha=0.3)

        # Mean +/- SD
        m = np.array(summary[f"{joint}_mean"])
        sd = np.array(summary[f"{joint}_std"])
        color = "#2171b5" if side == "left" else "#cb181d"
        ax.plot(x, m, color=color, linewidth=2.5, label=s["mean"])
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, label=s["sd_band"])

        # ROM
        rom = float(np.ptp(m))
        ax.set_title(f"{label} \u2014 ROM {s['mean'].lower()}: {rom:.1f}\u00b0", fontsize=10, fontweight="bold")
        ax.set_xlabel("% cycle")
        ax.set_ylabel(s["angle_deg"])
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


def _page_normative(pdf, cycles: dict, data: dict, s: dict):
    """Page 7: normative comparison."""
    from .plotting import plot_normative_comparison

    try:
        fig = plot_normative_comparison(data, cycles)
        fig.suptitle(s["normative_title"], fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)
    except Exception:
        logger.exception("Could not generate normative comparison page")
        fig = plt.figure(figsize=_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, s["no_data"], ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.suptitle(s["normative_title"], fontsize=14, fontweight="bold")
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)


def _has_frontal_data(cycles: dict) -> bool:
    """Return True if any cycle summary contains frontal angle keys."""
    summary = cycles.get("summary", {})
    frontal_keys = ("pelvis_list_mean", "hip_adduction_mean", "knee_valgus_mean")
    for side in ("left", "right"):
        side_summary = summary.get(side, {})
        for key in frontal_keys:
            if key in side_summary:
                return True
    return False


def _page_frontal(pdf, cycles: dict, data: dict, s: dict):
    """Frontal angles normative comparison page."""
    from .plotting import plot_normative_comparison

    title = s.get("frontal_title", "Frontal Plane Comparison")
    try:
        fig = plot_normative_comparison(data, cycles, plane="frontal")
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)
    except Exception:
        logger.exception("Could not generate frontal comparison page")
        fig = plt.figure(figsize=_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, s["no_data"], ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)


def _page_gvs(pdf, cycles: dict, data: dict, s: dict):
    """Page 8: GVS/MAP profile."""
    from .plotting import plot_gvs_profile

    try:
        fig = plot_gvs_profile(cycles)
        fig.suptitle(s["gvs_title"], fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)
    except Exception:
        logger.exception("Could not generate GVS profile page")
        fig = plt.figure(figsize=_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, s["no_data"], ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.suptitle(s["gvs_title"], fontsize=14, fontweight="bold")
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)


def _page_quality(pdf, data: dict, s: dict):
    """Page 9: quality dashboard."""
    from .plotting import plot_quality_dashboard

    try:
        fig = plot_quality_dashboard(data)
        fig.suptitle(s["quality_title"], fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)
    except Exception:
        logger.exception("Could not generate quality dashboard page")
        fig = plt.figure(figsize=_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, s["no_data"], ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.suptitle(s["quality_title"], fontsize=14, fontweight="bold")
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)


def _page_detailed_text(pdf, data: dict, cycles: dict, stats: dict, s: dict):
    """Last page: full text summary of all metrics."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.suptitle(s["detailed_title"], fontsize=14, fontweight="bold", y=0.99)
    ax = fig.add_subplot(111)
    ax.axis("off")

    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    var = stats.get("variability", {})
    flags = stats.get("pathology_flags", [])
    meta = data.get("meta", {})
    summary = cycles.get("summary", {})

    lines = []
    lines.append(s["full_report_header"])
    lines.append("\u2500" * 50)
    lines.append("")
    lines.append(f"{s['source']}: {meta.get('video_path', '?')}")
    lines.append(f"{s['fps']}: {meta.get('fps', '?')}  |  {s['resolution']}: {meta.get('width', '?')}x{meta.get('height', '?')}")
    lines.append(f"{s['model']}: {data.get('extraction', {}).get('model', '?')}  |  {s['correction']}: {data.get('angles', {}).get('correction_factor', '?')}")
    lines.append("")

    lines.append(s["spatiotemporal_header"])
    lines.append("\u2500" * 35)
    lines.append(f"  {s['cadence']}:         {st.get('cadence_steps_per_min', 'N/A')} {s['steps_per_min']}  ({s['normal_cadence']})")
    lines.append(f"  {s['stride_mean']}:    {st.get('stride_time_mean_s', 'N/A')} +/- {st.get('stride_time_std_s', 'N/A')} s")
    lines.append(f"  {s['step_mean']}:      {st.get('step_time_mean_s', 'N/A')} +/- {st.get('step_time_std_s', 'N/A')} s")
    lines.append(f"  {s['stance_left']}:        {st.get('stance_pct_left', 'N/A')}%    {s['stance_right']}: {st.get('stance_pct_right', 'N/A')}%  ({s['normal_stance']})")
    lines.append(f"  {s['double_support']}:  {st.get('double_support_pct', 'N/A')}%  ({s['normal_double_support']})")
    lines.append("")

    lines.append(s["rom_header"])
    lines.append("\u2500" * 35)
    for side_name, side_key in [(s["left"].upper(), "left"), (s["right"].upper(), "right")]:
        side_sum = summary.get(side_key)
        if side_sum:
            lines.append(f"  {side_name}:")
            for joint in ["hip", "knee", "ankle"]:
                m = side_sum.get(f"{joint}_mean")
                if m:
                    arr = np.array(m)
                    lines.append(f"    {joint.capitalize():8s}  {np.min(arr):6.1f} a {np.max(arr):6.1f}  (ROM: {np.ptp(arr):.1f})")
    lines.append("")

    lines.append(s["symmetry_header"])
    lines.append("\u2500" * 35)
    for k in ["hip_rom_si", "knee_rom_si", "ankle_rom_si", "step_time_si", "overall_si"]:
        v = sym.get(k, "N/A")
        lines.append(f"  {k:20s}  {v}%")
    lines.append(f"  {s['si_excellent']}")
    lines.append("")

    lines.append(s["variability_header"])
    lines.append("\u2500" * 35)
    lines.append(f"  {s['cycle_duration_cv']} CV:  {var.get('cycle_duration_cv', 'N/A')}%")
    lines.append(f"  {s['stance_pct_cv']} CV:        {var.get('stance_pct_cv', 'N/A')}%")
    lines.append("")

    if flags:
        lines.append(s["alerts"])
        lines.append("\u2500" * 35)
        for f in flags:
            lines.append(f"  * {f}")
    else:
        lines.append(s["no_alert"])

    lines.append("")
    lines.append(s["normal_ref"])
    lines.append(f"  {s['ref_values']}")
    lines.append("")
    lines.append(f"{s['generated_by']} v{data.get('myogait_version', '?')}")

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
    language: str = "fr",
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
    language : str, optional
        Report language: ``'fr'`` (default) or ``'en'``.

    Returns
    -------
    str
        Path to the generated PDF file.

    Raises
    ------
    ValueError
        If *data* has no angles or *language* is not supported.
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
    if language not in _STRINGS:
        raise ValueError(f"Unsupported language: {language}. Use one of: {list(_STRINGS.keys())}")

    s = _STRINGS[language]
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating PDF report: {output_path} (language={language})")

    with PdfPages(output_path) as pdf:
        _page_overview(pdf, data, cycles, s)
        _page_bilateral(pdf, cycles, data, s)
        _page_statistics(pdf, stats, s)
        _page_trunk_pelvis(pdf, data, s)
        _page_normalized_cycles(pdf, cycles, "left", data, s)
        _page_normalized_cycles(pdf, cycles, "right", data, s)
        _page_normative(pdf, cycles, data, s)
        if _has_frontal_data(cycles):
            _page_frontal(pdf, cycles, data, s)
        _page_gvs(pdf, cycles, data, s)
        _page_quality(pdf, data, s)
        _page_detailed_text(pdf, data, cycles, stats, s)

        d = pdf.infodict()
        d["Title"] = s["title"]
        d["Author"] = "myogait"
        d["Subject"] = "Gait Analysis"

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"PDF generated: {output_path} ({size_kb:.0f} KB)")

    return output_path


def generate_longitudinal_report(
    sessions: List[dict],
    output_path: str,
    language: str = "fr",
) -> str:
    """Generate a multi-session comparison PDF report.

    Parameters
    ----------
    sessions : list of dict
        Each dict must have keys ``'data'``, ``'cycles'``, ``'stats'``
        and optionally ``'label'`` (session label).
    output_path : str
        Path for the output PDF file.
    language : str, optional
        Report language: ``'fr'`` (default) or ``'en'``.

    Returns
    -------
    str
        Path to the generated PDF file.

    Raises
    ------
    ValueError
        If *sessions* is empty or *language* is not supported.
    """
    if not sessions:
        raise ValueError("sessions must not be empty")
    if language not in _STRINGS:
        raise ValueError(f"Unsupported language: {language}. Use one of: {list(_STRINGS.keys())}")

    s = _STRINGS[language]
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating longitudinal report: {output_path} ({len(sessions)} sessions)")

    with PdfPages(output_path) as pdf:
        # Page 1: Overlay of mean curves across sessions
        fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE)
        fig.suptitle(s["longitudinal_title"], fontsize=14, fontweight="bold", y=0.99)

        x = np.linspace(0, 100, 101)
        joints = [("hip", s["hip"], axes[0, 0]), ("knee", s["knee"], axes[0, 1]),
                  ("ankle", s["ankle"], axes[1, 0])]

        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(sessions), 1))

        for joint, label, ax in joints:
            for idx, session in enumerate(sessions):
                sess_label = session.get("label", f"{s['session']} {idx + 1}")
                cycles = session.get("cycles", {})
                for side in ("left", "right"):
                    summary = cycles.get("summary", {}).get(side)
                    if summary and f"{joint}_mean" in summary:
                        m = np.array(summary[f"{joint}_mean"])
                        linestyle = "-" if side == "left" else "--"
                        ax.plot(x, m, color=cmap(idx), linewidth=1.5, linestyle=linestyle,
                                label=f"{sess_label} ({s[side]})" if joint == "hip" else None)

            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel(s["pct_cycle"])
            ax.set_ylabel(s["angle_deg"])
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

        # Legend in the 4th subplot
        ax_legend = axes[1, 1]
        ax_legend.axis("off")
        handles, labels_list = axes[0, 0].get_legend_handles_labels()
        if handles:
            ax_legend.legend(handles, labels_list, loc="center", fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)

        # Page 2: Spatiotemporal comparison table
        fig = plt.figure(figsize=_FIG_SIZE)
        fig.suptitle(f"{s['longitudinal_title']} \u2014 {s['spatiotemporal_header']}", fontsize=14,
                     fontweight="bold", y=0.99)
        ax = fig.add_subplot(111)
        ax.axis("off")

        col_headers = [s["session"], s["cadence"], s["stride_mean"],
                       s["stance_left"], s["stance_right"]]
        cell_data = []
        for idx, session in enumerate(sessions):
            sess_label = session.get("label", f"{s['session']} {idx + 1}")
            st = session.get("stats", {}).get("spatiotemporal", {})
            cell_data.append([
                sess_label,
                f"{st.get('cadence_steps_per_min', 'N/A')}",
                f"{st.get('stride_time_mean_s', 'N/A')} s",
                f"{st.get('stance_pct_left', 'N/A')}%",
                f"{st.get('stance_pct_right', 'N/A')}%",
            ])

        table = ax.table(
            cellText=cell_data, colLabels=col_headers,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#D9E2F3" if key[0] % 2 == 0 else "white")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=_DPI)
        plt.close(fig)

        d = pdf.infodict()
        d["Title"] = s["longitudinal_title"]
        d["Author"] = "myogait"

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"Longitudinal PDF generated: {output_path} ({size_kb:.0f} KB)")

    return output_path
