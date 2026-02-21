"""Clinical gait scores: GVS, GPS-2D, GDI-2D, MAP.

Computes standardized gait deviation scores by comparing patient
kinematic curves to normative data.

IMPORTANT: The GPS-2D and GDI-2D scores are 2D sagittal-plane
adaptations using 4 variables. They are NOT equivalent to the
standard 9-variable, 3-plane GPS/GDI (Baker et al. 2009,
Schwartz & Rozumalski 2008). Use for screening only.

Standard GPS/GDI require 3D motion capture with:
    9 kinematic variables: pelvis tilt/obliquity/rotation,
    hip flex/abd/rotation, knee flex, ankle dorsi, foot progression.

Our 2D adaptation uses 4 sagittal variables:
    hip flexion, knee flexion, ankle dorsiflexion, trunk angle.

References:
    Baker R, McGinley JL, Schwartz MH, et al. The Gait Profile
    Score and Movement Analysis Profile. Gait Posture.
    2009;30(3):265-269. doi:10.1016/j.gaitpost.2009.05.020

    Schwartz MH, Rozumalski A. The Gait Deviation Index: a new
    comprehensive index of gait pathology. Gait Posture.
    2008;28(3):351-357. doi:10.1016/j.gaitpost.2008.05.001
"""

import numpy as np

from .normative import get_normative_curve

# Joints used in the 2D sagittal adaptation
_SCORE_JOINTS = ("hip", "knee", "ankle", "trunk")


# ── Helpers ──────────────────────────────────────────────────────────

def _rms(arr):
    """Root mean square of an array."""
    arr = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def _get_patient_mean(summary: dict, side: str, joint: str):
    """Extract a patient's mean curve from cycle summary.

    Returns
    -------
    np.ndarray or None
        101-point mean curve, or None if data is missing.
    """
    side_data = summary.get(side)
    if side_data is None:
        return None
    key = f"{joint}_mean"
    vals = side_data.get(key)
    if vals is None:
        return None
    return np.array(vals, dtype=float)


def _validate_cycles(cycles: dict):
    """Raise if the cycles dict lacks a summary section."""
    if not isinstance(cycles, dict):
        raise TypeError("cycles must be a dict")
    if "summary" not in cycles:
        raise ValueError(
            "cycles dict has no 'summary'. "
            "Run segment_cycles() first to produce summary curves."
        )


# ── Public API ───────────────────────────────────────────────────────

def gait_variable_scores(cycles: dict, stratum: str = "adult") -> dict:
    """Compute Gait Variable Score (GVS) per joint per side.

    For each side and each joint in (hip, knee, ankle, trunk), the
    GVS is the RMS difference between the patient's mean curve and
    the normative mean curve:

        GVS_j = RMS(patient_mean[j] - normative_mean[j])

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`, which must
        contain a ``'summary'`` key with per-side mean curves.
    stratum : str
        Normative stratum (default ``'adult'``).

    Returns
    -------
    dict
        ``{"left": {"hip": float, ...}, "right": {"hip": float, ...}}``.
        Missing joints are set to ``None``.

    Raises
    ------
    ValueError
        If *cycles* has no summary.

    References
    ----------
    Baker et al. (2009).
    """
    _validate_cycles(cycles)
    summary = cycles["summary"]

    result = {}
    for side in ("left", "right"):
        side_gvs = {}
        for joint in _SCORE_JOINTS:
            patient = _get_patient_mean(summary, side, joint)
            if patient is None:
                side_gvs[joint] = None
                continue
            norm = get_normative_curve(joint, stratum)
            norm_mean = np.array(norm["mean"])
            # Ensure same length
            if len(patient) != len(norm_mean):
                # Resample patient to 101 if needed
                from numpy import interp, linspace
                x_old = linspace(0, 100, len(patient))
                x_new = linspace(0, 100, 101)
                patient = interp(x_new, x_old, patient)
            side_gvs[joint] = round(_rms(patient - norm_mean), 2)
        result[side] = side_gvs

    return result


def gait_profile_score_2d(cycles: dict, stratum: str = "adult") -> dict:
    """Compute 2D sagittal-plane Gait Profile Score.

    GPS-2D is the RMS of all GVS values across the 4 sagittal
    joints (hip, knee, ankle, trunk) for each side:

        GPS-2D = sqrt(mean(GVS_hip^2, GVS_knee^2,
                           GVS_ankle^2, GVS_trunk^2))

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`.
    stratum : str
        Normative stratum (default ``'adult'``).

    Returns
    -------
    dict
        Keys: ``'gps_2d_left'``, ``'gps_2d_right'``,
        ``'gps_2d_overall'``, ``'variables_used'``, ``'note'``.

    References
    ----------
    Baker et al. (2009). Adapted for 2D sagittal-plane analysis.
    """
    gvs = gait_variable_scores(cycles, stratum)

    gps = {}
    all_gvs_values = []
    for side in ("left", "right"):
        side_vals = [
            v for v in gvs[side].values() if v is not None
        ]
        if side_vals:
            gps_val = float(np.sqrt(np.mean(np.array(side_vals) ** 2)))
            gps[f"gps_2d_{side}"] = round(gps_val, 2)
            all_gvs_values.extend(side_vals)
        else:
            gps[f"gps_2d_{side}"] = None

    if all_gvs_values:
        gps["gps_2d_overall"] = round(
            float(np.sqrt(np.mean(np.array(all_gvs_values) ** 2))), 2
        )
    else:
        gps["gps_2d_overall"] = None

    gps["variables_used"] = list(_SCORE_JOINTS)
    gps["note"] = (
        "2D sagittal-plane adaptation using 4 variables "
        "(hip, knee, ankle, trunk). Not equivalent to the standard "
        "9-variable 3-plane GPS (Baker et al. 2009). "
        "Use for screening only."
    )

    return gps


def gait_deviation_index_2d(cycles: dict, stratum: str = "adult") -> dict:
    """Compute 2D sagittal-plane Gait Deviation Index.

    GDI-2D is a scaled index where normal gait scores approximately
    100 and pathological gait scores below 100:

        GDI-2D = 100 - 10 * (GPS-2D - ref_gps) / ref_sd

    where ``ref_gps`` and ``ref_sd`` are estimated from the normative
    SD values. For a normal population, GPS-2D should equal
    approximately the mean SD of the normative curves (because each
    subject deviates by ~1 SD on average).

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`.
    stratum : str
        Normative stratum (default ``'adult'``).

    Returns
    -------
    dict
        Keys: ``'gdi_2d_left'``, ``'gdi_2d_right'``,
        ``'gdi_2d_overall'``, ``'note'``.

    References
    ----------
    Schwartz & Rozumalski (2008). Adapted for 2D sagittal-plane
    analysis.
    """
    gps = gait_profile_score_2d(cycles, stratum)

    # Reference GPS: the expected GPS-2D for a normal subject.
    # A normal subject's curve deviates from the population mean by
    # ~1 SD on average, so ref_gps approx mean(SD) across joints.
    ref_sds = []
    for joint in _SCORE_JOINTS:
        norm = get_normative_curve(joint, stratum)
        ref_sds.append(np.mean(norm["sd"]))
    ref_gps = float(np.mean(ref_sds))  # expected GPS for normal subject

    # Reference SD: inter-subject variability of GPS itself (~2-3 deg)
    # Estimated as half the mean SD (conservative estimate)
    ref_sd = ref_gps * 0.5
    if ref_sd < 0.5:
        ref_sd = 0.5  # floor to avoid division issues

    result = {}
    for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
        gps_val = gps[key]
        gdi_key = key.replace("gps_2d", "gdi_2d")
        if gps_val is not None:
            gdi = 100.0 - 10.0 * (gps_val - ref_gps) / ref_sd
            result[gdi_key] = round(gdi, 1)
        else:
            result[gdi_key] = None

    result["note"] = (
        "2D sagittal-plane adaptation of the Gait Deviation Index "
        "(Schwartz & Rozumalski 2008). Normal gait scores ~100; "
        "pathological gait scores below 100. Not equivalent to the "
        "standard 9-variable 3-plane GDI. Use for screening only."
    )

    return result


def movement_analysis_profile(cycles: dict, stratum: str = "adult") -> dict:
    """Compute Movement Analysis Profile for visualization.

    Organizes GVS values by joint for barplot visualization, sorted
    alphabetically by joint name.

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`.
    stratum : str
        Normative stratum (default ``'adult'``).

    Returns
    -------
    dict
        Keys: ``'joints'`` (list of joint names, sorted),
        ``'left'`` (list of GVS values), ``'right'`` (list of GVS
        values), ``'gps_2d'`` (overall GPS-2D score).

    References
    ----------
    Baker et al. (2009). Adapted for 2D sagittal-plane analysis.
    """
    gvs = gait_variable_scores(cycles, stratum)
    gps = gait_profile_score_2d(cycles, stratum)

    # Sort joints alphabetically for consistent visualization
    joints_sorted = sorted(_SCORE_JOINTS)

    left_vals = []
    right_vals = []
    for joint in joints_sorted:
        left_vals.append(gvs["left"].get(joint))
        right_vals.append(gvs["right"].get(joint))

    return {
        "joints": joints_sorted,
        "left": left_vals,
        "right": right_vals,
        "gps_2d": gps["gps_2d_overall"],
    }
