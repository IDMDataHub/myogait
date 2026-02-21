"""Clinical gait scores: GVS, GPS-2D, Sagittal Deviation Index, MAP.

Computes standardized gait deviation scores by comparing patient
kinematic curves to normative data.

IMPORTANT: The GPS-2D and Sagittal Deviation Index scores are 2D
sagittal-plane adaptations using 4 variables. They are NOT equivalent
to the standard 9-variable, 3-plane GPS/GDI (Baker et al. 2009,
Schwartz & Rozumalski 2008). Use for screening only.

Standard GPS/GDI require 3D motion capture with:
    9 kinematic variables: pelvis tilt/obliquity/rotation,
    hip flex/abd/rotation, knee flex, ankle dorsi, foot progression.

Our 2D adaptation uses 4 sagittal variables:
    hip flexion, knee flexion, ankle dorsiflexion, trunk angle.

When frontal-plane data is available (pelvis obliquity, hip adduction,
knee valgus), these are optionally included in score computations.

References:
    Baker R, McGinley JL, Schwartz MH, et al. The Gait Profile
    Score and Movement Analysis Profile. Gait Posture.
    2009;30(3):265-269. doi:10.1016/j.gaitpost.2009.05.020

    Schwartz MH, Rozumalski A. The Gait Deviation Index: a new
    comprehensive index of gait pathology. Gait Posture.
    2008;28(3):351-357. doi:10.1016/j.gaitpost.2008.05.001
"""

import warnings

import numpy as np

from .normative import get_normative_curve

# Joints used in the 2D sagittal adaptation
_SCORE_JOINTS = ("hip", "knee", "ankle", "trunk")

# Mapping from cycle summary data keys to normative joint keys for
# frontal plane variables.
_FRONTAL_KEY_MAP = {
    "pelvis_list": "pelvis_obliquity",
    "hip_adduction_L": "hip_adduction",
    "hip_adduction_R": "hip_adduction",
    "knee_valgus_L": "knee_valgus",
    "knee_valgus_R": "knee_valgus",
}

# Frontal keys expected per side in the cycle summary
_FRONTAL_KEYS_LEFT = ("pelvis_list", "hip_adduction_L", "knee_valgus_L")
_FRONTAL_KEYS_RIGHT = ("pelvis_list", "hip_adduction_R", "knee_valgus_R")


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

    When frontal plane data is present in the cycle summary (keys
    ``pelvis_list``, ``hip_adduction_L/R``, ``knee_valgus_L/R``),
    GVS values are also computed for those variables using the
    corresponding frontal normative curves.

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
        Missing joints are set to ``None``.  Frontal keys are included
        only when present in the input data.

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
        # Sagittal joints
        for joint in _SCORE_JOINTS:
            patient = _get_patient_mean(summary, side, joint)
            if patient is None:
                side_gvs[joint] = None
                continue
            norm = get_normative_curve(joint, stratum)
            norm_mean = np.array(norm["mean"])
            # Ensure same length
            if len(patient) != len(norm_mean):
                from numpy import interp, linspace
                x_old = linspace(0, 100, len(patient))
                x_new = linspace(0, 100, 101)
                patient = interp(x_new, x_old, patient)
            side_gvs[joint] = round(_rms(patient - norm_mean), 2)

        # Frontal plane joints (when present)
        frontal_keys = _FRONTAL_KEYS_LEFT if side == "left" else _FRONTAL_KEYS_RIGHT
        for fkey in frontal_keys:
            norm_key = _FRONTAL_KEY_MAP[fkey]
            patient = _get_patient_mean(summary, side, fkey)
            if patient is None:
                continue
            norm = get_normative_curve(norm_key, stratum)
            norm_mean = np.array(norm["mean"])
            if len(patient) != len(norm_mean):
                from numpy import interp, linspace
                x_old = linspace(0, 100, len(patient))
                x_new = linspace(0, 100, 101)
                patient = interp(x_new, x_old, patient)
            side_gvs[fkey] = round(_rms(patient - norm_mean), 2)

        result[side] = side_gvs

    return result


def gait_profile_score_2d(
    cycles: dict,
    stratum: str = "adult",
    include_frontal: bool = True,
) -> dict:
    """Compute 2D Gait Profile Score.

    GPS-2D is the RMS of all GVS values across the sagittal
    joints (hip, knee, ankle, trunk) for each side.  When
    *include_frontal* is ``True`` and frontal GVS values are
    available, they are included in the overall GPS computation:

        GPS-2D = sqrt(mean(GVS_hip^2, GVS_knee^2,
                           GVS_ankle^2, GVS_trunk^2[, GVS_frontal...]))

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`.
    stratum : str
        Normative stratum (default ``'adult'``).
    include_frontal : bool
        If ``True`` (default), include any available frontal-plane
        GVS values in the GPS calculation.

    Returns
    -------
    dict
        Keys: ``'gps_2d_left'``, ``'gps_2d_right'``,
        ``'gps_2d_overall'``, ``'variables_used'``, ``'note'``.

    References
    ----------
    Baker et al. (2009). Adapted for 2D analysis.
    """
    gvs = gait_variable_scores(cycles, stratum)

    # Determine which keys to include per side
    frontal_key_sets = {
        "left": set(_FRONTAL_KEYS_LEFT),
        "right": set(_FRONTAL_KEYS_RIGHT),
    }

    gps = {}
    all_gvs_values = []
    variables_used = set(_SCORE_JOINTS)

    for side in ("left", "right"):
        side_vals = []
        for key, val in gvs[side].items():
            if val is None:
                continue
            # Include sagittal joints always
            if key in _SCORE_JOINTS:
                side_vals.append(val)
            # Include frontal joints only if requested
            elif include_frontal and key in frontal_key_sets[side]:
                side_vals.append(val)
                variables_used.add(key)

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

    gps["variables_used"] = sorted(variables_used)

    if include_frontal and len(variables_used) > len(_SCORE_JOINTS):
        gps["note"] = (
            "2D adaptation using sagittal + frontal variables "
            f"({len(variables_used)} total). Not equivalent to the standard "
            "9-variable 3-plane GPS (Baker et al. 2009). "
            "Use for screening only."
        )
    else:
        gps["note"] = (
            "2D sagittal-plane adaptation using 4 variables "
            "(hip, knee, ankle, trunk). Not equivalent to the standard "
            "9-variable 3-plane GPS (Baker et al. 2009). "
            "Use for screening only."
        )

    return gps


def sagittal_deviation_index(
    cycles: dict,
    stratum: str = "adult",
    include_frontal: bool = True,
) -> dict:
    """Compute the myogait Sagittal Deviation Index (SDI).

    This is a myogait-specific 2D deviation index.
    It is **NOT** the GDI of Schwartz & Rozumalski (2008), which
    requires 9 kinematic variables from 3D motion capture.

    The SDI is a scaled index where normal gait scores approximately
    100 and pathological gait scores below 100:

        SDI = 100 - 10 * (GPS-2D - ref_gps) / ref_sd

    where ``ref_gps`` is the expected GPS-2D for a normative subject
    (mean of per-joint normative SDs) and ``ref_sd`` is the mean of
    those normative SDs (i.e. derived from real normative data rather
    than an arbitrary fraction).

    When *include_frontal* is ``True`` and frontal data is available,
    the GPS computation includes frontal variables, making the score
    more comprehensive.

    The result is clamped to [0, 120].

    Parameters
    ----------
    cycles : dict
        Output of :func:`~myogait.cycles.segment_cycles`.
    stratum : str
        Normative stratum (default ``'adult'``).
    include_frontal : bool
        If ``True`` (default), include frontal-plane variables
        when available in the GPS computation.

    Returns
    -------
    dict
        Keys: ``'gdi_2d_left'``, ``'gdi_2d_right'``,
        ``'gdi_2d_overall'``, ``'note'``.

    Notes
    -----
    This score was previously named ``gait_deviation_index_2d``.
    That name is kept as a deprecated alias for backwards
    compatibility.
    """
    gps = gait_profile_score_2d(cycles, stratum, include_frontal=include_frontal)

    # Determine the set of normative joints for computing reference values.
    # Always use sagittal joints; add frontal joints only if they were
    # actually used in the GPS computation.
    ref_joints = list(_SCORE_JOINTS)
    if include_frontal:
        # Add unique frontal normative keys that appeared in variables_used
        for var in gps.get("variables_used", []):
            norm_key = _FRONTAL_KEY_MAP.get(var)
            if norm_key and norm_key not in ref_joints:
                ref_joints.append(norm_key)

    # Reference GPS: the expected GPS-2D for a normal subject.
    norm_sds = {}
    for joint in ref_joints:
        norm = get_normative_curve(joint, stratum)
        norm_sds[joint] = float(np.mean(norm["sd"]))
    ref_gps = float(np.mean(list(norm_sds.values())))

    # Reference SD: derived from the normative SDs themselves
    ref_sd = max(float(np.std(list(norm_sds.values()))), 1.0)

    result = {}
    for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
        gps_val = gps[key]
        gdi_key = key.replace("gps_2d", "gdi_2d")
        if gps_val is not None:
            gdi = 100.0 - 10.0 * (gps_val - ref_gps) / ref_sd
            # Clamp to [0, 120] -- no negative or astronomic values
            gdi = float(np.clip(gdi, 0.0, 120.0))
            result[gdi_key] = round(gdi, 1)
        else:
            result[gdi_key] = None

    result["note"] = (
        "myogait Sagittal Deviation Index (SDI): a 2D sagittal-plane "
        "deviation index using 4 variables (hip, knee, ankle, trunk). "
        "Normal gait scores ~100; pathological gait scores below 100. "
        "This is NOT the GDI of Schwartz & Rozumalski (2008). "
        "Use for screening only."
    )

    return result


def gait_deviation_index_2d(cycles: dict, stratum: str = "adult") -> dict:
    """Deprecated alias for :func:`sagittal_deviation_index`.

    .. deprecated::
        Use :func:`sagittal_deviation_index` instead.
    """
    warnings.warn(
        "gait_deviation_index_2d is deprecated; use sagittal_deviation_index instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sagittal_deviation_index(cycles, stratum, include_frontal=False)


def movement_analysis_profile(cycles: dict, stratum: str = "adult") -> dict:
    """Compute Movement Analysis Profile for visualization.

    Organizes GVS values by joint for barplot visualization, sorted
    alphabetically by joint name.  When frontal-plane GVS values
    are present, they are included in the profile.

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
    Baker et al. (2009). Adapted for 2D analysis.
    """
    gvs = gait_variable_scores(cycles, stratum)
    gps = gait_profile_score_2d(cycles, stratum)

    # Collect all unique joint/variable keys across both sides
    all_keys = set(_SCORE_JOINTS)
    for side in ("left", "right"):
        all_keys.update(gvs[side].keys())

    # Sort joints alphabetically for consistent visualization
    joints_sorted = sorted(all_keys)

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
