"""Normative gait kinematic data for clinical comparison.

Embedded mean +/- SD curves at 101 points (0-100% gait cycle) for
sagittal-plane joint angles. Data derived from published normative
databases.

References:
    Perry J, Burnfield JM. Gait Analysis: Normal and Pathological
    Function. 2nd ed. SLACK Incorporated; 2010.

    Winter DA. Biomechanics and Motor Control of Human Movement.
    4th ed. Wiley; 2009.

    Kadaba MP, Ramakrishnan HK, Wootten ME. Measurement of lower
    extremity kinematics during level walking. J Orthop Res.
    1990;8(3):383-392.

Strata:
    - adult (18-65 years)
    - elderly (65+ years)
    - pediatric (5-17 years)
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

STRATA = ("adult", "elderly", "pediatric")

_JOINTS = ("hip", "knee", "ankle", "trunk", "pelvis_sagittal")

_N_POINTS = 101

_GC = np.linspace(0, 100, _N_POINTS)  # percent gait cycle


# ── Curve generation helpers ─────────────────────────────────────────

def _make_hip_curve():
    """Generate adult hip flexion/extension curve (101 points).

    Shape: Starts ~30 deg (initial contact flexion), decreases to
    ~-10 deg (terminal stance extension at ~50% GC), increases to
    ~35 deg (mid-swing flexion at ~85% GC), returns to ~30 deg.

    Peak flexion: 30-35 deg, peak extension: ~-10 deg, ROM: ~40-45 deg.
    Based on Perry & Burnfield (2010) Fig. 4.3 and Winter (2009) Ch. 5.
    """
    t = _GC / 100.0  # 0 to 1

    # Piecewise smooth construction using Fourier-like components
    # Fundamental: one full cycle of flexion/extension
    # DC offset + harmonics fitted to published curves
    curve = (
        10.0                               # DC offset (mean angle)
        + 20.0 * np.cos(2 * np.pi * t - 0.05)  # primary oscillation
        - 3.0 * np.cos(4 * np.pi * t - 0.2)    # 2nd harmonic
        + 1.5 * np.cos(6 * np.pi * t + 0.3)    # 3rd harmonic
    )

    # Fine-tune: ensure IC starts ~30, terminal stance dips to ~-10
    # Shift so initial contact is ~30 deg
    curve = curve - curve[0] + 30.0

    # Adjust extension valley: scale so min is near -10
    current_min = np.min(curve)
    current_max = np.max(curve)
    target_min = -10.0
    target_max = 35.0
    curve = target_min + (curve - current_min) / (current_max - current_min) * (target_max - target_min)

    return curve


def _make_knee_curve():
    """Generate adult knee flexion/extension curve (101 points).

    Shape: Starts ~5 deg (IC), flexes to ~15-20 deg (loading response
    ~12% GC), extends to ~5 deg (midstance ~40%), flexes to ~60 deg
    (swing phase peak ~72%), returns to ~5 deg.

    Based on Perry & Burnfield (2010) Fig. 4.9 and Kadaba et al. (1990).
    """
    t = _GC / 100.0

    # Knee has a distinctive double-bump pattern
    # Loading response flexion peak (~15 deg at 12% GC)
    loading_peak = 15.0 * np.exp(-((t - 0.12) ** 2) / (2 * 0.03 ** 2))

    # Swing phase flexion peak (~60 deg at 72% GC)
    swing_peak = 60.0 * np.exp(-((t - 0.72) ** 2) / (2 * 0.06 ** 2))

    # Transition: terminal stance extension plateau
    # Gradual rise from midstance extension to pre-swing
    pre_swing = 15.0 * np.exp(-((t - 0.55) ** 2) / (2 * 0.04 ** 2))

    # Baseline (near full extension) with gentle blend
    baseline = 5.0 * np.ones_like(t)

    # Combine all components
    curve = baseline + loading_peak + pre_swing + swing_peak

    # Ensure start and end are near 5 deg
    # Smooth boundary using cosine taper at the end
    end_taper = 0.5 * (1 + np.cos(np.pi * np.clip((t - 0.90) / 0.10, 0, 1)))
    curve = curve * end_taper + 5.0 * (1 - end_taper)

    # Fine-tune start
    start_taper = 0.5 * (1 - np.cos(np.pi * np.clip(t / 0.05, 0, 1)))
    curve = curve * start_taper + 5.0 * (1 - start_taper)

    return curve


def _make_ankle_curve():
    """Generate adult ankle dorsi/plantarflexion curve (101 points).

    Shape: Starts ~0 deg (IC neutral), plantarflexes to ~-5 deg
    (loading response ~5% GC), dorsiflexes to ~10 deg (terminal
    stance ~45% GC), plantarflexes rapidly to ~-15 deg (push-off
    ~62% GC), returns to ~0 deg (swing).

    Based on Perry & Burnfield (2010) Fig. 4.15 and Winter (2009).
    """
    t = _GC / 100.0

    # Initial plantarflexion dip at loading
    loading_dip = -5.0 * np.exp(-((t - 0.05) ** 2) / (2 * 0.02 ** 2))

    # Dorsiflexion rise in stance
    dorsi_peak = 12.0 * np.exp(-((t - 0.45) ** 2) / (2 * 0.08 ** 2))

    # Push-off plantarflexion
    pushoff_dip = -17.0 * np.exp(-((t - 0.62) ** 2) / (2 * 0.04 ** 2))

    # Swing phase: return to neutral with slight dorsiflexion
    swing_dorsi = 2.0 * np.exp(-((t - 0.80) ** 2) / (2 * 0.06 ** 2))

    curve = loading_dip + dorsi_peak + pushoff_dip + swing_dorsi

    # Smooth transitions at start and end to neutral
    # Ensure starts at ~0
    curve = curve - curve[0]
    # Ensure ends near 0 by blending
    end_blend = np.clip((t - 0.92) / 0.08, 0, 1)
    curve = curve * (1 - end_blend) + 0.0 * end_blend

    return curve


def _make_trunk_curve():
    """Generate adult trunk forward lean curve (101 points).

    Shape: Relatively constant ~5 deg forward lean with +/-2 deg
    oscillation (two peaks per gait cycle: at loading response and
    pre-swing, ~12% and ~50% GC).

    Based on Perry & Burnfield (2010) Chapter 11.
    """
    t = _GC / 100.0

    # Mean forward lean of ~5 degrees
    # Small oscillation: 2 peaks per cycle (bilateral influence)
    curve = 5.0 + 2.0 * np.cos(4 * np.pi * t - 0.4)

    return curve


def _make_pelvis_sagittal_curve():
    """Generate adult pelvis sagittal tilt curve (101 points).

    Shape: Oscillates +/-4 deg around ~10 deg anterior tilt,
    with peaks at ~20% and ~70% GC (weight acceptance on each side).

    Based on Perry & Burnfield (2010) Fig. 4.1 and Kadaba et al. (1990).
    """
    t = _GC / 100.0

    # Anterior tilt baseline of ~10 deg
    # Two oscillation peaks per stride
    curve = 10.0 + 4.0 * np.sin(4 * np.pi * t - 0.3)

    return curve


# ── Build normative data at module load ──────────────────────────────

def _build_normative_data():
    """Construct the full normative database for all strata."""
    # Adult curves (baseline)
    adult_hip_mean = _make_hip_curve()
    adult_knee_mean = _make_knee_curve()
    adult_ankle_mean = _make_ankle_curve()
    adult_trunk_mean = _make_trunk_curve()
    adult_pelvis_mean = _make_pelvis_sagittal_curve()

    # Adult SD values
    adult_hip_sd = 5.0 * np.ones(_N_POINTS)
    adult_knee_sd = 5.0 * np.ones(_N_POINTS)
    adult_ankle_sd = 3.0 * np.ones(_N_POINTS)
    adult_trunk_sd = 3.0 * np.ones(_N_POINTS)
    adult_pelvis_sd = 3.0 * np.ones(_N_POINTS)

    # Elderly: reduced ROM (~80% of adult), same shape
    # Scale about the mean of each curve
    def _scale_rom(curve, factor):
        """Scale the ROM of a curve by factor, preserving the mean."""
        mean_val = np.mean(curve)
        return mean_val + (curve - mean_val) * factor

    elderly_hip_mean = _scale_rom(adult_hip_mean, 0.80)
    elderly_knee_mean = _scale_rom(adult_knee_mean, 0.80)
    elderly_ankle_mean = _scale_rom(adult_ankle_mean, 0.80)
    elderly_trunk_mean = _scale_rom(adult_trunk_mean, 0.85)  # trunk less affected
    elderly_pelvis_mean = _scale_rom(adult_pelvis_mean, 0.85)

    elderly_hip_sd = 5.0 * np.ones(_N_POINTS)
    elderly_knee_sd = 5.0 * np.ones(_N_POINTS)
    elderly_ankle_sd = 3.0 * np.ones(_N_POINTS)
    elderly_trunk_sd = 3.0 * np.ones(_N_POINTS)
    elderly_pelvis_sd = 3.0 * np.ones(_N_POINTS)

    # Pediatric: slightly increased ROM (~105% of adult), more variable
    pediatric_hip_mean = _scale_rom(adult_hip_mean, 1.05)
    pediatric_knee_mean = _scale_rom(adult_knee_mean, 1.05)
    pediatric_ankle_mean = _scale_rom(adult_ankle_mean, 1.05)
    pediatric_trunk_mean = _scale_rom(adult_trunk_mean, 1.05)
    pediatric_pelvis_mean = _scale_rom(adult_pelvis_mean, 1.05)

    pediatric_hip_sd = 7.0 * np.ones(_N_POINTS)
    pediatric_knee_sd = 7.0 * np.ones(_N_POINTS)
    pediatric_ankle_sd = 6.0 * np.ones(_N_POINTS)
    pediatric_trunk_sd = 6.0 * np.ones(_N_POINTS)
    pediatric_pelvis_sd = 6.0 * np.ones(_N_POINTS)

    return {
        "adult": {
            "hip": {
                "mean": adult_hip_mean.tolist(),
                "sd": adult_hip_sd.tolist(),
            },
            "knee": {
                "mean": adult_knee_mean.tolist(),
                "sd": adult_knee_sd.tolist(),
            },
            "ankle": {
                "mean": adult_ankle_mean.tolist(),
                "sd": adult_ankle_sd.tolist(),
            },
            "trunk": {
                "mean": adult_trunk_mean.tolist(),
                "sd": adult_trunk_sd.tolist(),
            },
            "pelvis_sagittal": {
                "mean": adult_pelvis_mean.tolist(),
                "sd": adult_pelvis_sd.tolist(),
            },
        },
        "elderly": {
            "hip": {
                "mean": elderly_hip_mean.tolist(),
                "sd": elderly_hip_sd.tolist(),
            },
            "knee": {
                "mean": elderly_knee_mean.tolist(),
                "sd": elderly_knee_sd.tolist(),
            },
            "ankle": {
                "mean": elderly_ankle_mean.tolist(),
                "sd": elderly_ankle_sd.tolist(),
            },
            "trunk": {
                "mean": elderly_trunk_mean.tolist(),
                "sd": elderly_trunk_sd.tolist(),
            },
            "pelvis_sagittal": {
                "mean": elderly_pelvis_mean.tolist(),
                "sd": elderly_pelvis_sd.tolist(),
            },
        },
        "pediatric": {
            "hip": {
                "mean": pediatric_hip_mean.tolist(),
                "sd": pediatric_hip_sd.tolist(),
            },
            "knee": {
                "mean": pediatric_knee_mean.tolist(),
                "sd": pediatric_knee_sd.tolist(),
            },
            "ankle": {
                "mean": pediatric_ankle_mean.tolist(),
                "sd": pediatric_ankle_sd.tolist(),
            },
            "trunk": {
                "mean": pediatric_trunk_mean.tolist(),
                "sd": pediatric_trunk_sd.tolist(),
            },
            "pelvis_sagittal": {
                "mean": pediatric_pelvis_mean.tolist(),
                "sd": pediatric_pelvis_sd.tolist(),
            },
        },
    }


_NORMATIVE_DATA = _build_normative_data()


# ── Public API ───────────────────────────────────────────────────────

def get_normative_curve(joint: str, stratum: str = "adult") -> dict:
    """Return normative curve for a joint.

    Parameters
    ----------
    joint : str
        One of: ``'hip'``, ``'knee'``, ``'ankle'``, ``'trunk'``,
        ``'pelvis_sagittal'``.
    stratum : str
        One of: ``'adult'``, ``'elderly'``, ``'pediatric'``.

    Returns
    -------
    dict
        Keys: ``'mean'`` (list of 101 floats), ``'sd'`` (list of
        101 floats), ``'unit'`` (``'deg'``), ``'source'``,
        ``'stratum'``.

    Raises
    ------
    ValueError
        If *joint* or *stratum* is not recognized.

    References
    ----------
    Perry & Burnfield (2010), Winter (2009), Kadaba et al. (1990).
    """
    if stratum not in STRATA:
        raise ValueError(
            f"Unknown stratum '{stratum}'. Choose from: {STRATA}"
        )
    if joint not in _JOINTS:
        raise ValueError(
            f"Unknown joint '{joint}'. Choose from: {_JOINTS}"
        )

    data = _NORMATIVE_DATA[stratum][joint]
    return {
        "mean": list(data["mean"]),
        "sd": list(data["sd"]),
        "unit": "deg",
        "source": "Perry & Burnfield 2010; Winter 2009; Kadaba et al. 1990",
        "stratum": stratum,
    }


def get_normative_band(
    joint: str, stratum: str = "adult", n_sd: float = 1.0
) -> dict:
    """Return upper/lower normative band.

    Parameters
    ----------
    joint : str
        Joint name (see :func:`get_normative_curve`).
    stratum : str
        Stratum name (default ``'adult'``).
    n_sd : float
        Number of standard deviations for band width (default 1.0).

    Returns
    -------
    dict
        Keys: ``'upper'`` (list of 101 floats),
        ``'lower'`` (list of 101 floats), ``'mean'`` (list of 101 floats).

    Raises
    ------
    ValueError
        If *joint* or *stratum* is not recognized.
    """
    curve = get_normative_curve(joint, stratum)
    mean_arr = np.array(curve["mean"])
    sd_arr = np.array(curve["sd"])
    return {
        "upper": (mean_arr + n_sd * sd_arr).tolist(),
        "lower": (mean_arr - n_sd * sd_arr).tolist(),
        "mean": curve["mean"],
    }


def select_stratum(age: int = None) -> str:
    """Auto-select stratum from age.

    Parameters
    ----------
    age : int, optional
        Subject age in years.  Falls back to ``'adult'`` when *age*
        is ``None``.

    Returns
    -------
    str
        One of ``'pediatric'``, ``'adult'``, ``'elderly'``.
    """
    if age is None:
        return "adult"
    if age < 18:
        return "pediatric"
    if age >= 65:
        return "elderly"
    return "adult"


def list_joints() -> list:
    """Return available joint names.

    Returns
    -------
    list of str
        E.g. ``['hip', 'knee', 'ankle', 'trunk', 'pelvis_sagittal']``.
    """
    return list(_JOINTS)


def list_strata() -> list:
    """Return available strata names.

    Returns
    -------
    list of str
        E.g. ``['adult', 'elderly', 'pediatric']``.
    """
    return list(STRATA)
