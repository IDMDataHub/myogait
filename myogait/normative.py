"""Normative gait kinematic data for clinical comparison.

Embedded mean +/- SD curves at 101 points (0-100% gait cycle) for
sagittal-plane and frontal-plane joint angles.  Mean curves and
standard-deviation envelopes are interpolated from key values digitized
from published normative datasets.  Standard deviations vary across the
gait cycle to reflect the well-documented higher inter-subject
variability at range-of-motion extremes (e.g. push-off, swing peak)
compared with mid-stance.

References:
    Perry J, Burnfield JM. Gait Analysis: Normal and Pathological
    Function. 2nd ed. SLACK Incorporated; 2010.
    (Hip: Fig 4.3; Knee: Fig 4.9; Ankle: Fig 4.15; Pelvis: Fig 4.1;
    Trunk: Ch 11; Frontal: Ch 14, Fig 14.5)

    Winter DA. Biomechanics and Motor Control of Human Movement.
    4th ed. Wiley; 2009. (Ch 5 sagittal kinematics)

    Kadaba MP, Ramakrishnan HK, Wootten ME. Measurement of lower
    extremity kinematics during level walking. J Orthop Res.
    1990;8(3):383-392.

    Schwartz MH, Rozumalski A, Trost JP. The effect of walking speed
    on the gait of typically developing children. J Biomech.
    2008;41(8):1639-1650. (Pediatric SD scaling)

Strata:
    - adult (18-65 years)
    - elderly (65+ years)  -- ROM x 0.80, SD x 1.3
    - pediatric (5-17 years) -- ROM x 1.05, SD x 1.4
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

STRATA = ("adult", "elderly", "pediatric")

_JOINTS = (
    "hip", "knee", "ankle", "trunk", "pelvis_sagittal",
    "pelvis_obliquity", "hip_adduction", "knee_valgus",
)

_N_POINTS = 101

_GC = np.linspace(0, 100, _N_POINTS)  # percent gait cycle

# Strata scaling factors
_ELDERLY_ROM_FACTOR = 0.80
_ELDERLY_SD_FACTOR = 1.3
_PEDIATRIC_ROM_FACTOR = 1.05
_PEDIATRIC_SD_FACTOR = 1.4


# ── Digitized reference data (Perry & Burnfield 2010; Winter 2009) ──
#
# Each entry is (percent_gc, mean_deg, sd_deg) digitized from the
# published figures / tables.  Between key points, values are linearly
# interpolated to 101 equally-spaced points covering 0-100 %GC.

_HIP_KEYPOINTS = {
    # %GC    mean   sd
    "gc":   [  0,   10,   30,   50,   62,   75,   85,  100],
    "mean": [ 30,   25,   10,  -10,   -5,   25,   32,   30],
    "sd":   [  5,    5,    4,    5,    6,    6,    5,    5],
}

_KNEE_KEYPOINTS = {
    "gc":   [  0,   12,   25,   40,   55,   70,   85,   95,  100],
    "mean": [  5,   18,    5,    3,   35,   62,   25,    8,    5],
    "sd":   [  4,    5,    4,    4,    6,    7,    6,    5,    4],
}

_ANKLE_KEYPOINTS = {
    "gc":   [  0,    8,   25,   40,   55,   62,   70,   85,  100],
    "mean": [  0,   -5,    5,   10,    5,  -15,   -5,    0,    0],
    "sd":   [  3,    3,    3,    4,    4,    5,    4,    3,    3],
}

_TRUNK_KEYPOINTS = {
    "gc":   [  0,   30,   62,   85,  100],
    "mean": [  3,    1,    5,    2,    3],
    "sd":   [  3,    3,    4,    3,    3],
}

_PELVIS_SAGITTAL_KEYPOINTS = {
    "gc":   [  0,   30,   50,   75,  100],
    "mean": [ 10,    8,   12,    9,   10],
    "sd":   [  3,    3,    3,    3,    3],
}

# Frontal-plane digitized data (Perry & Burnfield 2010 Ch 14, Kadaba 1990)
# SD higher during swing (~4-5 deg), lower during mid-stance (~2-3 deg)

_PELVIS_OBLIQUITY_KEYPOINTS = {
    "gc":   [  0,   15,   30,   50,   60,   75,   90,  100],
    "mean": [  2,    4,    3,    0,   -1,    1,    3,    2],
    "sd":   [  3,    3,    2,    3,    4,    5,    4,    3],
}

_HIP_ADDUCTION_KEYPOINTS = {
    "gc":   [  0,   10,   30,   50,   60,   75,   90,  100],
    "mean": [  2,    6,    5,    4,    1,    2,    3,    2],
    "sd":   [  3,    3,    2,    3,    4,    5,    4,    3],
}

_KNEE_VALGUS_KEYPOINTS = {
    "gc":   [  0,   12,   30,   50,   60,   75,   90,  100],
    "mean": [  3,    5,    3,    2,    2,    1,    2,    3],
    "sd":   [  3,    3,    2,    3,    4,    5,    4,    3],
}


# ── Interpolation helper ────────────────────────────────────────────

def _interp_keypoints(keypoints: dict) -> tuple:
    """Linearly interpolate digitized key values to 101 gait-cycle points.

    Parameters
    ----------
    keypoints : dict
        Must contain ``'gc'``, ``'mean'``, and ``'sd'`` keys, each a
        list of floats.  ``'gc'`` values must span 0 to 100.

    Returns
    -------
    mean : np.ndarray, shape (101,)
    sd : np.ndarray, shape (101,)
    """
    gc_kp = np.asarray(keypoints["gc"], dtype=float)
    mean_kp = np.asarray(keypoints["mean"], dtype=float)
    sd_kp = np.asarray(keypoints["sd"], dtype=float)

    mean = np.interp(_GC, gc_kp, mean_kp)
    sd = np.interp(_GC, gc_kp, sd_kp)

    return mean, sd


# ── Build normative data at module load ──────────────────────────────

def _build_normative_data():
    """Construct the full normative database for all strata."""

    # ── Adult curves (digitized reference data) ──────────────────────
    adult_hip_mean, adult_hip_sd = _interp_keypoints(_HIP_KEYPOINTS)
    adult_knee_mean, adult_knee_sd = _interp_keypoints(_KNEE_KEYPOINTS)
    adult_ankle_mean, adult_ankle_sd = _interp_keypoints(_ANKLE_KEYPOINTS)
    adult_trunk_mean, adult_trunk_sd = _interp_keypoints(_TRUNK_KEYPOINTS)
    adult_pelvis_mean, adult_pelvis_sd = _interp_keypoints(
        _PELVIS_SAGITTAL_KEYPOINTS
    )

    # Frontal plane
    adult_pelvis_obliquity_mean, adult_pelvis_obliquity_sd = (
        _interp_keypoints(_PELVIS_OBLIQUITY_KEYPOINTS)
    )
    adult_hip_adduction_mean, adult_hip_adduction_sd = _interp_keypoints(
        _HIP_ADDUCTION_KEYPOINTS
    )
    adult_knee_valgus_mean, adult_knee_valgus_sd = _interp_keypoints(
        _KNEE_VALGUS_KEYPOINTS
    )

    # ── Strata helper ────────────────────────────────────────────────

    def _scale_rom(curve, factor):
        """Scale the ROM of a curve by *factor*, preserving the mean."""
        mean_val = np.mean(curve)
        return mean_val + (curve - mean_val) * factor

    # ── Elderly: reduced ROM, wider SD ───────────────────────────────
    elderly_hip_mean = _scale_rom(adult_hip_mean, _ELDERLY_ROM_FACTOR)
    elderly_knee_mean = _scale_rom(adult_knee_mean, _ELDERLY_ROM_FACTOR)
    elderly_ankle_mean = _scale_rom(adult_ankle_mean, _ELDERLY_ROM_FACTOR)
    elderly_trunk_mean = _scale_rom(adult_trunk_mean, _ELDERLY_ROM_FACTOR)
    elderly_pelvis_mean = _scale_rom(adult_pelvis_mean, _ELDERLY_ROM_FACTOR)

    elderly_hip_sd = adult_hip_sd * _ELDERLY_SD_FACTOR
    elderly_knee_sd = adult_knee_sd * _ELDERLY_SD_FACTOR
    elderly_ankle_sd = adult_ankle_sd * _ELDERLY_SD_FACTOR
    elderly_trunk_sd = adult_trunk_sd * _ELDERLY_SD_FACTOR
    elderly_pelvis_sd = adult_pelvis_sd * _ELDERLY_SD_FACTOR

    elderly_pelvis_obliquity_mean = _scale_rom(
        adult_pelvis_obliquity_mean, _ELDERLY_ROM_FACTOR
    )
    elderly_hip_adduction_mean = _scale_rom(
        adult_hip_adduction_mean, _ELDERLY_ROM_FACTOR
    )
    elderly_knee_valgus_mean = _scale_rom(
        adult_knee_valgus_mean, _ELDERLY_ROM_FACTOR
    )

    elderly_pelvis_obliquity_sd = adult_pelvis_obliquity_sd * _ELDERLY_SD_FACTOR
    elderly_hip_adduction_sd = adult_hip_adduction_sd * _ELDERLY_SD_FACTOR
    elderly_knee_valgus_sd = adult_knee_valgus_sd * _ELDERLY_SD_FACTOR

    # ── Pediatric: slightly increased ROM, wider SD ──────────────────
    pediatric_hip_mean = _scale_rom(adult_hip_mean, _PEDIATRIC_ROM_FACTOR)
    pediatric_knee_mean = _scale_rom(adult_knee_mean, _PEDIATRIC_ROM_FACTOR)
    pediatric_ankle_mean = _scale_rom(adult_ankle_mean, _PEDIATRIC_ROM_FACTOR)
    pediatric_trunk_mean = _scale_rom(adult_trunk_mean, _PEDIATRIC_ROM_FACTOR)
    pediatric_pelvis_mean = _scale_rom(
        adult_pelvis_mean, _PEDIATRIC_ROM_FACTOR
    )

    pediatric_hip_sd = adult_hip_sd * _PEDIATRIC_SD_FACTOR
    pediatric_knee_sd = adult_knee_sd * _PEDIATRIC_SD_FACTOR
    pediatric_ankle_sd = adult_ankle_sd * _PEDIATRIC_SD_FACTOR
    pediatric_trunk_sd = adult_trunk_sd * _PEDIATRIC_SD_FACTOR
    pediatric_pelvis_sd = adult_pelvis_sd * _PEDIATRIC_SD_FACTOR

    pediatric_pelvis_obliquity_mean = _scale_rom(
        adult_pelvis_obliquity_mean, _PEDIATRIC_ROM_FACTOR
    )
    pediatric_hip_adduction_mean = _scale_rom(
        adult_hip_adduction_mean, _PEDIATRIC_ROM_FACTOR
    )
    pediatric_knee_valgus_mean = _scale_rom(
        adult_knee_valgus_mean, _PEDIATRIC_ROM_FACTOR
    )

    pediatric_pelvis_obliquity_sd = (
        adult_pelvis_obliquity_sd * _PEDIATRIC_SD_FACTOR
    )
    pediatric_hip_adduction_sd = (
        adult_hip_adduction_sd * _PEDIATRIC_SD_FACTOR
    )
    pediatric_knee_valgus_sd = (
        adult_knee_valgus_sd * _PEDIATRIC_SD_FACTOR
    )

    # ── Assemble full database ───────────────────────────────────────

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
            "pelvis_obliquity": {
                "mean": adult_pelvis_obliquity_mean.tolist(),
                "sd": adult_pelvis_obliquity_sd.tolist(),
            },
            "hip_adduction": {
                "mean": adult_hip_adduction_mean.tolist(),
                "sd": adult_hip_adduction_sd.tolist(),
            },
            "knee_valgus": {
                "mean": adult_knee_valgus_mean.tolist(),
                "sd": adult_knee_valgus_sd.tolist(),
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
            "pelvis_obliquity": {
                "mean": elderly_pelvis_obliquity_mean.tolist(),
                "sd": elderly_pelvis_obliquity_sd.tolist(),
            },
            "hip_adduction": {
                "mean": elderly_hip_adduction_mean.tolist(),
                "sd": elderly_hip_adduction_sd.tolist(),
            },
            "knee_valgus": {
                "mean": elderly_knee_valgus_mean.tolist(),
                "sd": elderly_knee_valgus_sd.tolist(),
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
            "pelvis_obliquity": {
                "mean": pediatric_pelvis_obliquity_mean.tolist(),
                "sd": pediatric_pelvis_obliquity_sd.tolist(),
            },
            "hip_adduction": {
                "mean": pediatric_hip_adduction_mean.tolist(),
                "sd": pediatric_hip_adduction_sd.tolist(),
            },
            "knee_valgus": {
                "mean": pediatric_knee_valgus_mean.tolist(),
                "sd": pediatric_knee_valgus_sd.tolist(),
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
        ``'pelvis_sagittal'``, ``'pelvis_obliquity'``,
        ``'hip_adduction'``, ``'knee_valgus'``.
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
