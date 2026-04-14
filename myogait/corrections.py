"""Post-hoc angle corrections for myogait.

.. warning::
   **Bias corrections can hide pathological gait signatures.**

   The ``apply_{hip,knee,ankle}_bias_correction`` functions in this module
   apply frozen LASSO coefficients trained on *healthy young adults*
   vs Vicon ground truth.  They encode the **average bias** of pose
   estimators on typical gait.  When applied to a patient with
   neuromuscular disease (DMD, CMT, SMA, myotonic dystrophy, etc.) or
   any pathology that alters the kinematic pattern, they will
   artificially "restore" a healthy-looking curve at exactly the phases
   where the clinical sign is visible:

   - knee flexion swing peak (60–75 % cycle) — masked in DMD, CMT
   - ankle push-off plantaflexion (55–75 % cycle) — masked in drop foot
   - hip extension end-stance — masked in hip weakness compensations

   **Rule of thumb.**  Use these corrections only when you want to
   benchmark your pipeline against a healthy Vicon reference, or when
   the downstream question explicitly assumes a healthy population.
   **For clinical reading of pathological gait, skip the bias
   corrections entirely** and keep only :func:`apply_perspective_correction`
   (zero-parameter, pure geometry, session-local, safe on any
   population).  The uncorrected signal preserves pathological
   signatures.

   :func:`apply_perspective_correction` is always safe because it is
   physics-only: it undoes orthographic projection foreshortening using
   segment lengths from the current session.  It adds no prior from the
   training population.

This module provides two correction families applied to joint angles
after ``compute_angles()``:

**perspective_correction** — ``apply_perspective_correction(data)``
    Zero-parameter geometric correction for hip and knee flexion.
    Rationale: under orthographic projection, a segment tilted out of the
    sagittal plane by angle α has its projected length reduced by a
    factor ``cos α``.  The observed 2D sagittal joint angle
    θ\\ :sub:`2D` is related to the true 3D angle θ\\ :sub:`3D` by

    .. math::  \\theta_{3D} \\approx \\mathrm{atan2}(\\sin\\theta_{2D},
               \\cos\\theta_{2D} \\cdot \\cos\\alpha)

    with ``cos α`` recovered from observed segment length divided by its
    session 95-th percentile.  For the hip we use the thigh tilt alone;
    for the knee we take the most-foreshortened of (thigh, shank).
    The ankle is handled by ``apply_ankle_bias_correction`` instead.

    Typical gain: +10 to +20 % RMSE on hip/knee across Sapiens and
    MediaPipe on healthy adult gait.

**ankle_bias_correction** — ``apply_ankle_bias_correction(data, cycles)``
    Empirical correction for the ankle push-off underestimation that
    appears in all tested pose estimators.  Adds a two-term Fourier
    correction indexed by normalized gait phase:

    .. math::  \\theta_{\\text{corr}}(\\varphi) = \\theta(\\varphi)
               - \\bigl[ a_1 \\sin(2\\pi\\varphi)
                         + a_2 \\sin(4\\pi\\varphi) \\bigr]

    Coefficients were fitted with LASSO (α=0.3) on 9 healthy adult
    subjects × 2 pose estimators (Sapiens-quick, MediaPipe) and frozen
    as **ankle_bias_v1**.

    Typical gain: +30 % RMSE on held-out subjects.

    **Safety note.**  This is an empirical average bias.  It can mask
    real ankle anomalies in pathological gait (stiff ankle, drop-foot,
    ankle fusion).  Use it for healthy-reference comparison only; retain
    the uncorrected signal for clinical screening.

Both corrections operate in-place on the ``data["angles"]["frames"]``
list.  Calling either function twice is a no-op: a marker is set in
``data["angles"]`` to indicate which corrections have been applied.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Frozen LASSO coefficients (standardised features) ────────────────
# Fit: 9 subjects × 2 pose estimators × 2 sides = 36 cases.
# Target: ε(φ) = myogait_cycle_mean − vicon_cycle_mean at φ ∈ [0, 1].
# Features: [sin(2πφ), cos(2πφ), sin(4πφ), cos(4πφ)] then z-scored.
# Regularization: Lasso(alpha=0.3, max_iter=20000).

_SCALER_STD = 0.707106781  # std of sin/cos uniformly sampled on [0,1]
_SCALER_MEAN = [0.0, 0.0, 0.0, 0.0]
_SCALER_SCALE = [_SCALER_STD] * 4
_FOURIER_FEATS = ["sin_2pi", "cos_2pi", "sin_4pi", "cos_4pi"]

ANKLE_BIAS_V1 = {
    "name": "ankle_bias_v1",
    "description": (
        "Universal ankle push-off bias correction for pose estimators, "
        "fitted on 9 healthy adults, Sapiens+MediaPipe. Freeze date: "
        "2026-04-14."
    ),
    "feature_names": _FOURIER_FEATS,
    "coef_standardized": [-1.398, -0.000, +2.508, -0.056],
    "intercept": 0.0,
    "scaler_mean": list(_SCALER_MEAN),
    "scaler_scale": list(_SCALER_SCALE),
    "limitations": [
        "Valid for healthy adult gait at preferred walking speed.",
        "May mask pathological anomalies (stiff ankle, drop-foot, "
        "absent push-off in gastrocnemius weakness, CMT, early DMD). "
        "DO NOT apply when reading patient gait clinically — the push-off "
        "correction adds ~5° of plantaflexion at 60-75%% cycle that may "
        "not exist in the patient's real kinematics.",
        "Retain uncorrected signal for clinical screening.",
    ],
}

HIP_BIAS_V1 = {
    "name": "hip_bias_v1",
    "description": (
        "Universal hip flexion residual bias correction for pose estimators, "
        "fitted on 12 healthy adults, Sapiens+MediaPipe, after "
        "apply_perspective_correction (M1). Freeze date: 2026-04-14."
    ),
    "feature_names": _FOURIER_FEATS,
    "coef_standardized": [+0.208, +3.338, -0.000, -1.468],
    "intercept": 0.0,
    "scaler_mean": list(_SCALER_MEAN),
    "scaler_scale": list(_SCALER_SCALE),
    "limitations": [
        "Valid for healthy adult gait at preferred walking speed.",
        "Apply AFTER apply_perspective_correction — the M1 residual is the "
        "target the LASSO was trained on.",
        "May mask pathological anomalies (hip flexion contracture, "
        "antalgic compensations, Trendelenburg, etc.). "
        "DO NOT apply when reading patient gait clinically.",
        "Retain uncorrected signal for clinical screening.",
    ],
}

KNEE_BIAS_V1 = {
    "name": "knee_bias_v1",
    "description": (
        "Universal knee flexion residual bias correction for pose estimators, "
        "fitted on 12 healthy adults, Sapiens+MediaPipe, after "
        "apply_perspective_correction (M1). Freeze date: 2026-04-14."
    ),
    "feature_names": _FOURIER_FEATS,
    "coef_standardized": [+3.251, +1.207, -2.989, +4.170],
    "intercept": 0.0,
    "scaler_mean": list(_SCALER_MEAN),
    "scaler_scale": list(_SCALER_SCALE),
    "limitations": [
        "Valid for healthy adult gait at preferred walking speed.",
        "Apply AFTER apply_perspective_correction — the M1 residual is the "
        "target the LASSO was trained on.",
        "May mask pathological anomalies (reduced knee flex in DMD/CMT, "
        "stiff-knee gait, genu recurvatum). "
        "DO NOT apply when reading patient gait clinically — the swing "
        "peak at 60-75%% is precisely where this correction acts and "
        "where clinical signs of neuromuscular disease appear.",
        "Retain uncorrected signal for clinical screening.",
    ],
}

# Registry for generic lookup by joint
_BIAS_MODELS = {
    "hip_v1":   HIP_BIAS_V1,
    "knee_v1":  KNEE_BIAS_V1,
    "ankle_v1": ANKLE_BIAS_V1,
}


# ── Helpers ──────────────────────────────────────────────────────────

_LANDMARK_TRIPLETS = {
    "L": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "R": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
}


def _segment_lengths(data: dict) -> dict:
    """Return per-frame segment lengths (in pixel units) for L and R sides.

    Returns dict keyed by side with keys ``thigh`` and ``shank``, each a
    numpy array of length N = number of frames (NaN where landmarks are
    missing).
    """
    meta = data.get("meta") or {}
    w = float(meta.get("width", 1.0))
    h = float(meta.get("height", 1.0))
    frames = data.get("frames", [])
    N = len(frames)
    out: dict[str, dict[str, np.ndarray]] = {}
    for side, (hip_n, knee_n, ankle_n) in _LANDMARK_TRIPLETS.items():
        thigh = np.full(N, np.nan)
        shank = np.full(N, np.nan)
        for i, f in enumerate(frames):
            lm = f.get("landmarks") or {}
            h_lm = lm.get(hip_n); k_lm = lm.get(knee_n); a_lm = lm.get(ankle_n)
            if (isinstance(h_lm, dict) and isinstance(k_lm, dict)
                    and h_lm.get("x") is not None and k_lm.get("x") is not None):
                dx = (h_lm["x"] - k_lm["x"]) * w
                dy = (h_lm["y"] - k_lm["y"]) * h
                thigh[i] = float(np.hypot(dx, dy))
            if (isinstance(k_lm, dict) and isinstance(a_lm, dict)
                    and k_lm.get("x") is not None and a_lm.get("x") is not None):
                dx = (k_lm["x"] - a_lm["x"]) * w
                dy = (k_lm["y"] - a_lm["y"]) * h
                shank[i] = float(np.hypot(dx, dy))
        out[side] = {"thigh": thigh, "shank": shank}
    return out


def _cos_alpha(length: np.ndarray, *, floor: float = 0.3) -> np.ndarray:
    """Foreshortening factor cos α = L / L_p95, clipped to [floor, 1]."""
    valid = length[~np.isnan(length)]
    if valid.size < 5:
        return np.ones_like(length)
    ref = float(np.nanpercentile(valid, 95))
    if ref <= 0:
        return np.ones_like(length)
    return np.clip(length / ref, floor, 1.0)


def _apply_m1(theta_deg: float, cos_a: float, *, clip_deg: float = 80.0) -> float:
    """Inverse orthographic projection: θ_corr = atan2(sin θ, cos θ · cos α)."""
    if theta_deg is None or np.isnan(theta_deg):
        return theta_deg
    t = np.radians(float(np.clip(theta_deg, -clip_deg, clip_deg)))
    return float(np.degrees(np.arctan2(np.sin(t), np.cos(t) * cos_a)))


# ── Public API ───────────────────────────────────────────────────────


def apply_perspective_correction(data: dict) -> dict:
    """Apply zero-parameter M1 perspective correction to hip and knee.

    The correction assumes a sagittal camera view and healthy segment
    length statistics: ``cos α`` for each frame is estimated as the
    observed segment length divided by its session 95-th percentile.

    Parameters
    ----------
    data : dict
        Pivot JSON dict that has been through ``compute_angles()``.
        Modified in place: ``data["angles"]["frames"][i]["hip_{L,R}"]``
        and ``["knee_{L,R}"]`` are replaced by their corrected values.

    Returns
    -------
    dict
        The same *data* dict with corrections applied and marker
        ``data["angles"]["perspective_corrected"] = True`` set.

    Notes
    -----
    * Safe to call once.  If already applied (marker present), this
      function is a no-op.
    * The ankle is not touched — use :func:`apply_ankle_bias_correction`.
    """
    if "angles" not in data or "frames" not in data["angles"]:
        raise ValueError("apply_perspective_correction requires compute_angles() output.")

    angles_meta = data["angles"]
    if angles_meta.get("perspective_corrected"):
        logger.info("perspective correction already applied — skipping.")
        return data

    seg = _segment_lengths(data)
    cos_a_per_side = {}
    for side in ("L", "R"):
        cos_t = _cos_alpha(seg[side]["thigh"])
        cos_s = _cos_alpha(seg[side]["shank"])
        cos_a_per_side[side] = {
            "hip":  cos_t,            # hip depends on thigh only
            "knee": np.minimum(cos_t, cos_s),  # knee: most foreshortened
        }

    frames = angles_meta["frames"]
    for i, af in enumerate(frames):
        for side in ("L", "R"):
            hip_key = f"hip_{side}"
            knee_key = f"knee_{side}"
            ca = cos_a_per_side[side]
            if hip_key in af and af[hip_key] is not None:
                af[hip_key] = _apply_m1(af[hip_key], float(ca["hip"][i]))
            if knee_key in af and af[knee_key] is not None:
                af[knee_key] = _apply_m1(af[knee_key], float(ca["knee"][i]))

    angles_meta["perspective_corrected"] = True
    logger.info("Applied M1 perspective correction to hip_{L,R} and knee_{L,R}.")
    return data


def _phase_per_frame(data: dict, cycles: dict) -> dict:
    """Build per-frame phase arrays ∈ [0, 1] for each side.

    Phase is linear within each detected cycle (heel-strike → next
    heel-strike).  Frames outside any cycle get NaN.
    """
    frames = data["angles"]["frames"]
    N = len(frames)
    if not frames:
        return {"L": np.full(0, np.nan), "R": np.full(0, np.nan)}

    frame_idx = np.array([f.get("frame_idx", i) for i, f in enumerate(frames)])
    first_idx = int(frame_idx[0]) if N else 0
    offsets = frame_idx - first_idx

    phase = {"L": np.full(N, np.nan), "R": np.full(N, np.nan)}
    for c in cycles.get("cycles", []):
        side = "L" if c.get("side") == "left" else "R"
        sf = int(c.get("start_frame", 0)) - first_idx
        ef = int(c.get("end_frame", 0)) - first_idx
        if sf < 0 or ef >= N or ef <= sf:
            continue
        n = ef - sf + 1
        phase[side][sf:ef + 1] = np.linspace(0.0, 1.0, n)
    return phase


def _lasso_pred(phase: np.ndarray, model: dict) -> np.ndarray:
    """Evaluate the frozen LASSO correction on a phase vector.

    Returns predicted ε in degrees; NaN where phase is NaN.
    """
    out = np.full_like(phase, np.nan, dtype=float)
    ok = ~np.isnan(phase)
    if not ok.any():
        return out
    phi = phase[ok]
    feats = np.column_stack([
        np.sin(2 * np.pi * phi),
        np.cos(2 * np.pi * phi),
        np.sin(4 * np.pi * phi),
        np.cos(4 * np.pi * phi),
    ])
    scaler_mean = np.asarray(model["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(model["scaler_scale"], dtype=float)
    coef = np.asarray(model["coef_standardized"], dtype=float)
    intercept = float(model.get("intercept", 0.0))
    feats_std = (feats - scaler_mean) / scaler_scale
    out[ok] = feats_std @ coef + intercept
    return out


def _apply_bias_correction_generic(
    data: dict,
    cycles: dict,
    *,
    joint: str,
    model_key: str,
    marker_key: str,
) -> dict:
    """Shared implementation for per-joint Fourier bias corrections."""
    if model_key not in _BIAS_MODELS:
        raise ValueError(
            f"Unknown {joint} bias model '{model_key}'. "
            f"Available: {sorted(_BIAS_MODELS)}"
        )
    if "angles" not in data or "frames" not in data["angles"]:
        raise ValueError(
            f"apply_{joint}_bias_correction requires compute_angles() output."
        )

    angles_meta = data["angles"]
    if angles_meta.get(marker_key):
        logger.info("%s already applied — skipping.", marker_key)
        return data

    model = _BIAS_MODELS[model_key]
    phase = _phase_per_frame(data, cycles)
    eps_L = _lasso_pred(phase["L"], model)
    eps_R = _lasso_pred(phase["R"], model)

    key_L = f"{joint}_L"
    key_R = f"{joint}_R"
    frames = angles_meta["frames"]
    for i, af in enumerate(frames):
        v = af.get(key_L)
        if v is not None and not np.isnan(v) and not np.isnan(eps_L[i]):
            af[key_L] = float(v - eps_L[i])
        v = af.get(key_R)
        if v is not None and not np.isnan(v) and not np.isnan(eps_R[i]):
            af[key_R] = float(v - eps_R[i])

    angles_meta[marker_key] = model["name"]
    logger.info("Applied %s to %s_{L,R}.", model["name"], joint)
    return data


def apply_ankle_bias_correction(
    data: dict,
    cycles: dict,
    *,
    model: str = "v1",
) -> dict:
    """Apply the frozen Fourier LASSO correction to ankle_L and ankle_R.

    .. warning::
       **Do NOT apply to pathological gait for clinical reading.**
       The push-off plantaflexion dip at 60–75 % cycle is injected from
       the healthy reference and will mask drop-foot, gastrocnemius
       weakness and absent push-off in NMD patients. Use only for
       benchmarking vs a healthy Vicon reference.

    See :data:`ANKLE_BIAS_V1` for coefficient provenance and limitations.
    Does NOT require :func:`apply_perspective_correction` to have been
    called first — the ankle correction was trained on the un-M1 signal
    because M1 has a negligible effect on ankle amplitude.
    """
    return _apply_bias_correction_generic(
        data, cycles, joint="ankle", model_key=f"ankle_{model}",
        marker_key="ankle_bias_corrected",
    )


def apply_hip_bias_correction(
    data: dict,
    cycles: dict,
    *,
    model: str = "v1",
) -> dict:
    """Apply the frozen Fourier LASSO correction to hip_L and hip_R.

    .. important::
       This correction must be applied **after**
       :func:`apply_perspective_correction`.  The LASSO coefficients
       were trained on the residual of M1-corrected hip angles vs Vicon.
       Applying it to raw (non-M1) angles will double-count part of the
       projection correction.

    .. warning::
       **Do NOT apply to pathological gait for clinical reading.**
       The correction injects a healthy-population bias pattern and may
       mask hip compensations (Trendelenburg, antalgic, hyperlordosis).
       Use only for benchmarking vs a healthy Vicon reference.

    See :data:`HIP_BIAS_V1` for coefficient provenance and limitations.
    """
    return _apply_bias_correction_generic(
        data, cycles, joint="hip", model_key=f"hip_{model}",
        marker_key="hip_bias_corrected",
    )


def apply_knee_bias_correction(
    data: dict,
    cycles: dict,
    *,
    model: str = "v1",
) -> dict:
    """Apply the frozen Fourier LASSO correction to knee_L and knee_R.

    .. important::
       This correction must be applied **after**
       :func:`apply_perspective_correction`.  The LASSO coefficients
       were trained on the residual of M1-corrected knee angles vs Vicon.

    .. warning::
       **Do NOT apply to pathological gait for clinical reading.**
       This is the most dangerous of the three bias corrections for
       clinical use: it acts on the swing peak flexion (60–75 % cycle),
       which is precisely the phase where reduced knee flexion is the
       hallmark sign of DMD, CMT and stiff-knee gait. The correction
       will artificially restore a normal peak and mask these pathologies.
       Use only for benchmarking vs a healthy Vicon reference.

    See :data:`KNEE_BIAS_V1` for coefficient provenance and limitations.
    """
    return _apply_bias_correction_generic(
        data, cycles, joint="knee", model_key=f"knee_{model}",
        marker_key="knee_bias_corrected",
    )


__all__ = [
    "ANKLE_BIAS_V1",
    "HIP_BIAS_V1",
    "KNEE_BIAS_V1",
    "apply_perspective_correction",
    "apply_ankle_bias_correction",
    "apply_hip_bias_correction",
    "apply_knee_bias_correction",
]
