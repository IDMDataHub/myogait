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
from typing import Dict, List, Optional, Tuple

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
            h_lm = lm.get(hip_n)
            k_lm = lm.get(knee_n)
            a_lm = lm.get(ankle_n)
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


def apply_linear_detrend(
    data: dict,
    joints: Optional[List[str]] = None,
) -> dict:
    """Remove a linear drift from each joint-angle time series.

    Sagittal recordings often show a slow offset drift over the video
    (e.g. hip angle sliding by 10-30° from the first to the last cycle)
    that is not a real gait signal but the projection of the subject's
    distance to the camera changing across the frame. Fitting a line
    ``y = a·t + b`` to each joint signal and subtracting the slope
    removes that drift while preserving both the anatomical mean
    (kept intact) and the per-cycle ROM (unchanged).

    Parameters
    ----------
    data : dict
        Pivot JSON dict that has been through ``compute_angles()``.
    joints : list of str, optional
        Joint keys to detrend.  Defaults to the six lower-limb angles
        plus trunk: ``["hip_L", "hip_R", "knee_L", "knee_R",
        "ankle_L", "ankle_R", "trunk_angle"]``.

    Returns
    -------
    dict
        The same *data* dict with detrended angles and
        ``data["angles"]["linear_detrended"] = True`` set.

    Notes
    -----
    * Safe to call once.  If already applied, the function is a no-op.
    * Should be called AFTER perspective correction (``apply_perspective_correction``),
      which removes the frame-by-frame angular effect of perspective;
      this function then mops up the residual DC drift.
    """
    angles = data.get("angles", {})
    if angles.get("linear_detrended"):
        return data

    frames = angles.get("frames", [])
    if not frames:
        return data

    if joints is None:
        joints = ["hip_L", "hip_R", "knee_L", "knee_R",
                  "ankle_L", "ankle_R", "trunk_angle"]

    for key in joints:
        vals = np.array(
            [f.get(key) if f.get(key) is not None else np.nan for f in frames],
            dtype=float,
        )
        mask = ~np.isnan(vals)
        if mask.sum() < 20:
            continue
        idx = np.arange(len(vals))
        slope, intercept = np.polyfit(idx[mask], vals[mask], 1)
        trend = slope * idx + intercept
        mean_v = float(np.mean(vals[mask]))
        for i, f in enumerate(frames):
            v = f.get(key)
            if v is not None and not np.isnan(v):
                f[key] = float(vals[i] - trend[i] + mean_v)

    angles["linear_detrended"] = True
    return data


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


_DEFAULT_CORRECTABLE_LANDMARKS = (
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
)


def _landmark_xy(landmarks: dict, name: str):
    """Extract (x, y) as np.ndarray for a landmark, or None if unusable."""
    lm = landmarks.get(name)
    if lm is None: return None
    if isinstance(lm, dict):
        x, y = lm.get("x"), lm.get("y")
    elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
        x, y = lm[0], lm[1]
    else:
        return None
    if x is None or y is None: return None
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if np.isnan(xf) or np.isnan(yf): return None
    return np.array([xf, yf])


def _pose_anchor(landmarks: dict):
    """Return (mid_hip_xy, thigh_scale) for a landmark dict or None.

    Anchoring on mid-hip and rescaling by the hip↔knee segment length
    makes landmark comparisons camera-independent (kills translation
    and depth-dependent scale).  Falls back to None when any of the
    four required landmarks (LEFT/RIGHT × HIP/KNEE) is missing.
    """
    lh = _landmark_xy(landmarks, "LEFT_HIP")
    rh = _landmark_xy(landmarks, "RIGHT_HIP")
    lk = _landmark_xy(landmarks, "LEFT_KNEE")
    rk = _landmark_xy(landmarks, "RIGHT_KNEE")
    if any(v is None for v in (lh, rh, lk, rk)): return None
    mid_hip = (lh + rh) / 2.0
    mid_knee = (lk + rk) / 2.0
    scale = float(np.linalg.norm(mid_knee - mid_hip))
    if scale < 1e-6: return None
    return mid_hip, scale


def fit_landmark_bias_by_phase(
    sapiens_data: dict,
    vicon_data: dict,
    cycles_sapiens: dict,
    offset_s: float,
    n_bins: int = 10,
    landmarks: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Dict[str, list]]:
    """Fit a per-phase 2D bias between Sapiens landmarks and a Vicon C3D.

    Both dicts must already be aligned by ``offset_s`` such that
    ``vicon_time = sapiens_time + offset_s``.  The video-side
    ``cycles_sapiens`` (as returned by :func:`segment_cycles`) provides
    the phase 0-1 assignment per frame per side.

    Left-side landmarks are binned on the left cycle phase, right-side
    on the right cycle phase — the two feet are 50 % out of phase, so
    binning them on a single side would smear the swing/stance
    contrast that motivates the correction.

    Returns a dict::

        {landmark_name: {"dx": [b_0, ..., b_{n-1}],
                          "dy": [b_0, ..., b_{n-1}],
                          "n":  [count_0, ..., count_{n-1}]}}

    where (dx, dy) is expressed in *thigh-length units* in a mid-hip
    anchored frame (so ``apply_landmark_bias_correction`` can rescale
    it back to normalised image coordinates using the current frame's
    thigh length).
    """
    if landmarks is None:
        landmarks = _DEFAULT_CORRECTABLE_LANDMARKS

    mg_fps = float(sapiens_data.get("meta", {}).get("fps", 30.0))
    vc_fps = float(vicon_data.get("meta", {}).get("fps", 200.0))
    if mg_fps <= 0 or vc_fps <= 0:
        raise ValueError("Invalid fps on sapiens_data or vicon_data")

    phase = _phase_per_frame(sapiens_data, cycles_sapiens)  # {"L","R"} in [0,1]

    accum = {name: {"dx": [[] for _ in range(n_bins)],
                     "dy": [[] for _ in range(n_bins)]}
             for name in landmarks}

    for i, frame in enumerate(sapiens_data.get("frames", [])):
        t = frame.get("time_s")
        if t is None:
            t = frame.get("frame_idx", i) / mg_fps
        vc_i = int(round((float(t) + float(offset_s)) * vc_fps))
        if vc_i < 0 or vc_i >= len(vicon_data.get("frames", [])):
            continue

        mg_anchor = _pose_anchor(frame.get("landmarks", {}))
        vc_anchor = _pose_anchor(vicon_data["frames"][vc_i].get("landmarks", {}))
        if mg_anchor is None or vc_anchor is None:
            continue
        mg_mh, mg_s = mg_anchor
        vc_mh, vc_s = vc_anchor

        for name in landmarks:
            mg_p = _landmark_xy(frame.get("landmarks", {}), name)
            vc_p = _landmark_xy(vicon_data["frames"][vc_i].get("landmarks", {}), name)
            if mg_p is None or vc_p is None:
                continue
            mg_rel = (mg_p - mg_mh) / mg_s
            vc_rel = (vc_p - vc_mh) / vc_s
            d = mg_rel - vc_rel  # bias to subtract from sapiens

            side_key = "L" if name.startswith("LEFT_") else "R"
            phi = phase[side_key][i] if i < len(phase[side_key]) else np.nan
            if np.isnan(phi): continue
            b = min(int(phi * n_bins), n_bins - 1)
            accum[name]["dx"][b].append(float(d[0]))
            accum[name]["dy"][b].append(float(d[1]))

    out = {}
    for name in landmarks:
        dx_bins = [float(np.mean(v)) if v else np.nan
                   for v in accum[name]["dx"]]
        dy_bins = [float(np.mean(v)) if v else np.nan
                   for v in accum[name]["dy"]]
        n_bins_count = [len(v) for v in accum[name]["dx"]]
        out[name] = {"dx": dx_bins, "dy": dy_bins, "n": n_bins_count}
    return out


def _fill_nan_bins(arr: list) -> np.ndarray:
    """Fill NaN phase-bins by nearest-neighbour on the cycle circle."""
    a = np.array(arr, dtype=float)
    n = a.size
    ok = ~np.isnan(a)
    if ok.sum() == 0:
        return np.zeros(n)
    if ok.sum() == n:
        return a
    idx = np.arange(n)
    # Circular interp: duplicate the good values three times and pick middle
    idx_ext = np.concatenate([idx - n, idx[ok], idx + n * 2]) if False else None
    # Simpler: linear interp on the good ones, treating cycle as circular
    good_i = idx[ok]
    good_v = a[ok]
    # Add wrap-around anchors
    good_i_wrap = np.concatenate([good_i - n, good_i, good_i + n])
    good_v_wrap = np.concatenate([good_v, good_v, good_v])
    a[~ok] = np.interp(idx[~ok], good_i_wrap, good_v_wrap)
    return a


def apply_landmark_bias_correction(
    data: dict,
    bias: Dict[str, Dict[str, list]],
    cycles: dict,
    *,
    in_place: bool = False,
) -> dict:
    """Subtract a phase-binned landmark bias from a Sapiens myogait dict.

    For each frame that belongs to a detected cycle, looks up the
    left- or right-cycle phase, interpolates the bias linearly (with
    circular wrap) between the two nearest phase bins, rescales by
    the frame's current thigh length, and subtracts it from the
    landmark xy in normalised image coordinates.

    Frames outside any cycle, or where the mid-hip / thigh scale
    cannot be computed, are left unchanged.  Landmarks not present
    in ``bias`` are also untouched.  Compute-angles output already
    stored in ``data['angles']`` becomes stale after this call —
    re-run :func:`compute_angles` if you need updated joint angles.

    .. warning::
       Do **not** re-run :func:`normalize` after this correction — the
       Butterworth filter would smooth away the phase-binned offsets
       and cancel the fix.  Apply the correction *after* normalize,
       then recompute angles directly.

    .. warning::
       Correcting a **subset** of landmarks (e.g. knee only, or heel
       only) breaks the geometric coherence of the hip-knee-ankle
       triangle: the pose estimator's per-landmark biases are not
       independent errors — correcting one without the others pushes
       the joint angle in an unpredictable direction.  In practice:

       - ``(LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)`` is the
         recommended conservative set.  Empirically reduces knee RMSE
         by ~4° and hip RMSE by ~7° on Bath BioCV vs a Vicon C3D
         reference, at the cost of a temporary ~8° ankle degradation.
       - Adding heel / foot_index makes the ankle angle worse because
         the Sapiens "foot centre" landmark is not the anatomical
         equivalent of the Vicon MTP marker — the measured bias
         encodes marker-placement disagreement, not a fixable error.
    """
    if not in_place:
        data = _shallow_copy_with_frames(data)

    frames = data.get("frames", [])
    phase = _phase_per_frame(data, cycles)
    # Prefilled bins per landmark
    bias_arr = {}
    for name, b in bias.items():
        dx = _fill_nan_bins(b.get("dx", []))
        dy = _fill_nan_bins(b.get("dy", []))
        bias_arr[name] = (dx, dy)
    n_bins_for = {name: len(dx) for name, (dx, _) in bias_arr.items()}

    def _interp_bias(name: str, phi: float) -> Tuple[float, float]:
        dx, dy = bias_arr[name]
        n = len(dx)
        pos = phi * n  # [0, n)
        i0 = int(np.floor(pos)) % n
        i1 = (i0 + 1) % n
        w = pos - np.floor(pos)
        return (float((1 - w) * dx[i0] + w * dx[i1]),
                float((1 - w) * dy[i0] + w * dy[i1]))

    for i, frame in enumerate(frames):
        anchor = _pose_anchor(frame.get("landmarks", {}))
        if anchor is None: continue
        _, thigh_scale = anchor
        lm_dict = frame["landmarks"]
        for name in list(lm_dict.keys()):
            if name not in bias_arr: continue
            side_key = "L" if name.startswith("LEFT_") else "R"
            phi = phase[side_key][i] if i < len(phase[side_key]) else np.nan
            if np.isnan(phi): continue
            bx, by = _interp_bias(name, float(phi))
            corr_x = bx * thigh_scale
            corr_y = by * thigh_scale
            lm = lm_dict[name]
            if isinstance(lm, dict):
                if lm.get("x") is not None: lm["x"] = float(lm["x"]) - corr_x
                if lm.get("y") is not None: lm["y"] = float(lm["y"]) - corr_y
            elif isinstance(lm, list) and len(lm) >= 2:
                lm[0] = float(lm[0]) - corr_x
                lm[1] = float(lm[1]) - corr_y
    # Angles are now stale — drop them so callers must re-run compute_angles
    data.pop("angles", None)
    return data


def _shallow_copy_with_frames(data: dict) -> dict:
    """Shallow-copy the top-level dict but deep-copy every frame's landmark
    dict so the caller's original stays intact.
    """
    import copy
    out = dict(data)
    out["frames"] = [
        {**f, "landmarks": copy.deepcopy(f.get("landmarks", {}))}
        for f in data.get("frames", [])
    ]
    return out


__all__ = [
    "ANKLE_BIAS_V1",
    "HIP_BIAS_V1",
    "KNEE_BIAS_V1",
    "apply_perspective_correction",
    "apply_linear_detrend",
    "apply_ankle_bias_correction",
    "apply_hip_bias_correction",
    "apply_knee_bias_correction",
    "fit_landmark_bias_by_phase",
    "apply_landmark_bias_correction",
]
