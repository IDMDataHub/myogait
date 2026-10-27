"""Markerless sagittal kinetics: per-foot ground reaction forces (GRF), centre of
pressure (COP) and lower-limb joint moments from 2-D pose (video) or marker
kinematics, without force plates.

The chain, validated against force plates and marker-based inverse dynamics
on 12 gait datasets (healthy and pathological) and on video (fixed camera and
hand-held smartphone):

1. Optional vertical camera stabilisation from the background
   (:func:`background_vertical_shift`) and camera-motion detection
   (:func:`detect_camera_motion`), so a hand-held / tracking camera does not
   inject its own bounce into the centre of mass.
2. Whole-body centre of mass (de Leva 1996 segment parameters) and total
   vertical / antero-posterior GRF from Newton's second law
   (``F = m (a_CoM + g)``), low-pass filtered (4 Hz fixed camera, 2.5 Hz moving
   camera).
3. Per-foot vertical GRF: single support takes the total; in double support
   the load is shared with a cubic transfer law ``w(tau)`` learnt on force
   plates (:data:`DS_CUBIC`).
4. Per-foot antero-posterior GRF from a virtual pivot point
   (:data:`VPP_HEIGHT_M` above the CoM): the GRF line of action passes through
   the COP and the pivot, so ``F_ap = F_v (x_CoM - x_COP) / (z_CoM + h - z_COP)``.
5. Foot-rotation COP model (FR-COP): cubic in normalised stance time whose four coefficients
   depend linearly on three foot rotations measured relative to the foot-flat
   posture (initial contact, mid and late stance) — invariant to the points
   used to measure the foot (markers or pose keypoints) and to vertical camera
   motion (:data:`COP_FOOT_ROTATION`).
6. The pose ankle keypoint sits posterior to the anatomical ankle joint centre;
   it is moved along the heel→toe axis to :data:`ANKLE_FOOT_FRACTION` of the
   heel→MTP1 foot length (:func:`relocate_ankle`).
7. 2-D inverse dynamics (foot → shank → thigh, Winter), inertial terms from
   kinematics low-pass filtered at 6 Hz.

All forces are returned in newtons and body weights, moments as internal
moments (plantarflexor / extensor positive) in N·m and N·m/kg.

Reference: Fer F., Arlaud L., Feigean R. et al. (in preparation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

G = 9.81

#: De Leva (1996) mass fractions and CoM position from the proximal joint.
DE_LEVA = {
    "M": {"head": (0.0694, 0.5), "trunk": (0.4346, 0.5138), "thigh": (0.1416, 0.4095), "shank": (0.0433, 0.4459),
          "foot": (0.0137, 0.4415), "upperarm": (0.0271, 0.5772), "forearm": (0.0162, 0.4574)},
    "F": {"head": (0.0668, 0.5), "trunk": (0.4257, 0.4997), "thigh": (0.1478, 0.3612), "shank": (0.0481, 0.4416),
          "foot": (0.0129, 0.4014), "upperarm": (0.0255, 0.5754), "forearm": (0.0138, 0.4559)},
}
#: Sagittal inverse dynamics segments: (mass fraction, CoM from proximal, radius of gyration / length).
_ID_SEG = {"foot": (0.0137, 0.4415, 0.257), "shank": (0.0433, 0.4459, 0.255), "thigh": (0.1416, 0.4095, 0.329)}

#: Leading-foot share of the vertical load in double support, w(tau) = polyval(DS_CUBIC, tau),
#: tau = normalised double-support time (fitted on 616 195 double-support frames, 457 subjects, 12 datasets).
DS_CUBIC = (-0.13292, 0.18365, 0.78608, 0.14092)

#: Foot-rotation COP model (FR-COP). COP position along the foot r(tau) (0 = heel at foot flat, 1 = first metatarsal head),
#: r = sum_k (B[0,k] + B[1,k] a + B[2,k] rm + B[3,k] rl) tau^k, with a / rm / rl the foot rotations (rad)
#: at initial contact (tau 0-5 %), mid stance (50-80 %) and late stance (80-95 %) minus the foot-flat
#: rotation (median 15-45 %). Fitted on 11 384 stances, 563 subjects, 11 datasets.
COP_FOOT_ROTATION = np.array([[0.39468, 0.09275, 0.03184, 0.59476],
                    [0.91025, -3.78567, 5.21195, -2.01425],
                    [-0.27801, 12.97732, -26.83098, 12.86307],
                    [0.05070, -2.18379, 8.40579, -5.76788]])

VPP_HEIGHT_M = 0.30           #: virtual pivot point height above the CoM
ANKLE_FOOT_FRACTION = 0.35    #: anatomical ankle joint centre, fraction of heel→MTP1 length from the heel
MTP_TO_TOE_RATIO = 0.85       #: heel→MTP1 / heel→toe-tip (pose foot_index) length ratio
LOAD_BLEND_BW = 0.10          #: below this vertical load the COP is drawn back under the ankle
HS_BIAS_S = -0.035            #: median bias of kinematic (Zeni) heel strike vs force plate (s)
TO_BIAS_S = 0.010             #: median bias of kinematic toe off vs force plate (s)
FEMUR_HEIGHT_RATIO = 0.245

_KEYS = {"hip": "HIP", "knee": "KNEE", "ankle": "ANKLE", "heel": "HEEL", "foot_index": "FOOT_INDEX",
         "shoulder": "SHOULDER", "elbow": "ELBOW", "wrist": "WRIST"}
_HEAD = ("NOSE", "LEFT_EAR", "RIGHT_EAR", "LEFT_EYE", "RIGHT_EYE")


# --------------------------------------------------------------------------------------------------- utilities
def _lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 1:
        ok = np.isfinite(x)
        if ok.sum() < 3 * (order + 1):
            return x.copy()
        xi = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
        b, a = butter(order, min(fc / (fs / 2.0), 0.99))
        return filtfilt(b, a, xi)
    return np.column_stack([_lowpass(x[:, j], fs, fc, order) for j in range(x.shape[1])])


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).copy()
    if a.ndim == 1:
        ok = np.isfinite(a)
        if 2 <= ok.sum() < len(a):
            a[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), a[ok])
        return a
    return np.column_stack([_interp_nan(a[:, j]) for j in range(a.shape[1])])


def _despike(x: np.ndarray, win: int = 7, k: float = 3.0) -> np.ndarray:
    """Hampel filter along time: replace single-frame pose glitches by the local median."""
    x = np.asarray(x, float)
    size = (win,) + (1,) * (x.ndim - 1)
    med = median_filter(x, size=size, mode="nearest")
    mad = median_filter(np.abs(x - med), size=size, mode="nearest")
    out = x.copy(); bad = np.abs(x - med) > k * 1.4826 * np.maximum(mad, 1e-9); out[bad] = med[bad]
    return out


def _cross(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    return r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]


# --------------------------------------------------------------------------------------------------- pose → metric
def landmarks_to_metric(data: dict, height_m: Optional[float] = None, femur_mm: Optional[float] = None,
                        camera_dy_px: Optional[np.ndarray] = None) -> dict:
    """Pivot landmarks → metric sagittal coordinates.

    Returns ``{"t", "fs", "scale_m_per_px", "forward", "<side>_<joint>": (N, 2) [AP forward, vertical up] m,
    "head": (N, 2)}``. The scale is isotropic (source pixels), from the femur (``femur_mm``, else
    ``FEMUR_HEIGHT_RATIO * height_m``) measured as the inter-quartile median hip→knee length.
    ``camera_dy_px`` (N,) is the vertical image displacement of the background (pixels, image y down):
    it is subtracted from every landmark before conversion (vertical camera stabilisation).
    """
    frames = data["frames"]; meta = data.get("meta", {})
    W, H = float(meta.get("width", 1.0)), float(meta.get("height", 1.0)); n = len(frames)
    fs = float(meta.get("fps") or 30.0)
    t = np.array([f.get("time_s", i / fs) for i, f in enumerate(frames)], float)
    raw = {}
    for name in [f"{s}_{k}" for s in ("LEFT", "RIGHT") for k in _KEYS.values()] + list(_HEAD):
        a = np.full((n, 2), np.nan)
        for i, f in enumerate(frames):
            p = (f.get("landmarks") or {}).get(name)
            if p and p.get("x") is not None and p.get("y") is not None:
                a[i] = (p["x"] * W, p["y"] * H)
        raw[name] = a
    if camera_dy_px is not None:
        dy = np.asarray(camera_dy_px, float)
        for k in raw:
            raw[k][:, 1] = raw[k][:, 1] - dy
    fem = np.concatenate([np.linalg.norm(raw[f"{s}_HIP"] - raw[f"{s}_KNEE"], axis=1) for s in ("LEFT", "RIGHT")])
    fem = fem[np.isfinite(fem)]
    if fem.size == 0:
        raise ValueError("no hip/knee landmarks: cannot scale the pose")
    q1, q3 = np.percentile(fem, [25, 75]); fem_px = np.median(fem[(fem >= q1) & (fem <= q3)])
    if femur_mm:
        femur_m = femur_mm / 1000.0
    elif height_m:
        femur_m = FEMUR_HEIGHT_RATIO * height_m
    else:
        raise ValueError("height_m or femur_mm is required to scale the pose")
    scale = femur_m / fem_px
    dx = np.nanmedian(np.concatenate([raw[f"{s}_FOOT_INDEX"][:, 0] - raw[f"{s}_HEEL"][:, 0] for s in ("LEFT", "RIGHT")]))
    fwd = 1.0 if (not np.isfinite(dx) or dx >= 0) else -1.0
    out = {"t": t, "fs": fs, "scale_m_per_px": scale, "forward": fwd}
    for s in ("left", "right"):
        for k, nm in _KEYS.items():
            a = raw[f"{s.upper()}_{nm}"]
            out[f"{s}_{k}"] = _interp_nan(np.column_stack([fwd * a[:, 0] * scale, -a[:, 1] * scale]))
    hd = np.nanmean(np.stack([raw[h] for h in _HEAD]), 0)
    out["head"] = _interp_nan(np.column_stack([fwd * hd[:, 0] * scale, -hd[:, 1] * scale]))
    return out


def whole_body_com(J: dict, sex: str = "M") -> np.ndarray:
    """Whole-body CoM (N, 2) from segment CoMs (de Leva 1996)."""
    dl = DE_LEVA.get(sex, DE_LEVA["M"]); seg = lambda p, d, c: (1.0 - c) * p + c * d
    hip = 0.5 * (J["left_hip"] + J["right_hip"]); sho = 0.5 * (J["left_shoulder"] + J["right_shoulder"])
    head = J.get("head"); head = sho + (sho - hip) * 0.35 if head is None or not np.isfinite(head).any() else head
    parts = [(dl["trunk"][0], seg(hip, sho, dl["trunk"][1])), (dl["head"][0], head)]
    for s in ("left", "right"):
        parts += [(dl["thigh"][0], seg(J[f"{s}_hip"], J[f"{s}_knee"], dl["thigh"][1])),
                  (dl["shank"][0], seg(J[f"{s}_knee"], J[f"{s}_ankle"], dl["shank"][1])),
                  (dl["foot"][0], seg(J[f"{s}_heel"], J[f"{s}_foot_index"], dl["foot"][1])),
                  (dl["upperarm"][0], seg(J[f"{s}_shoulder"], J[f"{s}_elbow"], dl["upperarm"][1])),
                  (dl["forearm"][0], seg(J[f"{s}_elbow"], J[f"{s}_wrist"], dl["forearm"][1]))]
    m = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / m


# --------------------------------------------------------------------------------------------------- camera
def _gray_frames(video_path: str, width_small: int = 270, stride: int = 1, max_frames: Optional[int] = None):
    import cv2
    cap = cv2.VideoCapture(video_path); i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok or (max_frames is not None and i >= max_frames * stride):
                break
            if i % stride == 0:
                h, w = fr.shape[:2]; f = width_small / w
                yield i, cv2.cvtColor(cv2.resize(fr, (width_small, int(round(h * f))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), w / width_small
            i += 1
    finally:
        cap.release()


def _subject_box(data: dict, idx: int, shape) -> Optional[tuple]:
    frames = data.get("frames") if data else None
    if not frames or idx >= len(frames):
        return None
    L = frames[idx].get("landmarks") or {}
    xs = [p["x"] for p in L.values() if p and p.get("x") is not None]; ys = [p["y"] for p in L.values() if p and p.get("y") is not None]
    if not xs:
        return None
    h, w = shape; x0, x1, y0, y1 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
    mx, my = 0.2 * (x1 - x0), 0.1 * (y1 - y0)
    return int(max(0, x0 - mx)), int(min(w, x1 + mx)), int(max(0, y0 - my)), int(min(h, y1 + my))


def _flow_shift(prev, cur, mask):
    import cv2
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=150, qualityLevel=0.01, minDistance=7, mask=mask)
    if p0 is None or len(p0) < 8:
        return np.nan, np.nan
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return np.nan, np.nan
    d = (p1[ok] - p0[ok]).reshape(-1, 2)
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def background_vertical_shift(video_path: str, data: Optional[dict] = None, width_small: int = 270,
                              stride: int = 1) -> np.ndarray:
    """Cumulative vertical image displacement of the static background (source pixels, image y down), one
    value per video frame. Background = everything outside the subject's bounding box (pose landmarks + 20 %).
    Lucas-Kanade on a 270-px-wide grey image (~6 ms/frame). Subtract it from landmark y (see
    :func:`landmarks_to_metric`) to remove the bounce of a hand-held camera."""
    import cv2  # noqa: F401
    dys, idx, prev, f_up = [], [], None, 1.0
    for i, g, f_up in _gray_frames(video_path, width_small, stride):
        if prev is None:
            dys.append(0.0); idx.append(i); prev = g; continue
        mask = np.full(g.shape, 255, np.uint8); box = _subject_box(data, i, g.shape)
        if box:
            x0, x1, y0, y1 = box; mask[y0:y1, x0:x1] = 0
        _, dy = _flow_shift(prev, g, mask)
        dys.append(0.0 if not np.isfinite(dy) else dy); idx.append(i); prev = g
    cum = np.cumsum(dys) * f_up
    n = (data and len(data.get("frames", []))) or (idx[-1] + 1 if idx else 0)
    return np.interp(np.arange(n), idx, cum) if idx else np.zeros(n)


def detect_camera_motion(video_path: str, data: Optional[dict] = None, stride: int = 5, max_frames: int = 120,
                         threshold_px: float = 0.5) -> dict:
    """Cheap fixed-vs-moving camera test: background optical flow on one frame in ``stride`` over the first
    ``max_frames`` sampled frames. Returns ``{"moving": bool, "dx_px_per_frame", "dy_px_per_frame"}`` (median
    absolute background shift between consecutive analysed frames, source pixels, divided by ``stride``)."""
    shifts, prev = [], None
    for i, g, f_up in _gray_frames(video_path, 270, stride, max_frames):
        if prev is not None:
            mask = np.full(g.shape, 255, np.uint8); box = _subject_box(data, i, g.shape)
            if box:
                x0, x1, y0, y1 = box; mask[y0:y1, x0:x1] = 0
            dx, dy = _flow_shift(prev, g, mask)
            if np.isfinite(dx):
                shifts.append((abs(dx) * f_up / stride, abs(dy) * f_up / stride))
        prev = g
    if not shifts:
        return {"moving": None, "dx_px_per_frame": np.nan, "dy_px_per_frame": np.nan}
    s = np.array(shifts); mx, my = float(np.median(s[:, 0])), float(np.median(s[:, 1]))
    return {"moving": bool(max(mx, my) > threshold_px), "dx_px_per_frame": mx, "dy_px_per_frame": my}


# --------------------------------------------------------------------------------------------------- model blocks
def ds_share(tau) -> np.ndarray:
    """Leading-foot share of the vertical GRF in double support (cubic transfer law, clipped to [0, 1])."""
    return np.clip(np.polyval(DS_CUBIC, np.clip(tau, 0.0, 1.0)), 0.0, 1.0)


def _foot_rotation(heel: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """Foot pitch (rad): + when the forefoot is lower than the heel."""
    return np.arctan2(heel[:, 1] - toe[:, 1], toe[:, 0] - heel[:, 0])


def cop_foot_rotation_rel(heel: np.ndarray, toe: np.ndarray, on: int, off: int, coefs: np.ndarray = COP_FOOT_ROTATION) -> np.ndarray:
    """Foot-rotation COP model (FR-COP): COP along the foot (101 points of the stance, 0 = heel at foot flat, 1 = MTP1)."""
    tt = np.linspace(0, 1, off - on + 1); T = np.linspace(0, 1, 101)
    th = np.interp(T, tt, _foot_rotation(heel, toe)[on:off + 1]); thf = np.median(th[15:46])
    a, rm, rl = th[:6].mean() - thf, th[50:81].mean() - thf, th[80:96].mean() - thf
    base = np.stack([T ** k for k in range(4)], -1)
    return np.clip(base @ (coefs[0] + a * coefs[1] + rm * coefs[2] + rl * coefs[3]), -0.1, 1.15)


def cop_position(heel, toe, ankle, on, off, vgrf_bw, foot_length_mtp: Optional[float] = None,
                 rel101: Optional[np.ndarray] = None) -> np.ndarray:
    """Absolute COP (N, 2) for one stance: anchored on the forefoot (on the ground all stance long),
    COP_x = toe_x(t) + (heel_x − toe_x)|foot flat + r(tau)·L_MTP1, drawn back under the ankle at low load."""
    n = len(ankle); m = off - on; ff = slice(on + int(0.15 * m), on + max(int(0.45 * m), int(0.15 * m) + 4))
    L = foot_length_mtp if foot_length_mtp else MTP_TO_TOE_RATIO * np.nanmedian(toe[ff, 0] - heel[ff, 0])
    rel = cop_foot_rotation_rel(heel, toe, on, off) if rel101 is None else rel101
    rel_t = np.interp(np.clip((np.arange(n) - on) / max(m, 1), 0, 1), np.linspace(0, 1, 101), rel)
    cop = np.zeros((n, 2)); cop[:, 1] = np.nanmin(np.minimum(heel[on:off + 1, 1], toe[on:off + 1, 1]))
    cop[:, 0] = toe[:, 0] + np.nanmedian(heel[ff, 0] - toe[ff, 0]) + rel_t * L
    w = np.clip(np.asarray(vgrf_bw, float) / LOAD_BLEND_BW, 0, 1); cop[:, 0] = ankle[:, 0] + (cop[:, 0] - ankle[:, 0]) * w
    return cop


def relocate_ankle(heel, toe, ankle, on, off, k_target: float = ANKLE_FOOT_FRACTION) -> np.ndarray:
    """Move a pose ankle keypoint along the heel→toe axis so that it sits at ``k_target`` × heel→MTP1 length
    from the heel at foot flat (pose ankle keypoints lie 2-3 cm posterior to the joint centre)."""
    m = off - on; ff = slice(on + int(0.15 * m), on + max(int(0.45 * m), int(0.15 * m) + 4))
    L = MTP_TO_TOE_RATIO * np.nanmedian(toe[ff, 0] - heel[ff, 0]); k = np.nanmedian(ankle[ff, 0] - heel[ff, 0]) / L
    u = toe - heel; u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return ankle + (k_target - k) * L * u


def vpp_ap_force(fv: np.ndarray, cop: np.ndarray, com: np.ndarray, h: float = VPP_HEIGHT_M) -> np.ndarray:
    """AP force from the virtual pivot point: line of action through the COP and a point h above the CoM."""
    d = com[:, 1] + h - cop[:, 1]
    return np.nan_to_num(fv * (com[:, 0] - cop[:, 0]) / np.where(np.abs(d) > 0.3, d, np.nan))


def inverse_dynamics_2d(J: dict, side: str, grf: np.ndarray, cop: np.ndarray, mass: float, fs: float,
                        inertia_cutoff: Optional[float] = 6.0, despike: bool = True) -> dict:
    """Sagittal inverse dynamics foot → shank → thigh. Returns internal moments (N·m): ``ankle``
    (plantarflexor +), ``knee`` and ``hip`` (extensor +). Inertial terms from kinematics low-pass filtered
    at ``inertia_cutoff`` (moment arms stay unfiltered)."""
    from scipy.signal import savgol_filter
    A, K, H, TO = (np.asarray(J[f"{side}_{k}"], float) for k in ("ankle", "knee", "hip", "foot_index"))
    if despike:
        A, K, H, TO, cop = _despike(A), _despike(K), _despike(H), _despike(TO), _despike(np.asarray(cop, float))
    gv = np.array([0.0, -G]); seg = {"foot": (A, TO), "shank": (K, A), "thigh": (H, K)}; com, al, I, acc = {}, {}, {}, {}
    d2 = lambda x: savgol_filter(x, 11, 3, deriv=2, delta=1.0 / fs, axis=0, mode="interp")
    for name, (p, d) in seg.items():
        mf, cr, rg = _ID_SEG[name]; c = p + cr * (d - p); com[name] = c
        I[name] = mass * mf * (rg * np.linalg.norm(d - p, axis=1)) ** 2
        th = np.unwrap(np.arctan2((p - d)[:, 1], (p - d)[:, 0]))
        cs, ths = (c, th) if inertia_cutoff is None else (_lowpass(c, fs, inertia_cutoff), _lowpass(th, fs, inertia_cutoff))
        al[name] = np.clip(d2(ths), -400.0, 400.0); acc[name] = d2(cs)
    mf = mass * _ID_SEG["foot"][0]; F_a = mf * acc["foot"] - grf - mf * gv
    M_a = I["foot"] * al["foot"] - _cross(A - com["foot"], F_a) - _cross(cop - com["foot"], grf)
    ms = mass * _ID_SEG["shank"][0]; F_k = ms * acc["shank"] + F_a - ms * gv
    M_k = I["shank"] * al["shank"] + M_a - _cross(A - com["shank"], -F_a) - _cross(K - com["shank"], F_k)
    mt = mass * _ID_SEG["thigh"][0]; F_h = mt * acc["thigh"] + F_k - mt * gv
    M_h = I["thigh"] * al["thigh"] + M_k - _cross(K - com["thigh"], -F_k) - _cross(H - com["thigh"], F_h)
    return {"ankle": -M_a, "knee": M_k, "hip": -M_h}


# --------------------------------------------------------------------------------------------------- stances
def _ev_frames(events: dict, key: str) -> list:
    return sorted(int(e["frame"] if isinstance(e, dict) else e) for e in events.get(key, []))


def stances_from_events(events: dict, n: int, fs: float, hs_bias_s: float = HS_BIAS_S, to_bias_s: float = TO_BIAS_S) -> list:
    """Stances (side, on, off, contralateral toe off, contralateral heel strike) from myogait events,
    corrected for the median kinematic-vs-force-plate bias."""
    bh, bt = int(round(-hs_bias_s * fs)), int(round(-to_bias_s * fs)); out = []
    for side, other in (("left", "right"), ("right", "left")):
        hs, to = _ev_frames(events, f"{side}_hs"), _ev_frames(events, f"{side}_to")
        ohs, oto = _ev_frames(events, f"{other}_hs"), _ev_frames(events, f"{other}_to")
        for h in hs:
            t = next((x for x in to if x > h), None)
            if t is None:
                continue
            ct = next((x for x in oto if h - 2 <= x <= t), None); ch = next((x for x in ohs if (ct or h) < x < t), None)
            on, off = h + bh, t + bt
            if ct is None or ch is None:
                continue
            ct, ch = ct + bt, ch + bh
            if 0 <= on < ct < ch < off < n:
                out.append({"side": side, "on": on, "off": off, "contra_to": ct, "contra_hs": ch})
    return sorted(out, key=lambda s: s["on"])


def _foot_share(n, on, off, ct, ch) -> np.ndarray:
    m = off - on; tt = np.linspace(0, 1, m + 1); sh = np.ones(m + 1); f1, f2 = (ct - on) / m, (ch - on) / m
    d1 = tt <= f1; sh[d1] = ds_share(tt[d1] / f1)
    d2 = tt >= f2; sh[d2] = 1 - ds_share((tt[d2] - f2) / (1 - f2))
    out = np.zeros(n); out[on:off + 1] = sh
    return out


# --------------------------------------------------------------------------------------------------- main entry
def estimate_kinetics(data: dict, mass_kg: float, height_m: Optional[float] = None, femur_mm: Optional[float] = None,
                      sex: str = "M", events: Optional[dict] = None, video_path: Optional[str] = None,
                      camera_moving: Optional[bool] = None, stabilize: bool = True, com_cutoff_hz: Optional[float] = None,
                      relocate_pose_ankle: bool = True, camera_dy_px: Optional[np.ndarray] = None,
                      hs_bias_s: float = HS_BIAS_S, to_bias_s: float = TO_BIAS_S) -> dict:
    """Per-foot GRF, COP and sagittal joint moments from a myogait pivot (pose landmarks + events).

    Parameters
    ----------
    data : dict
        myogait pivot with ``frames`` (landmarks), ``meta`` (fps, width, height) and, unless ``events`` is
        given, ``data["events"]`` (``left_hs``, ``left_to``, ``right_hs``, ``right_to``).
    mass_kg, height_m / femur_mm, sex :
        Subject anthropometry (mass is required for forces in newtons and moments in N·m).
    video_path : str, optional
        Source video. If given, the camera is tested for motion (:func:`detect_camera_motion`) and, if it moves
        and ``stabilize`` is True, the pose is stabilised vertically from the background before the CoM is computed.
    camera_moving : bool, optional
        Overrides the camera test (e.g. known fixed laboratory camera).
    camera_dy_px : array, optional
        Precomputed background vertical shift (one value per pivot frame, source pixels, image y down), e.g. from
        :func:`background_vertical_shift`; used instead of re-reading the video.
    hs_bias_s, to_bias_s : float
        Kinematic event bias versus force plates, removed from the detected events.
    com_cutoff_hz : float, optional
        CoM low-pass cutoff before differentiation (default 4 Hz fixed camera, 2.5 Hz moving camera).

    Returns
    -------
    dict
        ``time`` (s), ``camera`` (dict), ``walking_speed_m_s`` (fixed camera only), ``grf_total_bw`` (N, 2),
        per side ``{side}``: ``grf_v_N``, ``grf_ap_N``, ``cop_x_m`` (N,), ``moments_Nm`` (dict of (N,)),
        and ``stances``: list of dicts with ``side``, ``on``, ``off`` and 101-point curves (``vgrf_bw``,
        ``apgrf_bw``, ``cop_minus_ankle_m``, ``ankle_Nm_kg``, ``knee_Nm_kg``, ``hip_Nm_kg``, ``ankle_power_W_kg``).
    """
    events = events or data.get("events") or {}
    camera = {"moving": camera_moving, "source": "user" if camera_moving is not None else None}
    dy = None if camera_dy_px is None else np.asarray(camera_dy_px, float)
    if dy is not None:
        camera.update({"moving": True if camera_moving is None else camera_moving, "vertical_stabilisation": True, "source": camera["source"] or "precomputed shift"})
    if video_path is not None and camera_moving is None and dy is None:
        camera.update(detect_camera_motion(video_path, data)); camera["source"] = "background optical flow"
    if video_path is not None and stabilize and camera.get("moving") and dy is None:
        dy = background_vertical_shift(video_path, data); camera["vertical_stabilisation"] = True
    J = landmarks_to_metric(data, height_m=height_m, femur_mm=femur_mm, camera_dy_px=dy)
    fs = J["fs"]; n = len(J["t"]); moving = bool(camera.get("moving"))
    fc = com_cutoff_hz or (2.5 if moving else 4.0)
    com = whole_body_com(J, sex); comf = _lowpass(com, fs, 6.0)
    Rv = (np.gradient(np.gradient(_lowpass(com[:, 1], fs, fc))) * fs ** 2 + G) / G
    Rap = np.gradient(np.gradient(_lowpass(com[:, 0], fs, fc))) * fs ** 2 / G
    speed = None
    if not moving:
        pel = 0.5 * (J["left_hip"][:, 0] + J["right_hip"][:, 0]); speed = float(abs(np.nanmedian(np.gradient(_lowpass(pel, fs, 2.0)) * fs)))
    res = {"time": J["t"], "camera": camera, "walking_speed_m_s": speed, "grf_total_bw": np.c_[Rap, Rv], "stances": []}
    for s in ("left", "right"):
        res[s] = {"grf_v_N": np.zeros(n), "grf_ap_N": np.zeros(n), "cop_x_m": np.full(n, np.nan), "moments_Nm": {k: np.full(n, np.nan) for k in ("ankle", "knee", "hip")}}
    T = np.linspace(0, 1, 101)
    for st in stances_from_events(events, n, fs, hs_bias_s, to_bias_s):
        s, on, off = st["side"], st["on"], st["off"]; sw = "right" if s == "left" else "left"
        Js = dict(J)
        if relocate_pose_ankle:
            Js[f"{s}_ankle"] = relocate_ankle(J[f"{s}_heel"], J[f"{s}_foot_index"], J[f"{s}_ankle"], on, off)
        share = _foot_share(n, on, off, st["contra_to"], st["contra_hs"])
        F = np.zeros((n, 2)); F[:, 1] = np.clip(Rv * share, 0, None) * mass_kg * G
        cop = cop_position(Js[f"{s}_heel"], Js[f"{s}_foot_index"], Js[f"{s}_ankle"], on, off, F[:, 1] / (mass_kg * G))
        F[:, 0] = vpp_ap_force(F[:, 1], cop, comf)
        F[:on] = 0; F[off + 1:] = 0
        M = inverse_dynamics_2d(Js, s, F, cop, mass_kg, fs)
        sl = slice(on, off + 1)
        res[s]["grf_v_N"][sl] = F[sl, 1]; res[s]["grf_ap_N"][sl] = F[sl, 0]; res[s]["cop_x_m"][sl] = cop[sl, 0]
        for k in M:
            res[s]["moments_Nm"][k][sl] = M[k][sl]
        sh, ft = Js[f"{s}_knee"] - Js[f"{s}_ankle"], Js[f"{s}_foot_index"] - Js[f"{s}_heel"]
        ang = np.unwrap(np.arctan2(ft[:, 1], ft[:, 0]) - np.arctan2(sh[:, 1], sh[:, 0]))
        power = -M["ankle"] * np.gradient(_lowpass(ang, fs, 6.0)) * fs          # ankle power, generation positive
        tt = np.linspace(0, 1, off - on + 1); cv = lambda y: np.interp(T, tt, np.asarray(y)[sl])
        res["stances"].append({**st, "vgrf_bw": cv(F[:, 1] / (mass_kg * G)), "apgrf_bw": cv(F[:, 0] / (mass_kg * G)),
                               "cop_minus_ankle_m": cv(cop[:, 0] - Js[f"{s}_ankle"][:, 0]), "ankle_Nm_kg": cv(M["ankle"] / mass_kg),
                               "knee_Nm_kg": cv(M["knee"] / mass_kg), "hip_Nm_kg": cv(M["hip"] / mass_kg), "ankle_power_W_kg": cv(power / mass_kg)})
    return res
