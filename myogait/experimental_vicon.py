"""Experimental VICON alignment and single-trial benchmark utilities.

This module is intentionally experimental and designed for AIM benchmark
workflows. It aligns one myogait result (single video) with one VICON trial
and computes comparison metrics on the overlapping temporal window.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy import signal

logger = logging.getLogger(__name__)


_ANGLE_CANDIDATES = {
    "hip_L": ("Lhip", "l_hip"),
    "hip_R": ("Rhip", "r_hip"),
    "knee_L": ("Lknee", "l_knee"),
    "knee_R": ("Rknee", "r_knee"),
    "ankle_L": ("Lankle", "l_ankle"),
    "ankle_R": ("Rankle", "r_ankle"),
}

# Joints considered for video<->VICON temporal synchronization, in priority
# order.  Knees first: they have the largest sagittal range of motion during
# gait, which gives the strongest cross-correlation signal.  Hips are kept as
# a fallback for recordings where knee tracking is partial or missing (common
# in slow pathological gait, e.g. DMD/SMA).
_SYNC_CANDIDATE_JOINTS = ("knee_L", "knee_R", "hip_L", "hip_R")

# Minimum number of samples on the best candidate signal for a meaningful
# cross-correlation lag estimate (10 samples at 30 fps ~= 0.33 s of motion).
_MIN_SYNC_SAMPLES = 10


class SyncError(ValueError):
    """Video<->VICON temporal synchronization failed.

    Carries a structured diagnostic (per-candidate sample counts) so
    callers such as the benchmark orchestrator can report WHY the sync
    failed instead of a bare message.  Inherits ValueError for
    backward compatibility with existing ``except ValueError`` handlers.
    """

    def __init__(self, message: str, candidates: dict, required: int):
        super().__init__(message)
        self.candidates = dict(candidates)
        self.required = int(required)

_LANDMARK_TO_VICON_MARKERS = {
    "LEFT_HIP": ("LHJC",),
    "RIGHT_HIP": ("RHJC",),
    "LEFT_KNEE": ("LLFE", "LMFE"),
    "RIGHT_KNEE": ("RLFE", "RMFE"),
    "LEFT_ANKLE": ("LLM", "LMM"),
    "RIGHT_ANKLE": ("RLM", "RMM"),
    "LEFT_HEEL": ("LCAL",),
    "RIGHT_HEEL": ("RCAL",),
    "LEFT_FOOT_INDEX": ("LTT2",),
    "RIGHT_FOOT_INDEX": ("RTT2",),
}


def _pick_struct_array(struct, names: Tuple[str, ...]) -> Optional[np.ndarray]:
    if struct is None:
        return None
    for name in names:
        if name in struct.dtype.names:
            arr = struct[name]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0:
                return arr
    return None


def _safe_float_array(values: List[Optional[float]]) -> np.ndarray:
    out = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    return out


def _interp_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y
    idx = np.arange(len(y))
    out = y.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return out


def load_vicon_trial_mat(trial_dir: str | Path, vicon_fps: float = 200.0) -> dict:
    """Load one VICON trial from MATLAB files used in premanip workflows.

    Expected files in ``trial_dir``:
    - ``res_angles_t.mat`` (optional but recommended)
    - ``points3D_t.mat`` (optional)
    - ``cycle.mat`` (optional)
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.exists():
        raise FileNotFoundError(f"VICON trial directory not found: {trial_dir}")

    angles_struct = None
    points_struct = None
    cycle_struct = None

    angles_path = trial_dir / "res_angles_t.mat"
    if angles_path.exists():
        angles_mat = loadmat(str(angles_path))
        angles_struct = angles_mat.get("res_angles_t", None)
        if angles_struct is not None:
            angles_struct = angles_struct[0, 0]

    points_path = trial_dir / "points3D_t.mat"
    if points_path.exists():
        points_mat = loadmat(str(points_path))
        points_struct = points_mat.get("points3D_t", None)
        if points_struct is not None:
            points_struct = points_struct[0, 0]

    cycle_path = trial_dir / "cycle.mat"
    if cycle_path.exists():
        cycle_mat = loadmat(str(cycle_path))
        cycle_struct = cycle_mat.get("cycle", None)
        if cycle_struct is not None:
            cycle_struct = cycle_struct[0, 0]

    angles = {}
    n_frames = 0
    for out_name, candidates in _ANGLE_CANDIDATES.items():
        arr = _pick_struct_array(angles_struct, candidates)
        if arr is None:
            angles[out_name] = np.array([], dtype=float)
            continue
        # Vicon Y-X-Z Euler decomposition: col 0 = transverse rotation,
        # col 1 = frontal (abd/add), col 2 = sagittal (flex/ext).
        if arr.ndim == 2 and arr.shape[1] >= 3:
            sig = arr[:, 2].astype(float)
        else:
            sig = arr.reshape(-1).astype(float)
        angles[out_name] = sig
        n_frames = max(n_frames, len(sig))

    landmarks = {}
    for lm_name, markers in _LANDMARK_TO_VICON_MARKERS.items():
        pts = []
        for mk in markers:
            if points_struct is not None and mk in points_struct.dtype.names:
                arr = points_struct[mk]
                if arr.ndim == 2 and arr.shape[1] == 3:
                    pts.append(arr.astype(float))
        if not pts:
            landmarks[lm_name] = np.empty((0, 3), dtype=float)
            continue
        if len(pts) == 1:
            merged = pts[0]
        else:
            merged = np.nanmean(np.stack(pts, axis=0), axis=0)
        landmarks[lm_name] = merged
        n_frames = max(n_frames, len(merged))

    events = {
        "left_hs": [],
        "right_hs": [],
        "left_to": [],
        "right_to": [],
    }
    if cycle_struct is not None:
        # Keep extraction permissive because cycle.mat structures can vary.
        for key, out in (("L", "left_hs"), ("R", "right_hs")):
            if key in cycle_struct.dtype.names:
                vals = cycle_struct[key].flatten()
                events[out] = [int(v) for v in vals if np.isfinite(v)]
        for key, out in (("stanceL", "left_to"), ("stanceR", "right_to")):
            if key in cycle_struct.dtype.names:
                vals = cycle_struct[key].flatten()
                # stance array is not always TO indices; keep raw as optional refs.
                events[out] = [int(v) for v in vals if np.isfinite(v)]

    return {
        "meta": {
            "source": "vicon_mat",
            "trial_name": trial_dir.name,
            "trial_dir": str(trial_dir),
            "fps": float(vicon_fps),
            "n_frames": int(n_frames),
            "duration_s": float(n_frames / vicon_fps) if vicon_fps > 0 else 0.0,
        },
        "angles": angles,
        "landmarks": landmarks,
        "events": events,
    }


def _myogait_angle_arrays(data: dict) -> Dict[str, np.ndarray]:
    frames = data.get("angles", {}).get("frames", [])
    out = {
        "hip_L": _safe_float_array([f.get("hip_L") for f in frames]),
        "hip_R": _safe_float_array([f.get("hip_R") for f in frames]),
        "knee_L": _safe_float_array([f.get("knee_L") for f in frames]),
        "knee_R": _safe_float_array([f.get("knee_R") for f in frames]),
        "ankle_L": _safe_float_array([f.get("ankle_L") for f in frames]),
        "ankle_R": _safe_float_array([f.get("ankle_R") for f in frames]),
    }
    return out


def _best_sync_signal(mg_angles: Dict[str, np.ndarray], vc_angles: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, np.ndarray]:
    best_name = None
    best_len = -1
    best_pair = (np.array([]), np.array([]))
    for name in _SYNC_CANDIDATE_JOINTS:
        a = _interp_nan(mg_angles.get(name, np.array([])))
        b = _interp_nan(vc_angles.get(name, np.array([])))
        valid = min(len(a), len(b))
        if valid > best_len:
            best_name = name
            best_len = valid
            best_pair = (a, b)
    return best_name or "knee_L", best_pair[0], best_pair[1]


def estimate_vicon_offset_seconds(
    myogait_data: dict,
    vicon_data: dict,
    max_lag_seconds: float = 10.0,
) -> dict:
    """Estimate temporal offset using cross-correlation on knee angles.

    Returns offset where:
    - ``offset_seconds > 0`` means VICON starts before myogait.
    - Mapping uses ``vicon_time = myogait_time + offset_seconds``.
    """
    fps_mg = float(myogait_data.get("meta", {}).get("fps", 30.0))
    fps_vc = float(vicon_data.get("meta", {}).get("fps", 200.0))
    if fps_mg <= 0 or fps_vc <= 0:
        raise ValueError("Invalid fps for synchronization")

    mg_angles = _myogait_angle_arrays(myogait_data)
    vc_angles = vicon_data.get("angles", {})
    signal_name, mg_sig, vc_sig = _best_sync_signal(mg_angles, vc_angles)
    if len(mg_sig) < _MIN_SYNC_SAMPLES or len(vc_sig) < _MIN_SYNC_SAMPLES:
        counts = {}
        for name in _SYNC_CANDIDATE_JOINTS:
            a = _interp_nan(mg_angles.get(name, np.array([])))
            b = _interp_nan(vc_angles.get(name, np.array([])))
            counts[name] = int(min(len(a), len(b)))
        detail = ", ".join(f"{name}={counts[name]}" for name in _SYNC_CANDIDATE_JOINTS)
        raise SyncError(
            "Not enough angle samples for synchronization "
            f"({detail}; need >= {_MIN_SYNC_SAMPLES} on the best candidate). "
            "Try a longer recording or check that compute_angles() ran on "
            "the video side.",
            candidates=counts,
            required=_MIN_SYNC_SAMPLES,
        )

    # Resample both to common frequency for stable lag estimate.
    common_fps = min(fps_mg, fps_vc)
    dur_mg = len(mg_sig) / fps_mg
    dur_vc = len(vc_sig) / fps_vc
    n_mg = max(10, int(round(dur_mg * common_fps)))
    n_vc = max(10, int(round(dur_vc * common_fps)))
    t_mg = np.linspace(0.0, dur_mg, len(mg_sig), endpoint=False)
    t_vc = np.linspace(0.0, dur_vc, len(vc_sig), endpoint=False)
    t_mg_r = np.linspace(0.0, dur_mg, n_mg, endpoint=False)
    t_vc_r = np.linspace(0.0, dur_vc, n_vc, endpoint=False)
    mg_r = np.interp(t_mg_r, t_mg, mg_sig)
    vc_r = np.interp(t_vc_r, t_vc, vc_sig)

    mg_n = (mg_r - np.nanmean(mg_r)) / (np.nanstd(mg_r) + 1e-8)
    vc_n = (vc_r - np.nanmean(vc_r)) / (np.nanstd(vc_r) + 1e-8)

    corr = signal.correlate(mg_n, vc_n, mode="full")
    lags = signal.correlation_lags(len(mg_n), len(vc_n), mode="full")
    max_lag = int(round(max_lag_seconds * common_fps))
    keep = np.abs(lags) <= max_lag
    corr = corr[keep]
    lags = lags[keep]

    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])
    # Positive lag means mg advanced vs vicon in this formulation.
    offset_seconds = -float(best_lag / common_fps)

    peak_corr = float(corr[best_idx] / (len(mg_n) + 1e-8))
    return {
        "signal_used": signal_name,
        "common_fps": float(common_fps),
        "lag_samples": best_lag,
        "offset_seconds": offset_seconds,
        "correlation_peak": peak_corr,
    }


def _vicon_sample_index(video_time_s: float, offset_s: float, vicon_fps: float) -> float:
    vicon_time_s = video_time_s + offset_s
    return vicon_time_s * vicon_fps


def align_vicon_to_myogait(
    myogait_data: dict,
    vicon_data: dict,
    offset_seconds: float,
) -> dict:
    """Align VICON angles/landmarks on myogait frames (single-trial overlap only)."""
    fps_mg = float(myogait_data.get("meta", {}).get("fps", 30.0))
    fps_vc = float(vicon_data.get("meta", {}).get("fps", 200.0))
    n_mg = int(myogait_data.get("meta", {}).get("n_frames", len(myogait_data.get("frames", []))))

    aligned_frames = []
    for i in range(n_mg):
        t = i / fps_mg if fps_mg > 0 else 0.0
        vc_idx_f = _vicon_sample_index(t, offset_seconds, fps_vc)
        if vc_idx_f < 0:
            continue
        vc_idx = int(round(vc_idx_f))
        if vc_idx >= int(vicon_data.get("meta", {}).get("n_frames", 0)):
            continue

        frame = {
            "frame_idx": i,
            "time_s": t,
            "vicon_frame": vc_idx,
            "angles": {},
            "landmarks": {},
        }
        for k in ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"):
            arr = vicon_data.get("angles", {}).get(k, np.array([]))
            if vc_idx < len(arr):
                val = float(arr[vc_idx])
                frame["angles"][k] = None if np.isnan(val) else val
            else:
                frame["angles"][k] = None

        for name, arr in vicon_data.get("landmarks", {}).items():
            if vc_idx < len(arr):
                xyz = arr[vc_idx]
                frame["landmarks"][name] = [
                    None if np.isnan(float(xyz[0])) else float(xyz[0]),
                    None if np.isnan(float(xyz[1])) else float(xyz[1]),
                    None if np.isnan(float(xyz[2])) else float(xyz[2]),
                ]
        aligned_frames.append(frame)

    return {
        "offset_seconds": float(offset_seconds),
        "n_aligned_frames": len(aligned_frames),
        "aligned_frames": aligned_frames,
    }


def _event_frames_myogait(data: dict, key: str) -> List[int]:
    out = []
    for e in data.get("events", {}).get(key, []):
        if isinstance(e, dict) and "frame" in e:
            out.append(int(e["frame"]))
    return sorted(out)


def _nearest_abs_diff(a: List[int], b: List[int], fps: float) -> List[float]:
    if not a or not b or fps <= 0:
        return []
    diffs = []
    b_arr = np.array(b, dtype=float)
    for x in a:
        j = int(np.argmin(np.abs(b_arr - x)))
        diffs.append(abs(float(x - b_arr[j])) * 1000.0 / fps)
    return diffs


def _angle_error_metrics(mg: np.ndarray, vc: np.ndarray) -> dict:
    mask = np.isfinite(mg) & np.isfinite(vc)
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "rmse_deg": None, "mae_deg": None, "bias_deg": None, "rom_diff_deg": None}
    d = mg[mask] - vc[mask]
    rom_mg = float(np.nanmax(mg[mask]) - np.nanmin(mg[mask]))
    rom_vc = float(np.nanmax(vc[mask]) - np.nanmin(vc[mask]))
    return {
        "n": int(mask.sum()),
        "rmse_deg": float(np.sqrt(np.mean(d ** 2))),
        "mae_deg": float(np.mean(np.abs(d))),
        "bias_deg": float(np.mean(d)),
        "rom_diff_deg": float(rom_mg - rom_vc),
    }


def compute_single_trial_benchmark_metrics(
    myogait_data: dict,
    vicon_data: dict,
    alignment: dict,
) -> dict:
    """Compute single-video benchmark metrics (no cohort aggregation)."""
    fps_mg = float(myogait_data.get("meta", {}).get("fps", 30.0))
    mg_angles = _myogait_angle_arrays(myogait_data)

    idx_pairs = [(f["frame_idx"], f["vicon_frame"]) for f in alignment.get("aligned_frames", [])]
    if not idx_pairs:
        return {"status": "no_overlap", "angle_metrics": {}, "event_metrics_ms": {}}

    angle_metrics = {}
    for name in ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"):
        mg_arr = mg_angles.get(name, np.array([]))
        vc_arr = vicon_data.get("angles", {}).get(name, np.array([]))
        mg_s = []
        vc_s = []
        for i_mg, i_vc in idx_pairs:
            if i_mg < len(mg_arr) and i_vc < len(vc_arr):
                mg_s.append(mg_arr[i_mg])
                vc_s.append(vc_arr[i_vc])
        angle_metrics[name] = _angle_error_metrics(np.array(mg_s, dtype=float), np.array(vc_s, dtype=float))

    vc_events = vicon_data.get("events", {})
    mg_events = {
        "left_hs": _event_frames_myogait(myogait_data, "left_hs"),
        "right_hs": _event_frames_myogait(myogait_data, "right_hs"),
        "left_to": _event_frames_myogait(myogait_data, "left_to"),
        "right_to": _event_frames_myogait(myogait_data, "right_to"),
    }
    event_metrics = {}
    for key in ("left_hs", "right_hs", "left_to", "right_to"):
        diffs = _nearest_abs_diff(mg_events.get(key, []), vc_events.get(key, []), fps_mg)
        event_metrics[key] = {
            "n_myogait": len(mg_events.get(key, [])),
            "n_vicon": len(vc_events.get(key, [])),
            "mae_ms": float(np.mean(diffs)) if diffs else None,
            "median_ms": float(np.median(diffs)) if diffs else None,
        }

    return {
        "status": "ok",
        "angle_metrics": angle_metrics,
        "event_metrics_ms": event_metrics,
    }


def attach_vicon_experimental_block(
    myogait_data: dict,
    vicon_data: dict,
    sync: dict,
    alignment: dict,
    metrics: dict,
) -> dict:
    """Attach VICON benchmark results to myogait JSON under experimental block."""
    if "experimental" not in myogait_data or myogait_data["experimental"] is None:
        myogait_data["experimental"] = {}

    myogait_data["experimental"]["vicon_benchmark"] = {
        "status": "experimental",
        "scope": "AIM benchmark only",
        "vicon_meta": vicon_data.get("meta", {}),
        "sync": sync,
        "alignment": {
            "offset_seconds": alignment.get("offset_seconds"),
            "n_aligned_frames": alignment.get("n_aligned_frames"),
        },
        "aligned_frames": alignment.get("aligned_frames", []),
        "metrics": metrics,
    }
    return myogait_data


def run_single_trial_vicon_benchmark(
    myogait_data: dict,
    trial_dir: str | Path,
    vicon_fps: float = 200.0,
    max_lag_seconds: float = 10.0,
) -> dict:
    """End-to-end helper: load VICON, sync, align, compute metrics, attach."""
    vicon = load_vicon_trial_mat(trial_dir, vicon_fps=vicon_fps)
    sync = estimate_vicon_offset_seconds(myogait_data, vicon, max_lag_seconds=max_lag_seconds)
    alignment = align_vicon_to_myogait(myogait_data, vicon, offset_seconds=sync["offset_seconds"])
    metrics = compute_single_trial_benchmark_metrics(myogait_data, vicon, alignment)
    return attach_vicon_experimental_block(myogait_data, vicon, sync, alignment, metrics)


# ── C3D loading ──────────────────────────────────────────────────────

DEFAULT_C3D_MARKER_MAP: Dict[str, List[str]] = {
    "LEFT_HIP":        ["LASIS", "LHJC"],
    "RIGHT_HIP":       ["RASIS", "RHJC"],
    "LEFT_KNEE":       ["LLFE", "LMFE"],
    "RIGHT_KNEE":      ["RLFE", "RMFE"],
    "LEFT_ANKLE":      ["LLM", "LMM"],
    "RIGHT_ANKLE":     ["RLM", "RMM"],
    "LEFT_HEEL":       ["LCAL"],
    "RIGHT_HEEL":      ["RCAL"],
    "LEFT_FOOT_INDEX": ["LTT2"],
    "RIGHT_FOOT_INDEX": ["RTT2"],
    "LEFT_SHOULDER":   ["LASIS"],
    "RIGHT_SHOULDER":  ["RASIS"],
    "NOSE":            ["LASIS", "RASIS"],
}


# ── Registered C3D marker conventions ─────────────────────────────────
# Registry of the marker-naming conventions we see most in the wild.
# Autodetection scores each convention by how many of the required
# lower-limb markers it can resolve in the C3D file; the best match
# is picked automatically by ``load_c3d`` when *marker_mapping* is
# omitted.  Add new conventions here — one flat dict per convention,
# keyed by the myogait landmark name, value = list of accepted C3D
# labels (first match wins, all found markers are averaged).

C3D_MARKER_CONVENTIONS: Dict[str, Dict[str, List[str]]] = {
    # Vicon Plug-in Gait — by far the most common in clinical labs.
    "plug_in_gait": {
        "LEFT_HIP":         ["LASI", "LPSI", "LHJC"],
        "RIGHT_HIP":        ["RASI", "RPSI", "RHJC"],
        "LEFT_KNEE":        ["LKNE", "LTHI"],
        "RIGHT_KNEE":       ["RKNE", "RTHI"],
        "LEFT_ANKLE":       ["LANK", "LTIB"],
        "RIGHT_ANKLE":      ["RANK", "RTIB"],
        "LEFT_HEEL":        ["LHEE"],
        "RIGHT_HEEL":       ["RHEE"],
        "LEFT_FOOT_INDEX":  ["LTOE"],
        "RIGHT_FOOT_INDEX": ["RTOE"],
        "LEFT_SHOULDER":    ["LSHO"],
        "RIGHT_SHOULDER":   ["RSHO"],
        "NOSE":             ["C7", "CLAV", "STRN"],
    },
    # ISB medial+lateral bony landmarks (the default used to be this).
    "iso_biomechanics": DEFAULT_C3D_MARKER_MAP,
    # Helen Hayes / Newington modified PIG (foot markers renamed).
    "helen_hayes": {
        "LEFT_HIP":         ["LASI", "LPSI"],
        "RIGHT_HIP":        ["RASI", "RPSI"],
        "LEFT_KNEE":        ["LKNE"],
        "RIGHT_KNEE":       ["RKNE"],
        "LEFT_ANKLE":       ["LANK"],
        "RIGHT_ANKLE":      ["RANK"],
        "LEFT_HEEL":        ["LHEEL", "LHEE"],
        "RIGHT_HEEL":       ["RHEEL", "RHEE"],
        "LEFT_FOOT_INDEX":  ["LFT2", "LMT2", "LTOE"],
        "RIGHT_FOOT_INDEX": ["RFT2", "RMT2", "RTOE"],
        "LEFT_SHOULDER":    ["LSHO"],
        "RIGHT_SHOULDER":   ["RSHO"],
        "NOSE":             ["C7", "CLAV"],
    },
    # Bath BioCV dataset (Cotton et al.) — explicit anatomical labels
    # with LEFT_/RIGHT_ prefixes and _L / _R suffixes.  Uses the
    # already-computed joint centres when available (LEFT_HIP etc.).
    "bath_biocv": {
        "LEFT_HIP":         ["LEFT_HIP", "ILCREST_L", "ASIS_L"],
        "RIGHT_HIP":        ["RIGHT_HIP", "ILCREST_R", "ASIS_R"],
        "LEFT_KNEE":        ["LEFT_KNEE", "KNEE_LAT_L", "KNEE_MED_L"],
        "RIGHT_KNEE":       ["RIGHT_KNEE", "KNEE_LAT_R", "KNEE_MED_R"],
        "LEFT_ANKLE":       ["LEFT_ANKLE", "MAL_LAT_L", "MAL_MED_L", "LAJC"],
        "RIGHT_ANKLE":      ["RIGHT_ANKLE", "MAL_LAT_R", "MAL_MED_R", "RAJC"],
        "LEFT_HEEL":        ["HEEL_L"],
        "RIGHT_HEEL":       ["HEEL_R"],
        "LEFT_FOOT_INDEX":  ["LEFT_MTP", "TOE_L", "MTP1_L", "MTP5_L", "LTOE"],
        "RIGHT_FOOT_INDEX": ["RIGHT_MTP", "TOE_R", "MTP1_R", "MTP5_R", "RTOE"],
        "LEFT_SHOULDER":    ["LEFT_SHO", "ACROM_L"],
        "RIGHT_SHOULDER":   ["RIGHT_SHO", "ACROM_R"],
        "NOSE":             ["C7", "CLAV"],
    },
    # Underscore-prefixed variant (some lab exports).
    "iso_biomechanics_underscore": {
        "LEFT_HIP":         ["L_ASIS", "L_HJC"],
        "RIGHT_HIP":        ["R_ASIS", "R_HJC"],
        "LEFT_KNEE":        ["L_LFE",  "L_MFE"],
        "RIGHT_KNEE":       ["R_LFE",  "R_MFE"],
        "LEFT_ANKLE":       ["L_LM",   "L_MM"],
        "RIGHT_ANKLE":      ["R_LM",   "R_MM"],
        "LEFT_HEEL":        ["L_CAL"],
        "RIGHT_HEEL":       ["R_CAL"],
        "LEFT_FOOT_INDEX":  ["L_TT2"],
        "RIGHT_FOOT_INDEX": ["R_TT2"],
        "LEFT_SHOULDER":    ["L_SHO"],
        "RIGHT_SHOULDER":   ["R_SHO"],
        "NOSE":             ["L_ASIS", "R_ASIS"],
    },
}


# Regex families used as the last-resort fuzzy fallback when no
# registered convention matches enough markers.  Anchored on the
# side prefix (L / R / LEFT_ / RIGHT_ / L_ / R_) and the anatomical
# keyword (ASI, HIP, HJC → hip ; KNE, LFE, MFE, KJC → knee ; ...).
_C3D_FUZZY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "LEFT_HIP":         ("ASIS?", "HJC", "HIP", "ILIAC", "PELV"),
    "RIGHT_HIP":        ("ASIS?", "HJC", "HIP", "ILIAC", "PELV"),
    "LEFT_KNEE":        ("KNE", "KJC", "LFE", "MFE", "FEP"),
    "RIGHT_KNEE":       ("KNE", "KJC", "LFE", "MFE", "FEP"),
    "LEFT_ANKLE":       ("ANK", "AJC", "LM$", "MM$", "MALL"),
    "RIGHT_ANKLE":      ("ANK", "AJC", "LM$", "MM$", "MALL"),
    "LEFT_HEEL":        ("HEE", "HEEL", "CAL"),
    "RIGHT_HEEL":       ("HEE", "HEEL", "CAL"),
    "LEFT_FOOT_INDEX":  ("TOE", "TT2", "MT2", "MTP", "FT2"),
    "RIGHT_FOOT_INDEX": ("TOE", "TT2", "MT2", "MTP", "FT2"),
    "LEFT_SHOULDER":    ("SHO", "AC$", "ACR"),
    "RIGHT_SHOULDER":   ("SHO", "AC$", "ACR"),
    "NOSE":             ("CLAV", "C7", "STRN", "NOSE"),
}

_REQUIRED_LANDMARKS = (
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
)


def _side_prefix(landmark: str) -> str:
    """Return the side prefix regex for a landmark name (LEFT_/RIGHT_)."""
    if landmark.startswith("LEFT_"):
        return r"^(L(?![A-Z])|LEFT[_ ]?|L_)"
    if landmark.startswith("RIGHT_"):
        return r"^(R(?![A-Z])|RIGHT[_ ]?|R_)"
    return r"^"


def _fuzzy_marker_map(labels: List[str]) -> Dict[str, List[str]]:
    """Build a mapping by regex-matching each landmark to available labels."""
    import re
    mapping: Dict[str, List[str]] = {}
    for lm, keywords in _C3D_FUZZY_PATTERNS.items():
        side_re = _side_prefix(lm)
        candidates: List[str] = []
        for lbl in labels:
            up = lbl.upper()
            if not re.search(side_re, up):
                continue
            if any(re.search(kw, up) for kw in keywords):
                candidates.append(lbl)
        if candidates:
            mapping[lm] = candidates
    return mapping


def detect_c3d_convention(
    labels: List[str],
) -> Tuple[str, Dict[str, List[str]], Dict[str, int]]:
    """Autodetect the marker convention used in a C3D label list.

    Scores each registered convention (:data:`C3D_MARKER_CONVENTIONS`)
    by how many *required* lower-limb landmarks (hip, knee, ankle,
    both sides) it can resolve.  Falls back to a regex-based fuzzy
    match when no registered convention resolves enough landmarks.

    Parameters
    ----------
    labels : list of str
        Marker labels from the C3D ``POINT/LABELS`` section.

    Returns
    -------
    (name, mapping, scores)
        - ``name``: the picked convention name (``"fuzzy_fallback"`` if
          no registered convention was good enough).
        - ``mapping``: the marker mapping actually used
          ({landmark: [C3D labels]}).
        - ``scores``: per-convention count of required landmarks
          resolved, for diagnostics.
    """
    upper_labels = [lbl.upper() for lbl in labels]
    label_set = set(upper_labels)

    scores: Dict[str, int] = {}
    resolved: Dict[str, Dict[str, List[str]]] = {}
    for name, mapping in C3D_MARKER_CONVENTIONS.items():
        found = {}
        for lm, candidates in mapping.items():
            matched = [c for c in candidates if c.upper() in label_set]
            if matched:
                # Return the original-case labels
                found[lm] = [
                    labels[upper_labels.index(m.upper())] for m in matched
                ]
        scores[name] = sum(1 for lm in _REQUIRED_LANDMARKS if lm in found)
        resolved[name] = found

    # Pick the best convention that resolves at least 4/6 required landmarks
    best_name = max(scores, key=lambda k: scores[k])
    if scores[best_name] >= 4:
        return best_name, resolved[best_name], scores

    # Fallback: fuzzy regex match on the raw labels
    fuzzy = _fuzzy_marker_map(labels)
    scores["fuzzy_fallback"] = sum(1 for lm in _REQUIRED_LANDMARKS if lm in fuzzy)
    return "fuzzy_fallback", fuzzy, scores


def load_c3d(
    c3d_path: str | Path,
    marker_mapping: dict | None = None,
    ap_axis: int = 1,
    vertical_axis: int = 2,
) -> dict:
    """Load a C3D file and return a myogait-compatible pivot dict.

    Reads 3-D marker trajectories with *ezc3d*, projects them into a 2-D
    sagittal plane (antero-posterior → x, vertical → y inverted), normalises
    coordinates to [0, 1] and builds the standard pivot structure expected by
    :func:`normalize`, :func:`compute_angles`, etc.

    Parameters
    ----------
    c3d_path : str or Path
        Path to the ``.c3d`` file.
    marker_mapping : dict, optional
        Custom ``{landmark_name: [marker1, marker2, ...]}`` mapping.  When
        *None*, :data:`DEFAULT_C3D_MARKER_MAP` is used.
    ap_axis : int
        Index of the antero-posterior axis in the C3D coordinate system
        (default ``1`` → Y).
    vertical_axis : int
        Index of the vertical axis (default ``2`` → Z).

    Returns
    -------
    dict
        Pivot dict with ``meta``, ``extraction`` and ``frames`` keys.
    """
    c3d_path = Path(c3d_path)
    if not c3d_path.exists():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    try:
        import ezc3d
    except ImportError:
        raise ImportError(
            "ezc3d is required to read C3D files. "
            "Install it with: pip install myogait[c3d]"
        )

    c3d = ezc3d.c3d(str(c3d_path))

    # ── Extract marker labels and point data ──
    labels = [lbl.strip() for lbl in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    points = c3d["data"]["points"]  # (4, n_markers, n_frames) — X,Y,Z,residual
    fps = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    n_frames = points.shape[2]

    label_idx = {lbl: i for i, lbl in enumerate(labels)}

    # ── Resolve which marker convention to use ──
    if marker_mapping is not None:
        mapping = marker_mapping
        convention_name = "user_supplied"
        conv_scores: Dict[str, int] = {}
    else:
        convention_name, mapping, conv_scores = detect_c3d_convention(labels)
        n_resolved = sum(1 for lm in _REQUIRED_LANDMARKS if lm in mapping)
        logger.info(
            "C3D %s: detected convention '%s' — %d/6 required lower-limb "
            "landmarks resolved  (scores: %s)",
            c3d_path.name, convention_name, n_resolved,
            {k: v for k, v in sorted(conv_scores.items(), key=lambda x: -x[1])},
        )
        if n_resolved < 4:
            raise ValueError(
                f"Could not resolve enough lower-limb markers in "
                f"{c3d_path.name}.\n"
                f"  Best convention '{convention_name}' resolved "
                f"{n_resolved}/6 required landmarks.\n"
                f"  Convention scores: {conv_scores}\n"
                f"  Available labels: {labels}\n"
                f"  Pass an explicit marker_mapping={{'LEFT_HIP': [...], ...}} "
                f"or add your convention to myogait.experimental_vicon."
                f"C3D_MARKER_CONVENTIONS."
            )

    # ── Resolve landmarks → 2-D sagittal coords per frame ──
    # For each landmark: pick the first available marker or average all found.
    # Then project: x = ap_axis component, y = -vertical_axis component.
    raw: Dict[str, np.ndarray] = {}  # landmark → (n_frames, 2)
    for lm_name, candidates in mapping.items():
        found = [label_idx[mk] for mk in candidates if mk in label_idx]
        if not found:
            logger.debug("C3D: no marker found for %s (tried %s)", lm_name, candidates)
            continue
        # points shape: (4, n_markers, n_frames) → take XYZ (first 3 rows)
        pts = points[:3, found, :]  # (3, n_found, n_frames)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(pts, axis=1)  # (3, n_frames)
        raw[lm_name] = np.column_stack([avg[ap_axis], avg[vertical_axis]])

    if not raw:
        raise ValueError(
            f"No matching markers found in {c3d_path.name}. "
            f"Available labels: {labels}"
        )

    # ── Normalise to [0, 1] using global bounds ──
    all_xy = np.vstack(list(raw.values()))  # (total_pts, 2)
    x_min, y_min = np.nanmin(all_xy, axis=0)
    x_max, y_max = np.nanmax(all_xy, axis=0)
    x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

    virtual_w = 1000
    virtual_h = 1000

    # ── Build frames ──
    frames = []
    for fi in range(n_frames):
        landmarks = {}
        conf_sum = 0.0
        for lm_name, xy in raw.items():
            nx = float((xy[fi, 0] - x_min) / x_range)
            # Invert vertical so that up in world = small y in image space
            ny = 1.0 - float((xy[fi, 1] - y_min) / y_range)
            vis = 0.0 if np.isnan(xy[fi, 0]) or np.isnan(xy[fi, 1]) else 1.0
            landmarks[lm_name] = {"x": nx, "y": ny, "visibility": vis}
            conf_sum += vis
        confidence = conf_sum / max(len(raw), 1)
        frames.append({
            "frame_idx": fi,
            "time_s": float(fi / fps) if fps > 0 else 0.0,
            "confidence": confidence,
            "landmarks": landmarks,
        })

    logger.info(
        "Loaded C3D %s: %d frames, %d landmarks, %.1f fps",
        c3d_path.name, n_frames, len(raw), fps,
    )

    return {
        "meta": {
            "fps": fps,
            "n_frames": n_frames,
            "duration_s": float(n_frames / fps) if fps > 0 else 0.0,
            "source": "c3d",
            "width": virtual_w,
            "height": virtual_h,
        },
        "extraction": {
            "model": "vicon",
            "keypoint_format": "mediapipe33",
            "n_landmarks": len(raw),
            "source_file": str(c3d_path),
            "c3d_convention": convention_name,
            "c3d_convention_scores": conv_scores,
            "c3d_marker_mapping": {lm: mks for lm, mks in mapping.items()
                                     if lm in raw},
        },
        "frames": frames,
    }
