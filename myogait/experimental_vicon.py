"""Experimental VICON alignment and single-trial benchmark utilities.

This module is intentionally experimental and designed for AIM benchmark
workflows. It aligns one myogait result (single video) with one VICON trial
and computes comparison metrics on the overlapping temporal window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy import signal


_ANGLE_CANDIDATES = {
    "hip_L": ("Lhip", "l_hip"),
    "hip_R": ("Rhip", "r_hip"),
    "knee_L": ("Lknee", "l_knee"),
    "knee_R": ("Rknee", "r_knee"),
    "ankle_L": ("Lankle", "l_ankle"),
    "ankle_R": ("Rankle", "r_ankle"),
}

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
    candidates = ("knee_L", "knee_R")
    best_name = None
    best_len = -1
    best_pair = (np.array([]), np.array([]))
    for name in candidates:
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
    if len(mg_sig) < 10 or len(vc_sig) < 10:
        raise ValueError("Not enough angle samples for synchronization")

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
