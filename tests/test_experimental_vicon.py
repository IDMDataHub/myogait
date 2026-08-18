"""Tests for experimental VICON single-trial benchmark helpers."""

import numpy as np
import pytest

from myogait.experimental_vicon import (
    estimate_vicon_offset_seconds,
    align_vicon_to_myogait,
    compute_single_trial_benchmark_metrics,
    attach_vicon_experimental_block,
    load_vicon_trial_mat,
)


def _make_synthetic_pair(offset_s: float = 0.4):
    fps_mg = 50.0
    fps_vc = 200.0
    n_mg = 300
    n_vc = 1200

    t_mg = np.arange(n_mg) / fps_mg
    t_vc = np.arange(n_vc) / fps_vc
    base = lambda t: 30.0 + 15.0 * np.sin(2 * np.pi * 1.2 * t)  # noqa: E731

    # Mapping in module: vicon_time = myogait_time + offset
    mg_knee = base(t_mg)
    vc_knee = base(t_vc - offset_s)

    myogait = {
        "meta": {"fps": fps_mg, "n_frames": n_mg},
        "angles": {
            "frames": [
                {
                    "hip_L": 0.0,
                    "hip_R": 0.0,
                    "knee_L": float(mg_knee[i]),
                    "knee_R": float(mg_knee[i]),
                    "ankle_L": 0.0,
                    "ankle_R": 0.0,
                }
                for i in range(n_mg)
            ]
        },
        "events": {
            "left_hs": [{"frame": 50}, {"frame": 100}, {"frame": 150}],
            "right_hs": [{"frame": 75}, {"frame": 125}, {"frame": 175}],
            "left_to": [{"frame": 65}, {"frame": 115}],
            "right_to": [{"frame": 90}, {"frame": 140}],
        },
    }

    vicon = {
        "meta": {"fps": fps_vc, "n_frames": n_vc, "trial_name": "synthetic"},
        "angles": {
            "hip_L": np.zeros(n_vc),
            "hip_R": np.zeros(n_vc),
            "knee_L": vc_knee,
            "knee_R": vc_knee,
            "ankle_L": np.zeros(n_vc),
            "ankle_R": np.zeros(n_vc),
        },
        "landmarks": {},
        "events": {
            "left_hs": [50, 100, 150],
            "right_hs": [75, 125, 175],
            "left_to": [65, 115],
            "right_to": [90, 140],
        },
    }
    return myogait, vicon, offset_s


def test_load_vicon_trial_mat_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vicon_trial_mat(tmp_path / "nope")


def test_estimate_offset_synthetic():
    myogait, vicon, expected = _make_synthetic_pair(offset_s=0.4)
    sync = estimate_vicon_offset_seconds(myogait, vicon, max_lag_seconds=2.0)
    assert sync["signal_used"] in ("knee_L", "knee_R")
    assert sync["offset_seconds"] == pytest.approx(expected, abs=0.06)


def test_alignment_and_metrics_synthetic():
    myogait, vicon, expected = _make_synthetic_pair(offset_s=0.4)
    alignment = align_vicon_to_myogait(myogait, vicon, offset_seconds=expected)
    assert alignment["n_aligned_frames"] > 0

    metrics = compute_single_trial_benchmark_metrics(myogait, vicon, alignment)
    assert metrics["status"] == "ok"
    assert "knee_L" in metrics["angle_metrics"]
    assert metrics["angle_metrics"]["knee_L"]["rmse_deg"] is not None
    assert metrics["angle_metrics"]["knee_L"]["rmse_deg"] < 1.5


def test_attach_experimental_block():
    myogait, vicon, expected = _make_synthetic_pair(offset_s=0.3)
    sync = {"offset_seconds": expected, "correlation_peak": 0.9}
    alignment = align_vicon_to_myogait(myogait, vicon, offset_seconds=expected)
    metrics = compute_single_trial_benchmark_metrics(myogait, vicon, alignment)
    out = attach_vicon_experimental_block(myogait, vicon, sync, alignment, metrics)
    assert "experimental" in out
    assert "vicon_benchmark" in out["experimental"]
    assert out["experimental"]["vicon_benchmark"]["status"] == "experimental"


def test_sync_candidates_constant_includes_hips():
    from myogait.experimental_vicon import _SYNC_CANDIDATE_JOINTS, _MIN_SYNC_SAMPLES
    assert _SYNC_CANDIDATE_JOINTS[:2] == ("knee_L", "knee_R")  # knees first (best signal)
    assert "hip_L" in _SYNC_CANDIDATE_JOINTS and "hip_R" in _SYNC_CANDIDATE_JOINTS
    assert isinstance(_MIN_SYNC_SAMPLES, int) and _MIN_SYNC_SAMPLES >= 10


def test_best_sync_falls_back_to_hip_when_knees_missing():
    from myogait.experimental_vicon import _best_sync_signal
    mg = {"hip_L": np.zeros(50)}
    vc = {"hip_L": np.zeros(50), "knee_L": np.zeros(60)}
    name, a, b = _best_sync_signal(mg, vc)
    assert name == "hip_L"
    assert len(a) == 50


def test_insufficient_samples_raises_syncerror_with_diagnostic():
    from myogait.experimental_vicon import SyncError, estimate_vicon_offset_seconds
    mg = {"meta": {"fps": 30.0}, "angles": {"frames": [
        {"knee_L": 1.0, "knee_R": 1.0, "hip_L": 0.0, "hip_R": 0.0}
        for _ in range(3)]}}
    vc = {"meta": {"fps": 200.0, "n_frames": 400}, "angles": {
        "knee_L": np.zeros(400), "knee_R": np.zeros(400),
        "hip_L": np.zeros(400), "hip_R": np.zeros(400)}}
    with pytest.raises(SyncError) as exc_info:
        estimate_vicon_offset_seconds(mg, vc)
    err = exc_info.value
    assert err.required == 10
    assert err.candidates["knee_L"] == 3
    assert err.candidates["hip_L"] == 3
    msg = str(err)
    assert "Not enough angle samples" in msg
    assert "knee_L=3" in msg
    assert "longer recording" in msg
