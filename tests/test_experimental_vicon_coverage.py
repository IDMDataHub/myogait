"""Useful tests for experimental_vicon alignment/metrics behavior."""

import numpy as np
import pytest


def _make_mg_with_angles(n=120, fps=30.0):
    t = np.arange(n) / fps
    knee = np.sin(2 * np.pi * 1.2 * t) * 20.0
    frames = [{"knee_L": float(k), "knee_R": float(-k), "hip_L": 0.0, "hip_R": 0.0, "ankle_L": 0.0, "ankle_R": 0.0} for k in knee]
    return {"meta": {"fps": fps, "n_frames": n}, "angles": {"frames": frames}, "events": {}}


def _make_vc_with_angles(n=800, fps=200.0):
    t = np.arange(n) / fps
    knee = np.sin(2 * np.pi * 1.2 * t) * 20.0
    return {
        "meta": {"fps": fps, "n_frames": n},
        "angles": {
            "knee_L": knee,
            "knee_R": -knee,
            "hip_L": np.zeros_like(knee),
            "hip_R": np.zeros_like(knee),
            "ankle_L": np.zeros_like(knee),
            "ankle_R": np.zeros_like(knee),
        },
        "landmarks": {"LEFT_HIP": np.column_stack([np.ones(n), np.ones(n), np.ones(n)])},
        "events": {"left_hs": [50, 150], "right_hs": [100, 200], "left_to": [], "right_to": []},
    }


def test_interp_nan_and_nearest_abs_diff():
    from myogait.experimental_vicon import _interp_nan, _nearest_abs_diff

    y = np.array([0.0, np.nan, 2.0, np.nan, 4.0], dtype=float)
    yi = _interp_nan(y)
    assert np.isfinite(yi).all()

    diffs = _nearest_abs_diff([10, 20], [12, 23], fps=100.0)
    assert diffs == [20.0, 30.0]


def test_estimate_vicon_offset_invalid_fps_raises():
    from myogait.experimental_vicon import estimate_vicon_offset_seconds

    mg = {"meta": {"fps": 0.0}, "angles": {"frames": []}}
    vc = {"meta": {"fps": 200.0}, "angles": {}}
    with pytest.raises(ValueError, match="Invalid fps"):
        estimate_vicon_offset_seconds(mg, vc)


def test_estimate_vicon_offset_not_enough_samples_raises():
    from myogait.experimental_vicon import estimate_vicon_offset_seconds

    mg = {"meta": {"fps": 30.0}, "angles": {"frames": [{"knee_L": 1.0}] * 5}}
    vc = {"meta": {"fps": 200.0}, "angles": {"knee_L": np.ones(5), "knee_R": np.ones(5)}}
    with pytest.raises(ValueError, match="Not enough angle samples"):
        estimate_vicon_offset_seconds(mg, vc)


def test_estimate_vicon_offset_and_alignment_success():
    from myogait.experimental_vicon import estimate_vicon_offset_seconds, align_vicon_to_myogait

    mg = _make_mg_with_angles()
    vc = _make_vc_with_angles()
    sync = estimate_vicon_offset_seconds(mg, vc, max_lag_seconds=2.0)
    assert "offset_seconds" in sync
    assert "signal_used" in sync

    aligned = align_vicon_to_myogait(mg, vc, sync["offset_seconds"])
    assert aligned["n_aligned_frames"] > 0
    assert aligned["aligned_frames"][0]["angles"]["knee_L"] is not None


def test_angle_error_metrics_and_no_overlap_status():
    from myogait.experimental_vicon import _angle_error_metrics, compute_single_trial_benchmark_metrics

    m = np.array([1.0, 2.0, 3.0])
    v = np.array([1.5, 2.5, 3.5])
    metrics = _angle_error_metrics(m, v)
    assert metrics["n"] == 3
    assert metrics["mae_deg"] == pytest.approx(0.5)

    mg = _make_mg_with_angles()
    vc = _make_vc_with_angles()
    out = compute_single_trial_benchmark_metrics(mg, vc, {"aligned_frames": []})
    assert out["status"] == "no_overlap"


def test_attach_vicon_experimental_block_shape():
    from myogait.experimental_vicon import attach_vicon_experimental_block

    mg = {"experimental": None}
    vc = {"meta": {"trial_name": "t1"}}
    sync = {"offset_seconds": 0.2}
    alignment = {"offset_seconds": 0.2, "n_aligned_frames": 3, "aligned_frames": [{"frame_idx": 0}]}
    metrics = {"status": "ok"}

    out = attach_vicon_experimental_block(mg, vc, sync, alignment, metrics)
    blk = out["experimental"]["vicon_benchmark"]
    assert blk["status"] == "experimental"
    assert blk["alignment"]["n_aligned_frames"] == 3


def test_run_single_trial_pipeline_with_monkeypatch(monkeypatch):
    from myogait.experimental_vicon import run_single_trial_vicon_benchmark

    mg = {"meta": {"fps": 30.0}, "angles": {"frames": []}}

    monkeypatch.setattr(
        "myogait.experimental_vicon.load_vicon_trial_mat",
        lambda *a, **k: {"meta": {"fps": 200.0, "n_frames": 10}, "angles": {}, "landmarks": {}, "events": {}},
    )
    monkeypatch.setattr(
        "myogait.experimental_vicon.estimate_vicon_offset_seconds",
        lambda *a, **k: {"offset_seconds": 0.0},
    )
    monkeypatch.setattr(
        "myogait.experimental_vicon.align_vicon_to_myogait",
        lambda *a, **k: {"offset_seconds": 0.0, "n_aligned_frames": 0, "aligned_frames": []},
    )
    monkeypatch.setattr(
        "myogait.experimental_vicon.compute_single_trial_benchmark_metrics",
        lambda *a, **k: {"status": "no_overlap"},
    )

    out = run_single_trial_vicon_benchmark(mg, "/tmp/does-not-matter")
    assert "experimental" in out
    assert "vicon_benchmark" in out["experimental"]
