"""I/O and edge-case tests for experimental_vicon helpers."""

import numpy as np

from myogait.experimental_vicon import (
    _pick_struct_array,
    align_vicon_to_myogait,
    compute_single_trial_benchmark_metrics,
    load_c3d,
    load_vicon_trial_mat,
)


def test_load_vicon_trial_mat_empty_existing_dir_returns_empty_payload(tmp_path):
    out = load_vicon_trial_mat(tmp_path)

    assert out["meta"]["trial_dir"] == str(tmp_path)
    assert out["meta"]["n_frames"] == 0
    assert isinstance(out["angles"], dict)
    assert isinstance(out["landmarks"], dict)
    assert isinstance(out["events"], dict)


def test_pick_struct_array_returns_none_for_missing_or_invalid_shape():
    dt = np.dtype([("Lhip", object)])
    struct = np.zeros((1,), dtype=dt)
    struct[0]["Lhip"] = np.zeros((3,), dtype=float)  # invalid: not 2D

    assert _pick_struct_array(struct, ("Lhip",)) is None
    assert _pick_struct_array(None, ("Lhip",)) is None


def test_align_vicon_to_myogait_skips_out_of_range_frames_with_negative_offset():
    mg = {"meta": {"fps": 10.0, "n_frames": 5}, "frames": []}
    vc = {
        "meta": {"fps": 10.0, "n_frames": 5},
        "angles": {"knee_L": np.arange(5, dtype=float), "knee_R": np.arange(5, dtype=float)},
        "landmarks": {},
    }

    aligned = align_vicon_to_myogait(mg, vc, offset_seconds=-0.3)

    assert aligned["n_aligned_frames"] < 5
    assert all(f["vicon_frame"] >= 0 for f in aligned["aligned_frames"])


def test_compute_single_trial_benchmark_metrics_event_metrics_none_when_no_events():
    mg = {
        "meta": {"fps": 30.0},
        "angles": {"frames": [{"knee_L": 1.0, "knee_R": 1.0}]},
        "events": {},
    }
    vc = {
        "angles": {"knee_L": np.array([1.0]), "knee_R": np.array([1.0])},
        "events": {},
    }
    alignment = {"aligned_frames": [{"frame_idx": 0, "vicon_frame": 0}]}

    out = compute_single_trial_benchmark_metrics(mg, vc, alignment)

    assert out["status"] == "ok"
    for key in ("left_hs", "right_hs", "left_to", "right_to"):
        assert out["event_metrics_ms"][key]["mae_ms"] is None
        assert out["event_metrics_ms"][key]["median_ms"] is None


def test_load_c3d_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.c3d"
    try:
        load_c3d(missing)
    except FileNotFoundError as exc:
        assert "C3D file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_load_c3d_import_error_is_clear(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ezc3d":
            raise ImportError("blocked ezc3d")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    c3d_path = tmp_path / "trial.c3d"
    c3d_path.write_bytes(b"dummy")

    try:
        load_c3d(c3d_path)
    except ImportError as exc:
        assert "ezc3d is required" in str(exc)
    else:
        raise AssertionError("Expected ImportError")
