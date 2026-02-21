"""Tests for experimental single-pair benchmark orchestration."""

import json
from pathlib import Path

import pandas as pd

from myogait.experimental_benchmark import (
    build_single_pair_benchmark_config,
    run_single_pair_benchmark,
)


def test_build_single_pair_config_merges_defaults():
    cfg = build_single_pair_benchmark_config(
        {
            "models": ["mediapipe", "yolo"],
            "vicon": {"vicon_fps": 100.0},
        }
    )
    assert cfg["models"] == ["mediapipe", "yolo"]
    assert cfg["vicon"]["vicon_fps"] == 100.0
    assert "max_lag_seconds" in cfg["vicon"]
    assert cfg["degradation_variants"][0]["name"] == "none"


def test_run_single_pair_benchmark_generates_json_and_csv(tmp_path, monkeypatch):
    calls = {"extract": 0, "normalize": 0, "angles": 0, "events": 0, "vicon": 0}

    def fake_extract(video_path, model, experimental, **kwargs):
        calls["extract"] += 1
        return {
            "meta": {"fps": 30.0, "n_frames": 3},
            "frames": [{"frame_idx": i, "landmarks": {}} for i in range(3)],
            "extraction": {"model": model},
            "experimental_input": experimental,
            "extract_kwargs": kwargs,
        }

    def fake_normalize(data, **kwargs):
        calls["normalize"] += 1
        data["normalization"] = {"kwargs": kwargs}
        return data

    def fake_compute_angles(data, **kwargs):
        calls["angles"] += 1
        data["angles"] = {
            "frames": [
                {
                    "hip_L": 0.0, "hip_R": 0.0,
                    "knee_L": 10.0, "knee_R": 11.0,
                    "ankle_L": 2.0, "ankle_R": 3.0,
                }
                for _ in range(3)
            ]
        }
        data["angle_kwargs"] = kwargs
        return data

    def fake_detect_events(data, method):
        calls["events"] += 1
        data["events"] = {
            "method": method,
            "left_hs": [{"frame": 1}],
            "right_hs": [{"frame": 2}],
            "left_to": [{"frame": 1}],
            "right_to": [{"frame": 2}],
        }
        return data

    def fake_vicon(data, trial_dir, vicon_fps, max_lag_seconds):
        calls["vicon"] += 1
        data.setdefault("experimental", {})
        data["experimental"]["vicon_benchmark"] = {
            "sync": {
                "offset_seconds": 0.12,
                "correlation_peak": 0.9,
                "signal_used": "knee_L",
            },
            "metrics": {
                "angle_metrics": {
                    "knee_L": {"rmse_deg": 1.5, "mae_deg": 1.1, "bias_deg": 0.2, "rom_diff_deg": 0.3},
                    "knee_R": {"rmse_deg": 1.7, "mae_deg": 1.2, "bias_deg": 0.1, "rom_diff_deg": 0.4},
                },
                "event_metrics_ms": {
                    "left_hs": {"mae_ms": 18.0, "median_ms": 16.0},
                    "right_hs": {"mae_ms": 20.0, "median_ms": 18.0},
                },
            },
        }
        return data

    monkeypatch.setattr("myogait.experimental_benchmark.extract", fake_extract)
    monkeypatch.setattr("myogait.experimental_benchmark.normalize", fake_normalize)
    monkeypatch.setattr("myogait.experimental_benchmark.compute_angles", fake_compute_angles)
    monkeypatch.setattr("myogait.experimental_benchmark.detect_events", fake_detect_events)
    monkeypatch.setattr("myogait.experimental_benchmark.run_single_trial_vicon_benchmark", fake_vicon)

    cfg = {
        "models": ["mediapipe", "yolo"],
        "event_methods": ["zeni", "velocity"],
        "normalization_variants": [
            {"name": "none", "enabled": False, "kwargs": {}},
            {"name": "bw", "enabled": True, "kwargs": {"filters": ["butterworth"]}},
        ],
        "degradation_variants": [
            {"name": "none", "experimental": {"enabled": False}},
            {"name": "lowres", "experimental": {"enabled": True, "downscale": 0.7}},
        ],
    }

    out = run_single_pair_benchmark(
        video_path=tmp_path / "video.mp4",
        vicon_trial_dir=tmp_path / "vicon",
        output_dir=tmp_path / "bench",
        benchmark_config=cfg,
    )

    expected_runs = 2 * 2 * 2 * 2
    assert out["n_runs"] == expected_runs
    assert out["n_ok"] == expected_runs
    assert out["n_error"] == 0
    assert calls["extract"] == expected_runs
    assert calls["normalize"] == expected_runs // 2
    assert calls["angles"] == expected_runs
    assert calls["events"] == expected_runs
    assert calls["vicon"] == expected_runs

    summary_csv = Path(out["summary_csv"])
    assert summary_csv.exists()
    df = pd.read_csv(summary_csv)
    assert len(df) == expected_runs
    assert set(df["status"]) == {"ok"}
    assert "knee_L_rmse_deg" in df.columns
    assert "left_hs_mae_ms" in df.columns

    runs_dir = Path(out["runs_dir"])
    run_json_files = sorted(runs_dir.glob("*.json"))
    assert len(run_json_files) == expected_runs
    first = json.loads(run_json_files[0].read_text(encoding="utf-8"))
    assert first["experimental"]["benchmark_run"]["status"] == "experimental"


def test_run_single_pair_benchmark_continue_on_error(tmp_path, monkeypatch):
    def fake_extract(video_path, model, experimental, **kwargs):
        return {
            "meta": {"fps": 30.0, "n_frames": 2},
            "frames": [{"frame_idx": 0, "landmarks": {}}, {"frame_idx": 1, "landmarks": {}}],
            "extraction": {"model": model},
        }

    def fake_compute_angles(data, **kwargs):
        data["angles"] = {"frames": [{"knee_L": 10.0}, {"knee_L": 12.0}]}
        return data

    def fake_detect_events(data, method):
        if method == "bad":
            raise RuntimeError("bad detector")
        data["events"] = {"left_hs": [{"frame": 0}], "right_hs": [], "left_to": [], "right_to": []}
        return data

    def fake_vicon(data, trial_dir, vicon_fps, max_lag_seconds):
        data.setdefault("experimental", {})
        data["experimental"]["vicon_benchmark"] = {"sync": {}, "metrics": {}}
        return data

    monkeypatch.setattr("myogait.experimental_benchmark.extract", fake_extract)
    monkeypatch.setattr("myogait.experimental_benchmark.compute_angles", fake_compute_angles)
    monkeypatch.setattr("myogait.experimental_benchmark.detect_events", fake_detect_events)
    monkeypatch.setattr("myogait.experimental_benchmark.run_single_trial_vicon_benchmark", fake_vicon)

    cfg = {
        "models": ["mediapipe"],
        "event_methods": ["bad", "zeni"],
        "normalization_variants": [{"name": "none", "enabled": False, "kwargs": {}}],
        "degradation_variants": [{"name": "none", "experimental": {"enabled": False}}],
        "continue_on_error": True,
    }
    out = run_single_pair_benchmark(
        video_path=tmp_path / "video.mp4",
        vicon_trial_dir=tmp_path / "vicon",
        output_dir=tmp_path / "bench_err",
        benchmark_config=cfg,
    )
    assert out["n_runs"] == 2
    assert out["n_ok"] == 1
    assert out["n_error"] == 1

    df = pd.read_csv(out["summary_csv"])
    assert set(df["status"]) == {"ok", "error"}
    assert "bad detector" in " ".join(str(v) for v in df["error"].dropna().tolist())
