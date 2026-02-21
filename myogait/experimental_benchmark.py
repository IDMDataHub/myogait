"""Experimental single-pair benchmark orchestration (video + VICON).

This module is intentionally experimental and intended for AIM benchmark
workflows. It runs multiple configurable pipeline variants on one video and
compares each run against one VICON trial.
"""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .extract import extract
from .normalize import normalize
from .angles import compute_angles
from .events import detect_events, list_event_methods
from .schema import save_json
from .models import list_models
from .experimental_vicon import run_single_trial_vicon_benchmark


DEFAULT_SINGLE_PAIR_BENCHMARK_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "scope": "AIM benchmark only",
    "models": ["mediapipe"],  # or "all"
    "event_methods": "all",   # or list[str]
    "normalization_variants": [
        {
            "name": "none",
            "enabled": False,
            "kwargs": {},
        },
        {
            "name": "butterworth_default",
            "enabled": True,
            "kwargs": {"filters": ["butterworth"]},
        },
    ],
    "degradation_variants": [
        {
            "name": "none",
            "experimental": {"enabled": False},
        }
    ],
    "extract_kwargs": {
        "max_frames": None,
        "flip_if_right": True,
        "correct_inversions": True,
    },
    "angle_kwargs": {
        "method": "sagittal_vertical_axis",
        "correction_factor": 0.8,
        "calibrate": True,
    },
    "vicon": {
        "vicon_fps": 200.0,
        "max_lag_seconds": 10.0,
    },
    "continue_on_error": True,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_single_pair_benchmark_config(config: Optional[dict] = None) -> dict:
    """Build benchmark config with defaults."""
    if config is None:
        return deepcopy(DEFAULT_SINGLE_PAIR_BENCHMARK_CONFIG)
    return _deep_merge(DEFAULT_SINGLE_PAIR_BENCHMARK_CONFIG, config)


def _resolve_models(models_cfg: Any) -> List[str]:
    if models_cfg == "all":
        return sorted(list_models())
    if not isinstance(models_cfg, list) or not models_cfg:
        raise ValueError("benchmark.models must be 'all' or a non-empty list")
    return [str(m) for m in models_cfg]


def _resolve_event_methods(event_cfg: Any) -> List[str]:
    if event_cfg == "all":
        return sorted(list_event_methods())
    if not isinstance(event_cfg, list) or not event_cfg:
        raise ValueError("benchmark.event_methods must be 'all' or a non-empty list")
    return [str(m) for m in event_cfg]


def _flatten_row_metrics(row: dict, data: dict) -> dict:
    block = ((data.get("experimental") or {}).get("vicon_benchmark") or {})
    sync = block.get("sync", {})
    metrics = block.get("metrics", {})
    angle_metrics = metrics.get("angle_metrics", {})
    event_metrics = metrics.get("event_metrics_ms", {})

    row["sync_offset_seconds"] = sync.get("offset_seconds")
    row["sync_correlation_peak"] = sync.get("correlation_peak")
    row["sync_signal_used"] = sync.get("signal_used")

    for joint in ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"):
        jm = angle_metrics.get(joint, {})
        row[f"{joint}_rmse_deg"] = jm.get("rmse_deg")
        row[f"{joint}_mae_deg"] = jm.get("mae_deg")
        row[f"{joint}_bias_deg"] = jm.get("bias_deg")
        row[f"{joint}_rom_diff_deg"] = jm.get("rom_diff_deg")

    for ev in ("left_hs", "right_hs", "left_to", "right_to"):
        em = event_metrics.get(ev, {})
        row[f"{ev}_mae_ms"] = em.get("mae_ms")
        row[f"{ev}_median_ms"] = em.get("median_ms")

    return row


def run_single_pair_benchmark(
    video_path: str | Path,
    vicon_trial_dir: str | Path,
    output_dir: str | Path,
    benchmark_config: Optional[dict] = None,
) -> dict:
    """Run experimental benchmark for one video/VICON pair.

    Produces:
    - One JSON result per run in ``<output_dir>/runs``
    - One CSV summary in ``<output_dir>/benchmark_summary.csv``
    """
    cfg = build_single_pair_benchmark_config(benchmark_config)
    out_dir = Path(output_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    models = _resolve_models(cfg.get("models"))
    event_methods = _resolve_event_methods(cfg.get("event_methods"))
    norm_variants = cfg.get("normalization_variants", [])
    deg_variants = cfg.get("degradation_variants", [])
    if not norm_variants:
        raise ValueError("benchmark.normalization_variants cannot be empty")
    if not deg_variants:
        raise ValueError("benchmark.degradation_variants cannot be empty")

    extract_kwargs = dict(cfg.get("extract_kwargs", {}))
    angle_kwargs = dict(cfg.get("angle_kwargs", {}))
    vicon_cfg = dict(cfg.get("vicon", {}))
    continue_on_error = bool(cfg.get("continue_on_error", True))

    rows = []
    run_index = 0

    for model, deg, norm, event_method in product(models, deg_variants, norm_variants, event_methods):
        run_index += 1
        deg_name = str(deg.get("name", "deg"))
        norm_name = str(norm.get("name", "norm"))
        run_id = f"{run_index:03d}_{model}__{deg_name}__{norm_name}__{event_method}"
        run_json = runs_dir / f"{run_id}.json"

        row = {
            "run_id": run_id,
            "status": "ok",
            "video_path": str(video_path),
            "vicon_trial_dir": str(vicon_trial_dir),
            "model": model,
            "degradation_variant": deg_name,
            "normalization_variant": norm_name,
            "event_method": event_method,
            "json_path": str(run_json),
            "error": None,
        }

        try:
            data = extract(
                str(video_path),
                model=model,
                experimental=deg.get("experimental", {"enabled": False}),
                **extract_kwargs,
            )

            if bool(norm.get("enabled", False)):
                data = normalize(data, **dict(norm.get("kwargs", {})))

            data = compute_angles(data, **angle_kwargs)
            data = detect_events(data, method=event_method)

            data = run_single_trial_vicon_benchmark(
                data,
                trial_dir=vicon_trial_dir,
                vicon_fps=float(vicon_cfg.get("vicon_fps", 200.0)),
                max_lag_seconds=float(vicon_cfg.get("max_lag_seconds", 10.0)),
            )
            data.setdefault("experimental", {})
            data["experimental"]["benchmark_run"] = {
                "status": "experimental",
                "scope": "AIM benchmark only",
                "run_id": run_id,
                "config": {
                    "model": model,
                    "degradation_variant": deg,
                    "normalization_variant": norm,
                    "event_method": event_method,
                    "extract_kwargs": extract_kwargs,
                    "angle_kwargs": angle_kwargs,
                    "vicon": vicon_cfg,
                },
            }

            save_json(data, run_json)
            row = _flatten_row_metrics(row, data)

        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            if not continue_on_error:
                rows.append(row)
                summary_path = out_dir / "benchmark_summary.csv"
                pd.DataFrame(rows).to_csv(summary_path, index=False)
                raise

        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = out_dir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)

    manifest = {
        "status": "experimental",
        "scope": "AIM benchmark only",
        "video_path": str(video_path),
        "vicon_trial_dir": str(vicon_trial_dir),
        "n_runs": int(len(rows)),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_error": int((df["status"] == "error").sum()),
        "summary_csv": str(summary_path),
        "runs_dir": str(runs_dir),
        "config": cfg,
    }
    with open(out_dir / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
