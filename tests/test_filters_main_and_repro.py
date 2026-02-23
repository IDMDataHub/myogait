"""Tests for filters wrapper, module entrypoint, and numeric reproducibility."""

import runpy

import numpy as np
import pandas as pd

from myogait.analysis import _cv, _rom, _symmetry_index
from myogait.filters import apply_filters_pipeline
from myogait.normalize import filter_median, filter_moving_mean


def test_apply_filters_pipeline_ignores_unknown_and_empty_types():
    df = pd.DataFrame({"A_x": [0.0, 1.0, 2.0], "A_y": [0.0, 1.0, 2.0]})
    result = apply_filters_pipeline(
        df,
        filter_configs=[
            {"type": ""},
            {"type": "unknown"},
        ],
        framerate=30.0,
    )

    pd.testing.assert_frame_equal(result, df)


def test_apply_filters_pipeline_sets_fs_for_butterworth():
    df = pd.DataFrame({"A_x": [0.0, 1.0, 2.0], "A_y": [0.0, 1.0, 2.0]})

    result = apply_filters_pipeline(
        df,
        filter_configs=[
            {"type": "moving_mean", "params": {"window": 3}},
            {"type": "butterworth", "params": {"order": 2, "cutoff": 3.0}},
        ],
        framerate=120.0,
    )

    assert list(result.columns) == list(df.columns)
    assert np.isfinite(result.to_numpy()).all()


def test_python_m_myogait_entrypoint_calls_cli_main(monkeypatch):
    called = {"n": 0}

    def fake_main():
        called["n"] += 1

    monkeypatch.setattr("myogait.cli.main", fake_main)
    runpy.run_module("myogait", run_name="__main__")

    assert called["n"] == 1


def test_numeric_helpers_are_stable_and_guard_zero_division():
    assert _symmetry_index(0.0, 0.0) == 0.0
    assert _cv([1.0]) == 0.0
    assert _cv([0.0, 0.0, 0.0]) == 0.0
    assert _rom([np.nan, None]) == 0.0


def test_filter_outputs_are_deterministic_for_identical_input():
    rng = np.random.RandomState(123)
    values = rng.randn(200)
    df = pd.DataFrame({"P_x": values, "P_y": values * 0.5})

    out1 = filter_median(filter_moving_mean(df, window=5), kernel_size=3)
    out2 = filter_median(filter_moving_mean(df, window=5), kernel_size=3)

    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), atol=1e-12)


def test_filter_pipeline_handles_spikes_without_nan_propagation():
    values = np.zeros(101)
    values[50] = 1000.0  # outlier spike
    df = pd.DataFrame({"P_x": values, "P_y": values})

    out = filter_median(df, kernel_size=5)

    assert np.isfinite(out.to_numpy()).all()
    assert out["P_x"].iloc[50] < 1000.0
