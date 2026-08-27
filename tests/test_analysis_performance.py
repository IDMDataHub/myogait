"""Numerical regression tests for efficient analysis primitives."""

from __future__ import annotations

import numpy as np
import pytest

from myogait.analysis import _frame_rate, _ordered_heel_strikes, _positive_autocorrelation


@pytest.mark.parametrize("fps", [0, -25, "unknown", float("nan"), float("inf"), None])
def test_frame_rate_uses_a_safe_default_for_invalid_metadata(fps):
    assert _frame_rate({"meta": {"fps": fps}}) == 30.0


def test_frame_rate_accepts_numeric_metadata():
    assert _frame_rate({"meta": {"fps": "60"}}) == 60.0


def test_ordered_heel_strikes_interleaves_both_sides_by_frame():
    events = {
        "left_hs": [{"frame": 30}, {"frame": 10}],
        "right_hs": [{"frame": 20}],
    }

    assert _ordered_heel_strikes(events) == [(10, "left"), (20, "right"), (30, "left")]


def test_positive_autocorrelation_matches_numpy_reference_for_a_long_signal():
    signal = np.sin(np.linspace(0, 20 * np.pi, 3_000))
    signal += 0.01 * np.random.default_rng(42).normal(size=signal.size)

    expected = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
    expected /= expected[0]

    actual = _positive_autocorrelation(signal)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_positive_autocorrelation_handles_a_zero_signal_without_nan():
    actual = _positive_autocorrelation(np.zeros(60))

    assert np.array_equal(actual, np.zeros(60))
