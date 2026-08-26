"""Numerical regression tests for efficient analysis primitives."""

from __future__ import annotations

import numpy as np

from myogait.analysis import _positive_autocorrelation


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
