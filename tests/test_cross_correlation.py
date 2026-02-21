"""Tests for cross_correlation_lag and align_signals (normalize module)."""

import numpy as np
import pytest

from myogait.normalize import cross_correlation_lag, align_signals


# ── 1. Identical signals should produce zero lag ─────────────────────

def test_identical_signals_zero_lag():
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(100)
    result = cross_correlation_lag(signal, signal)
    assert result["optimal_lag"] == 0
    assert result["max_correlation"] == pytest.approx(1.0, abs=1e-10)


# ── 2. Known positive shift ─────────────────────────────────────────

def test_known_shift():
    rng = np.random.default_rng(0)
    n = 200
    shift = 10
    base = np.sin(2 * np.pi * np.arange(n) / 50) + 0.1 * rng.standard_normal(n)
    signal_a = base.copy()
    # signal_b is signal_a shifted right by `shift` samples
    signal_b = np.roll(base, shift)
    result = cross_correlation_lag(signal_a, signal_b, max_lag=30)
    assert result["optimal_lag"] == shift


# ── 3. Negative shift ───────────────────────────────────────────────

def test_negative_shift():
    rng = np.random.default_rng(1)
    n = 200
    shift = -8
    base = np.sin(2 * np.pi * np.arange(n) / 40) + 0.1 * rng.standard_normal(n)
    signal_a = base.copy()
    signal_b = np.roll(base, shift)
    result = cross_correlation_lag(signal_a, signal_b, max_lag=30)
    assert result["optimal_lag"] == shift


# ── 4. max_lag constraint ────────────────────────────────────────────

def test_max_lag_constraint():
    rng = np.random.default_rng(2)
    n = 200
    base = np.sin(2 * np.pi * np.arange(n) / 50) + 0.1 * rng.standard_normal(n)
    signal_a = base.copy()
    # Shift by 20 samples, but constrain search to max_lag=5
    signal_b = np.roll(base, 20)
    result = cross_correlation_lag(signal_a, signal_b, max_lag=5)
    assert abs(result["optimal_lag"]) <= 5
    # The returned lags array should only span [-5, 5]
    assert result["lags"].min() >= -5
    assert result["lags"].max() <= 5


# ── 5. Correlation coefficient range ────────────────────────────────

def test_correlation_coefficient_range():
    rng = np.random.default_rng(3)
    signal_a = rng.standard_normal(150)
    signal_b = rng.standard_normal(150)
    result = cross_correlation_lag(signal_a, signal_b)
    assert -1.0 <= result["max_correlation"] <= 1.0
    # The entire correlation curve should also be within [-1, 1]
    assert np.all(result["correlation_curve"] >= -1.0 - 1e-10)
    assert np.all(result["correlation_curve"] <= 1.0 + 1e-10)


# ── 6. Unequal lengths raise ValueError ─────────────────────────────

def test_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        cross_correlation_lag(np.ones(10), np.ones(12))


# ── 7. NaN in signal raises ValueError ──────────────────────────────

def test_nan_in_signal_raises():
    a = np.ones(10)
    b = np.ones(10)
    b[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cross_correlation_lag(a, b)

    # Also test NaN in signal_a
    a2 = np.ones(10)
    a2[5] = np.nan
    b2 = np.ones(10)
    with pytest.raises(ValueError, match="NaN"):
        cross_correlation_lag(a2, b2)


# ── 8. align_signals basic ──────────────────────────────────────────

def test_align_signals_basic():
    n = 100
    shift = 7
    base = np.sin(2 * np.pi * np.arange(n) / 25)
    signal_a = base.copy()
    signal_b = np.roll(base, shift)

    result = align_signals(signal_a, signal_b, max_lag=20)

    assert result["optimal_lag"] == shift
    # After alignment, the two signals should be highly correlated
    corr = np.corrcoef(result["aligned_a"], result["aligned_b"])[0, 1]
    assert corr > 0.95


# ── 9. align_signals preserves shape ────────────────────────────────

def test_align_signals_preserves_shape():
    n = 100
    shift = 5
    base = np.sin(2 * np.pi * np.arange(n) / 30)
    signal_a = base.copy()
    signal_b = np.roll(base, shift)

    result = align_signals(signal_a, signal_b, max_lag=20)

    # Both aligned outputs must have same length
    assert len(result["aligned_a"]) == len(result["aligned_b"])
    # Length should be n - |shift|
    assert len(result["aligned_a"]) == n - abs(result["optimal_lag"])


# ── 10. Sine wave phase detection ───────────────────────────────────

def test_sine_wave_phase_detection():
    n = 500
    freq = 5.0  # Hz
    fs = 100.0  # sampling rate
    t = np.arange(n) / fs
    phase_shift_samples = 3  # 3 samples offset

    signal_a = np.sin(2 * np.pi * freq * t)
    signal_b = np.sin(2 * np.pi * freq * (t - phase_shift_samples / fs))

    result = cross_correlation_lag(signal_a, signal_b, max_lag=20)

    # The optimal lag should match the known phase shift
    assert result["optimal_lag"] == phase_shift_samples
    assert result["max_correlation"] > 0.99
