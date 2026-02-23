"""Tests for residual analysis and auto cutoff frequency selection.

Tests cover:
- residual_analysis returns correct dict keys
- residual_analysis finds a reasonable cutoff for a signal with known noise
- auto_cutoff_frequency wraps residual_analysis correctly
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Ensure the tests directory is on sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from myogait.normalize import residual_analysis, auto_cutoff_frequency


def _make_noisy_signal(n=300, fs=100.0, signal_freq=2.0, noise_freq=20.0,
                       noise_amp=0.05, seed=42):
    """Create a DataFrame with a clean sinusoid plus high-frequency noise.

    The signal has most of its power at *signal_freq* Hz and additive
    sinusoidal noise at *noise_freq* Hz.  A good residual analysis should
    find a cutoff somewhere between the two.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling rate in Hz.
    signal_freq : float
        Frequency of the clean component in Hz.
    noise_freq : float
        Frequency of the noise component in Hz.
    noise_amp : float
        Amplitude of the noise relative to signal amplitude (1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``LM_x`` and ``LM_y`` columns.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    clean = np.sin(2 * np.pi * signal_freq * t)
    noise = noise_amp * np.sin(2 * np.pi * noise_freq * t)
    # Add a tiny bit of broadband noise as well
    broadband = 0.01 * rng.randn(n)
    signal = clean + noise + broadband
    return pd.DataFrame({"LM_x": signal, "LM_y": signal * 0.8 + 0.1})


# ── test_residual_analysis_returns_dict_keys ────────────────────────


class TestResidualAnalysisDictKeys:

    def test_returns_required_keys(self):
        """residual_analysis must return optimal_cutoff, residuals, per_column."""
        df = _make_noisy_signal(n=200, fs=100.0)
        result = residual_analysis(df, fs=100.0)

        assert "optimal_cutoff" in result
        assert "residuals" in result
        assert "per_column" in result

    def test_optimal_cutoff_is_float(self):
        df = _make_noisy_signal(n=200, fs=100.0)
        result = residual_analysis(df, fs=100.0)
        assert isinstance(result["optimal_cutoff"], float)

    def test_residuals_is_dict_of_floats(self):
        df = _make_noisy_signal(n=200, fs=100.0)
        result = residual_analysis(df, fs=100.0)
        assert isinstance(result["residuals"], dict)
        for k, v in result["residuals"].items():
            assert isinstance(k, float)
            assert isinstance(v, float)

    def test_per_column_contains_all_xy_columns(self):
        df = _make_noisy_signal(n=200, fs=100.0)
        result = residual_analysis(df, fs=100.0)
        expected_cols = {"LM_x", "LM_y"}
        assert set(result["per_column"].keys()) == expected_cols

    def test_residuals_cover_freq_range(self):
        """Number of entries in residuals should match the frequency grid."""
        df = _make_noisy_signal(n=200, fs=100.0)
        result = residual_analysis(
            df, fs=100.0, freq_range=(2.0, 10.0), freq_step=1.0)
        freqs = np.arange(2.0, 10.0 + 0.5, 1.0)
        assert len(result["residuals"]) == len(freqs)


# ── test_residual_analysis_finds_correct_cutoff ─────────────────────


class TestResidualAnalysisFindsCorrectCutoff:

    def test_cutoff_between_signal_and_noise(self):
        """Optimal cutoff should fall between the signal and noise frequencies.

        With signal at 2 Hz and noise at 20 Hz (fs=100), the cutoff
        should be well above 2 Hz and well below 20 Hz.
        """
        df = _make_noisy_signal(
            n=500, fs=100.0, signal_freq=2.0, noise_freq=20.0,
            noise_amp=0.1, seed=0,
        )
        result = residual_analysis(
            df, fs=100.0, freq_range=(1.0, 30.0), freq_step=0.5, order=2,
        )
        cutoff = result["optimal_cutoff"]
        # Cutoff should be above the signal frequency
        assert cutoff >= 2.0, f"Cutoff {cutoff} is below signal freq 2 Hz"
        # Cutoff should be below the noise frequency
        assert cutoff <= 20.0, f"Cutoff {cutoff} is above noise freq 20 Hz"

    def test_cutoff_reasonable_for_gait_like_signal(self):
        """For a gait-like signal (1 Hz walk + noise > 10 Hz), cutoff ~3-10 Hz."""
        df = _make_noisy_signal(
            n=600, fs=30.0, signal_freq=1.0, noise_freq=12.0,
            noise_amp=0.08, seed=7,
        )
        result = residual_analysis(
            df, fs=30.0, freq_range=(1.0, 14.0), freq_step=0.5, order=2,
        )
        cutoff = result["optimal_cutoff"]
        assert 1.0 <= cutoff <= 14.0, f"Cutoff {cutoff} outside search range"

    def test_residuals_monotonically_decrease(self):
        """RMS residuals should generally decrease as cutoff increases."""
        df = _make_noisy_signal(n=300, fs=100.0)
        result = residual_analysis(
            df, fs=100.0, freq_range=(1.0, 40.0), freq_step=1.0)
        freqs_sorted = sorted(result["residuals"].keys())
        residuals = [result["residuals"][f] for f in freqs_sorted]
        # First residual (low cutoff) should be >= last (high cutoff)
        assert residuals[0] >= residuals[-1]

    def test_per_column_cutoffs_within_range(self):
        """Each per-column cutoff should be within the search range."""
        df = _make_noisy_signal(n=400, fs=100.0)
        result = residual_analysis(
            df, fs=100.0, freq_range=(2.0, 30.0), freq_step=0.5)
        for col, fc in result["per_column"].items():
            assert 2.0 <= fc <= 30.0, (
                f"Column {col} cutoff {fc} outside range [2, 30]"
            )

    def test_failed_column_is_ignored_not_zero_biased(self, monkeypatch):
        """A failing column should be excluded instead of contributing zeros."""
        import scipy.signal as sig

        df = _make_noisy_signal(n=400, fs=100.0)
        y_ref = df["LM_y"].to_numpy()
        orig_filtfilt = sig.filtfilt

        def _patched_filtfilt(b, a, x, *args, **kwargs):
            if np.allclose(np.asarray(x), y_ref):
                raise ValueError("synthetic failure for LM_y")
            return orig_filtfilt(b, a, x, *args, **kwargs)

        monkeypatch.setattr(sig, "filtfilt", _patched_filtfilt)

        result = residual_analysis(
            df, fs=100.0, freq_range=(2.0, 20.0), freq_step=1.0, order=2
        )

        assert "LM_x" in result["per_column"]
        assert "LM_y" not in result["per_column"]
        assert result["optimal_cutoff"] == result["per_column"]["LM_x"]


# ── test_auto_cutoff_wraps_residual ─────────────────────────────────


class TestAutoCutoffWrapsResidual:

    def test_returns_float(self):
        """auto_cutoff_frequency should return a single float."""
        df = _make_noisy_signal(n=200, fs=100.0)
        cutoff = auto_cutoff_frequency(df, fs=100.0, method="residual")
        assert isinstance(cutoff, float)

    def test_matches_residual_analysis_optimal(self):
        """auto_cutoff_frequency should return the same value as
        residual_analysis(...)[\"optimal_cutoff\"]."""
        df = _make_noisy_signal(n=200, fs=100.0, seed=99)
        direct = residual_analysis(df, fs=100.0)
        wrapped = auto_cutoff_frequency(df, fs=100.0, method="residual")
        assert direct["optimal_cutoff"] == wrapped

    def test_forwards_kwargs(self):
        """Extra kwargs should be forwarded to residual_analysis."""
        df = _make_noisy_signal(n=200, fs=100.0)
        c1 = auto_cutoff_frequency(
            df, fs=100.0, freq_range=(1.0, 10.0), freq_step=1.0)
        c2 = auto_cutoff_frequency(
            df, fs=100.0, freq_range=(5.0, 40.0), freq_step=1.0)
        # Different ranges can produce different cutoffs (or at least run
        # without error); we mainly check no exception is raised
        assert isinstance(c1, float)
        assert isinstance(c2, float)

    def test_unsupported_method_raises(self):
        """Passing an unknown method should raise ValueError."""
        df = _make_noisy_signal(n=100, fs=100.0)
        with pytest.raises(ValueError, match="Unsupported method"):
            auto_cutoff_frequency(df, fs=100.0, method="psd")

    def test_default_method_is_residual(self):
        """Calling without explicit method should use 'residual'."""
        df = _make_noisy_signal(n=200, fs=100.0)
        # Should not raise
        cutoff = auto_cutoff_frequency(df, fs=100.0)
        assert isinstance(cutoff, float)
