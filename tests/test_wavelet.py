"""Tests for filter_wavelet wavelet denoising."""

import numpy as np
import pandas as pd
import pytest


def test_wavelet_removes_noise():
    """Wavelet denoising should reduce noise substantially."""
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    clean = np.sin(2 * np.pi * 2 * t) * 30
    noisy = clean + np.random.randn(200) * 5
    df = pd.DataFrame({"signal": noisy})
    from myogait.normalize import filter_wavelet

    result = filter_wavelet(df)
    # Denoised should be closer to clean than noisy was
    noisy_rmse = np.sqrt(np.mean((noisy - clean) ** 2))
    denoised_rmse = np.sqrt(np.mean((result["signal"].values - clean) ** 2))
    assert denoised_rmse < noisy_rmse


def test_wavelet_preserves_clean_signal():
    """Clean signal should not be distorted much."""
    t = np.linspace(0, 1, 200)
    clean = np.sin(2 * np.pi * 2 * t) * 30
    df = pd.DataFrame({"signal": clean})
    from myogait.normalize import filter_wavelet

    result = filter_wavelet(df)
    rmse = np.sqrt(np.mean((result["signal"].values - clean) ** 2))
    assert rmse < 1.0  # Very small distortion


def test_wavelet_soft_vs_hard():
    """Both soft and hard thresholding modes produce valid output."""
    np.random.seed(123)
    t = np.linspace(0, 1, 200)
    noisy = np.sin(2 * np.pi * 3 * t) * 20 + np.random.randn(200) * 3
    df = pd.DataFrame({"signal": noisy})
    from myogait.normalize import filter_wavelet

    result_soft = filter_wavelet(df, threshold_mode="soft")
    result_hard = filter_wavelet(df, threshold_mode="hard")

    # Both should produce finite values with the same shape
    assert result_soft.shape == df.shape
    assert result_hard.shape == df.shape
    assert np.all(np.isfinite(result_soft["signal"].values))
    assert np.all(np.isfinite(result_hard["signal"].values))

    # Soft and hard should generally produce different results
    # (unless the signal is trivial)
    assert not np.allclose(
        result_soft["signal"].values, result_hard["signal"].values
    )


def test_wavelet_different_families():
    """Test with different wavelet families: sym4, coif3."""
    np.random.seed(99)
    t = np.linspace(0, 1, 200)
    noisy = np.sin(2 * np.pi * 2 * t) * 25 + np.random.randn(200) * 4
    clean = np.sin(2 * np.pi * 2 * t) * 25
    df = pd.DataFrame({"signal": noisy})
    from myogait.normalize import filter_wavelet

    noisy_rmse = np.sqrt(np.mean((noisy - clean) ** 2))

    for wav in ("sym4", "coif3"):
        result = filter_wavelet(df, wavelet=wav)
        assert result.shape == df.shape
        denoised_rmse = np.sqrt(
            np.mean((result["signal"].values - clean) ** 2)
        )
        assert denoised_rmse < noisy_rmse, (
            f"Wavelet {wav} did not reduce noise"
        )


def test_wavelet_handles_nan():
    """NaN values in the middle should be preserved after denoising."""
    np.random.seed(7)
    t = np.linspace(0, 1, 200)
    signal = np.sin(2 * np.pi * 2 * t) * 30 + np.random.randn(200) * 2
    signal[50] = np.nan
    signal[100] = np.nan
    signal[101] = np.nan
    df = pd.DataFrame({"signal": signal})
    from myogait.normalize import filter_wavelet

    result = filter_wavelet(df)

    # NaN positions should still be NaN
    assert np.isnan(result["signal"].iloc[50])
    assert np.isnan(result["signal"].iloc[100])
    assert np.isnan(result["signal"].iloc[101])

    # Non-NaN positions should be finite
    non_nan_mask = ~np.isnan(signal)
    assert np.all(np.isfinite(result["signal"].values[non_nan_mask]))


def test_wavelet_custom_level():
    """Explicit decomposition level parameter should work."""
    import warnings
    np.random.seed(55)
    t = np.linspace(0, 1, 200)
    noisy = np.sin(2 * np.pi * 2 * t) * 30 + np.random.randn(200) * 5
    df = pd.DataFrame({"signal": noisy})
    from myogait.normalize import filter_wavelet

    # Low level = less denoising; high level = more aggressive
    result_low = filter_wavelet(df, level=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_high = filter_wavelet(df, level=5)

    assert result_low.shape == df.shape
    assert result_high.shape == df.shape
    assert np.all(np.isfinite(result_low["signal"].values))
    assert np.all(np.isfinite(result_high["signal"].values))


def test_wavelet_import_error():
    """Mocking pywt import failure should raise ImportError with message."""
    import builtins
    from unittest.mock import patch

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pywt":
            raise ImportError("No module named 'pywt'")
        return real_import(name, *args, **kwargs)

    df = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0, 5.0]})

    # Need to reload the module to ensure the import inside the function
    # is actually called (it's not cached at module level)
    from myogait.normalize import filter_wavelet

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="PyWavelets"):
            filter_wavelet(df)


def test_wavelet_registered_in_steps():
    """The 'wavelet' key should exist in NORMALIZE_STEPS."""
    from myogait.normalize import NORMALIZE_STEPS, filter_wavelet

    assert "wavelet" in NORMALIZE_STEPS
    assert NORMALIZE_STEPS["wavelet"] is filter_wavelet


def test_wavelet_output_same_shape():
    """Output DataFrame should have the same shape as input."""
    np.random.seed(0)
    df = pd.DataFrame({
        "col_a": np.random.randn(150),
        "col_b": np.random.randn(150),
        "col_c": np.random.randn(150),
    })
    from myogait.normalize import filter_wavelet

    result = filter_wavelet(df)
    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)


def test_wavelet_multi_column():
    """Wavelet denoising should work with multiple columns simultaneously."""
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    clean_a = np.sin(2 * np.pi * 2 * t) * 30
    clean_b = np.cos(2 * np.pi * 3 * t) * 20
    noisy_a = clean_a + np.random.randn(200) * 5
    noisy_b = clean_b + np.random.randn(200) * 5
    df = pd.DataFrame({"signal_a": noisy_a, "signal_b": noisy_b})
    from myogait.normalize import filter_wavelet

    result = filter_wavelet(df)

    # Both columns should be denoised
    rmse_a_noisy = np.sqrt(np.mean((noisy_a - clean_a) ** 2))
    rmse_b_noisy = np.sqrt(np.mean((noisy_b - clean_b) ** 2))
    rmse_a_denoised = np.sqrt(
        np.mean((result["signal_a"].values - clean_a) ** 2)
    )
    rmse_b_denoised = np.sqrt(
        np.mean((result["signal_b"].values - clean_b) ** 2)
    )

    assert rmse_a_denoised < rmse_a_noisy
    assert rmse_b_denoised < rmse_b_noisy
    assert result.shape == df.shape
