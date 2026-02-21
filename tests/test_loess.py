"""Tests for the LOESS/LOWESS smoothing filter in normalize.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_walking_data
from myogait.normalize import filter_loess, NORMALIZE_STEPS


class TestLoess:

    def _make_noisy_df(self, n=200, noise_std=0.02, seed=42):
        """Build a DataFrame with a clean sine trend plus Gaussian noise."""
        rng = np.random.RandomState(seed)
        t = np.arange(n, dtype=float)
        clean = 0.5 + 0.1 * np.sin(2 * np.pi * t / 30)
        noisy = clean + rng.normal(0, noise_std, n)
        df = pd.DataFrame({
            "frame_idx": t,
            "time_s": t / 30.0,
            "PT_x": noisy,
            "PT_y": noisy * 0.8,
        })
        return df, clean

    # ── test_loess_smooths_noisy_signal ──────────────────────────────

    def test_loess_smooths_noisy_signal(self):
        """LOESS output should be closer to the true signal than the noisy input."""
        df, clean = self._make_noisy_df(n=300, noise_std=0.05)
        smoothed = filter_loess(df, frac=0.05, it=3)

        raw_mse = np.mean((df["PT_x"].values - clean) ** 2)
        smooth_mse = np.mean((smoothed["PT_x"].values - clean) ** 2)

        # Smoothed signal should have lower MSE to the true signal
        assert smooth_mse < raw_mse, (
            f"Smoothed MSE ({smooth_mse:.6f}) should be less than "
            f"raw MSE ({raw_mse:.6f})"
        )

    # ── test_loess_preserves_trend ───────────────────────────────────

    def test_loess_preserves_trend(self):
        """LOESS should preserve the overall trend (mean and shape)."""
        n = 300
        t = np.arange(n, dtype=float)
        # Linear trend with small noise
        trend = 0.3 + 0.001 * t
        rng = np.random.RandomState(99)
        noisy = trend + rng.normal(0, 0.005, n)
        df = pd.DataFrame({
            "frame_idx": t,
            "time_s": t / 30.0,
            "PT_x": noisy,
            "PT_y": noisy,
        })
        smoothed = filter_loess(df, frac=0.15, it=3)

        # Mean of smoothed should be close to mean of true trend
        assert abs(smoothed["PT_x"].mean() - trend.mean()) < 0.01

        # Correlation with the true trend should be very high
        corr = np.corrcoef(smoothed["PT_x"].values, trend)[0, 1]
        assert corr > 0.99, f"Correlation with trend is {corr:.4f}, expected > 0.99"

    # ── test_loess_registered_in_steps ────────────────────────────────

    def test_loess_registered_in_steps(self):
        """filter_loess must be registered as 'loess' in NORMALIZE_STEPS."""
        assert "loess" in NORMALIZE_STEPS
        assert NORMALIZE_STEPS["loess"] is filter_loess

    # ── test_loess_import_error ───────────────────────────────────────

    def test_loess_import_error(self):
        """If statsmodels is unavailable, filter_loess raises ImportError."""
        df, _ = self._make_noisy_df(n=50)
        # Block the statsmodels import inside filter_loess
        import builtins
        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "statsmodels.nonparametric.smoothers_lowess":
                raise ImportError("mocked missing statsmodels")
            return _real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            with pytest.raises(ImportError, match="pip install statsmodels"):
                filter_loess(df)

    # ── test_loess_handles_nan ────────────────────────────────────────

    def test_loess_handles_nan(self):
        """LOESS should handle NaN values gracefully by skipping them."""
        df, clean = self._make_noisy_df(n=100, noise_std=0.01)
        # Inject NaN gap
        df.loc[40:49, "PT_x"] = np.nan
        df.loc[40:49, "PT_y"] = np.nan

        smoothed = filter_loess(df, frac=0.15, it=3)

        # Result should have no NaN (LOWESS interpolates through gaps)
        assert smoothed["PT_x"].notna().all()
        assert smoothed["PT_y"].notna().all()

    # ── test_loess_via_normalize_pipeline ─────────────────────────────

    def test_loess_via_normalize_pipeline(self):
        """LOESS should work end-to-end through the normalize() orchestrator."""
        from myogait import normalize
        data = make_walking_data(n_frames=100)
        result = normalize(data, filters=["loess"])
        assert "loess" in result["normalization"]["steps_applied"]
