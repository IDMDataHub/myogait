"""Tests for filter_median in normalize.py.

Covers spike removal, smooth-signal preservation, input validation,
and registry integration.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Ensure the tests directory is on sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from myogait.normalize import filter_median, NORMALIZE_STEPS


# ── helpers ──────────────────────────────────────────────────────────


def _smooth_df(n: int = 100) -> pd.DataFrame:
    """Return a DataFrame with smooth sinusoidal _x / _y columns."""
    t = np.linspace(0, 4 * np.pi, n)
    return pd.DataFrame({
        "LM_x": np.sin(t) * 0.1 + 0.5,
        "LM_y": np.cos(t) * 0.1 + 0.5,
    })


# ── test_median_removes_spike ────────────────────────────────────────


class TestMedianRemovesSpike:

    def test_single_spike_is_suppressed(self):
        """A single-frame spike should be removed by median filtering."""
        df = _smooth_df(100)
        original_x = df["LM_x"].copy()

        # Inject a large spike at frame 50
        df.loc[50, "LM_x"] = 5.0

        filtered = filter_median(df, kernel_size=3)

        # The spiked value should now be close to the original smooth value
        assert abs(filtered["LM_x"].iloc[50] - original_x.iloc[50]) < 0.05

    def test_multiple_isolated_spikes(self):
        """Multiple isolated spikes at different positions are all removed."""
        df = _smooth_df(200)
        original_x = df["LM_x"].copy()

        spike_indices = [20, 80, 140, 180]
        for idx in spike_indices:
            df.loc[idx, "LM_x"] = 10.0

        filtered = filter_median(df, kernel_size=3)

        for idx in spike_indices:
            assert abs(filtered["LM_x"].iloc[idx] - original_x.iloc[idx]) < 0.05, (
                f"Spike at index {idx} was not removed"
            )

    def test_spike_in_y_column(self):
        """Spikes in _y columns are also removed."""
        df = _smooth_df(100)
        original_y = df["LM_y"].copy()

        df.loc[30, "LM_y"] = -3.0

        filtered = filter_median(df, kernel_size=3)

        assert abs(filtered["LM_y"].iloc[30] - original_y.iloc[30]) < 0.05

    def test_larger_kernel_removes_wider_spike(self):
        """A kernel_size=5 filter can remove a 2-frame spike."""
        df = _smooth_df(100)
        original_x = df["LM_x"].copy()

        df.loc[50, "LM_x"] = 5.0
        df.loc[51, "LM_x"] = 5.0

        filtered = filter_median(df, kernel_size=5)

        assert abs(filtered["LM_x"].iloc[50] - original_x.iloc[50]) < 0.1
        assert abs(filtered["LM_x"].iloc[51] - original_x.iloc[51]) < 0.1


# ── test_median_preserves_smooth_signal ──────────────────────────────


class TestMedianPreservesSmoothSignal:

    def test_smooth_sinusoid_unchanged(self):
        """A smooth sinusoidal signal should pass through nearly unchanged.

        A median filter on a finely-sampled smooth signal introduces only
        tiny deviations near curvature extrema (peaks/troughs).  We allow
        up to 1e-2 absolute tolerance, which is negligible compared to the
        signal amplitude of 0.1.
        """
        df = _smooth_df(100)
        original_x = df["LM_x"].values.copy()
        original_y = df["LM_y"].values.copy()

        filtered = filter_median(df, kernel_size=3)

        # Interior points of a smooth signal should be very close
        # (small deviations at peaks are expected due to median quantization)
        np.testing.assert_allclose(
            filtered["LM_x"].values[1:-1],
            original_x[1:-1],
            atol=1e-2,
        )
        np.testing.assert_allclose(
            filtered["LM_y"].values[1:-1],
            original_y[1:-1],
            atol=1e-2,
        )

    def test_constant_signal_unchanged(self):
        """A constant signal should be completely unchanged."""
        n = 50
        df = pd.DataFrame({
            "PT_x": np.full(n, 0.5),
            "PT_y": np.full(n, 0.3),
        })

        filtered = filter_median(df, kernel_size=5)

        np.testing.assert_array_equal(filtered["PT_x"].values, df["PT_x"].values)
        np.testing.assert_array_equal(filtered["PT_y"].values, df["PT_y"].values)

    def test_non_coordinate_columns_untouched(self):
        """Columns that do not end in _x or _y should not be modified."""
        df = _smooth_df(50)
        df["frame_idx"] = np.arange(50)
        df["time_s"] = np.linspace(0, 1, 50)

        filtered = filter_median(df, kernel_size=3)

        np.testing.assert_array_equal(
            filtered["frame_idx"].values, df["frame_idx"].values
        )
        np.testing.assert_array_equal(
            filtered["time_s"].values, df["time_s"].values
        )


# ── test_median_even_kernel_raises ───────────────────────────────────


class TestMedianEvenKernelRaises:

    def test_even_kernel_raises_value_error(self):
        """An even kernel_size should raise ValueError."""
        df = _smooth_df(50)
        with pytest.raises(ValueError, match="positive odd integer"):
            filter_median(df, kernel_size=4)

    def test_zero_kernel_raises_value_error(self):
        """kernel_size=0 should raise ValueError."""
        df = _smooth_df(50)
        with pytest.raises(ValueError, match="positive odd integer"):
            filter_median(df, kernel_size=0)

    def test_negative_kernel_raises_value_error(self):
        """Negative kernel_size should raise ValueError."""
        df = _smooth_df(50)
        with pytest.raises(ValueError, match="positive odd integer"):
            filter_median(df, kernel_size=-3)

    def test_odd_kernel_does_not_raise(self):
        """Odd kernel sizes (1, 3, 5, 7) should work without error."""
        df = _smooth_df(50)
        for ks in (1, 3, 5, 7):
            result = filter_median(df, kernel_size=ks)
            assert len(result) == len(df)


# ── test_median_registered_in_steps ──────────────────────────────────


class TestMedianRegisteredInSteps:

    def test_median_key_exists(self):
        """'median' should be a key in NORMALIZE_STEPS."""
        assert "median" in NORMALIZE_STEPS

    def test_median_maps_to_filter_median(self):
        """The 'median' entry should point to the filter_median function."""
        assert NORMALIZE_STEPS["median"] is filter_median

    def test_median_callable(self):
        """The registered function should be callable."""
        assert callable(NORMALIZE_STEPS["median"])

    def test_median_usable_via_normalize_pipeline(self):
        """filter_median should work when invoked through normalize()."""
        from conftest import make_walking_data
        from myogait import normalize

        data = make_walking_data(n_frames=100)

        # Inject a spike into the raw data
        data["frames"][50]["landmarks"]["LEFT_ANKLE"]["x"] = 5.0

        normalize(data, steps=[{"type": "median", "kernel_size": 3}])

        assert "median" in data["normalization"]["steps_applied"]

        # The spike should have been suppressed
        ankle_x = data["frames"][50]["landmarks"]["LEFT_ANKLE"]["x"]
        assert abs(ankle_x - 5.0) > 1.0, "Spike should have been removed by median filter"
