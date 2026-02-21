"""Tests for data quality functions in normalize.py.

Tests cover confidence filtering, outlier detection, quality scoring,
gap handling, and integration with the normalize() pipeline.
"""


import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure the tests directory is on sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    make_walking_data,
    make_standing_data,
    make_fake_data,
    make_walking_data_with_low_confidence,
)
from myogait.normalize import (
    confidence_filter,
    detect_outliers,
    data_quality_score,
    frames_to_dataframe,
    NORMALIZE_STEPS,
)
from myogait import normalize


# ── confidence_filter ────────────────────────────────────────────────


class TestConfidenceFilter:

    def test_confidence_filter_removes_low_vis(self):
        """Landmarks with visibility below threshold should become NaN."""
        data = make_walking_data_with_low_confidence(n_frames=50)
        df = frames_to_dataframe(data["frames"])

        # Count non-NaN before
        before_valid = df[[c for c in df.columns if c.endswith("_x")]].notna().sum().sum()

        filtered = confidence_filter(df, threshold=0.3, _data_frames=data["frames"])

        after_valid = filtered[[c for c in filtered.columns if c.endswith("_x")]].notna().sum().sum()

        # Some coordinates should have been set to NaN
        assert after_valid < before_valid

    def test_confidence_filter_preserves_high_vis(self):
        """Landmarks with visibility >= threshold should be preserved."""
        data = make_standing_data(20)
        # All landmarks in standing data have visibility=1.0
        df = frames_to_dataframe(data["frames"])

        filtered = confidence_filter(df, threshold=0.3, _data_frames=data["frames"])

        # All x/y values should be unchanged (no NaN introduced)
        xcols = [c for c in df.columns if c.endswith("_x")]
        assert filtered[xcols].notna().all().all()

    def test_confidence_filter_threshold_zero(self):
        """With threshold=0, no landmarks should be filtered out."""
        data = make_walking_data_with_low_confidence(n_frames=50)
        df = frames_to_dataframe(data["frames"])

        before_valid = df[[c for c in df.columns if c.endswith("_x")]].notna().sum().sum()

        filtered = confidence_filter(df, threshold=0.0, _data_frames=data["frames"])

        after_valid = filtered[[c for c in filtered.columns if c.endswith("_x")]].notna().sum().sum()

        # Nothing should be removed when threshold is 0
        assert after_valid == before_valid


# ── detect_outliers ──────────────────────────────────────────────────


class TestDetectOutliers:

    def test_detect_outliers_removes_spikes(self):
        """A large spike should be detected and interpolated away."""
        np.random.seed(42)
        n = 100
        x = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.1 + 0.5
        y = np.cos(np.linspace(0, 4 * np.pi, n)) * 0.1 + 0.5

        # Insert spike outlier
        x_spiked = x.copy()
        x_spiked[50] = 5.0  # Way outside normal range

        df = pd.DataFrame({"LM_x": x_spiked, "LM_y": y})
        cleaned = detect_outliers(df, z_thresh=3.0)

        # The spike should have been replaced; value should be close to original
        assert abs(cleaned["LM_x"].iloc[50] - x[50]) < 0.5
        # Other values should be essentially unchanged
        assert np.allclose(cleaned["LM_y"].values, y, atol=1e-10)

    def test_detect_outliers_preserves_clean(self):
        """Clean data without outliers should not be changed."""
        np.random.seed(42)
        n = 100
        x = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.1 + 0.5
        y = np.cos(np.linspace(0, 4 * np.pi, n)) * 0.1 + 0.5

        df = pd.DataFrame({"LM_x": x, "LM_y": y})
        cleaned = detect_outliers(df, z_thresh=3.0)

        # Values should be identical (no outliers to remove)
        np.testing.assert_array_almost_equal(cleaned["LM_x"].values, x, decimal=10)
        np.testing.assert_array_almost_equal(cleaned["LM_y"].values, y, decimal=10)


# ── data_quality_score ───────────────────────────────────────────────


class TestDataQualityScore:

    def test_data_quality_score_perfect_data(self):
        """Perfect data should have a high quality score."""
        data = make_walking_data(n_frames=100)
        result = data_quality_score(data)

        assert result["overall_score"] >= 80.0
        assert result["detection_rate"] == 1.0
        assert result["mean_confidence"] > 0.9
        assert result["gap_pct"] == 0.0

    def test_data_quality_score_low_quality(self):
        """Data with many NaN frames should have a low quality score."""
        data = make_walking_data(n_frames=100)

        # Set most frames to NaN landmarks and low confidence
        for i in range(80):
            for name in list(data["frames"][i]["landmarks"].keys()):
                data["frames"][i]["landmarks"][name] = {
                    "x": float("nan"),
                    "y": float("nan"),
                    "visibility": 0.0,
                }
            data["frames"][i]["confidence"] = 0.0

        result = data_quality_score(data)

        assert result["overall_score"] < 50.0
        assert result["detection_rate"] < 0.5
        assert result["gap_pct"] > 0.5

    def test_data_quality_score_returns_dict_keys(self):
        """Quality score dict should have all expected keys."""
        data = make_fake_data(10)
        result = data_quality_score(data)

        assert "overall_score" in result
        assert "detection_rate" in result
        assert "mean_confidence" in result
        assert "gap_pct" in result
        assert "jitter_score" in result

        # Should also be stored in data["quality"]
        assert data["quality"] is result


# ── normalize() integration ──────────────────────────────────────────


class TestNormalizeIntegration:

    def test_normalize_with_confidence_filter(self):
        """Confidence filter should run as a step within normalize()."""
        data = make_walking_data_with_low_confidence(n_frames=100)

        normalize(data, steps=[
            {"type": "confidence_filter", "threshold": 0.3},
            {"type": "butterworth", "cutoff": 4.0, "order": 2},
        ])

        assert "confidence_filter" in data["normalization"]["steps_applied"]
        assert "butterworth" in data["normalization"]["steps_applied"]

    def test_normalize_gap_max_frames(self):
        """Gaps longer than gap_max_frames should be preserved as NaN."""
        data = make_walking_data(n_frames=100)

        # Create a long gap (15 frames) at frames 20-34
        for i in range(20, 35):
            for name in list(data["frames"][i]["landmarks"].keys()):
                data["frames"][i]["landmarks"][name] = {
                    "x": float("nan"),
                    "y": float("nan"),
                    "visibility": 0.0,
                }

        normalize(data, filters=["butterworth"], gap_max_frames=10)

        # The long gap should still have NaN values after normalization
        # Check a landmark in the middle of the gap
        mid_frame = data["frames"][27]
        ankle_x = mid_frame["landmarks"]["LEFT_ANKLE"]["x"]
        assert np.isnan(ankle_x), "Long gap should remain NaN after normalization"

    def test_normalize_gap_metadata(self):
        """Normalization should record gap information in metadata."""
        data = make_walking_data(n_frames=100)

        # Create a long gap (15 frames)
        for i in range(20, 35):
            for name in list(data["frames"][i]["landmarks"].keys()):
                data["frames"][i]["landmarks"][name] = {
                    "x": float("nan"),
                    "y": float("nan"),
                    "visibility": 0.0,
                }

        normalize(data, filters=["moving_mean"], gap_max_frames=10)

        assert "gaps" in data["normalization"]
        assert "gap_max_frames" in data["normalization"]
        assert data["normalization"]["gap_max_frames"] == 10

        # There should be gap entries for the long gap
        gaps = data["normalization"]["gaps"]
        assert len(gaps) > 0

        # At least one gap should span frames 20-34
        long_gaps = [g for g in gaps if g["length"] > 10]
        assert len(long_gaps) > 0


# ── Step registry ───────────────────────────────────────────────────


class TestStepRegistry:

    def test_quality_registered_in_steps(self):
        """confidence_filter and detect_outliers should be in NORMALIZE_STEPS."""
        assert "confidence_filter" in NORMALIZE_STEPS
        assert "detect_outliers" in NORMALIZE_STEPS
        assert callable(NORMALIZE_STEPS["confidence_filter"])
        assert callable(NORMALIZE_STEPS["detect_outliers"])
