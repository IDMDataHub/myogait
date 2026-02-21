"""Tests for new features: analysis, export, and normalize additions."""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myogait.analysis import (
    segment_lengths,
    instantaneous_cadence,
    compute_rom_summary,
    estimate_center_of_mass,
    postural_sway,
)
from myogait.export import (
    to_dataframe,
    export_summary_json,
)
from myogait.normalize import fill_gaps
from conftest import make_walking_data, run_full_pipeline


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def pipeline_data():
    """Full pipeline data (data, cycles, stats)."""
    return run_full_pipeline(n_frames=300, fps=30.0)


@pytest.fixture
def data_and_cycles(pipeline_data):
    """Return (data, cycles) tuple."""
    data, cycles, _ = pipeline_data
    return data, cycles


@pytest.fixture
def walking_data():
    """Raw walking data without pipeline processing."""
    return make_walking_data(n_frames=300, fps=30.0)


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS TESTS
# ══════════════════════════════════════════════════════════════════════


# ── segment_lengths ──────────────────────────────────────────────────


class TestSegmentLengths:

    def test_returns_dict(self, walking_data):
        result = segment_lengths(walking_data)
        assert isinstance(result, dict)

    def test_default_segments_present(self, walking_data):
        result = segment_lengths(walking_data)
        expected_keys = [
            "femur_L", "femur_R", "tibia_L", "tibia_R",
            "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R",
            "trunk_L", "trunk_R",
        ]
        for key in expected_keys:
            assert key in result, f"Missing segment: {key}"

    def test_has_quality_flags(self, walking_data):
        result = segment_lengths(walking_data)
        assert "quality_flags" in result
        assert isinstance(result["quality_flags"], list)

    def test_cv_non_negative(self, walking_data):
        result = segment_lengths(walking_data)
        for key, val in result.items():
            if key == "quality_flags":
                continue
            assert val["cv"] >= 0, f"CV for {key} is negative"

    def test_segment_stats_structure(self, walking_data):
        result = segment_lengths(walking_data)
        for key, val in result.items():
            if key == "quality_flags":
                continue
            assert "mean" in val
            assert "std" in val
            assert "cv" in val
            assert "time_series" in val
            assert isinstance(val["time_series"], list)

    def test_mean_positive(self, walking_data):
        result = segment_lengths(walking_data)
        for key, val in result.items():
            if key == "quality_flags":
                continue
            assert val["mean"] > 0, f"Mean for {key} should be positive"

    def test_custom_segments(self, walking_data):
        custom = [("LEFT_HIP", "LEFT_ANKLE", "full_leg_L")]
        result = segment_lengths(walking_data, segments=custom)
        assert "full_leg_L" in result
        assert "quality_flags" in result

    def test_with_height_m(self, walking_data):
        result_norm = segment_lengths(walking_data, unit="normalized")
        result_m = segment_lengths(walking_data, unit="m", height_m=1.75)
        # Meter values should be larger since height_m > 1
        for key in ["femur_L", "tibia_L"]:
            assert result_m[key]["mean"] > result_norm[key]["mean"]


# ── instantaneous_cadence ────────────────────────────────────────────


class TestInstantaneousCadence:

    def test_returns_dict(self, pipeline_data):
        data, _, _ = pipeline_data
        result = instantaneous_cadence(data)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, pipeline_data):
        data, _, _ = pipeline_data
        result = instantaneous_cadence(data)
        for key in ("times", "cadence", "mean", "std", "cv", "trend_slope"):
            assert key in result, f"Missing key: {key}"

    def test_mean_positive(self, pipeline_data):
        data, _, _ = pipeline_data
        result = instantaneous_cadence(data)
        assert result["mean"] > 0

    def test_trend_is_float(self, pipeline_data):
        data, _, _ = pipeline_data
        result = instantaneous_cadence(data)
        assert isinstance(result["trend_slope"], float)

    def test_cadence_list_not_empty(self, pipeline_data):
        data, _, _ = pipeline_data
        result = instantaneous_cadence(data)
        assert len(result["cadence"]) > 0
        assert len(result["times"]) == len(result["cadence"])

    def test_empty_events(self):
        data = {"events": {}, "meta": {"fps": 30.0}}
        result = instantaneous_cadence(data)
        assert result["mean"] == 0.0
        assert result["cadence"] == []


# ── compute_rom_summary ──────────────────────────────────────────────


class TestComputeRomSummary:

    def test_returns_dict(self, pipeline_data):
        data, cycles, _ = pipeline_data
        result = compute_rom_summary(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, pipeline_data):
        data, cycles, _ = pipeline_data
        result = compute_rom_summary(data, cycles)
        expected = ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_rom_structure(self, pipeline_data):
        data, cycles, _ = pipeline_data
        result = compute_rom_summary(data, cycles)
        for key, val in result.items():
            assert "rom_per_cycle" in val
            assert "rom_mean" in val
            assert "rom_std" in val
            assert "rom_cv" in val
            assert isinstance(val["rom_per_cycle"], list)

    def test_rom_mean_non_negative(self, pipeline_data):
        data, cycles, _ = pipeline_data
        result = compute_rom_summary(data, cycles)
        for key, val in result.items():
            assert val["rom_mean"] >= 0


# ── estimate_center_of_mass ──────────────────────────────────────────


class TestEstimateCenterOfMass:

    def test_returns_dict(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        for key in ("com_x", "com_y", "vertical_excursion", "smoothness"):
            assert key in result, f"Missing key: {key}"

    def test_com_lists(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        assert isinstance(result["com_x"], list)
        assert isinstance(result["com_y"], list)
        assert len(result["com_x"]) == len(walking_data["frames"])
        assert len(result["com_y"]) == len(walking_data["frames"])

    def test_vertical_excursion_positive(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        assert result["vertical_excursion"] >= 0

    def test_smoothness_positive(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        assert result["smoothness"] >= 0

    def test_com_values_are_finite(self, walking_data):
        result = estimate_center_of_mass(walking_data)
        for x, y in zip(result["com_x"], result["com_y"]):
            assert np.isfinite(x)
            assert np.isfinite(y)


# ── postural_sway ────────────────────────────────────────────────────


class TestPosturalSway:

    def test_returns_dict(self, walking_data):
        result = postural_sway(walking_data)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, walking_data):
        result = postural_sway(walking_data)
        for key in ("cop_x", "cop_y", "ellipse_area", "sway_velocity",
                     "ml_range", "ap_range"):
            assert key in result, f"Missing key: {key}"

    def test_ellipse_area_non_negative(self, walking_data):
        result = postural_sway(walking_data)
        assert result["ellipse_area"] >= 0

    def test_sway_velocity_non_negative(self, walking_data):
        result = postural_sway(walking_data)
        assert result["sway_velocity"] >= 0

    def test_cop_lists_length(self, walking_data):
        result = postural_sway(walking_data)
        assert len(result["cop_x"]) == len(walking_data["frames"])
        assert len(result["cop_y"]) == len(walking_data["frames"])

    def test_with_frame_range(self, walking_data):
        result = postural_sway(walking_data, start_frame=10, end_frame=50)
        assert len(result["cop_x"]) == 40

    def test_ranges_non_negative(self, walking_data):
        result = postural_sway(walking_data)
        assert result["ml_range"] >= 0
        assert result["ap_range"] >= 0


# ══════════════════════════════════════════════════════════════════════
# EXPORT TESTS
# ══════════════════════════════════════════════════════════════════════


# ── to_dataframe ─────────────────────────────────────────────────────


class TestToDataframe:

    def test_angles_returns_dataframe(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="angles")
        assert isinstance(df, pd.DataFrame)

    def test_angles_columns(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="angles")
        expected_cols = ["frame_idx", "hip_L", "hip_R", "knee_L", "knee_R",
                         "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_landmarks_returns_dataframe(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="landmarks")
        assert isinstance(df, pd.DataFrame)

    def test_landmarks_has_coordinate_columns(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="landmarks")
        assert "frame_idx" in df.columns
        # Should have _x, _y, _vis columns for landmarks
        x_cols = [c for c in df.columns if c.endswith("_x")]
        y_cols = [c for c in df.columns if c.endswith("_y")]
        vis_cols = [c for c in df.columns if c.endswith("_vis")]
        assert len(x_cols) > 0
        assert len(y_cols) > 0
        assert len(vis_cols) > 0

    def test_events_returns_dataframe(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="events")
        assert isinstance(df, pd.DataFrame)

    def test_events_columns(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="events")
        expected_cols = ["event_type", "frame", "time", "confidence"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_returns_dict(self, pipeline_data):
        data, _, _ = pipeline_data
        result = to_dataframe(data, what="all")
        assert isinstance(result, dict)
        assert "angles" in result
        assert "landmarks" in result
        assert "events" in result
        assert isinstance(result["angles"], pd.DataFrame)
        assert isinstance(result["landmarks"], pd.DataFrame)
        assert isinstance(result["events"], pd.DataFrame)

    def test_invalid_what_raises(self, pipeline_data):
        data, _, _ = pipeline_data
        with pytest.raises(ValueError):
            to_dataframe(data, what="invalid")

    def test_angles_row_count(self, pipeline_data):
        data, _, _ = pipeline_data
        df = to_dataframe(data, what="angles")
        n_angle_frames = len(data.get("angles", {}).get("frames", []))
        assert len(df) == n_angle_frames


# ── export_summary_json ──────────────────────────────────────────────


class TestExportSummaryJson:

    def test_creates_json_file(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            result = export_summary_json(data, cycles, stats, path)
            assert os.path.exists(result)

    def test_returns_path(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            result = export_summary_json(data, cycles, stats, path)
            assert isinstance(result, str)
            assert result.endswith("summary.json")

    def test_valid_json(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            export_summary_json(data, cycles, stats, path)
            with open(path) as f:
                content = json.load(f)
            assert isinstance(content, dict)

    def test_json_has_metadata(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            export_summary_json(data, cycles, stats, path)
            with open(path) as f:
                content = json.load(f)
            assert "metadata" in content
            assert "date" in content["metadata"]

    def test_json_has_cycles(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            export_summary_json(data, cycles, stats, path)
            with open(path) as f:
                content = json.load(f)
            assert "cycles" in content
            assert "n_total" in content["cycles"]

    def test_json_has_spatiotemporal(self, pipeline_data):
        data, cycles, stats = pipeline_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            export_summary_json(data, cycles, stats, path)
            with open(path) as f:
                content = json.load(f)
            assert "spatiotemporal" in content


# ══════════════════════════════════════════════════════════════════════
# NORMALIZE TESTS
# ══════════════════════════════════════════════════════════════════════


# ── fill_gaps ────────────────────────────────────────────────────────


class TestFillGaps:

    def _make_data_with_gaps(self, gap_frames=None, gap_landmarks=None):
        """Create walking data with specific gaps."""
        data = make_walking_data(n_frames=100, fps=30.0)
        if gap_frames is None:
            gap_frames = [10, 11, 12, 13, 14]
        if gap_landmarks is None:
            gap_landmarks = ["LEFT_ANKLE", "RIGHT_ANKLE"]
        for i in gap_frames:
            if i < len(data["frames"]):
                for lm_name in gap_landmarks:
                    data["frames"][i]["landmarks"][lm_name] = {
                        "x": float("nan"),
                        "y": float("nan"),
                        "visibility": 0.0,
                    }
        return data

    def test_returns_dict(self):
        data = self._make_data_with_gaps()
        result = fill_gaps(data)
        assert isinstance(result, dict)

    def test_fills_short_gaps_spline(self):
        data = self._make_data_with_gaps(gap_frames=[10, 11, 12])
        fill_gaps(data, method="spline", max_gap_frames=10)
        # Check that the gap was filled
        for i in [10, 11, 12]:
            lm = data["frames"][i]["landmarks"]["LEFT_ANKLE"]
            assert not np.isnan(lm["x"]), f"Gap at frame {i} was not filled"
            assert not np.isnan(lm["y"]), f"Gap at frame {i} was not filled"

    def test_fills_short_gaps_linear(self):
        data = self._make_data_with_gaps(gap_frames=[20, 21, 22])
        fill_gaps(data, method="linear", max_gap_frames=10)
        for i in [20, 21, 22]:
            lm = data["frames"][i]["landmarks"]["LEFT_ANKLE"]
            assert not np.isnan(lm["x"])
            assert not np.isnan(lm["y"])

    def test_fills_with_zero(self):
        data = self._make_data_with_gaps(gap_frames=[30, 31])
        fill_gaps(data, method="zero", max_gap_frames=10)
        for i in [30, 31]:
            lm = data["frames"][i]["landmarks"]["LEFT_ANKLE"]
            assert lm["x"] == 0.0
            assert lm["y"] == 0.0

    def test_respects_max_gap_frames(self):
        # Create a gap longer than max_gap_frames
        long_gap = list(range(10, 25))  # 15 frames
        data = self._make_data_with_gaps(gap_frames=long_gap)
        fill_gaps(data, method="linear", max_gap_frames=5)
        # The gap should NOT be filled
        lm = data["frames"][15]["landmarks"]["LEFT_ANKLE"]
        assert np.isnan(lm["x"]), "Long gap should not be filled"

    def test_report_flag(self):
        data = self._make_data_with_gaps(gap_frames=[10, 11, 12])
        fill_gaps(data, method="spline", max_gap_frames=10, report=True)
        assert "gap_fill_report" in data
        report = data["gap_fill_report"]
        assert "total_gaps_filled" in report
        assert "landmarks_with_gaps" in report
        assert report["total_gaps_filled"] > 0

    def test_no_report_by_default(self):
        data = self._make_data_with_gaps()
        fill_gaps(data, method="linear")
        assert "gap_fill_report" not in data

    def test_invalid_method_raises(self):
        data = self._make_data_with_gaps()
        with pytest.raises(ValueError):
            fill_gaps(data, method="invalid")

    def test_no_gaps_does_nothing(self):
        data = make_walking_data(n_frames=50, fps=30.0)
        original_x = data["frames"][10]["landmarks"]["LEFT_ANKLE"]["x"]
        fill_gaps(data, method="linear")
        assert data["frames"][10]["landmarks"]["LEFT_ANKLE"]["x"] == original_x

    def test_report_contains_affected_landmarks(self):
        data = self._make_data_with_gaps(
            gap_frames=[5, 6, 7],
            gap_landmarks=["LEFT_KNEE"]
        )
        fill_gaps(data, method="linear", max_gap_frames=10, report=True)
        report = data["gap_fill_report"]
        assert "LEFT_KNEE" in report["landmarks_with_gaps"]
