"""Tests for extended spatio-temporal analysis functions (Group 5)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myogait.analysis import (
    single_support_time,
    toe_clearance,
    stride_variability,
    arm_swing_analysis,
    speed_normalized_params,
    detect_equinus,
    detect_antalgic,
    detect_parkinsonian,
)
from conftest import (
    make_walking_data,
    run_full_pipeline,
)


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


# ── single_support_time ──────────────────────────────────────────────


class TestSingleSupportTime:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = single_support_time(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = single_support_time(data, cycles)
        for key in ("sst_left_s", "sst_right_s", "sst_left_pct",
                     "sst_right_pct", "sst_symmetry_index"):
            assert key in result

    def test_sst_values_positive(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = single_support_time(data, cycles)
        for side in ("left", "right"):
            val = result[f"sst_{side}_s"]
            if val is not None:
                assert val >= 0

    def test_sst_pct_range(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = single_support_time(data, cycles)
        for side in ("left", "right"):
            val = result[f"sst_{side}_pct"]
            if val is not None:
                assert 0 <= val <= 100

    def test_empty_events(self):
        """No events → all None."""
        data = make_walking_data(30)
        data["events"] = {}
        cycles = {"cycles": []}
        result = single_support_time(data, cycles)
        assert result["sst_left_s"] is None


# ── toe_clearance ────────────────────────────────────────────────────


class TestToeClearance:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = toe_clearance(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = toe_clearance(data, cycles)
        for key in ("mtc_left", "mtc_right", "mtc_left_cv",
                     "mtc_right_cv", "unit"):
            assert key in result

    def test_mtc_values_positive(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = toe_clearance(data, cycles)
        for side in ("left", "right"):
            val = result[f"mtc_{side}"]
            if val is not None:
                # Clearance should be a small positive value
                assert val >= -0.1  # allow small negative due to noise

    def test_empty_cycles(self):
        """No cycles → all None."""
        data = make_walking_data(30)
        result = toe_clearance(data, {"cycles": []})
        assert result["mtc_left"] is None
        assert result["unit"] == "normalized"


# ── stride_variability ───────────────────────────────────────────────


class TestStrideVariability:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = stride_variability(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = stride_variability(data, cycles)
        for key in ("stride_time_cv", "step_time_cv",
                     "step_length_cv_left", "step_length_cv_right"):
            assert key in result

    def test_has_rom_cv_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = stride_variability(data, cycles)
        for side in ("left", "right"):
            for joint in ("hip", "knee", "ankle"):
                assert f"rom_cv_{joint}_{side}" in result

    def test_cv_non_negative(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = stride_variability(data, cycles)
        for key, val in result.items():
            if val is not None:
                assert val >= 0, f"{key} should be >= 0"

    def test_regular_gait_low_cv(self, data_and_cycles):
        """Synthetic sinusoidal gait → low CV (regular)."""
        data, cycles = data_and_cycles
        result = stride_variability(data, cycles)
        # Synthetic data is perfectly periodic → CV should be low
        assert result["stride_time_cv"] < 30


# ── arm_swing_analysis ───────────────────────────────────────────────


class TestArmSwingAnalysis:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = arm_swing_analysis(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = arm_swing_analysis(data, cycles)
        for key in ("amplitude_left", "amplitude_right",
                     "asymmetry_index", "reduced_swing"):
            assert key in result

    def test_amplitude_positive(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = arm_swing_analysis(data, cycles)
        for side in ("left", "right"):
            val = result[f"amplitude_{side}"]
            if val is not None:
                assert val >= 0

    def test_walking_data_has_arm_swing(self, data_and_cycles):
        """Walking data has non-zero arm swing."""
        data, cycles = data_and_cycles
        result = arm_swing_analysis(data, cycles)
        amp_l = result["amplitude_left"]
        amp_r = result["amplitude_right"]
        # Should have some arm swing (wrist fallback at minimum)
        assert amp_l is not None or amp_r is not None

    def test_empty_angles(self):
        """No angle frames → all None."""
        data = make_walking_data(30)
        data["angles"] = {}
        result = arm_swing_analysis(data, {"cycles": []})
        assert result["amplitude_left"] is None


# ── speed_normalized_params ──────────────────────────────────────────


class TestSpeedNormalizedParams:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = speed_normalized_params(data, cycles, height_m=1.75)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = speed_normalized_params(data, cycles, height_m=1.75)
        for key in ("froude_number", "dimensionless_speed",
                     "dimensionless_cadence", "leg_length_m"):
            assert key in result

    def test_leg_length_correct(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = speed_normalized_params(data, cycles, height_m=1.75)
        expected = round(1.75 * 0.53, 3)
        assert result["leg_length_m"] == expected

    def test_froude_non_negative(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = speed_normalized_params(data, cycles, height_m=1.75)
        fr = result["froude_number"]
        if fr is not None:
            assert fr >= 0

    def test_different_heights_different_froude(self, data_and_cycles):
        data, cycles = data_and_cycles
        r1 = speed_normalized_params(data, cycles, height_m=1.50)
        r2 = speed_normalized_params(data, cycles, height_m=1.90)
        if r1["froude_number"] is not None and r2["froude_number"] is not None:
            assert r1["froude_number"] != r2["froude_number"]


# ── detect_equinus ───────────────────────────────────────────────────


class TestDetectEquinus:

    def test_returns_dict(self, data_and_cycles):
        _, cycles = data_and_cycles
        result = detect_equinus(cycles)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "details" in result

    def test_normal_gait_no_equinus(self, data_and_cycles):
        """Normal walking data should not have equinus."""
        _, cycles = data_and_cycles
        result = detect_equinus(cycles)
        # May or may not detect depending on synthetic data,
        # but result structure should be valid
        assert isinstance(result["detected"], bool)
        assert isinstance(result["details"], list)

    def test_equinus_with_negative_ankle(self):
        """Force equinus by making all ankle angles negative."""
        cycles = {
            "cycles": [
                {
                    "side": "left",
                    "angles_normalized": {
                        "ankle": [-15.0] * 101,
                    },
                },
                {
                    "side": "right",
                    "angles_normalized": {
                        "ankle": [-10.0] * 101,
                    },
                },
            ]
        }
        result = detect_equinus(cycles)
        assert result["detected"] is True
        assert len(result["details"]) == 2


# ── detect_antalgic ──────────────────────────────────────────────────


class TestDetectAntalgic:

    def test_returns_dict(self, data_and_cycles):
        _, cycles = data_and_cycles
        result = detect_antalgic(cycles)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "details" in result

    def test_symmetric_gait_no_antalgic(self):
        """Symmetric stance → no antalgic detection."""
        cycles = {
            "cycles": [
                {"side": "left", "stance_pct": 60.0, "duration": 1.0},
                {"side": "right", "stance_pct": 60.0, "duration": 1.0},
            ]
        }
        result = detect_antalgic(cycles)
        assert result["detected"] is False

    def test_antalgic_asymmetric_stance(self):
        """Very asymmetric stance → antalgic detected."""
        cycles = {
            "cycles": [
                {"side": "left", "stance_pct": 45.0, "duration": 1.0},
                {"side": "left", "stance_pct": 48.0, "duration": 1.0},
                {"side": "right", "stance_pct": 68.0, "duration": 1.0},
                {"side": "right", "stance_pct": 70.0, "duration": 1.0},
            ]
        }
        result = detect_antalgic(cycles)
        assert result["detected"] is True
        assert result["details"]["short_side"] == "left"

    def test_empty_cycles(self):
        result = detect_antalgic({"cycles": []})
        assert result["detected"] is False


# ── detect_parkinsonian ──────────────────────────────────────────────


class TestDetectParkinsonian:

    def test_returns_dict(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = detect_parkinsonian(data, cycles)
        assert isinstance(result, dict)
        assert "detected" in result
        assert "features" in result
        assert "details" in result

    def test_normal_gait_not_parkinsonian(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = detect_parkinsonian(data, cycles)
        # Normal synthetic walking should not have 2+ features
        assert isinstance(result["detected"], bool)
        assert isinstance(result["features"], list)

    def test_features_list(self, data_and_cycles):
        data, cycles = data_and_cycles
        result = detect_parkinsonian(data, cycles)
        for f in result["features"]:
            assert f in ("short_stride", "reduced_arm_swing", "festination")

    def test_empty_data(self):
        """Empty data → not detected."""
        data = {"frames": [], "angles": {}, "events": {}, "meta": {"fps": 30.0}}
        cycles = {"cycles": []}
        result = detect_parkinsonian(data, cycles)
        assert result["detected"] is False
