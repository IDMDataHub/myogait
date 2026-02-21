"""Tests for extended validation functions in myogait.validation.

Covers stratified_ranges, model_accuracy_info, and
validate_biomechanical_stratified.
"""

import pytest

from myogait.validation import (
    ANGLE_RANGES,
    SPATIOTEMPORAL_RANGES,
    stratified_ranges,
    model_accuracy_info,
    validate_biomechanical,
    validate_biomechanical_stratified,
)
from conftest import run_full_pipeline


# ── stratified_ranges ────────────────────────────────────────────────


class TestStratifiedRangesDefault:
    """No arguments should return the standard (adult) ranges."""

    def test_returns_dict_with_expected_keys(self):
        result = stratified_ranges()
        assert "angle_ranges" in result
        assert "spatiotemporal_ranges" in result

    def test_default_angle_ranges_match_module_constants(self):
        result = stratified_ranges()
        for joint, ranges in ANGLE_RANGES.items():
            assert result["angle_ranges"][joint]["full"] == ranges["full"]

    def test_default_spatiotemporal_match_module_constants(self):
        result = stratified_ranges()
        for key, val in SPATIOTEMPORAL_RANGES.items():
            assert result["spatiotemporal_ranges"][key] == val


class TestStratifiedRangesElderly:
    """age >= 65 should widen ROM ranges and lower cadence."""

    def test_hip_full_range_wider(self):
        default = stratified_ranges()
        elderly = stratified_ranges(age=70)
        # Elderly hip lower bound should be lower (more negative)
        assert elderly["angle_ranges"]["hip_L"]["full"][0] < default["angle_ranges"]["hip_L"]["full"][0]

    def test_cadence_lower_bound_reduced(self):
        default = stratified_ranges()
        elderly = stratified_ranges(age=70)
        assert elderly["spatiotemporal_ranges"]["cadence_steps_per_min"][0] < \
               default["spatiotemporal_ranges"]["cadence_steps_per_min"][0]

    def test_stride_time_upper_extended(self):
        default = stratified_ranges()
        elderly = stratified_ranges(age=70)
        assert elderly["spatiotemporal_ranges"]["stride_time_mean_s"][1] > \
               default["spatiotemporal_ranges"]["stride_time_mean_s"][1]

    def test_ankle_range_wider(self):
        default = stratified_ranges()
        elderly = stratified_ranges(age=70)
        default_width = (default["angle_ranges"]["ankle_R"]["full"][1]
                         - default["angle_ranges"]["ankle_R"]["full"][0])
        elderly_width = (elderly["angle_ranges"]["ankle_R"]["full"][1]
                         - elderly["angle_ranges"]["ankle_R"]["full"][0])
        assert elderly_width > default_width


class TestStratifiedRangesPediatric:
    """age <= 17 should return pediatric ranges."""

    def test_hip_range_wider_than_adult(self):
        default = stratified_ranges()
        pediatric = stratified_ranges(age=10)
        default_width = (default["angle_ranges"]["hip_R"]["full"][1]
                         - default["angle_ranges"]["hip_R"]["full"][0])
        pediatric_width = (pediatric["angle_ranges"]["hip_R"]["full"][1]
                           - pediatric["angle_ranges"]["hip_R"]["full"][0])
        assert pediatric_width > default_width

    def test_cadence_upper_bound_higher(self):
        default = stratified_ranges()
        pediatric = stratified_ranges(age=10)
        assert pediatric["spatiotemporal_ranges"]["cadence_steps_per_min"][1] > \
               default["spatiotemporal_ranges"]["cadence_steps_per_min"][1]

    def test_knee_full_range_wider(self):
        default = stratified_ranges()
        pediatric = stratified_ranges(age=10)
        default_hi = default["angle_ranges"]["knee_L"]["full"][1]
        pediatric_hi = pediatric["angle_ranges"]["knee_L"]["full"][1]
        assert pediatric_hi > default_hi


# ── model_accuracy_info ──────────────────────────────────────────────


class TestModelAccuracyMediapipe:
    """mediapipe should return correct dict structure."""

    def test_returns_dict(self):
        result = model_accuracy_info("mediapipe")
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        result = model_accuracy_info("mediapipe")
        for key in ("model", "mae_px", "pck_05", "reference", "notes"):
            assert key in result, f"Missing key: {key}"

    def test_model_name_correct(self):
        result = model_accuracy_info("mediapipe")
        assert result["model"] == "mediapipe"

    def test_mae_px_value(self):
        result = model_accuracy_info("mediapipe")
        assert result["mae_px"] == pytest.approx(5.0)

    def test_pck_05_value(self):
        result = model_accuracy_info("mediapipe")
        assert result["pck_05"] == pytest.approx(0.92)


class TestModelAccuracyUnknown:
    """Unknown model should return dict with None values."""

    def test_returns_dict(self):
        result = model_accuracy_info("nonexistent_model")
        assert isinstance(result, dict)

    def test_model_name_preserved(self):
        result = model_accuracy_info("nonexistent_model")
        assert result["model"] == "nonexistent_model"

    def test_mae_px_is_none(self):
        result = model_accuracy_info("nonexistent_model")
        assert result["mae_px"] is None

    def test_pck_05_is_none(self):
        result = model_accuracy_info("nonexistent_model")
        assert result["pck_05"] is None

    def test_reference_is_none(self):
        result = model_accuracy_info("nonexistent_model")
        assert result["reference"] is None

    def test_notes_is_none(self):
        result = model_accuracy_info("nonexistent_model")
        assert result["notes"] is None


class TestModelAccuracyAllModels:
    """All 7 supported models should return valid dicts."""

    MODELS = ["mediapipe", "yolo", "vitpose", "sapiens", "rtmw", "mmpose", "hrnet"]

    @pytest.mark.parametrize("model_name", MODELS)
    def test_returns_valid_dict(self, model_name):
        result = model_accuracy_info(model_name)
        assert isinstance(result, dict)
        assert result["model"] == model_name
        assert isinstance(result["mae_px"], (int, float))
        assert isinstance(result["pck_05"], float)
        assert isinstance(result["reference"], str)
        assert isinstance(result["notes"], str)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_pck_between_0_and_1(self, model_name):
        result = model_accuracy_info(model_name)
        assert 0.0 < result["pck_05"] <= 1.0

    @pytest.mark.parametrize("model_name", MODELS)
    def test_mae_positive(self, model_name):
        result = model_accuracy_info(model_name)
        assert result["mae_px"] > 0


# ── validate_biomechanical_stratified ────────────────────────────────


class TestValidateStratifiedReturnsDict:
    """Basic integration test: function runs and returns expected structure."""

    def test_returns_dict(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles)
        for key in ("valid", "violations", "summary", "stratum", "adjusted_ranges"):
            assert key in result, f"Missing key: {key}"

    def test_valid_is_bool(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles)
        assert isinstance(result["valid"], bool)

    def test_default_stratum_is_adult(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles)
        assert result["stratum"] == "adult"

    def test_summary_counts_consistent(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles)
        s = result["summary"]
        assert s["total"] == s["critical"] + s["warning"] + s["info"]


class TestValidateStratifiedElderly:
    """Elderly ranges may produce different results than adult defaults."""

    def test_stratum_is_elderly(self):
        data, cycles, _stats = run_full_pipeline()
        result = validate_biomechanical_stratified(data, cycles, age=70)
        assert result["stratum"] == "elderly"

    def test_adjusted_ranges_differ_from_default(self):
        data, cycles, _stats = run_full_pipeline()
        default_result = validate_biomechanical_stratified(data, cycles)
        elderly_result = validate_biomechanical_stratified(data, cycles, age=70)
        # The adjusted_ranges should differ between default and elderly
        default_cad = default_result["adjusted_ranges"]["spatiotemporal_ranges"]["cadence_steps_per_min"]
        elderly_cad = elderly_result["adjusted_ranges"]["spatiotemporal_ranges"]["cadence_steps_per_min"]
        assert default_cad != elderly_cad

    def test_elderly_may_have_fewer_violations(self):
        """Wider ranges may reduce the number of violations (or at least not increase them)."""
        data, cycles, _stats = run_full_pipeline()
        default_result = validate_biomechanical_stratified(data, cycles)
        elderly_result = validate_biomechanical_stratified(data, cycles, age=70)
        # Elderly should have equal or fewer violations due to wider ranges
        assert elderly_result["summary"]["total"] <= default_result["summary"]["total"]

    def test_raises_on_non_dict_data(self):
        with pytest.raises(TypeError, match="data must be a dict"):
            validate_biomechanical_stratified("not a dict", None, age=70)


class TestValidateBiomechanicalContext:
    """Context-aware behavior in validate_biomechanical."""

    def test_auto_stratified_mode_when_subject_metadata_present(self):
        data, cycles, _stats = run_full_pipeline()
        data["subject"] = {"age": 70}
        report = validate_biomechanical(data, cycles)
        assert report["mode"] == "stratified_auto"
        assert report["stratum"] == "elderly"

    def test_context_caveats_are_included(self):
        data, cycles, _stats = run_full_pipeline()
        data["extraction"] = {"treadmill": True}
        data["subject"] = {"height_m": None}

        report = validate_biomechanical(data, cycles)
        context_params = {
            v.get("parameter")
            for v in report["violations"]
            if v.get("type") == "context"
        }
        assert "treadmill_context" in context_params
        assert "calibration" in context_params
        assert "pathology_screening" in context_params
