"""Tests for myogait.scores -- clinical gait deviation scores."""

import warnings

import pytest

from conftest import run_full_pipeline

from myogait.scores import (
    gait_variable_scores,
    gait_profile_score_2d,
    gait_deviation_index_2d,
    sagittal_deviation_index,
    movement_analysis_profile,
)
from myogait.normative import get_normative_curve


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def pipeline_cycles():
    """Run full pipeline and return cycles dict with summary."""
    _data, cycles, _stats = run_full_pipeline()
    return cycles


@pytest.fixture
def normative_cycles():
    """Create a synthetic cycles dict whose mean curves equal the normative adult data.

    This simulates a patient with perfectly normal gait, so GVS should be ~0.
    """
    summary = {}
    for side in ("left", "right"):
        side_data = {"n_cycles": 5}
        for joint in ("hip", "knee", "ankle", "trunk"):
            norm = get_normative_curve(joint, "adult")
            side_data[f"{joint}_mean"] = list(norm["mean"])
            side_data[f"{joint}_std"] = list(norm["sd"])
        summary[side] = side_data
    return {"cycles": [], "summary": summary}


@pytest.fixture
def pathological_cycles():
    """Synthetic cycles with large deviations from normative data.

    Add 30 degrees to every joint curve (clearly abnormal).
    """
    summary = {}
    for side in ("left", "right"):
        side_data = {"n_cycles": 5}
        for joint in ("hip", "knee", "ankle", "trunk"):
            norm = get_normative_curve(joint, "adult")
            shifted = [v + 30.0 for v in norm["mean"]]
            side_data[f"{joint}_mean"] = shifted
            side_data[f"{joint}_std"] = list(norm["sd"])
        summary[side] = side_data
    return {"cycles": [], "summary": summary}


# ── GVS tests ────────────────────────────────────────────────────────

class TestGVSStructure:
    """Test gait_variable_scores return structure."""

    def test_has_left_and_right(self, pipeline_cycles):
        result = gait_variable_scores(pipeline_cycles)
        assert "left" in result
        assert "right" in result

    def test_each_side_has_joints(self, pipeline_cycles):
        result = gait_variable_scores(pipeline_cycles)
        for side in ("left", "right"):
            for joint in ("hip", "knee", "ankle", "trunk"):
                assert joint in result[side]


class TestGVSValuesPositive:
    """Test that GVS values are non-negative."""

    def test_all_positive(self, pipeline_cycles):
        result = gait_variable_scores(pipeline_cycles)
        for side in ("left", "right"):
            for joint, val in result[side].items():
                if val is not None:
                    assert val >= 0, f"GVS {side}/{joint} = {val} is negative"


class TestGVSNormativeNearZero:
    """When patient data equals normative data, GVS should be ~0."""

    def test_gvs_near_zero(self, normative_cycles):
        result = gait_variable_scores(normative_cycles)
        for side in ("left", "right"):
            for joint, val in result[side].items():
                if val is not None:
                    assert val < 0.5, (
                        f"GVS {side}/{joint} = {val}, expected ~0 "
                        "for normative data"
                    )


# ── GPS-2D tests ─────────────────────────────────────────────────────

class TestGPS2DStructure:
    """Test gait_profile_score_2d return structure."""

    def test_has_required_keys(self, pipeline_cycles):
        result = gait_profile_score_2d(pipeline_cycles)
        for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall",
                     "variables_used", "note"):
            assert key in result, f"Missing key: {key}"

    def test_variables_used_list(self, pipeline_cycles):
        result = gait_profile_score_2d(pipeline_cycles)
        assert isinstance(result["variables_used"], list)
        assert len(result["variables_used"]) == 4


class TestGPS2DValueRange:
    """Test that GPS-2D values are in plausible range."""

    def test_values_positive(self, pipeline_cycles):
        result = gait_profile_score_2d(pipeline_cycles)
        for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
            val = result[key]
            if val is not None:
                assert val >= 0, f"{key} = {val} is negative"

    def test_values_not_extreme(self, pipeline_cycles):
        """GPS-2D should be <90 degrees for any realistic input."""
        result = gait_profile_score_2d(pipeline_cycles)
        for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
            val = result[key]
            if val is not None:
                assert val < 90, f"{key} = {val} is unreasonably large"


class TestGPS2DNormativeNearZero:
    """When patient = normative, GPS-2D should be ~0."""

    def test_gps_near_zero(self, normative_cycles):
        result = gait_profile_score_2d(normative_cycles)
        for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
            val = result[key]
            if val is not None:
                assert val < 0.5, f"{key} = {val}, expected ~0"


class TestGPS2DNoteMentions2D:
    """GPS-2D note should mention '2D'."""

    def test_note_contains_2d(self, pipeline_cycles):
        result = gait_profile_score_2d(pipeline_cycles)
        assert "2D" in result["note"] or "2d" in result["note"].lower()

    def test_note_mentions_screening(self, pipeline_cycles):
        result = gait_profile_score_2d(pipeline_cycles)
        assert "screening" in result["note"].lower()


# ── GDI-2D tests ─────────────────────────────────────────────────────

class TestGDI2DStructure:
    """Test sagittal_deviation_index return structure."""

    def test_has_required_keys(self, pipeline_cycles):
        result = sagittal_deviation_index(pipeline_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall", "note"):
            assert key in result, f"Missing key: {key}"


class TestGDI2DNormalAround100:
    """Normal gait should produce GDI-2D around 100."""

    def test_normative_gdi_near_100(self, normative_cycles):
        result = sagittal_deviation_index(normative_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                # Normative data should score near or above 100
                # (synthetic data may score slightly above 100 since it
                # can deviate less than a real population average)
                assert 80 <= val <= 130, (
                    f"{key} = {val}, expected ~100 for normative data"
                )


class TestGDI2DPathologicalBelow100:
    """Pathological gait (large deviations) should score below 100."""

    def test_pathological_gdi_below_100(self, pathological_cycles):
        result = sagittal_deviation_index(pathological_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert val < 100, (
                    f"{key} = {val}, expected <100 for pathological data"
                )


class TestGDI2DNoteMentions2D:
    """GDI-2D note should mention '2D'."""

    def test_note_contains_2d(self, pipeline_cycles):
        result = sagittal_deviation_index(pipeline_cycles)
        assert "2D" in result["note"] or "2d" in result["note"].lower()


# ── MAP tests ────────────────────────────────────────────────────────

class TestMAPStructure:
    """Test movement_analysis_profile return structure."""

    def test_has_required_keys(self, pipeline_cycles):
        result = movement_analysis_profile(pipeline_cycles)
        for key in ("joints", "left", "right", "gps_2d"):
            assert key in result, f"Missing key: {key}"


class TestMAPJointsList:
    """Test MAP joints list."""

    def test_joints_sorted(self, pipeline_cycles):
        result = movement_analysis_profile(pipeline_cycles)
        assert result["joints"] == sorted(result["joints"])

    def test_joints_count(self, pipeline_cycles):
        result = movement_analysis_profile(pipeline_cycles)
        assert len(result["joints"]) == 4

    def test_left_right_same_length(self, pipeline_cycles):
        result = movement_analysis_profile(pipeline_cycles)
        assert len(result["left"]) == len(result["joints"])
        assert len(result["right"]) == len(result["joints"])


# ── Integration / real pipeline tests ────────────────────────────────

class TestScoresWithRealPipeline:
    """Test scores with data from the full processing pipeline."""

    def test_full_pipeline_gvs(self):
        _data, cycles, _stats = run_full_pipeline()
        result = gait_variable_scores(cycles)
        assert "left" in result or "right" in result

    def test_full_pipeline_gps(self):
        _data, cycles, _stats = run_full_pipeline()
        result = gait_profile_score_2d(cycles)
        assert result["gps_2d_overall"] is not None or (
            result["gps_2d_left"] is None and result["gps_2d_right"] is None
        )

    def test_full_pipeline_gdi(self):
        _data, cycles, _stats = run_full_pipeline()
        result = sagittal_deviation_index(cycles)
        # Overall should be computed if any side has data
        assert "gdi_2d_overall" in result

    def test_full_pipeline_map(self):
        _data, cycles, _stats = run_full_pipeline()
        result = movement_analysis_profile(cycles)
        assert "joints" in result
        assert isinstance(result["gps_2d"], (float, int, type(None)))


# ── Error handling ───────────────────────────────────────────────────

class TestScoresMissingSummary:
    """Test error handling when cycles dict lacks summary."""

    def test_gvs_no_summary_raises(self):
        with pytest.raises(ValueError, match="summary"):
            gait_variable_scores({"cycles": []})

    def test_gps_no_summary_raises(self):
        with pytest.raises(ValueError, match="summary"):
            gait_profile_score_2d({"cycles": []})

    def test_gdi_no_summary_raises(self):
        with pytest.raises(ValueError, match="summary"):
            sagittal_deviation_index({"cycles": []})

    def test_map_no_summary_raises(self):
        with pytest.raises(ValueError, match="summary"):
            movement_analysis_profile({"cycles": []})

    def test_gvs_not_dict_raises(self):
        with pytest.raises(TypeError):
            gait_variable_scores("not a dict")


# ── sagittal_deviation_index (E2 fix) ───────────────────────────────


class TestSagittalDeviationIndexExists:
    """sagittal_deviation_index should exist and work."""

    def test_function_exists(self):
        """sagittal_deviation_index should be importable."""
        assert callable(sagittal_deviation_index)

    def test_returns_dict(self, pipeline_cycles):
        result = sagittal_deviation_index(pipeline_cycles)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pipeline_cycles):
        result = sagittal_deviation_index(pipeline_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall", "note"):
            assert key in result, f"Missing key: {key}"

    def test_note_mentions_sagittal(self, pipeline_cycles):
        """Note should mention Sagittal Deviation Index, not claim to be GDI."""
        result = sagittal_deviation_index(pipeline_cycles)
        assert "Sagittal Deviation Index" in result["note"]
        assert "NOT" in result["note"]


class TestGDI2DIsDeprecatedAlias:
    """gait_deviation_index_2d should still work but emit a deprecation warning."""

    def test_alias_returns_same_result(self, normative_cycles):
        """Alias should return the same result as the new name."""
        direct = sagittal_deviation_index(normative_cycles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            alias = gait_deviation_index_2d(normative_cycles)
        assert direct == alias

    def test_alias_emits_deprecation_warning(self, normative_cycles):
        """Calling the alias should raise DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gait_deviation_index_2d(normative_cycles)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1


class TestSDIClamped:
    """SDI values should be clamped between 0 and 120."""

    def test_normative_clamped_upper(self, normative_cycles):
        """Normative data: SDI should not exceed 120."""
        result = sagittal_deviation_index(normative_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert val <= 120.0, f"{key} = {val} exceeds upper clamp"

    def test_pathological_clamped_lower(self, pathological_cycles):
        """Pathological data: SDI should not go below 0."""
        result = sagittal_deviation_index(pathological_cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert val >= 0.0, f"{key} = {val} below lower clamp"

    def test_extreme_deviation_clamped_at_zero(self):
        """With extreme deviation (100 deg offset), SDI should be clamped at 0."""
        summary = {}
        for side in ("left", "right"):
            side_data = {"n_cycles": 5}
            for joint in ("hip", "knee", "ankle", "trunk"):
                norm = get_normative_curve(joint, "adult")
                shifted = [v + 100.0 for v in norm["mean"]]
                side_data[f"{joint}_mean"] = shifted
                side_data[f"{joint}_std"] = list(norm["sd"])
            summary[side] = side_data
        cycles = {"cycles": [], "summary": summary}
        result = sagittal_deviation_index(cycles)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert val >= 0.0, f"{key} = {val} below zero"
                assert val <= 120.0, f"{key} = {val} exceeds 120"
