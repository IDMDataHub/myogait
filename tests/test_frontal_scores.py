"""Tests for frontal plane variables in myogait.scores."""

import numpy as np
import pytest

from myogait.normative import get_normative_curve
from myogait.scores import (
    gait_variable_scores,
    gait_profile_score_2d,
    sagittal_deviation_index,
    movement_analysis_profile,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_sagittal_only_cycles():
    """Cycle summary with only sagittal data (no frontal)."""
    summary = {}
    for side in ("left", "right"):
        side_data = {"n_cycles": 5}
        for joint in ("hip", "knee", "ankle", "trunk"):
            norm = get_normative_curve(joint, "adult")
            side_data[f"{joint}_mean"] = list(norm["mean"])
            side_data[f"{joint}_std"] = list(norm["sd"])
        summary[side] = side_data
    return {"cycles": [], "summary": summary}


def _make_frontal_cycles():
    """Cycle summary with both sagittal and frontal data."""
    summary = {}
    for side in ("left", "right"):
        side_data = {"n_cycles": 5}
        # Sagittal joints
        for joint in ("hip", "knee", "ankle", "trunk"):
            norm = get_normative_curve(joint, "adult")
            side_data[f"{joint}_mean"] = list(norm["mean"])
            side_data[f"{joint}_std"] = list(norm["sd"])
        # Frontal joints
        norm_po = get_normative_curve("pelvis_obliquity", "adult")
        side_data["pelvis_list_mean"] = list(norm_po["mean"])
        side_data["pelvis_list_std"] = list(norm_po["sd"])

        norm_ha = get_normative_curve("hip_adduction", "adult")
        suffix = "L" if side == "left" else "R"
        side_data[f"hip_adduction_{suffix}_mean"] = list(norm_ha["mean"])
        side_data[f"hip_adduction_{suffix}_std"] = list(norm_ha["sd"])

        norm_kv = get_normative_curve("knee_valgus", "adult")
        side_data[f"knee_valgus_{suffix}_mean"] = list(norm_kv["mean"])
        side_data[f"knee_valgus_{suffix}_std"] = list(norm_kv["sd"])

        summary[side] = side_data
    return {"cycles": [], "summary": summary}


def _make_deviated_frontal_cycles():
    """Cycle summary with deviated frontal data (shifted by 10 deg)."""
    summary = {}
    for side in ("left", "right"):
        side_data = {"n_cycles": 5}
        # Sagittal joints (normative)
        for joint in ("hip", "knee", "ankle", "trunk"):
            norm = get_normative_curve(joint, "adult")
            side_data[f"{joint}_mean"] = list(norm["mean"])
            side_data[f"{joint}_std"] = list(norm["sd"])
        # Frontal joints (shifted by 10 degrees)
        norm_po = get_normative_curve("pelvis_obliquity", "adult")
        side_data["pelvis_list_mean"] = [v + 10.0 for v in norm_po["mean"]]
        side_data["pelvis_list_std"] = list(norm_po["sd"])

        norm_ha = get_normative_curve("hip_adduction", "adult")
        suffix = "L" if side == "left" else "R"
        side_data[f"hip_adduction_{suffix}_mean"] = [v + 10.0 for v in norm_ha["mean"]]
        side_data[f"hip_adduction_{suffix}_std"] = list(norm_ha["sd"])

        norm_kv = get_normative_curve("knee_valgus", "adult")
        side_data[f"knee_valgus_{suffix}_mean"] = [v + 10.0 for v in norm_kv["mean"]]
        side_data[f"knee_valgus_{suffix}_std"] = list(norm_kv["sd"])

        summary[side] = side_data
    return {"cycles": [], "summary": summary}


# ── GVS with frontal data ───────────────────────────────────────────


class TestGVSWithFrontalData:
    """Test GVS when frontal data is present."""

    def test_gvs_includes_frontal_keys(self):
        cycles = _make_frontal_cycles()
        result = gait_variable_scores(cycles)
        # Left side should have frontal keys
        assert "pelvis_list" in result["left"]
        assert "hip_adduction_L" in result["left"]
        assert "knee_valgus_L" in result["left"]
        # Right side should have its own frontal keys
        assert "pelvis_list" in result["right"]
        assert "hip_adduction_R" in result["right"]
        assert "knee_valgus_R" in result["right"]

    def test_gvs_frontal_near_zero_for_normative(self):
        """When frontal data matches normative, GVS should be ~0."""
        cycles = _make_frontal_cycles()
        result = gait_variable_scores(cycles)
        for fkey in ("pelvis_list", "hip_adduction_L", "knee_valgus_L"):
            val = result["left"].get(fkey)
            if val is not None:
                assert val < 0.5, f"GVS left/{fkey} = {val}, expected ~0"
        for fkey in ("pelvis_list", "hip_adduction_R", "knee_valgus_R"):
            val = result["right"].get(fkey)
            if val is not None:
                assert val < 0.5, f"GVS right/{fkey} = {val}, expected ~0"

    def test_gvs_frontal_positive_for_deviated(self):
        """Deviated frontal data should produce non-zero GVS."""
        cycles = _make_deviated_frontal_cycles()
        result = gait_variable_scores(cycles)
        for fkey in ("pelvis_list", "hip_adduction_L", "knee_valgus_L"):
            val = result["left"].get(fkey)
            if val is not None:
                assert val > 5.0, f"GVS left/{fkey} = {val}, expected >>0 for deviated data"


# ── GVS backward compatibility (no frontal) ─────────────────────────


class TestGVSBackwardCompat:
    """GVS without frontal data should work identically to before."""

    def test_gvs_no_frontal_keys(self):
        cycles = _make_sagittal_only_cycles()
        result = gait_variable_scores(cycles)
        # Should have only sagittal joints
        for side in ("left", "right"):
            assert set(result[side].keys()) == {"hip", "knee", "ankle", "trunk"}

    def test_gvs_values_near_zero(self):
        cycles = _make_sagittal_only_cycles()
        result = gait_variable_scores(cycles)
        for side in ("left", "right"):
            for joint, val in result[side].items():
                if val is not None:
                    assert val < 0.5, f"GVS {side}/{joint} = {val}, expected ~0"


# ── GPS-2D with include_frontal ──────────────────────────────────────


class TestGPS2DWithFrontal:
    """Test GPS-2D with frontal inclusion."""

    def test_gps_include_frontal_true_has_more_variables(self):
        cycles = _make_frontal_cycles()
        result = gait_profile_score_2d(cycles, include_frontal=True)
        assert len(result["variables_used"]) > 4

    def test_gps_include_frontal_false_has_4_variables(self):
        cycles = _make_frontal_cycles()
        result = gait_profile_score_2d(cycles, include_frontal=False)
        assert len(result["variables_used"]) == 4

    def test_gps_frontal_false_same_as_sagittal_only(self):
        """GPS with include_frontal=False should match sagittal-only."""
        sag_cycles = _make_sagittal_only_cycles()
        front_cycles = _make_frontal_cycles()
        sag_result = gait_profile_score_2d(sag_cycles)
        front_result = gait_profile_score_2d(front_cycles, include_frontal=False)
        assert sag_result["gps_2d_overall"] == front_result["gps_2d_overall"]

    def test_gps_values_not_nan(self):
        cycles = _make_frontal_cycles()
        result = gait_profile_score_2d(cycles, include_frontal=True)
        for key in ("gps_2d_left", "gps_2d_right", "gps_2d_overall"):
            val = result[key]
            if val is not None:
                assert not np.isnan(val), f"{key} is NaN"
                assert val >= 0, f"{key} = {val} is negative"

    def test_gps_note_mentions_frontal_when_included(self):
        cycles = _make_frontal_cycles()
        result = gait_profile_score_2d(cycles, include_frontal=True)
        assert "frontal" in result["note"].lower()

    def test_gps_note_sagittal_when_excluded(self):
        cycles = _make_frontal_cycles()
        result = gait_profile_score_2d(cycles, include_frontal=False)
        assert "sagittal" in result["note"].lower()


# ── SDI with frontal data ───────────────────────────────────────────


class TestSDIWithFrontal:
    """Test SDI with frontal data."""

    def test_sdi_with_frontal_returns_dict(self):
        cycles = _make_frontal_cycles()
        result = sagittal_deviation_index(cycles, include_frontal=True)
        assert isinstance(result, dict)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall", "note"):
            assert key in result

    def test_sdi_normative_near_100(self):
        """Normative data with frontal should score near 100."""
        cycles = _make_frontal_cycles()
        result = sagittal_deviation_index(cycles, include_frontal=True)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert 80 <= val <= 120, f"{key} = {val}, expected ~100"

    def test_sdi_not_nan(self):
        cycles = _make_frontal_cycles()
        result = sagittal_deviation_index(cycles, include_frontal=True)
        for key in ("gdi_2d_left", "gdi_2d_right", "gdi_2d_overall"):
            val = result[key]
            if val is not None:
                assert not np.isnan(val), f"{key} is NaN"


# ── MAP with frontal data ───────────────────────────────────────────


class TestMAPWithFrontal:
    """Test MAP includes frontal variables when present."""

    def test_map_joints_include_frontal(self):
        cycles = _make_frontal_cycles()
        result = movement_analysis_profile(cycles)
        joints = result["joints"]
        assert "pelvis_list" in joints
        # Check for hip_adduction_L or hip_adduction_R
        has_hip_add = any("hip_adduction" in j for j in joints)
        assert has_hip_add, "MAP should include hip_adduction when present"

    def test_map_sagittal_only_has_4_joints(self):
        cycles = _make_sagittal_only_cycles()
        result = movement_analysis_profile(cycles)
        assert len(result["joints"]) == 4

    def test_map_left_right_same_length_as_joints(self):
        cycles = _make_frontal_cycles()
        result = movement_analysis_profile(cycles)
        assert len(result["left"]) == len(result["joints"])
        assert len(result["right"]) == len(result["joints"])

    def test_map_gps_not_nan(self):
        cycles = _make_frontal_cycles()
        result = movement_analysis_profile(cycles)
        val = result["gps_2d"]
        if val is not None:
            assert not np.isnan(val)
            assert val >= 0
