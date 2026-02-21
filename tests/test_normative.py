"""Tests for myogait.normative -- normative gait kinematic database."""

import numpy as np
import pytest

from myogait.normative import (
    STRATA,
    get_normative_curve,
    get_normative_band,
    select_stratum,
    list_joints,
    list_strata,
)


class TestGetNormativeCurveAdultHip:
    """Test normative hip curve for adult stratum."""

    def test_returns_dict(self):
        result = get_normative_curve("hip", "adult")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_normative_curve("hip", "adult")
        for key in ("mean", "sd", "unit", "source", "stratum"):
            assert key in result, f"Missing key: {key}"

    def test_mean_101_points(self):
        result = get_normative_curve("hip", "adult")
        assert len(result["mean"]) == 101

    def test_sd_101_points(self):
        result = get_normative_curve("hip", "adult")
        assert len(result["sd"]) == 101

    def test_unit_is_deg(self):
        result = get_normative_curve("hip", "adult")
        assert result["unit"] == "deg"

    def test_stratum_is_adult(self):
        result = get_normative_curve("hip", "adult")
        assert result["stratum"] == "adult"

    def test_hip_rom_reasonable(self):
        """Adult hip ROM should be ~40-45 degrees."""
        result = get_normative_curve("hip", "adult")
        mean = np.array(result["mean"])
        rom = np.max(mean) - np.min(mean)
        assert 30 <= rom <= 55, f"Hip ROM {rom:.1f} out of expected range"

    def test_hip_peak_flexion(self):
        """Peak hip flexion should be ~30-35 degrees."""
        result = get_normative_curve("hip", "adult")
        mean = np.array(result["mean"])
        assert 25 <= np.max(mean) <= 40, f"Peak flexion {np.max(mean):.1f} out of range"

    def test_hip_peak_extension(self):
        """Peak hip extension should be near -10 degrees."""
        result = get_normative_curve("hip", "adult")
        mean = np.array(result["mean"])
        assert -20 <= np.min(mean) <= 0, f"Peak extension {np.min(mean):.1f} out of range"

    def test_hip_sd_positive(self):
        result = get_normative_curve("hip", "adult")
        sd = np.array(result["sd"])
        assert np.all(sd > 0)


class TestGetNormativeCurveAllJoints:
    """Test normative curves for all joints."""

    @pytest.mark.parametrize("joint", ["hip", "knee", "ankle", "trunk", "pelvis_sagittal"])
    def test_curve_exists(self, joint):
        result = get_normative_curve(joint, "adult")
        assert isinstance(result, dict)
        assert len(result["mean"]) == 101
        assert len(result["sd"]) == 101

    def test_knee_peak_swing_flexion(self):
        """Knee swing flexion peak should be ~55-65 degrees."""
        result = get_normative_curve("knee", "adult")
        mean = np.array(result["mean"])
        peak = np.max(mean)
        assert 45 <= peak <= 70, f"Knee peak flexion {peak:.1f} out of range"

    def test_ankle_peak_plantarflexion(self):
        """Ankle push-off plantarflexion should be ~-10 to -20 degrees."""
        result = get_normative_curve("ankle", "adult")
        mean = np.array(result["mean"])
        pf = np.min(mean)
        assert -25 <= pf <= -5, f"Ankle plantarflexion {pf:.1f} out of range"

    def test_trunk_mean_forward_lean(self):
        """Trunk should average ~5 deg forward lean."""
        result = get_normative_curve("trunk", "adult")
        mean = np.array(result["mean"])
        avg = np.mean(mean)
        assert 2 <= avg <= 8, f"Trunk mean {avg:.1f} out of range"

    def test_pelvis_anterior_tilt(self):
        """Pelvis should average ~10 deg anterior tilt."""
        result = get_normative_curve("pelvis_sagittal", "adult")
        mean = np.array(result["mean"])
        avg = np.mean(mean)
        assert 7 <= avg <= 13, f"Pelvis mean {avg:.1f} out of range"


class TestGetNormativeCurveAllStrata:
    """Test normative curves across all strata."""

    @pytest.mark.parametrize("stratum", ["adult", "elderly", "pediatric"])
    def test_hip_curve_exists(self, stratum):
        result = get_normative_curve("hip", stratum)
        assert len(result["mean"]) == 101

    def test_elderly_reduced_rom(self):
        """Elderly ROM should be less than adult ROM."""
        adult = np.array(get_normative_curve("hip", "adult")["mean"])
        elderly = np.array(get_normative_curve("hip", "elderly")["mean"])
        adult_rom = np.ptp(adult)
        elderly_rom = np.ptp(elderly)
        assert elderly_rom < adult_rom

    def test_pediatric_increased_rom(self):
        """Pediatric ROM should be slightly more than adult ROM."""
        adult = np.array(get_normative_curve("hip", "adult")["mean"])
        pediatric = np.array(get_normative_curve("hip", "pediatric")["mean"])
        adult_rom = np.ptp(adult)
        pediatric_rom = np.ptp(pediatric)
        assert pediatric_rom > adult_rom

    def test_pediatric_larger_sd(self):
        """Pediatric SD should be larger (more variable)."""
        adult_sd = np.array(get_normative_curve("hip", "adult")["sd"])
        ped_sd = np.array(get_normative_curve("hip", "pediatric")["sd"])
        assert np.mean(ped_sd) > np.mean(adult_sd)

    def test_invalid_stratum_raises(self):
        with pytest.raises(ValueError, match="Unknown stratum"):
            get_normative_curve("hip", "toddler")

    def test_invalid_joint_raises(self):
        with pytest.raises(ValueError, match="Unknown joint"):
            get_normative_curve("wrist", "adult")


class TestNormativeCurve101Points:
    """Verify all curves have exactly 101 points."""

    @pytest.mark.parametrize("joint", ["hip", "knee", "ankle", "trunk", "pelvis_sagittal"])
    @pytest.mark.parametrize("stratum", ["adult", "elderly", "pediatric"])
    def test_101_points(self, joint, stratum):
        result = get_normative_curve(joint, stratum)
        assert len(result["mean"]) == 101, f"{joint}/{stratum} mean != 101"
        assert len(result["sd"]) == 101, f"{joint}/{stratum} sd != 101"


class TestNormativeBandContainsMean:
    """Test that normative band contains the mean curve."""

    @pytest.mark.parametrize("joint", ["hip", "knee", "ankle", "trunk", "pelvis_sagittal"])
    def test_band_brackets_mean(self, joint):
        band = get_normative_band(joint, "adult", n_sd=1.0)
        upper = np.array(band["upper"])
        lower = np.array(band["lower"])
        mean = np.array(band["mean"])
        assert np.all(upper >= mean), "Upper band below mean"
        assert np.all(lower <= mean), "Lower band above mean"

    def test_band_width_scales_with_sd(self):
        band1 = get_normative_band("hip", "adult", n_sd=1.0)
        band2 = get_normative_band("hip", "adult", n_sd=2.0)
        width1 = np.array(band1["upper"]) - np.array(band1["lower"])
        width2 = np.array(band2["upper"]) - np.array(band2["lower"])
        # 2-SD band should be twice as wide as 1-SD band
        np.testing.assert_allclose(width2, width1 * 2.0, atol=1e-10)

    def test_band_has_101_points(self):
        band = get_normative_band("knee", "adult")
        assert len(band["upper"]) == 101
        assert len(band["lower"]) == 101
        assert len(band["mean"]) == 101


class TestSelectStratum:
    """Test automatic stratum selection from age."""

    def test_adult_age_30(self):
        assert select_stratum(30) == "adult"

    def test_adult_age_18(self):
        assert select_stratum(18) == "adult"

    def test_adult_age_64(self):
        assert select_stratum(64) == "adult"

    def test_elderly_age_65(self):
        assert select_stratum(65) == "elderly"

    def test_elderly_age_80(self):
        assert select_stratum(80) == "elderly"

    def test_pediatric_age_10(self):
        assert select_stratum(10) == "pediatric"

    def test_pediatric_age_5(self):
        assert select_stratum(5) == "pediatric"

    def test_pediatric_age_17(self):
        assert select_stratum(17) == "pediatric"

    def test_none_defaults_to_adult(self):
        assert select_stratum(None) == "adult"

    def test_no_argument_defaults_to_adult(self):
        assert select_stratum() == "adult"


class TestListJoints:
    """Test list_joints function."""

    def test_returns_list(self):
        result = list_joints()
        assert isinstance(result, list)

    def test_contains_expected_joints(self):
        result = list_joints()
        for j in ["hip", "knee", "ankle", "trunk", "pelvis_sagittal"]:
            assert j in result

    def test_length(self):
        assert len(list_joints()) == 8


class TestListStrata:
    """Test list_strata function."""

    def test_returns_list(self):
        result = list_strata()
        assert isinstance(result, list)

    def test_contains_expected_strata(self):
        result = list_strata()
        for s in ["adult", "elderly", "pediatric"]:
            assert s in result

    def test_length(self):
        assert len(list_strata()) == 3


# ── Variable SD tests (E3 fix) ──────────────────────────────────────


class TestVariableSD:
    """After the E3 fix, SDs should vary across the gait cycle for
    hip, knee, and ankle (not constant)."""

    def test_hip_sd_not_constant(self):
        """Adult hip SD should not be the same value everywhere."""
        result = get_normative_curve("hip", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Hip SD is constant -- expected variation"

    def test_knee_sd_not_constant(self):
        """Adult knee SD should not be the same value everywhere."""
        result = get_normative_curve("knee", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Knee SD is constant -- expected variation"

    def test_ankle_sd_not_constant(self):
        """Adult ankle SD should not be the same value everywhere."""
        result = get_normative_curve("ankle", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Ankle SD is constant -- expected variation"

    def test_trunk_sd_constant(self):
        """Trunk SD can remain constant (acceptable)."""
        result = get_normative_curve("trunk", "adult")
        sd = np.array(result["sd"])
        np.testing.assert_allclose(sd, sd[0], atol=1e-10)

    def test_pelvis_sd_constant(self):
        """Pelvis SD can remain constant (acceptable)."""
        result = get_normative_curve("pelvis_sagittal", "adult")
        sd = np.array(result["sd"])
        np.testing.assert_allclose(sd, sd[0], atol=1e-10)

    def test_hip_sd_higher_at_extremes(self):
        """Hip SD should be higher at cycle extremes (near 0% and 55-65%)
        than at mid-stance (40-50%)."""
        result = get_normative_curve("hip", "adult")
        sd = np.array(result["sd"])
        mid_stance_sd = np.mean(sd[40:51])   # 40-50% GC
        extreme_sd = np.mean(sd[0:11])        # 0-10% GC
        assert extreme_sd > mid_stance_sd, (
            f"Extreme SD ({extreme_sd:.2f}) should exceed mid-stance SD ({mid_stance_sd:.2f})"
        )

    def test_knee_sd_higher_in_swing(self):
        """Knee SD should be higher in swing phase (65-85%) than stance."""
        result = get_normative_curve("knee", "adult")
        sd = np.array(result["sd"])
        stance_sd = np.mean(sd[20:50])    # mid-stance
        swing_sd = np.mean(sd[65:86])     # swing phase
        assert swing_sd > stance_sd, (
            f"Swing SD ({swing_sd:.2f}) should exceed stance SD ({stance_sd:.2f})"
        )

    def test_ankle_sd_higher_at_pushoff(self):
        """Ankle SD should be higher at push-off (55-65%) than baseline."""
        result = get_normative_curve("ankle", "adult")
        sd = np.array(result["sd"])
        baseline_sd = np.mean(sd[20:40])    # mid-stance baseline
        pushoff_sd = np.mean(sd[55:66])     # push-off region
        assert pushoff_sd > baseline_sd, (
            f"Push-off SD ({pushoff_sd:.2f}) should exceed baseline SD ({baseline_sd:.2f})"
        )

    def test_all_sd_positive(self):
        """All SD values must be strictly positive for all joints/strata."""
        for joint in ["hip", "knee", "ankle", "trunk", "pelvis_sagittal"]:
            for stratum in ["adult", "elderly", "pediatric"]:
                result = get_normative_curve(joint, stratum)
                sd = np.array(result["sd"])
                assert np.all(sd > 0), f"{joint}/{stratum}: found non-positive SD"

    def test_elderly_sd_varies(self):
        """Elderly hip SD should also vary (not constant)."""
        result = get_normative_curve("hip", "elderly")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min()

    def test_pediatric_sd_varies(self):
        """Pediatric hip SD should also vary (not constant)."""
        result = get_normative_curve("hip", "pediatric")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min()
