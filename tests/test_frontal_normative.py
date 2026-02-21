"""Tests for frontal plane normative curves in myogait.normative."""

import numpy as np
import pytest

from myogait.normative import (
    STRATA,
    get_normative_curve,
    get_normative_band,
    list_joints,
)


# ── Frontal joints in list_joints ────────────────────────────────────


class TestFrontalJointsInListJoints:
    """Frontal plane joints should appear in list_joints()."""

    def test_pelvis_obliquity_in_list(self):
        assert "pelvis_obliquity" in list_joints()

    def test_hip_adduction_in_list(self):
        assert "hip_adduction" in list_joints()

    def test_knee_valgus_in_list(self):
        assert "knee_valgus" in list_joints()

    def test_total_joint_count(self):
        """Should now have 8 joints (5 sagittal + 3 frontal)."""
        assert len(list_joints()) == 8


# ── get_normative_curve for frontal joints ───────────────────────────


class TestFrontalCurvePelvisObliquity:
    """Test pelvis obliquity normative curve."""

    def test_returns_dict(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        for key in ("mean", "sd", "unit", "source", "stratum"):
            assert key in result

    def test_101_points_mean(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        assert len(result["mean"]) == 101

    def test_101_points_sd(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        assert len(result["sd"]) == 101

    def test_mean_around_2_degrees(self):
        """Pelvis obliquity mean should be near 2 degrees."""
        result = get_normative_curve("pelvis_obliquity", "adult")
        mean = np.array(result["mean"])
        avg = np.mean(mean)
        assert -2 <= avg <= 6, f"Pelvis obliquity mean {avg:.1f} out of range"

    def test_sd_positive(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        sd = np.array(result["sd"])
        assert np.all(sd > 0)


class TestFrontalCurveHipAdduction:
    """Test hip adduction normative curve."""

    def test_101_points(self):
        result = get_normative_curve("hip_adduction", "adult")
        assert len(result["mean"]) == 101
        assert len(result["sd"]) == 101

    def test_mean_positive_adduction(self):
        """Hip adduction mean should be mostly positive (adducted)."""
        result = get_normative_curve("hip_adduction", "adult")
        mean = np.array(result["mean"])
        avg = np.mean(mean)
        assert avg > 0, f"Hip adduction mean {avg:.1f} should be positive"

    def test_stance_higher_than_swing(self):
        """Hip adduction should be higher in stance (~5 deg) than swing (~2 deg)."""
        result = get_normative_curve("hip_adduction", "adult")
        mean = np.array(result["mean"])
        stance_avg = np.mean(mean[10:50])  # 10-50% GC
        swing_avg = np.mean(mean[70:95])   # 70-95% GC
        assert stance_avg > swing_avg, (
            f"Stance adduction ({stance_avg:.1f}) should exceed swing ({swing_avg:.1f})"
        )

    def test_sd_in_range(self):
        """Hip adduction SD should be ~3-4 degrees on average."""
        result = get_normative_curve("hip_adduction", "adult")
        sd = np.array(result["sd"])
        avg_sd = np.mean(sd)
        assert 2.0 <= avg_sd <= 5.0, f"Hip adduction SD {avg_sd:.1f} out of range"


class TestFrontalCurveKneeValgus:
    """Test knee valgus normative curve."""

    def test_101_points(self):
        result = get_normative_curve("knee_valgus", "adult")
        assert len(result["mean"]) == 101
        assert len(result["sd"]) == 101

    def test_mean_positive_valgus(self):
        """Knee valgus mean should be mostly positive (~3 deg)."""
        result = get_normative_curve("knee_valgus", "adult")
        mean = np.array(result["mean"])
        avg = np.mean(mean)
        assert avg > 0, f"Knee valgus mean {avg:.1f} should be positive"
        assert 1.0 <= avg <= 5.0, f"Knee valgus mean {avg:.1f} out of expected range"


# ── get_normative_band for frontal joints ────────────────────────────


class TestFrontalNormativeBand:
    """Test normative bands for frontal joints."""

    @pytest.mark.parametrize("joint", ["pelvis_obliquity", "hip_adduction", "knee_valgus"])
    def test_band_brackets_mean(self, joint):
        band = get_normative_band(joint, "adult", n_sd=1.0)
        upper = np.array(band["upper"])
        lower = np.array(band["lower"])
        mean = np.array(band["mean"])
        assert np.all(upper >= mean), f"Upper band below mean for {joint}"
        assert np.all(lower <= mean), f"Lower band above mean for {joint}"

    @pytest.mark.parametrize("joint", ["pelvis_obliquity", "hip_adduction", "knee_valgus"])
    def test_band_101_points(self, joint):
        band = get_normative_band(joint, "adult")
        assert len(band["upper"]) == 101
        assert len(band["lower"]) == 101
        assert len(band["mean"]) == 101


# ── All strata have frontal data ─────────────────────────────────────


class TestFrontalAllStrata:
    """All strata should have frontal plane data."""

    @pytest.mark.parametrize("joint", ["pelvis_obliquity", "hip_adduction", "knee_valgus"])
    @pytest.mark.parametrize("stratum", ["adult", "elderly", "pediatric"])
    def test_curve_exists(self, joint, stratum):
        result = get_normative_curve(joint, stratum)
        assert len(result["mean"]) == 101
        assert len(result["sd"]) == 101

    def test_pediatric_hip_adduction_rom_wider(self):
        """Pediatric hip adduction ROM should be wider than adult.

        The _scale_rom approach preserves the global mean while amplifying
        excursion, so the pediatric ROM (peak-to-peak) must exceed the
        adult ROM rather than the mean being higher.
        """
        adult = np.array(get_normative_curve("hip_adduction", "adult")["mean"])
        pediatric = np.array(get_normative_curve("hip_adduction", "pediatric")["mean"])
        assert np.ptp(pediatric) > np.ptp(adult)

    def test_pediatric_sd_wider(self):
        """Pediatric frontal SD should be wider than adult."""
        for joint in ("pelvis_obliquity", "hip_adduction", "knee_valgus"):
            adult_sd = np.array(get_normative_curve(joint, "adult")["sd"])
            ped_sd = np.array(get_normative_curve(joint, "pediatric")["sd"])
            assert np.mean(ped_sd) > np.mean(adult_sd), (
                f"Pediatric {joint} SD ({np.mean(ped_sd):.1f}) should exceed "
                f"adult ({np.mean(adult_sd):.1f})"
            )

    def test_elderly_sd_wider(self):
        """Elderly frontal SD should be wider than adult."""
        for joint in ("pelvis_obliquity", "hip_adduction", "knee_valgus"):
            adult_sd = np.array(get_normative_curve(joint, "adult")["sd"])
            eld_sd = np.array(get_normative_curve(joint, "elderly")["sd"])
            assert np.mean(eld_sd) > np.mean(adult_sd), (
                f"Elderly {joint} SD ({np.mean(eld_sd):.1f}) should exceed "
                f"adult ({np.mean(adult_sd):.1f})"
            )


# ── Variable SD tests for frontal ────────────────────────────────────


class TestFrontalVariableSD:
    """Frontal plane SDs should vary across the gait cycle."""

    def test_pelvis_obliquity_sd_varies(self):
        result = get_normative_curve("pelvis_obliquity", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Pelvis obliquity SD is constant"

    def test_hip_adduction_sd_varies(self):
        result = get_normative_curve("hip_adduction", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Hip adduction SD is constant"

    def test_knee_valgus_sd_varies(self):
        result = get_normative_curve("knee_valgus", "adult")
        sd = np.array(result["sd"])
        assert sd.max() > sd.min(), "Knee valgus SD is constant"

    def test_all_frontal_sd_positive(self):
        """All frontal SD values must be strictly positive for all strata."""
        for joint in ("pelvis_obliquity", "hip_adduction", "knee_valgus"):
            for stratum in ("adult", "elderly", "pediatric"):
                result = get_normative_curve(joint, stratum)
                sd = np.array(result["sd"])
                assert np.all(sd > 0), f"{joint}/{stratum}: found non-positive SD"
