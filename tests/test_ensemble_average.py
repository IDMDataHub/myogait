"""Tests for ensemble_average multi-trial averaging.

Verifies that ensemble_average() correctly computes grand mean,
inter-trial SD, and intra-trial SD across multiple segment_cycles()
outputs, following the OpenCap (Uhlrich et al. 2023) convention.
"""

import copy

import numpy as np
import pytest

from myogait.cycles import ensemble_average


# ── Helpers ──────────────────────────────────────────────────────────


def _make_trial_output(side_summaries, cycles=None):
    """Build a minimal segment_cycles()-style output dict.

    Parameters
    ----------
    side_summaries : dict
        Mapping from side ("left"/"right") to a dict of summary fields.
        Example: {"left": {"n_cycles": 5, "hip_mean": [...], "hip_std": [...]}}
    cycles : list, optional
        List of individual cycle dicts.  Defaults to an empty list.
    """
    return {
        "cycles": cycles if cycles is not None else [],
        "summary": side_summaries,
    }


def _constant_curve(value, n_points=101):
    """Return a flat array of *n_points* filled with *value*."""
    return np.full(n_points, value).tolist()


def _sine_curve(amplitude=1.0, offset=0.0, n_points=101):
    """Return a sine wave from 0 to 2*pi with given amplitude and offset."""
    x = np.linspace(0, 2 * np.pi, n_points)
    return (offset + amplitude * np.sin(x)).tolist()


# ── Test: single trial ───────────────────────────────────────────────


class TestEnsembleAverageSingleTrial:
    """With a single trial the grand mean should equal the trial mean."""

    def test_grand_mean_equals_trial_mean(self):
        hip_mean = _sine_curve(amplitude=20.0, offset=10.0)
        hip_std = _constant_curve(2.0)
        knee_mean = _sine_curve(amplitude=40.0, offset=30.0)
        knee_std = _constant_curve(3.5)

        trial = _make_trial_output({
            "left": {
                "n_cycles": 8,
                "hip_mean": hip_mean,
                "hip_std": hip_std,
                "knee_mean": knee_mean,
                "knee_std": knee_std,
            },
        })

        result = ensemble_average([trial])

        assert "left" in result
        hip = result["left"]["hip"]
        knee = result["left"]["knee"]

        np.testing.assert_allclose(hip["grand_mean"], np.array(hip_mean))
        np.testing.assert_allclose(knee["grand_mean"], np.array(knee_mean))

    def test_inter_trial_sd_zero_for_single_trial(self):
        """SD across one sample is 0."""
        trial = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(15.0),
                "hip_std": _constant_curve(1.0),
            },
        })

        result = ensemble_average([trial])
        np.testing.assert_allclose(
            result["left"]["hip"]["inter_trial_sd"],
            np.zeros(101),
        )

    def test_intra_trial_sd_equals_trial_std(self):
        std_vals = _constant_curve(2.5)
        trial = _make_trial_output({
            "left": {
                "n_cycles": 6,
                "hip_mean": _constant_curve(10.0),
                "hip_std": std_vals,
            },
        })

        result = ensemble_average([trial])
        np.testing.assert_allclose(
            result["left"]["hip"]["intra_trial_sd"],
            np.array(std_vals),
        )

    def test_n_trials_and_n_total_cycles(self):
        trial = _make_trial_output({
            "right": {
                "n_cycles": 7,
                "knee_mean": _constant_curve(20.0),
                "knee_std": _constant_curve(1.0),
            },
        })

        result = ensemble_average([trial])
        assert result["right"]["knee"]["n_trials"] == 1
        assert result["right"]["knee"]["n_total_cycles"] == 7


# ── Test: two identical trials ───────────────────────────────────────


class TestEnsembleAverageTwoIdenticalTrials:
    """Two identical trials should yield inter_trial_sd ~ 0."""

    @pytest.fixture()
    def identical_result(self):
        hip_mean = _sine_curve(amplitude=25.0, offset=5.0)
        hip_std = _constant_curve(3.0)
        trial = _make_trial_output({
            "left": {
                "n_cycles": 4,
                "hip_mean": hip_mean,
                "hip_std": hip_std,
                "knee_mean": _constant_curve(30.0),
                "knee_std": _constant_curve(2.0),
            },
            "right": {
                "n_cycles": 3,
                "hip_mean": hip_mean,
                "hip_std": hip_std,
            },
        })
        trial2 = copy.deepcopy(trial)
        return ensemble_average([trial, trial2])

    def test_inter_trial_sd_near_zero(self, identical_result):
        for side in ("left", "right"):
            if side not in identical_result:
                continue
            for joint_name, joint_data in identical_result[side].items():
                np.testing.assert_allclose(
                    joint_data["inter_trial_sd"],
                    np.zeros(101),
                    atol=1e-12,
                    err_msg=f"inter_trial_sd not zero for {side}/{joint_name}",
                )

    def test_grand_mean_unchanged(self, identical_result):
        expected = np.array(_sine_curve(amplitude=25.0, offset=5.0))
        np.testing.assert_allclose(
            identical_result["left"]["hip"]["grand_mean"],
            expected,
        )

    def test_intra_trial_sd_is_within_trial_std(self, identical_result):
        """Intra-trial SD should be the mean of the (identical) trial SDs."""
        np.testing.assert_allclose(
            identical_result["left"]["hip"]["intra_trial_sd"],
            np.full(101, 3.0),
        )

    def test_n_trials_two(self, identical_result):
        assert identical_result["left"]["hip"]["n_trials"] == 2

    def test_n_total_cycles_summed(self, identical_result):
        # 4 cycles per trial * 2 trials = 8
        assert identical_result["left"]["hip"]["n_total_cycles"] == 8


# ── Test: two different trials (verify arithmetic) ───────────────────


class TestEnsembleAverageTwoDifferentTrials:
    """Verify correct arithmetic when trials differ."""

    def test_grand_mean_is_average_of_trial_means(self):
        trial_a = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
            },
        })
        trial_b = _make_trial_output({
            "left": {
                "n_cycles": 3,
                "hip_mean": _constant_curve(20.0),
                "hip_std": _constant_curve(3.0),
            },
        })

        result = ensemble_average([trial_a, trial_b])
        np.testing.assert_allclose(
            result["left"]["hip"]["grand_mean"],
            np.full(101, 15.0),
        )

    def test_inter_trial_sd_correct(self):
        trial_a = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
            },
        })
        trial_b = _make_trial_output({
            "left": {
                "n_cycles": 3,
                "hip_mean": _constant_curve(20.0),
                "hip_std": _constant_curve(3.0),
            },
        })

        result = ensemble_average([trial_a, trial_b])
        # SD of [10, 20] with ddof=0 = 5.0
        np.testing.assert_allclose(
            result["left"]["hip"]["inter_trial_sd"],
            np.full(101, 5.0),
        )

    def test_intra_trial_sd_is_mean_of_stds(self):
        trial_a = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
            },
        })
        trial_b = _make_trial_output({
            "left": {
                "n_cycles": 3,
                "hip_mean": _constant_curve(20.0),
                "hip_std": _constant_curve(3.0),
            },
        })

        result = ensemble_average([trial_a, trial_b])
        # Mean of [1.0, 3.0] = 2.0
        np.testing.assert_allclose(
            result["left"]["hip"]["intra_trial_sd"],
            np.full(101, 2.0),
        )


# ── Test: missing joint handling ─────────────────────────────────────


class TestEnsembleAverageMissingJoint:
    """Joints absent in some trials should be handled gracefully."""

    def test_missing_joint_in_one_trial(self):
        """If one trial lacks a joint, only the other contributes."""
        trial_a = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
                "knee_mean": _constant_curve(30.0),
                "knee_std": _constant_curve(2.0),
            },
        })
        trial_b = _make_trial_output({
            "left": {
                "n_cycles": 4,
                "hip_mean": _constant_curve(20.0),
                "hip_std": _constant_curve(3.0),
                # No knee data in this trial
            },
        })

        result = ensemble_average([trial_a, trial_b])

        # hip should use both trials
        assert result["left"]["hip"]["n_trials"] == 2
        np.testing.assert_allclose(
            result["left"]["hip"]["grand_mean"],
            np.full(101, 15.0),
        )

        # knee should use only trial_a
        assert result["left"]["knee"]["n_trials"] == 1
        np.testing.assert_allclose(
            result["left"]["knee"]["grand_mean"],
            np.full(101, 30.0),
        )
        assert result["left"]["knee"]["n_total_cycles"] == 5

    def test_joint_missing_in_all_trials_is_omitted(self):
        """A joint not found in any trial should not appear in output."""
        trial = _make_trial_output({
            "left": {
                "n_cycles": 3,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
            },
        })

        result = ensemble_average([trial])
        assert "knee" not in result["left"]

    def test_side_missing_in_one_trial(self):
        """A side absent from one trial should still aggregate the other."""
        trial_a = _make_trial_output({
            "left": {
                "n_cycles": 4,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
            },
            "right": {
                "n_cycles": 3,
                "hip_mean": _constant_curve(20.0),
                "hip_std": _constant_curve(2.0),
            },
        })
        trial_b = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(15.0),
                "hip_std": _constant_curve(1.5),
            },
            # No right side in this trial
        })

        result = ensemble_average([trial_a, trial_b])

        # Left uses both
        assert result["left"]["hip"]["n_trials"] == 2
        # Right uses only trial_a
        assert result["right"]["hip"]["n_trials"] == 1
        np.testing.assert_allclose(
            result["right"]["hip"]["grand_mean"],
            np.full(101, 20.0),
        )


# ── Test: joints filter parameter ────────────────────────────────────


class TestEnsembleAverageJointsFilter:
    """The joints parameter should restrict which joints appear."""

    def test_filter_to_single_joint(self):
        trial = _make_trial_output({
            "left": {
                "n_cycles": 5,
                "hip_mean": _constant_curve(10.0),
                "hip_std": _constant_curve(1.0),
                "knee_mean": _constant_curve(30.0),
                "knee_std": _constant_curve(2.0),
                "ankle_mean": _constant_curve(5.0),
                "ankle_std": _constant_curve(0.5),
            },
        })

        result = ensemble_average([trial], joints=["hip"])
        assert "hip" in result["left"]
        assert "knee" not in result["left"]
        assert "ankle" not in result["left"]


# ── Test: error conditions ───────────────────────────────────────────


class TestEnsembleAverageErrors:
    """Input validation."""

    def test_raises_on_non_list(self):
        with pytest.raises(TypeError):
            ensemble_average({"summary": {}})

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            ensemble_average([])

    def test_empty_summaries_return_empty(self):
        """Trials with no summary data should produce an empty result."""
        trial = _make_trial_output({})
        result = ensemble_average([trial])
        assert result == {}
