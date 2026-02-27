"""Regression tests: COCO-17 models with estimated foot landmarks.

Verifies that toe_clearance, velocity/oconnor event detection, ankle
angles, and the full pipeline all work correctly when foot landmarks
(HEEL, FOOT_INDEX) are estimated from ankle/knee geometry instead of
being natively detected.

These tests catch regressions where COCO-17 backends (yolo, hrnet,
mmpose, vitpose, openpose, detectron2, alphapose) would silently
produce None/NaN for foot-dependent metrics.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_coco17_walking_data, make_walking_data

from myogait.extract import _estimate_missing_foot_landmarks, _is_landmark_nan


# ── Helpers ────────────────────────────────────────────────────────────


def _apply_foot_estimation(data):
    """Apply foot landmark estimation to all frames (simulates extract pipeline)."""
    for frame in data["frames"]:
        _estimate_missing_foot_landmarks(frame)


def _run_pipeline_with_events(data, method="zeni"):
    """Run normalize → angles → events pipeline."""
    from myogait import normalize, compute_angles, detect_events
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data, method=method)
    return data


def _run_full_pipeline(data):
    """Run complete pipeline through analysis."""
    from myogait import (
        normalize, compute_angles, detect_events,
        segment_cycles, analyze_gait,
    )
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    return data, cycles, stats


# ── Unit tests: estimation mechanics ──────────────────────────────────


class TestEstimationMechanics:

    def test_coco17_frames_get_estimated_heel(self):
        """COCO-17 frames should get estimated HEEL after estimation."""
        data = make_coco17_walking_data(n_frames=10)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            assert not _is_landmark_nan(frame["landmarks"].get("LEFT_HEEL"))
            assert not _is_landmark_nan(frame["landmarks"].get("RIGHT_HEEL"))

    def test_coco17_frames_get_estimated_foot_index(self):
        """COCO-17 frames should get estimated FOOT_INDEX after estimation."""
        data = make_coco17_walking_data(n_frames=10)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            assert not _is_landmark_nan(frame["landmarks"].get("LEFT_FOOT_INDEX"))
            assert not _is_landmark_nan(frame["landmarks"].get("RIGHT_FOOT_INDEX"))

    def test_estimated_visibility_always_half(self):
        """All estimated foot landmarks should have visibility=0.5."""
        data = make_coco17_walking_data(n_frames=10)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            for name in ("LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
                assert frame["landmarks"][name]["visibility"] == pytest.approx(0.5)

    def test_source_flag_set_to_estimated(self):
        """foot_landmarks_source should be 'estimated' for COCO-17."""
        data = make_coco17_walking_data(n_frames=5)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            assert frame.get("foot_landmarks_source") == "estimated"

    def test_mediapipe_data_not_modified(self):
        """MediaPipe data (with real foot landmarks) should not be changed."""
        data = make_walking_data(n_frames=5)
        original_heels = [
            frame["landmarks"]["LEFT_HEEL"]["x"] for frame in data["frames"]
        ]
        _apply_foot_estimation(data)
        for i, frame in enumerate(data["frames"]):
            assert frame["landmarks"]["LEFT_HEEL"]["x"] == pytest.approx(original_heels[i])
            assert "foot_landmarks_source" not in frame

    def test_one_side_nan_other_valid(self):
        """If only one side has NaN foot landmarks, only that side is estimated."""
        data = make_walking_data(n_frames=5)
        # Make only LEFT foot landmarks NaN
        for frame in data["frames"]:
            frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
            frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _apply_foot_estimation(data)
        for frame in data["frames"]:
            # LEFT should be estimated (vis=0.5)
            assert frame["landmarks"]["LEFT_HEEL"]["visibility"] == pytest.approx(0.5)
            # RIGHT should keep original (vis=1.0)
            assert frame["landmarks"]["RIGHT_HEEL"]["visibility"] == pytest.approx(1.0)

    def test_estimation_idempotent(self):
        """Running estimation twice should not change results."""
        data = make_coco17_walking_data(n_frames=5)
        _apply_foot_estimation(data)
        first_pass = [
            (frame["landmarks"]["LEFT_HEEL"]["x"], frame["landmarks"]["LEFT_HEEL"]["y"])
            for frame in data["frames"]
        ]
        _apply_foot_estimation(data)
        for i, frame in enumerate(data["frames"]):
            assert frame["landmarks"]["LEFT_HEEL"]["x"] == pytest.approx(first_pass[i][0])
            assert frame["landmarks"]["LEFT_HEEL"]["y"] == pytest.approx(first_pass[i][1])

    def test_estimated_heel_near_ankle(self):
        """Estimated HEEL should be close to ankle (within foot length)."""
        data = make_coco17_walking_data(n_frames=5)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            ankle_x = frame["landmarks"]["LEFT_ANKLE"]["x"]
            ankle_y = frame["landmarks"]["LEFT_ANKLE"]["y"]
            heel_x = frame["landmarks"]["LEFT_HEEL"]["x"]
            heel_y = frame["landmarks"]["LEFT_HEEL"]["y"]
            dist = np.sqrt((heel_x - ankle_x)**2 + (heel_y - ankle_y)**2)
            # Foot length is ~25% of shank; heel offset is ~30% of that
            assert dist < 0.10  # should be within 10% of image

    def test_estimated_foot_index_near_ankle(self):
        """Estimated FOOT_INDEX should be close to ankle."""
        data = make_coco17_walking_data(n_frames=5)
        _apply_foot_estimation(data)
        for frame in data["frames"]:
            ankle_x = frame["landmarks"]["LEFT_ANKLE"]["x"]
            ankle_y = frame["landmarks"]["LEFT_ANKLE"]["y"]
            fi_x = frame["landmarks"]["LEFT_FOOT_INDEX"]["x"]
            fi_y = frame["landmarks"]["LEFT_FOOT_INDEX"]["y"]
            dist = np.sqrt((fi_x - ankle_x)**2 + (fi_y - ankle_y)**2)
            assert dist < 0.10


# ── Regression: Ankle angles with COCO-17 ────────────────────────────


class TestAnkleAnglesRegression:

    def test_ankle_angles_not_none_with_coco17(self):
        """Ankle angles should be computed (not None) with COCO-17 + estimation."""
        from myogait import compute_angles
        data = make_coco17_walking_data(n_frames=100)
        _apply_foot_estimation(data)
        compute_angles(data, correction_factor=1.0, calibrate=False)

        non_none_l = sum(1 for af in data["angles"]["frames"] if af.get("ankle_L") is not None)
        non_none_r = sum(1 for af in data["angles"]["frames"] if af.get("ankle_R") is not None)
        assert non_none_l > 50, f"Only {non_none_l} non-None ankle_L angles"
        assert non_none_r > 50, f"Only {non_none_r} non-None ankle_R angles"

    def test_ankle_angles_same_count_as_mediapipe(self):
        """COCO-17 with estimation should produce similar ankle angle coverage as MediaPipe."""
        from myogait import compute_angles

        # MediaPipe (native foot landmarks)
        data_mp = make_walking_data(n_frames=100)
        compute_angles(data_mp, correction_factor=1.0, calibrate=False)
        mp_count = sum(1 for af in data_mp["angles"]["frames"] if af.get("ankle_L") is not None)

        # COCO-17 with estimation
        data_coco = make_coco17_walking_data(n_frames=100)
        _apply_foot_estimation(data_coco)
        compute_angles(data_coco, correction_factor=1.0, calibrate=False)
        coco_count = sum(1 for af in data_coco["angles"]["frames"] if af.get("ankle_L") is not None)

        # Should be very similar (both should produce ~100 valid angles)
        assert coco_count >= mp_count * 0.8, (
            f"COCO-17 ankle coverage ({coco_count}) much lower than MediaPipe ({mp_count})"
        )


# ── Regression: Event detection with COCO-17 ─────────────────────────


class TestEventDetectionRegression:

    def test_velocity_events_with_coco17(self):
        """Velocity method should produce events with COCO-17 + estimation."""
        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        _run_pipeline_with_events(data, method="velocity")

        events = data.get("events", {})
        # Should detect at least some heel strikes
        left_hs = events.get("left_hs", [])
        right_hs = events.get("right_hs", [])
        assert len(left_hs) > 0, "No left heel strikes detected with velocity method"
        assert len(right_hs) > 0, "No right heel strikes detected with velocity method"

    def test_oconnor_events_with_coco17(self):
        """O'Connor method should produce events with COCO-17 + estimation."""
        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        _run_pipeline_with_events(data, method="oconnor")

        events = data.get("events", {})
        left_hs = events.get("left_hs", [])
        right_hs = events.get("right_hs", [])
        assert len(left_hs) > 0, "No left heel strikes detected with oconnor method"
        assert len(right_hs) > 0, "No right heel strikes detected with oconnor method"

    def test_zeni_events_work_regardless(self):
        """Zeni method (ANKLE-based) should work with or without estimation."""
        data = make_coco17_walking_data(n_frames=300)
        # No foot estimation - Zeni uses ankle directly
        _run_pipeline_with_events(data, method="zeni")

        events = data.get("events", {})
        assert len(events.get("left_hs", [])) > 0
        assert len(events.get("right_hs", [])) > 0

    def test_velocity_events_count_reasonable(self):
        """Velocity events with COCO-17 estimation should detect a reasonable count.

        Estimated heel positions (geometric from ankle/knee) produce a
        different vertical velocity profile than real heel landmarks,
        so exact count comparison with MediaPipe is not meaningful.
        The key requirement is that a usable number of events is detected.
        """
        data_coco = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data_coco)
        _run_pipeline_with_events(data_coco, method="velocity")
        coco_hs = len(data_coco.get("events", {}).get("left_hs", []))

        # 300 frames at 30fps = 10s → ~10 gait cycles → expect at least 5 HS
        assert coco_hs >= 5, (
            f"COCO-17 velocity HS count too low: {coco_hs} (expected >= 5)"
        )


# ── Regression: Toe clearance with COCO-17 ───────────────────────────


class TestToeClearanceRegression:

    def test_toe_clearance_has_foot_index_access_with_estimation(self):
        """toe_clearance should be able to read FOOT_INDEX with COCO-17 + estimation.

        Note: the synthetic walking data has constant y for foot landmarks,
        so toe_clearance may still return None (no swing phase variation).
        This test verifies that the landmarks are at least accessible
        (not NaN), which is the precondition for real video data to work.
        """
        from myogait import (
            normalize, compute_angles, detect_events,
            segment_cycles,
        )

        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        cycles = segment_cycles(data)

        if cycles.get("cycles"):
            # Verify FOOT_INDEX and HEEL are accessible (not NaN)
            for frame in data["frames"]:
                for name in ("LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
                    assert not _is_landmark_nan(frame["landmarks"].get(name)), (
                        f"{name} is still NaN after estimation"
                    )

    def test_toe_clearance_reads_estimated_heel_for_ground(self):
        """toe_clearance should find ground level from estimated HEEL landmarks.

        With COCO-17 + estimation, HEEL is available at visibility=0.5.
        The function should use it to compute ground_y percentile,
        even if the synthetic data doesn't produce actual clearance values
        (no vertical swing phase in synthetic walking data).
        """
        from myogait.analysis import toe_clearance

        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)

        # Create mock cycles with to_frame to test the code path
        mock_cycles = {
            "cycles": [
                {"side": "left", "start_frame": 0, "to_frame": 15,
                 "end_frame": 30},
            ],
        }

        result = toe_clearance(data, mock_cycles)
        # The function should at least run without error and produce the dict
        assert "mtc_left" in result
        assert "unit" in result

    def test_toe_clearance_was_none_without_estimation(self):
        """Without estimation, COCO-17 toe_clearance should return None (baseline)."""
        from myogait import (
            normalize, compute_angles, detect_events,
            segment_cycles,
        )
        from myogait.analysis import toe_clearance

        data = make_coco17_walking_data(n_frames=300)
        # NO foot estimation
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        cycles = segment_cycles(data)

        if cycles.get("cycles"):
            result = toe_clearance(data, cycles)
            # Both sides should be None (HEEL and FOOT_INDEX are NaN)
            assert result.get("mtc_left") is None
            assert result.get("mtc_right") is None


# ── Regression: Full pipeline with COCO-17 ───────────────────────────


class TestFullPipelineRegression:

    def test_full_pipeline_coco17_with_estimation(self):
        """Full pipeline should complete without errors with COCO-17 + estimation."""
        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        data, cycles, stats = _run_full_pipeline(data)

        assert "angles" in data
        assert "events" in data
        assert cycles.get("cycles") is not None
        assert stats is not None

    def test_full_pipeline_coco17_vs_mediapipe_keys(self):
        """COCO-17 pipeline should produce same stat keys as MediaPipe pipeline."""
        # MediaPipe
        data_mp = make_walking_data(n_frames=300)
        data_mp, cycles_mp, stats_mp = _run_full_pipeline(data_mp)

        # COCO-17 with estimation
        data_coco = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data_coco)
        data_coco, cycles_coco, stats_coco = _run_full_pipeline(data_coco)

        # Both should have the same top-level stat keys
        mp_keys = set(stats_mp.keys()) if stats_mp else set()
        coco_keys = set(stats_coco.keys()) if stats_coco else set()
        # COCO stats should at least contain the core keys from MediaPipe
        missing = mp_keys - coco_keys
        assert len(missing) == 0, f"COCO-17 stats missing keys: {missing}"

    def test_step_length_works_with_coco17(self):
        """step_length should work with COCO-17 (uses ANKLE, not HEEL)."""
        from myogait import (
            normalize, compute_angles, detect_events, segment_cycles,
        )
        from myogait.analysis import step_length

        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        cycles = segment_cycles(data)

        result = step_length(data, cycles)
        # Should return numeric values
        assert result.get("step_length_left") is not None or result.get("step_length_right") is not None

    def test_walking_speed_works_with_coco17(self):
        """walking_speed should work with COCO-17."""
        from myogait import (
            normalize, compute_angles, detect_events, segment_cycles,
        )
        from myogait.analysis import walking_speed

        data = make_coco17_walking_data(n_frames=300)
        _apply_foot_estimation(data)
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        cycles = segment_cycles(data)

        result = walking_speed(data, cycles)
        assert "speed_mean" in result


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_estimation_with_nan_ankle(self):
        """Frames where ankle is also NaN should not crash."""
        data = make_coco17_walking_data(n_frames=5)
        # Make ankle NaN too
        for frame in data["frames"]:
            frame["landmarks"]["LEFT_ANKLE"] = {
                "x": float("nan"), "y": float("nan"), "visibility": 0.0,
            }
        _apply_foot_estimation(data)
        # LEFT should still be NaN (can't estimate without ankle)
        for frame in data["frames"]:
            assert _is_landmark_nan(frame["landmarks"]["LEFT_HEEL"])
            # RIGHT should be estimated (right ankle is valid)
            assert not _is_landmark_nan(frame["landmarks"]["RIGHT_HEEL"])

    def test_estimation_with_zero_length_shank(self):
        """Knee == Ankle position should not crash (zero-length shank)."""
        data = make_coco17_walking_data(n_frames=3)
        for frame in data["frames"]:
            frame["landmarks"]["LEFT_KNEE"] = {
                "x": 0.50, "y": 0.80, "visibility": 1.0,  # same as ankle
            }
        _apply_foot_estimation(data)
        # Should not crash; LEFT heel may or may not be estimated
        # (zero-length shank means no direction vector)

    def test_estimation_preserves_other_landmarks(self):
        """Estimation should not modify non-foot landmarks."""
        data = make_coco17_walking_data(n_frames=3)
        original_nose = data["frames"][0]["landmarks"]["NOSE"]["x"]
        original_ankle = data["frames"][0]["landmarks"]["LEFT_ANKLE"]["x"]

        _apply_foot_estimation(data)

        assert data["frames"][0]["landmarks"]["NOSE"]["x"] == pytest.approx(original_nose)
        assert data["frames"][0]["landmarks"]["LEFT_ANKLE"]["x"] == pytest.approx(original_ankle)

    def test_estimation_with_mixed_nan_frames(self):
        """Mixed frames (some valid, some NaN ankle) should work."""
        data = make_coco17_walking_data(n_frames=10)
        # Make frames 3-5 have NaN ankles
        for i in range(3, 6):
            data["frames"][i]["landmarks"]["LEFT_ANKLE"] = {
                "x": float("nan"), "y": float("nan"), "visibility": 0.0,
            }
            data["frames"][i]["landmarks"]["LEFT_KNEE"] = {
                "x": float("nan"), "y": float("nan"), "visibility": 0.0,
            }

        _apply_foot_estimation(data)

        # Frames 0-2 and 6-9 should have estimated LEFT foot
        for i in [0, 1, 2, 6, 7, 8, 9]:
            assert not _is_landmark_nan(data["frames"][i]["landmarks"]["LEFT_HEEL"])
        # Frames 3-5 should still have NaN LEFT foot (ankle was NaN)
        for i in [3, 4, 5]:
            assert _is_landmark_nan(data["frames"][i]["landmarks"]["LEFT_HEEL"])
