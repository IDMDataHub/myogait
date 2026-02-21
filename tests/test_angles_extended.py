"""Tests for extended angle functions in angles.py.

Tests cover head angle, arm angles, pelvis sagittal tilt,
depth-enhanced angles, frontal-plane angles, walking direction
detection, foot progression angle, unwrap-without-median fix,
and the compute_extended_angles() public API.
"""


import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    make_walking_data,
    make_walking_data_with_depth,
    walking_data_with_angles,
)
from myogait.angles import (
    _head_angle,
    _arm_angles,
    _pelvis_sagittal_tilt,
    _depth_enhanced_angles,
    _detect_walking_direction,
    _unwrap_angles,
    compute_frontal_angles,
    compute_extended_angles,
    compute_angles,
    foot_progression_angle,
)


# ── Helper to build a single frame ──────────────────────────────────


def _make_standing_frame():
    """Build a single standing-pose frame dict."""
    return {
        "frame_idx": 0,
        "time_s": 0.0,
        "landmarks": {
            "NOSE":             {"x": 0.50, "y": 0.08, "visibility": 1.0},
            "LEFT_EYE":         {"x": 0.49, "y": 0.07, "visibility": 1.0},
            "RIGHT_EYE":        {"x": 0.51, "y": 0.07, "visibility": 1.0},
            "LEFT_EAR":         {"x": 0.48, "y": 0.10, "visibility": 1.0},
            "RIGHT_EAR":        {"x": 0.52, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": 0.45, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": 0.55, "y": 0.25, "visibility": 1.0},
            "LEFT_ELBOW":       {"x": 0.45, "y": 0.37, "visibility": 1.0},
            "RIGHT_ELBOW":      {"x": 0.55, "y": 0.37, "visibility": 1.0},
            "LEFT_WRIST":       {"x": 0.45, "y": 0.48, "visibility": 1.0},
            "RIGHT_WRIST":      {"x": 0.55, "y": 0.48, "visibility": 1.0},
            "LEFT_HIP":         {"x": 0.47, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP":        {"x": 0.53, "y": 0.50, "visibility": 1.0},
            "LEFT_KNEE":        {"x": 0.47, "y": 0.65, "visibility": 1.0},
            "RIGHT_KNEE":       {"x": 0.53, "y": 0.65, "visibility": 1.0},
            "LEFT_ANKLE":       {"x": 0.47, "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE":      {"x": 0.53, "y": 0.80, "visibility": 1.0},
            "LEFT_HEEL":        {"x": 0.48, "y": 0.82, "visibility": 1.0},
            "RIGHT_HEEL":       {"x": 0.54, "y": 0.82, "visibility": 1.0},
            "LEFT_FOOT_INDEX":  {"x": 0.44, "y": 0.82, "visibility": 1.0},
            "RIGHT_FOOT_INDEX": {"x": 0.50, "y": 0.82, "visibility": 1.0},
        },
        "confidence": 0.95,
    }


def _make_forward_head_frame():
    """Build a frame where the nose is significantly forward of the ears."""
    frame = _make_standing_frame()
    # Move nose far forward (larger x = forward in sagittal view)
    frame["landmarks"]["NOSE"]["x"] = 0.65
    return frame


def _make_forward_lean_frame():
    """Build a frame where the trunk is leaning forward."""
    frame = _make_standing_frame()
    # Move shoulders forward (larger x)
    frame["landmarks"]["LEFT_SHOULDER"]["x"] = 0.55
    frame["landmarks"]["RIGHT_SHOULDER"]["x"] = 0.65
    return frame


# ── _head_angle ──────────────────────────────────────────────────────


class TestHeadAngle:

    def test_head_angle_standing(self):
        """Standing posture: head angle should be small."""
        frame = _make_standing_frame()
        angle = _head_angle(frame)
        assert not np.isnan(angle)
        # In a well-aligned standing posture, head angle should be moderate
        assert abs(angle) < 90

    def test_head_angle_forward(self):
        """Forward head posture should produce a larger angle."""
        standing = _make_standing_frame()
        forward = _make_forward_head_frame()

        angle_standing = _head_angle(standing)
        angle_forward = _head_angle(forward)

        # Forward head should have a different (larger magnitude) angle
        assert abs(angle_forward) > abs(angle_standing) or abs(angle_forward - angle_standing) > 5


# ── _arm_angles ──────────────────────────────────────────────────────


class TestArmAngles:

    def test_arm_angles_standing(self):
        """Standing with arms hanging down: shoulder flexion should be small."""
        frame = _make_standing_frame()
        result = _arm_angles(frame)

        # Arms hanging straight down aligned with trunk -> angle near 0
        assert not np.isnan(result["shoulder_flex_L"])
        assert not np.isnan(result["shoulder_flex_R"])
        assert result["shoulder_flex_L"] < 30
        assert result["shoulder_flex_R"] < 30

    def test_arm_angles_keys(self):
        """Arm angles should return all expected keys."""
        frame = _make_standing_frame()
        result = _arm_angles(frame)

        assert "shoulder_flex_L" in result
        assert "shoulder_flex_R" in result
        assert "elbow_flex_L" in result
        assert "elbow_flex_R" in result

    def test_arm_angles_side_symmetry(self):
        """In a symmetric standing pose, left and right arm angles should be similar."""
        # Use a perfectly symmetric frame
        frame = {
            "frame_idx": 0,
            "time_s": 0.0,
            "landmarks": {
                "LEFT_SHOULDER":  {"x": 0.45, "y": 0.25, "visibility": 1.0},
                "RIGHT_SHOULDER": {"x": 0.55, "y": 0.25, "visibility": 1.0},
                "LEFT_ELBOW":     {"x": 0.45, "y": 0.37, "visibility": 1.0},
                "RIGHT_ELBOW":    {"x": 0.55, "y": 0.37, "visibility": 1.0},
                "LEFT_WRIST":     {"x": 0.45, "y": 0.48, "visibility": 1.0},
                "RIGHT_WRIST":    {"x": 0.55, "y": 0.48, "visibility": 1.0},
                "LEFT_HIP":       {"x": 0.47, "y": 0.50, "visibility": 1.0},
                "RIGHT_HIP":      {"x": 0.53, "y": 0.50, "visibility": 1.0},
            },
        }
        result = _arm_angles(frame)

        # Shoulder flexion should be nearly identical
        assert abs(result["shoulder_flex_L"] - result["shoulder_flex_R"]) < 5
        # Elbow flexion should be nearly identical
        assert abs(result["elbow_flex_L"] - result["elbow_flex_R"]) < 5


# ── _pelvis_sagittal_tilt ────────────────────────────────────────────


class TestPelvisSagittalTilt:

    def test_pelvis_sagittal_tilt_standing(self):
        """Upright standing: sagittal tilt should be small."""
        frame = _make_standing_frame()
        angle = _pelvis_sagittal_tilt(frame)
        assert not np.isnan(angle)
        # Near-vertical trunk should give a small angle
        assert abs(angle) < 30

    def test_pelvis_sagittal_tilt_forward_lean(self):
        """Forward lean should produce a larger sagittal tilt."""
        standing = _make_standing_frame()
        forward = _make_forward_lean_frame()

        angle_standing = _pelvis_sagittal_tilt(standing)
        angle_forward = _pelvis_sagittal_tilt(forward)

        # Forward lean should produce a larger tilt
        assert abs(angle_forward) > abs(angle_standing)


# ── compute_extended_angles ──────────────────────────────────────────


class TestComputeExtendedAngles:

    def test_compute_extended_angles_adds_keys(self):
        """compute_extended_angles should add head, arm, pelvis keys to angle frames."""
        data = walking_data_with_angles(n_frames=60)
        compute_extended_angles(data)

        af = data["angles"]["frames"][10]
        assert "head_angle" in af
        assert "shoulder_flex_L" in af
        assert "shoulder_flex_R" in af
        assert "elbow_flex_L" in af
        assert "elbow_flex_R" in af
        assert "pelvis_sagittal_tilt" in af

    def test_compute_extended_angles_requires_angles(self):
        """compute_extended_angles should raise if angles not computed."""
        data = make_walking_data(n_frames=30)
        # No compute_angles() called
        with pytest.raises(ValueError, match="No angles"):
            compute_extended_angles(data)

    def test_extended_angles_full_pipeline(self):
        """Full pipeline: normalize -> angles -> extended angles."""
        from myogait import normalize

        data = make_walking_data(n_frames=100)
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        compute_extended_angles(data)

        # Verify all angle frames have extended keys
        for af in data["angles"]["frames"]:
            assert "head_angle" in af
            assert "shoulder_flex_L" in af
            assert "pelvis_sagittal_tilt" in af

    def test_extended_angles_nan_handling(self):
        """Frames with missing landmarks should produce None for extended angles."""
        from myogait.schema import create_empty

        data = create_empty("test.mp4", fps=30.0, width=100, height=100, n_frames=5)
        data["extraction"] = {"model": "mediapipe"}
        frames = []
        for i in range(5):
            # Only provide minimal landmarks (no ears/nose for head angle)
            lm = {
                "LEFT_SHOULDER":    {"x": 0.5, "y": 0.25, "visibility": 1.0},
                "RIGHT_SHOULDER":   {"x": 0.5, "y": 0.25, "visibility": 1.0},
                "LEFT_HIP":         {"x": 0.5, "y": 0.50, "visibility": 1.0},
                "RIGHT_HIP":        {"x": 0.5, "y": 0.50, "visibility": 1.0},
            }
            frames.append({
                "frame_idx": i,
                "time_s": i / 30.0,
                "landmarks": lm,
                "confidence": 0.5,
            })
        data["frames"] = frames
        compute_angles(data, correction_factor=1.0, calibrate=False)
        compute_extended_angles(data)

        # Head angle should be None (missing ears/nose)
        af = data["angles"]["frames"][2]
        assert af["head_angle"] is None


# ── _depth_enhanced_angles ───────────────────────────────────────────


class TestDepthEnhancedAngles:

    def test_depth_enhanced_angles_with_depth(self):
        """Frame with depth data should return correction factors."""
        frame = _make_standing_frame()
        frame["landmark_depths"] = {
            "LEFT_HIP": 1.5,
            "RIGHT_HIP": 1.5,
            "LEFT_KNEE": 1.6,
            "RIGHT_KNEE": 1.6,
            "LEFT_ANKLE": 1.7,
            "RIGHT_ANKLE": 1.7,
            "LEFT_FOOT_INDEX": 1.7,
            "RIGHT_FOOT_INDEX": 1.7,
        }

        result = _depth_enhanced_angles(frame)
        assert result is not None
        assert "hip_L_correction" in result
        assert "knee_L_correction" in result
        assert "ankle_L_correction" in result
        # Correction factors should be >= 1.0 (depth adds length)
        assert result["hip_L_correction"] >= 1.0
        assert result["knee_L_correction"] >= 1.0

    def test_depth_enhanced_angles_without_depth(self):
        """Frame without depth data should return None."""
        frame = _make_standing_frame()
        result = _depth_enhanced_angles(frame)
        assert result is None


# ── compute_frontal_angles ───────────────────────────────────────────


class TestComputeFrontalAngles:

    def test_compute_frontal_angles_no_depth(self):
        """Without depth data, frontal angles should be None."""
        data = make_walking_data(n_frames=30)
        compute_frontal_angles(data)
        assert data["angles_frontal"] is None

    def test_compute_frontal_angles_with_depth(self):
        """With depth data, frontal angles should be computed."""
        data = make_walking_data_with_depth(n_frames=30)
        compute_frontal_angles(data)

        assert data["angles_frontal"] is not None
        assert "frames" in data["angles_frontal"]
        assert len(data["angles_frontal"]["frames"]) == 30

        # Check structure
        af = data["angles_frontal"]["frames"][10]
        assert "hip_abduction_L" in af
        assert "hip_abduction_R" in af
        assert "knee_valgus_L" in af
        assert "knee_valgus_R" in af

        # With depth data, at least some values should be non-None
        non_none = sum(
            1 for f in data["angles_frontal"]["frames"]
            if f["hip_abduction_L"] is not None
        )
        assert non_none > 0


# ── _unwrap_angles (E1 fix) ─────────────────────────────────────────


class TestUnwrapAnglesNoMedianRecenter:
    """After the E1 fix, _unwrap_angles must NOT recenter on the median."""

    def test_preserves_absolute_reference(self):
        """A constant series at 30 deg should stay at 30 deg, not be zeroed."""
        vals = [30.0] * 20
        result = _unwrap_angles(vals)
        # Should still be 30, not shifted to 0
        np.testing.assert_allclose(result, 30.0, atol=1e-6)

    def test_preserves_offset_hip_signal(self):
        """A hip signal oscillating around 15 deg should keep that offset."""
        t = np.linspace(0, 2 * np.pi, 100)
        signal = 15.0 + 10.0 * np.sin(t)  # oscillates 5..25
        result = _unwrap_angles(signal.tolist())
        result_arr = np.array(result)
        # The mean should still be around 15, not shifted to 0
        assert abs(np.mean(result_arr) - 15.0) < 2.0

    def test_unwrap_removes_discontinuity(self):
        """Unwrap should still fix +-180 jumps."""
        vals = [170.0, 175.0, 179.0, -179.0, -175.0, -170.0]
        result = _unwrap_angles(vals)
        arr = np.array(result)
        # After unwrap the series should be monotonically increasing
        diffs = np.diff(arr)
        assert np.all(diffs > 0), "Unwrap did not remove the discontinuity"

    def test_nan_handling(self):
        """NaN values should pass through unchanged."""
        vals = [10.0, float('nan'), 12.0]
        result = _unwrap_angles(vals)
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])


# ── _detect_walking_direction (W1) ──────────────────────────────────


class TestDetectWalkingDirection:

    def test_left_to_right(self):
        """Hip center x increasing => left_to_right."""
        data = _make_direction_data(start_x=0.2, end_x=0.8)
        assert _detect_walking_direction(data) == "left_to_right"

    def test_right_to_left(self):
        """Hip center x decreasing => right_to_left."""
        data = _make_direction_data(start_x=0.8, end_x=0.2)
        assert _detect_walking_direction(data) == "right_to_left"

    def test_stationary_defaults_left_to_right(self):
        """No displacement => left_to_right (default)."""
        data = _make_direction_data(start_x=0.5, end_x=0.5)
        assert _detect_walking_direction(data) == "left_to_right"

    def test_empty_frames_defaults(self):
        """No frames => left_to_right (default)."""
        data = {"frames": []}
        assert _detect_walking_direction(data) == "left_to_right"

    def test_single_frame_defaults(self):
        """Only one frame => left_to_right (default)."""
        data = {"frames": [{"landmarks": {
            "LEFT_HIP": {"x": 0.5, "y": 0.5},
            "RIGHT_HIP": {"x": 0.5, "y": 0.5},
        }}]}
        assert _detect_walking_direction(data) == "left_to_right"

    def test_direction_stored_in_angles(self):
        """compute_angles should store walking_direction in data['angles']."""
        data = make_walking_data(n_frames=60)
        compute_angles(data, correction_factor=1.0, calibrate=False)
        assert "walking_direction" in data["angles"]
        assert data["angles"]["walking_direction"] in (
            "left_to_right", "right_to_left"
        )


def _make_direction_data(start_x, end_x, n_frames=50):
    """Build minimal data with hip center moving from start_x to end_x."""
    frames = []
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        x = start_x + (end_x - start_x) * frac
        frames.append({
            "frame_idx": i,
            "time_s": i / 30.0,
            "landmarks": {
                "LEFT_HIP": {"x": x - 0.01, "y": 0.5, "visibility": 1.0},
                "RIGHT_HIP": {"x": x + 0.01, "y": 0.5, "visibility": 1.0},
            },
        })
    return {"frames": frames}


# ── foot_progression_angle (Feature 16) ─────────────────────────────


class TestFootProgressionAngle:

    def test_returns_both_sides(self):
        """Result should have foot_angle_L and foot_angle_R keys."""
        data = make_walking_data(n_frames=30)
        result = foot_progression_angle(data)
        assert "foot_angle_L" in result
        assert "foot_angle_R" in result
        assert len(result["foot_angle_L"]) == 30
        assert len(result["foot_angle_R"]) == 30

    def test_values_are_numeric(self):
        """Angles should be numeric (float) or None."""
        data = make_walking_data(n_frames=30)
        result = foot_progression_angle(data)
        for val in result["foot_angle_L"]:
            assert val is None or isinstance(val, float)

    def test_straight_ahead_foot(self):
        """A foot pointing perfectly horizontally (toe to the right of heel)
        should give angle near 0 degrees."""
        data = {
            "frames": [{
                "frame_idx": 0,
                "landmarks": {
                    "LEFT_HEEL": {"x": 0.4, "y": 0.8, "visibility": 1.0},
                    "LEFT_FOOT_INDEX": {"x": 0.5, "y": 0.8, "visibility": 1.0},
                    "RIGHT_HEEL": {"x": 0.4, "y": 0.8, "visibility": 1.0},
                    "RIGHT_FOOT_INDEX": {"x": 0.5, "y": 0.8, "visibility": 1.0},
                },
            }],
        }
        result = foot_progression_angle(data)
        # Horizontal foot => angle ~0
        assert abs(result["foot_angle_L"][0]) < 5.0
        assert abs(result["foot_angle_R"][0]) < 5.0

    def test_missing_landmarks_gives_none(self):
        """Missing heel or toe landmarks should produce None."""
        data = {
            "frames": [{
                "frame_idx": 0,
                "landmarks": {
                    "LEFT_HEEL": {"x": 0.4, "y": 0.8, "visibility": 1.0},
                    # LEFT_FOOT_INDEX missing
                    "RIGHT_HEEL": {"x": 0.4, "y": 0.8, "visibility": 1.0},
                    "RIGHT_FOOT_INDEX": {"x": 0.5, "y": 0.8, "visibility": 1.0},
                },
            }],
        }
        result = foot_progression_angle(data)
        assert result["foot_angle_L"][0] is None
        assert result["foot_angle_R"][0] is not None

    def test_empty_data(self):
        """Empty frames should return empty lists."""
        data = {"frames": []}
        result = foot_progression_angle(data)
        assert result["foot_angle_L"] == []
        assert result["foot_angle_R"] == []
