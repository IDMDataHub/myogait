"""Tests for extended angle functions in angles.py.

Tests cover head angle, arm angles, pelvis sagittal tilt,
depth-enhanced angles, frontal-plane angles, and the
compute_extended_angles() public API.
"""

import copy

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    make_walking_data,
    make_standing_data,
    make_fake_data,
    make_walking_data_with_depth,
    walking_data_with_angles,
)
from myogait.angles import (
    _head_angle,
    _arm_angles,
    _pelvis_sagittal_tilt,
    _depth_enhanced_angles,
    compute_frontal_angles,
    compute_extended_angles,
    compute_angles,
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
