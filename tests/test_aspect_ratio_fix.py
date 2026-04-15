"""Tests for the aspect-ratio correction in compute_angles().

When pose-estimator landmarks are stored in normalised image coordinates
(``x``, ``y`` in [0, 1]) and the source image is non-square, the X and
Y axes have different metric units.  Computing angles directly in this
distorted space biases the result whenever the two segment vectors do
not share a single image axis — most strongly the ankle (vertical
shank versus horizontal foot).

These tests verify that:
  1. ``_pixelify_frame`` returns landmarks scaled by ``(width, height)``.
  2. ``compute_angles(apply_aspect_ratio=True)`` (the default) is a
     no-op for square images.
  3. For a portrait image the corrected ankle angle differs from the
     uncorrected one and matches the value computed directly in pixel
     space.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myogait.angles import (
    _pixelify_frame,
    compute_angles,
    _angle_between,
)


def _frame_with_landmarks(landmarks, frame_idx=0):
    return {
        "frame_idx": frame_idx,
        "time_s": 0.0,
        "confidence": 0.95,
        "landmarks": landmarks,
    }


def _make_min_data(frames, width, height, model="mediapipe"):
    return {
        "frames": frames,
        "meta": {"width": width, "height": height, "fps": 30.0},
        "extraction": {"model": model},
    }


def _ankle_dorsiflexed_landmarks_normalised():
    """A right leg with the foot dorsiflexed by ~10° (toe raised).

    Returns landmarks for HIP, KNEE, ANKLE, HEEL, FOOT_INDEX in
    normalised image coordinates.  The shank is vertical, the foot is
    tilted up at the toe end.
    """
    return {
        "RIGHT_HIP":        {"x": 0.50, "y": 0.30, "visibility": 1.0},
        "RIGHT_KNEE":       {"x": 0.50, "y": 0.55, "visibility": 1.0},
        "RIGHT_ANKLE":      {"x": 0.50, "y": 0.80, "visibility": 1.0},
        "RIGHT_HEEL":       {"x": 0.48, "y": 0.81, "visibility": 1.0},
        "RIGHT_FOOT_INDEX": {"x": 0.56, "y": 0.78, "visibility": 1.0},
        # The other side & trunk landmarks are required by some methods
        "LEFT_HIP":         {"x": 0.50, "y": 0.30, "visibility": 1.0},
        "LEFT_KNEE":        {"x": 0.50, "y": 0.55, "visibility": 1.0},
        "LEFT_ANKLE":       {"x": 0.50, "y": 0.80, "visibility": 1.0},
        "LEFT_HEEL":        {"x": 0.48, "y": 0.81, "visibility": 1.0},
        "LEFT_FOOT_INDEX":  {"x": 0.56, "y": 0.78, "visibility": 1.0},
        "LEFT_SHOULDER":    {"x": 0.45, "y": 0.10, "visibility": 1.0},
        "RIGHT_SHOULDER":   {"x": 0.55, "y": 0.10, "visibility": 1.0},
    }


# ── _pixelify_frame ──────────────────────────────────────────────────


class TestPixelifyFrame:
    def test_no_op_for_unit_scale(self):
        lm = {"NOSE": {"x": 0.5, "y": 0.4, "visibility": 1.0}}
        frame = _frame_with_landmarks(lm)
        out = _pixelify_frame(frame, 1.0, 1.0)
        # Should return the input frame unchanged
        assert out is frame

    def test_scales_by_width_height(self):
        lm = {"NOSE": {"x": 0.5, "y": 0.4, "visibility": 1.0}}
        frame = _frame_with_landmarks(lm)
        out = _pixelify_frame(frame, 1080.0, 1920.0)
        assert out is not frame
        assert out["landmarks"]["NOSE"]["x"] == pytest.approx(540.0)
        assert out["landmarks"]["NOSE"]["y"] == pytest.approx(768.0)
        # Visibility is preserved
        assert out["landmarks"]["NOSE"]["visibility"] == 1.0

    def test_does_not_mutate_input(self):
        lm = {"NOSE": {"x": 0.5, "y": 0.4, "visibility": 1.0}}
        frame = _frame_with_landmarks(lm)
        _ = _pixelify_frame(frame, 1080.0, 1920.0)
        # Original landmark untouched
        assert frame["landmarks"]["NOSE"]["x"] == 0.5
        assert frame["landmarks"]["NOSE"]["y"] == 0.4

    def test_preserves_top_level_fields(self):
        lm = {"NOSE": {"x": 0.5, "y": 0.4, "visibility": 1.0}}
        frame = _frame_with_landmarks(lm, frame_idx=42)
        out = _pixelify_frame(frame, 800.0, 600.0)
        assert out["frame_idx"] == 42
        assert out["time_s"] == frame["time_s"]
        assert out["confidence"] == frame["confidence"]


# ── compute_angles aspect-ratio behaviour ────────────────────────────


class TestComputeAnglesAspectRatio:
    def test_square_image_is_unchanged(self):
        """For a square image, the fix is a no-op."""
        frames = [_frame_with_landmarks(_ankle_dorsiflexed_landmarks_normalised())]
        # Square image: width == height
        d_on = _make_min_data(frames, 1000, 1000)
        d_off = _make_min_data([f for f in frames], 1000, 1000)
        out_on = compute_angles(d_on, apply_aspect_ratio=True,
                                  correction_factor=1.0,
                                  calibrate=False,
                                  correct_ankle_sliding=False)
        out_off = compute_angles(d_off, apply_aspect_ratio=False,
                                   correction_factor=1.0,
                                   calibrate=False,
                                   correct_ankle_sliding=False)
        a_on = out_on["angles"]["frames"][0]["ankle_R"]
        a_off = out_off["angles"]["frames"][0]["ankle_R"]
        assert a_on == pytest.approx(a_off, abs=1e-9)

    def test_portrait_image_changes_ankle(self):
        """For a portrait image the ankle angle differs when the
        aspect-ratio correction is applied vs not."""
        frames = [_frame_with_landmarks(_ankle_dorsiflexed_landmarks_normalised())]
        # Portrait image: width != height
        d_on = _make_min_data([dict(f) for f in frames], 1080, 1920)
        d_off = _make_min_data([dict(f) for f in frames], 1080, 1920)
        out_on = compute_angles(d_on, apply_aspect_ratio=True,
                                  correction_factor=1.0,
                                  calibrate=False,
                                  correct_ankle_sliding=False)
        out_off = compute_angles(d_off, apply_aspect_ratio=False,
                                   correction_factor=1.0,
                                   calibrate=False,
                                   correct_ankle_sliding=False)
        a_on = out_on["angles"]["frames"][0]["ankle_R"]
        a_off = out_off["angles"]["frames"][0]["ankle_R"]
        # The aspect-ratio correction should change the ankle angle by
        # at least a few degrees on a 1080x1920 image with a clear
        # foot tilt — the bias is largest precisely for the ankle.
        assert not math.isnan(a_on)
        assert not math.isnan(a_off)
        assert abs(a_on - a_off) > 1.0, (
            f"Aspect-ratio correction should change the ankle angle on a "
            f"non-square image: on={a_on:.2f}, off={a_off:.2f}"
        )

    def test_corrected_matches_pixel_space(self):
        """The corrected ankle angle must equal what we get by computing
        the angle directly in pixel space."""
        landmarks = _ankle_dorsiflexed_landmarks_normalised()
        frame = _frame_with_landmarks(landmarks)
        W, H = 1080.0, 1920.0
        d = _make_min_data([frame], W, H)
        out = compute_angles(d, apply_aspect_ratio=True,
                              correction_factor=1.0,
                              calibrate=False,
                              correct_ankle_sliding=False)
        a_corrected = out["angles"]["frames"][0]["ankle_R"]

        # Reference: compute the same angle directly in pixel space
        lm = landmarks
        knee = np.array([lm["RIGHT_KNEE"]["x"] * W, lm["RIGHT_KNEE"]["y"] * H])
        ankle = np.array([lm["RIGHT_ANKLE"]["x"] * W, lm["RIGHT_ANKLE"]["y"] * H])
        heel = np.array([lm["RIGHT_HEEL"]["x"] * W, lm["RIGHT_HEEL"]["y"] * H])
        foot = np.array([lm["RIGHT_FOOT_INDEX"]["x"] * W,
                          lm["RIGHT_FOOT_INDEX"]["y"] * H])
        shank_dir = knee - ankle
        foot_dir = foot - heel
        unsigned = _angle_between(shank_dir, foot_dir)
        a_reference = 90.0 - unsigned

        assert a_corrected == pytest.approx(a_reference, abs=1e-6)
