"""Pixel-to-metre scale must calibrate over the whole recording.

Regression guard for the subject-following / panning bug: when the subject is
still entering the frame (far, small) in the opening frames, calibrating the
femur from only the first 60 frames under-measures the segment and inflates the
metre scale, so step/stride came out ~1.5x too long vs Vicon. The scale must be
taken from the full recording instead.
"""
from __future__ import annotations

from myogait.analysis import _estimate_pixel_to_meter_scale


def _frames_with_femur(px_by_frame, width=1000.0, height=1000.0):
    """Build minimal frames whose LEFT_HIP->LEFT_KNEE spans a given pixel length.

    The segment is vertical, so its source-pixel length is ``dy * height``.
    """
    frames = []
    for px in px_by_frame:
        dy = px / height
        frames.append({"landmarks": {
            "LEFT_HIP": {"x": 0.5, "y": 0.4},
            "LEFT_KNEE": {"x": 0.5, "y": 0.4 + dy},
        }})
    return frames, width, height


def test_scale_uses_whole_recording_not_first_60_frames():
    # First 60 frames: subject far (femur 100 px). Remaining 60: full size (200 px).
    frames, w, h = _frames_with_femur([100.0] * 60 + [200.0] * 60)
    femur_mm = 400.0

    scale = _estimate_pixel_to_meter_scale(frames, femur_mm=femur_mm, width=w, height=h)

    # Whole-recording median femur is 150 px -> 0.4 m / 150 px.
    expected_full = (femur_mm / 1000.0) / 150.0
    first60_only = (femur_mm / 1000.0) / 100.0  # the old, inflated behaviour
    assert abs(scale - expected_full) < 1e-6
    assert scale < first60_only  # not the inflated first-60 scale


def test_scale_stable_when_subject_full_size_throughout():
    # Fixed-camera case: same femur everywhere -> unchanged by the fix.
    frames, w, h = _frames_with_femur([180.0] * 120)
    scale = _estimate_pixel_to_meter_scale(frames, femur_mm=400.0, width=w, height=h)
    assert abs(scale - (0.400 / 180.0)) < 1e-6
