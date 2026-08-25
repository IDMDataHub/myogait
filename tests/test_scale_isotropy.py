"""Regression tests for isotropic pixel→metre scaling (v0.8.2).

Landmarks are normalised per axis (x / width, y / height). The metric
scale is derived mostly from the (vertical) femur, but step / stride
length is a (horizontal) antero-posterior distance. On a non-square
frame the per-axis normalisation makes one x-unit and one y-unit span
different real distances, so the scale must de-normalise to source
pixels to stay isotropic.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myogait.analysis import _estimate_pixel_to_meter_scale


def _vertical_femur_frames(femur_norm_y, n=5):
    """Frames with a purely vertical femur of the given normalised length."""
    return [
        {
            "landmarks": {
                "LEFT_HIP": {"x": 0.5, "y": 0.4},
                "LEFT_KNEE": {"x": 0.5, "y": 0.4 + femur_norm_y},
            }
        }
        for _ in range(n)
    ]


def test_scale_is_metres_per_source_pixel():
    """A vertical femur on a 1920×1080 frame yields metres-per-pixel."""
    height_m = 1.75
    femur_m = 0.245 * height_m
    frames = _vertical_femur_frames(0.30)  # 0.30·1080 = 324 px
    scale = _estimate_pixel_to_meter_scale(
        frames, height_m=height_m, width=1920, height=1080
    )
    assert scale == pytest.approx(femur_m / (0.30 * 1080))


def test_horizontal_distance_isotropic_on_landscape():
    """The same real step measured horizontally recovers its true length.

    A horizontal displacement of 0.10 x-units on a 1920-wide frame spans
    0.10·1920 = 192 source pixels; multiplied by the femur scale it must
    give the true metric distance — the pre-0.8.2 bug under-estimated it
    by the frame aspect ratio.
    """
    height_m = 1.75
    femur_m = 0.245 * height_m
    frames = _vertical_femur_frames(0.30)
    scale = _estimate_pixel_to_meter_scale(
        frames, height_m=height_m, width=1920, height=1080
    )
    step_px = 0.10 * 1920
    dist_m = step_px * scale
    expected = (0.10 * 1920) * femur_m / (0.30 * 1080)
    assert dist_m == pytest.approx(expected)

    # The pre-0.8.2 code applied the femur scale (derived on unit
    # dimensions) directly to the normalised x-displacement, collapsing
    # the horizontal axis by the frame aspect ratio. The fix must restore
    # exactly that factor (1920 / 1080).
    scale_buggy = _estimate_pixel_to_meter_scale(frames, height_m=height_m)
    dist_buggy = 0.10 * scale_buggy
    assert dist_m / dist_buggy == pytest.approx(1920 / 1080)


def test_unit_dimensions_preserve_legacy_behaviour():
    """width = height = 1 reproduces the historical normalised-unit scale."""
    height_m = 1.75
    femur_m = 0.245 * height_m
    frames = _vertical_femur_frames(0.30)
    scale = _estimate_pixel_to_meter_scale(frames, height_m=height_m)
    assert scale == pytest.approx(femur_m / 0.30)
