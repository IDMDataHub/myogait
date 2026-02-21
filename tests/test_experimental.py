"""Tests for experimental video degradation utilities."""

import numpy as np
import pytest

from myogait.experimental import (
    VIDEO_DEGRADATION_DEFAULTS,
    build_video_degradation_config,
    is_video_degradation_active,
    compute_fps_sampling,
    degraded_resolution,
    apply_video_degradation,
)


def test_defaults_are_neutral():
    cfg = build_video_degradation_config(VIDEO_DEGRADATION_DEFAULTS)
    assert is_video_degradation_active(cfg) is False


def test_active_when_parameter_changes():
    cfg = build_video_degradation_config({"downscale": 0.7})
    assert is_video_degradation_active(cfg) is True


def test_compute_fps_sampling_downsamples():
    stride, fps = compute_fps_sampling(30.0, 10.0)
    assert stride == 3
    assert fps == pytest.approx(10.0, abs=1e-6)


def test_degraded_resolution_changes_size():
    cfg = build_video_degradation_config({"downscale": 0.5, "aspect_ratio": 1.2})
    w, h = degraded_resolution(640, 480, cfg)
    assert w == 384
    assert h == 240


def test_apply_video_degradation_noop_when_neutral():
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    cfg = build_video_degradation_config({})
    out = apply_video_degradation(frame, cfg)
    assert out.shape == frame.shape
    np.testing.assert_array_equal(out, frame)


def test_apply_video_degradation_changes_pixels_and_shape():
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[:, :80] = 100
    cfg = build_video_degradation_config(
        {
            "downscale": 0.5,
            "contrast": 0.6,
            "aspect_ratio": 1.25,
            "perspective_x": 0.25,
            "perspective_y": 0.15,
        }
    )
    out = apply_video_degradation(frame, cfg)
    assert out.shape == (60, 100, 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame[:60, :100])


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        build_video_degradation_config({"downscale": 1.5})
    with pytest.raises(ValueError):
        build_video_degradation_config({"contrast": 0.0})
    with pytest.raises(ValueError):
        build_video_degradation_config({"perspective_x": 1.5})
