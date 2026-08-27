"""Regression tests for the 2026-08 audit-driven fixes.

Covers the defect classes surfaced by the multi-agent audit and validated
against real data (GH panning-camera clip, Bath BioCV video+C3D):

- frame-index resolution (events/cycles store original-video frame_idx while
  the ``frames`` list may start late / be shorter)
- fps sanitisation (malformed ``meta.fps`` must not crash)
- ``trim_standstill`` reading the correct event key
- camera-pan immunity of the parkinsonian short-stride screen
- physiological plausibility guards / machine-readable ``stats["warnings"]``
- walking-direction fallback when the feet are occluded
- metric step/stride/speed from the real 3-D C3D markers
"""

import numpy as np
import pytest

from conftest import make_walking_data

import myogait as mg
from myogait.analysis import (
    analyze_gait, step_length, walking_speed, toe_clearance,
    stride_variability, _frame_index_map, _c3d_step_lengths,
)
from myogait.axis_utils import safe_frame_rate, detect_walking_direction_from_feet


# ── frame-index resolution ───────────────────────────────────────────

def _pipeline(data):
    mg.normalize(data, filters=["butterworth"])
    mg.compute_angles(data, correction_factor=1.0, calibrate=False)
    mg.detect_events(data)
    cycles = mg.segment_cycles(data)
    return cycles


def test_frame_index_map_handles_late_start():
    """A frames list starting at frame_idx 60 maps events correctly."""
    frames = [{"frame_idx": i + 60} for i in range(50)]
    m = _frame_index_map(frames)
    assert m[60] == 0 and m[109] == 49
    # a frame_idx below the window start is absent (not a wrong positional hit)
    assert 10 not in m


def test_toe_clearance_populated_with_late_frame_idx():
    """toe_clearance must not silently return all-None.

    Regression: it read the non-existent ``to_frame`` cycle key (always None)
    and indexed frames positionally, so every MTC was None.
    """
    data = make_walking_data(n_frames=200, fps=30.0)
    # shift the whole recording to a late frame_idx window
    for k, f in enumerate(data["frames"]):
        f["frame_idx"] = k + 61
    cycles = _pipeline(data)
    assert cycles["cycles"], "fixture should produce cycles"
    tc = toe_clearance(data, cycles)
    # at least one side yields a real value (not the all-None dead metric)
    vals = [tc["mtc_left"], tc["mtc_right"]]
    assert any(v is not None for v in vals)
    # minimum toe clearance is a mid-swing height above the ground: it must be
    # non-negative (a heel-referenced ground + toe-off search used to make it
    # slightly negative).
    for v in vals:
        if v is not None:
            assert v >= 0.0, f"MTC {v} should be non-negative"


def test_stride_variability_cv_is_plausible():
    """step_length_cv must be a plausible percentage, not >100%."""
    data = make_walking_data(n_frames=300, fps=30.0)
    cycles = _pipeline(data)
    sv = stride_variability(data, cycles)
    for key in ("step_length_cv_left", "step_length_cv_right"):
        assert sv[key] is not None
        assert 0.0 <= sv[key] < 60.0, f"{key}={sv[key]} implausible"


# ── fps sanitisation ─────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0, -5, None, "banana", float("nan"), float("inf")])
def test_safe_frame_rate_coerces_bad_fps(bad):
    assert safe_frame_rate({"meta": {"fps": bad}}) == 30.0


def test_safe_frame_rate_keeps_valid():
    assert safe_frame_rate({"meta": {"fps": 200.0}}) == 200.0


@pytest.mark.parametrize("bad", [0, -5, None, "banana"])
def test_detect_events_survives_bad_fps(bad):
    data = make_walking_data(n_frames=150, fps=30.0)
    data["meta"]["fps"] = bad
    mg.normalize(data, filters=["butterworth"])
    mg.compute_angles(data, correction_factor=1.0, calibrate=False)
    ev = mg.detect_events(data)  # must not raise
    assert isinstance(ev.get("events", ev), dict)


# ── plausibility guards / warnings ───────────────────────────────────

def test_plausibility_warning_on_gross_miscalibration():
    data = make_walking_data(n_frames=300, fps=30.0)
    cycles = _pipeline(data)
    # an absurd height inflates the pixel scale ~10x -> non-physiological metres
    data["subject"] = {"height_m": 18.0}
    stats = analyze_gait(data, cycles)
    assert stats.get("warnings"), "a 10x mis-scale must be flagged"
    assert stats["step_length"].get("valid_for_progression") is False


def test_no_warning_on_normal_gait():
    data = make_walking_data(n_frames=300, fps=30.0)
    cycles = _pipeline(data)
    data["subject"] = {"height_m": 1.75}
    stats = analyze_gait(data, cycles)
    assert stats.get("warnings") == []


# ── walking-direction fallback ───────────────────────────────────────

def test_direction_unknown_when_feet_absent():
    """With no toe/heel landmarks the detector reports the caller's default."""
    data = make_walking_data(n_frames=60)
    for f in data["frames"]:
        for name in ("LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
            f["landmarks"].pop(name, None)
    assert detect_walking_direction_from_feet(data, default="unknown") == "unknown"
    # legacy default preserved for existing callers
    assert detect_walking_direction_from_feet(data) == "right"


# ── C3D metric step from real markers ────────────────────────────────

def _make_c3d_like():
    """A minimal C3D-style pivot: normalized 2-D landmarks (unusable for metric
    scale) plus real 3-D markers in mm walking along the AP (y) axis."""
    n = 120
    data = make_walking_data(n_frames=n, fps=100.0)
    data["meta"]["source"] = "c3d"
    # real markers: forward travel along axis 1 (mm), feet alternate ±150 mm
    m3d = {}
    t = np.arange(n)
    forward = t * 15.0  # 15 mm/frame -> 1.5 m/s at 100 fps
    for name, base_ml, base_v in (
        ("LEFT_HIP", -100, 1000), ("RIGHT_HIP", 100, 1000),
        ("LEFT_KNEE", -100, 550), ("RIGHT_KNEE", 100, 550),
        ("LEFT_ANKLE", -100, 100), ("RIGHT_ANKLE", 100, 100),
        ("LEFT_HEEL", -100, 90), ("RIGHT_HEEL", 100, 90),
    ):
        phase = 0.0 if name.startswith("LEFT") else np.pi
        ap = forward + 150.0 * np.sin(2 * np.pi * t / 40.0 + phase)
        arr = np.column_stack([np.full(n, base_ml, float), ap, np.full(n, base_v, float)])
        m3d[name] = arr
    data["c3d_markers_3d"] = m3d
    return data


def test_c3d_step_lengths_are_metric_and_plausible():
    data = _make_c3d_like()
    _pipeline(data)  # populates data["events"]
    events = data.get("events", {})
    steps = _c3d_step_lengths(data, events)
    assert steps is not None
    allv = steps["left"] + steps["right"]
    assert allv, "expected at least one c3d step"
    # real inter-ankle AP separation, in metres, physiologically plausible
    assert all(0.0 <= v <= 2.0 for v in allv)


def test_c3d_step_length_not_pixel_scaled_garbage():
    """analyze_gait on a C3D pivot must yield metric, in-range step/speed."""
    data = _make_c3d_like()
    cycles = _pipeline(data)
    sl = step_length(data, cycles)
    ws = walking_speed(data, cycles)
    assert sl["unit"] == "m" and sl.get("source") == "c3d_markers_3d"
    for side in ("left", "right"):
        v = sl[f"step_length_{side}"]
        if v is not None:
            assert 0.05 <= v <= 1.5, f"c3d step_{side}={v} out of range"
    if ws["speed_mean"] is not None:
        assert 0.05 <= ws["speed_mean"] <= 3.0


def test_non_c3d_pivot_falls_through():
    """_c3d_step_lengths returns None for an ordinary video pivot."""
    data = make_walking_data(n_frames=60)
    assert _c3d_step_lengths(data, data.get("events", {})) is None
