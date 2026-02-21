"""Tests for gaitkit integration in myogait event detection.

All tests in this module require the optional ``gaitkit`` package.
They will be automatically skipped if gaitkit is not installed.
"""

import pytest
import numpy as np

gaitkit = pytest.importorskip("gaitkit")

from conftest import make_walking_data, walking_data_with_angles, run_full_pipeline
from myogait.events import (
    _convert_to_gaitkit_frames,
    _detect_gaitkit,
    _detect_gaitkit_ensemble,
    _detect_gaitkit_structured,
    _gaitkit_result_to_myogait,
    _is_gaitkit_available,
    _GAITKIT_METHODS,
    detect_events,
    list_event_methods,
    event_consensus,
    EVENT_METHODS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def walking_data():
    """Walking data with 10 seconds at 30 fps."""
    return make_walking_data(300, fps=30.0)


@pytest.fixture
def walking_data_angles():
    """Walking data with angles computed."""
    return walking_data_with_angles(300, fps=30.0)


# ---------------------------------------------------------------------------
# Tests: _is_gaitkit_available
# ---------------------------------------------------------------------------


def test_gaitkit_is_available():
    """gaitkit should be importable (enforced by importorskip)."""
    assert _is_gaitkit_available() is True


# ---------------------------------------------------------------------------
# Tests: _convert_to_gaitkit_frames
# ---------------------------------------------------------------------------


def test_convert_to_gaitkit_frames_basic(walking_data):
    """Conversion should produce list of dicts with gaitkit keys."""
    gk_frames = _convert_to_gaitkit_frames(walking_data)
    assert isinstance(gk_frames, list)
    assert len(gk_frames) == len(walking_data["frames"])


def test_convert_to_gaitkit_frames_field_names(walking_data):
    """Each converted frame should have gaitkit-required field names."""
    gk_frames = _convert_to_gaitkit_frames(walking_data)
    required_keys = [
        "frame_index",
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_ankle_angle", "right_ankle_angle",
        "landmark_positions",
    ]
    for frame in gk_frames[:5]:  # check first 5 frames
        for key in required_keys:
            assert key in frame, f"Missing key: {key}"


def test_convert_to_gaitkit_frames_frame_index(walking_data):
    """frame_index should map from myogait frame_idx."""
    gk_frames = _convert_to_gaitkit_frames(walking_data)
    for i, frame in enumerate(gk_frames):
        assert frame["frame_index"] == i


def test_convert_to_gaitkit_frames_landmarks(walking_data):
    """Landmark positions should be tuples of (x, y, z)."""
    gk_frames = _convert_to_gaitkit_frames(walking_data)
    for frame in gk_frames[:5]:
        lp = frame["landmark_positions"]
        assert isinstance(lp, dict)
        # Should have at least ankle landmarks from raw frames
        for name in ["left_ankle", "right_ankle", "left_hip", "right_hip"]:
            if name in lp:
                coords = lp[name]
                assert len(coords) == 3, f"Expected 3 coords for {name}"
                assert isinstance(coords[0], float)
                assert isinstance(coords[1], float)
                assert isinstance(coords[2], float)


def test_convert_to_gaitkit_frames_with_angles(walking_data_angles):
    """When angles are present, angle values should be populated."""
    gk_frames = _convert_to_gaitkit_frames(walking_data_angles)
    # With angles, hip values should be non-zero for walking data
    has_nonzero = False
    for frame in gk_frames:
        if frame["left_hip_angle"] != 0.0 or frame["right_hip_angle"] != 0.0:
            has_nonzero = True
            break
    assert has_nonzero, "Expected non-zero angle values when angles are computed"


def test_convert_to_gaitkit_frames_angle_mapping(walking_data_angles):
    """Verify the field name mapping from myogait to gaitkit."""
    gk_frames = _convert_to_gaitkit_frames(walking_data_angles)
    angle_frames = walking_data_angles["angles"]["frames"]
    # Check first non-NaN frame
    for i in range(min(10, len(gk_frames))):
        af = angle_frames[i]
        gf = gk_frames[i]
        if af.get("hip_L") is not None and not np.isnan(af.get("hip_L", float("nan"))):
            assert abs(gf["left_hip_angle"] - float(af["hip_L"])) < 1e-6
        if af.get("knee_R") is not None and not np.isnan(af.get("knee_R", float("nan"))):
            assert abs(gf["right_knee_angle"] - float(af["knee_R"])) < 1e-6


# ---------------------------------------------------------------------------
# Tests: _detect_gaitkit (individual methods)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bike", "zeni", "oconnor"])
def test_detect_gaitkit_basic(walking_data, method):
    """gaitkit detection should return dict with required event keys."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit(walking_data, fps, method=method)
    assert isinstance(events, dict)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        assert key in events
        assert isinstance(events[key], list)


@pytest.mark.parametrize("method", ["bike", "zeni", "oconnor"])
def test_detect_gaitkit_event_format(walking_data, method):
    """Each event should have frame, time, and confidence."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit(walking_data, fps, method=method)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in events[key]:
            assert "frame" in ev, f"Missing 'frame' in {key} event"
            assert "time" in ev, f"Missing 'time' in {key} event"
            assert "confidence" in ev, f"Missing 'confidence' in {key} event"
            assert isinstance(ev["frame"], int)
            assert isinstance(ev["time"], float)
            assert isinstance(ev["confidence"], float)


@pytest.mark.parametrize("method", ["bike", "zeni", "oconnor"])
def test_detect_gaitkit_detects_events(walking_data, method):
    """gaitkit should detect at least some events on walking data."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit(walking_data, fps, method=method)
    n_events = sum(len(v) for v in events.values())
    assert n_events > 0, f"gaitkit method '{method}' detected no events"


@pytest.mark.parametrize("method", ["bike", "zeni", "oconnor"])
def test_detect_gaitkit_confidence_range(walking_data, method):
    """Confidence scores should be between 0 and 1."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit(walking_data, fps, method=method)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in events[key]:
            assert 0.0 <= ev["confidence"] <= 1.0, (
                f"Confidence {ev['confidence']} out of range in {key}"
            )


# Test all 10 gaitkit methods
@pytest.mark.parametrize("method", _GAITKIT_METHODS)
def test_detect_gaitkit_all_methods(walking_data, method):
    """All 10 gaitkit methods should run without error."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit(walking_data, fps, method=method)
    assert isinstance(events, dict)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        assert key in events


# ---------------------------------------------------------------------------
# Tests: gk_ensemble
# ---------------------------------------------------------------------------


def test_detect_gaitkit_ensemble_basic(walking_data):
    """Ensemble detection should return standard event format."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit_ensemble(walking_data, fps)
    assert isinstance(events, dict)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        assert key in events
        assert isinstance(events[key], list)


def test_detect_gaitkit_ensemble_event_format(walking_data):
    """Ensemble events should have frame, time, and confidence."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit_ensemble(walking_data, fps)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in events[key]:
            assert "frame" in ev
            assert "time" in ev
            assert "confidence" in ev


def test_detect_gaitkit_ensemble_custom_methods(walking_data):
    """Ensemble should accept custom method list."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit_ensemble(
        walking_data, fps,
        methods=["bike", "zeni"],
        min_votes=1,
    )
    assert isinstance(events, dict)


# ---------------------------------------------------------------------------
# Tests: detect_events() with gk_* methods
# ---------------------------------------------------------------------------


def test_detect_events_gk_bike(walking_data):
    """detect_events(method='gk_bike') should work end-to-end."""
    data = detect_events(walking_data, method="gk_bike")
    assert "events" in data
    assert data["events"]["method"] == "gk_bike"
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        assert key in data["events"]


def test_detect_events_gk_zeni(walking_data):
    """detect_events(method='gk_zeni') should work end-to-end."""
    data = detect_events(walking_data, method="gk_zeni")
    assert "events" in data
    assert data["events"]["method"] == "gk_zeni"


def test_detect_events_gk_oconnor(walking_data):
    """detect_events(method='gk_oconnor') should work end-to-end."""
    data = detect_events(walking_data, method="gk_oconnor")
    assert "events" in data
    assert data["events"]["method"] == "gk_oconnor"


def test_detect_events_gk_ensemble(walking_data):
    """detect_events(method='gk_ensemble') should work end-to-end."""
    data = detect_events(walking_data, method="gk_ensemble")
    assert "events" in data
    assert data["events"]["method"] == "gk_ensemble"


@pytest.mark.parametrize("method", [f"gk_{m}" for m in _GAITKIT_METHODS])
def test_detect_events_all_gk_methods(walking_data, method):
    """detect_events should work with all gk_* method names."""
    data = detect_events(walking_data, method=method)
    assert "events" in data
    assert data["events"]["method"] == method


def test_detect_events_gk_returns_valid_events(walking_data):
    """gk_bike events should be valid (frame in range, time positive)."""
    data = detect_events(walking_data, method="gk_bike")
    n_frames = len(walking_data["frames"])
    fps = walking_data["meta"]["fps"]
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert 0 <= ev["frame"] < n_frames
            assert ev["time"] >= 0.0
            assert ev["time"] <= n_frames / fps + 0.1


# ---------------------------------------------------------------------------
# Tests: list_event_methods includes gaitkit
# ---------------------------------------------------------------------------


def test_list_event_methods_includes_gaitkit():
    """When gaitkit is installed, list_event_methods should include gk_* methods."""
    methods = list_event_methods()
    for gk_method in _GAITKIT_METHODS:
        assert f"gk_{gk_method}" in methods, f"gk_{gk_method} not in list_event_methods()"
    assert "gk_ensemble" in methods


def test_list_event_methods_includes_builtin():
    """list_event_methods should still include built-in methods."""
    methods = list_event_methods()
    for builtin in ["zeni", "crossing", "velocity", "oconnor"]:
        assert builtin in methods


# ---------------------------------------------------------------------------
# Tests: backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat_zeni(walking_data):
    """Built-in 'zeni' method should still work as before."""
    data = detect_events(walking_data, method="zeni")
    assert data["events"]["method"] == "zeni"
    n_events = sum(
        len(data["events"][k])
        for k in ["left_hs", "right_hs", "left_to", "right_to"]
    )
    assert n_events > 0


def test_backward_compat_oconnor(walking_data):
    """Built-in 'oconnor' method should still work."""
    data = detect_events(walking_data, method="oconnor")
    assert data["events"]["method"] == "oconnor"
    n_events = sum(
        len(data["events"][k])
        for k in ["left_hs", "right_hs", "left_to", "right_to"]
    )
    assert n_events > 0


def test_backward_compat_velocity(walking_data):
    """Built-in 'velocity' method should still work."""
    data = detect_events(walking_data, method="velocity")
    assert data["events"]["method"] == "velocity"


def test_backward_compat_crossing(walking_data):
    """Built-in 'crossing' method should still work."""
    data = detect_events(walking_data, method="crossing")
    assert data["events"]["method"] == "crossing"


def test_backward_compat_consensus(walking_data):
    """event_consensus with built-in methods should still work."""
    result = event_consensus(walking_data, methods=["zeni", "oconnor", "crossing"])
    assert result["events"]["method"] == "consensus"


def test_backward_compat_event_format(walking_data):
    """Events from built-in methods should have same format as before."""
    data = detect_events(walking_data, method="zeni")
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert "frame" in ev
            assert "time" in ev
            assert "confidence" in ev
            assert isinstance(ev["frame"], int)


def test_backward_compat_unknown_method(walking_data):
    """Unknown method should still raise ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        detect_events(walking_data, method="nonexistent_method")


# ---------------------------------------------------------------------------
# Tests: event_consensus with gaitkit methods
# ---------------------------------------------------------------------------


def test_event_consensus_with_gaitkit_methods(walking_data):
    """event_consensus should work with gk_* methods."""
    result = event_consensus(
        walking_data,
        methods=["gk_bike", "gk_zeni", "gk_oconnor"],
    )
    assert result["events"]["method"] == "consensus"
    assert result["events"]["methods_used"] == ["gk_bike", "gk_zeni", "gk_oconnor"]


def test_event_consensus_mixed_methods(walking_data):
    """event_consensus should work with mix of built-in and gaitkit methods."""
    result = event_consensus(
        walking_data,
        methods=["zeni", "gk_bike", "oconnor"],
    )
    assert result["events"]["method"] == "consensus"
    assert result["events"]["n_methods"] == 3


# ---------------------------------------------------------------------------
# Tests: confidence scores from gaitkit
# ---------------------------------------------------------------------------


def test_gaitkit_events_have_confidence(walking_data):
    """Events from gaitkit should include confidence scores."""
    data = detect_events(walking_data, method="gk_bike")
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert "confidence" in ev
            assert isinstance(ev["confidence"], float)
            assert 0.0 <= ev["confidence"] <= 1.0


def test_gaitkit_ensemble_confidence(walking_data):
    """Ensemble events should have confidence reflecting agreement."""
    data = detect_events(walking_data, method="gk_ensemble")
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert "confidence" in ev
            assert isinstance(ev["confidence"], float)


# ---------------------------------------------------------------------------
# Tests: _detect_gaitkit_structured
# ---------------------------------------------------------------------------


def test_detect_gaitkit_structured_basic(walking_data):
    """Structured API should return standard myogait event format."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit_structured(walking_data, fps, method="bike")
    assert isinstance(events, dict)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        assert key in events
        assert isinstance(events[key], list)


def test_detect_gaitkit_structured_event_format(walking_data):
    """Structured API events should have standard fields."""
    fps = walking_data["meta"]["fps"]
    events = _detect_gaitkit_structured(walking_data, fps, method="bike")
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in events[key]:
            assert "frame" in ev
            assert "time" in ev
            assert "confidence" in ev


# ---------------------------------------------------------------------------
# Tests: gk_* methods in EVENT_METHODS registry
# ---------------------------------------------------------------------------


def test_gk_methods_registered():
    """All gk_* methods should be in EVENT_METHODS when gaitkit is available."""
    for gk_method in _GAITKIT_METHODS:
        name = f"gk_{gk_method}"
        assert name in EVENT_METHODS, f"{name} not registered in EVENT_METHODS"
    assert "gk_ensemble" in EVENT_METHODS


def test_gk_registered_methods_are_callable():
    """Registered gk_* methods should be callable."""
    for gk_method in _GAITKIT_METHODS:
        name = f"gk_{gk_method}"
        assert callable(EVENT_METHODS[name])
    assert callable(EVENT_METHODS["gk_ensemble"])


# ---------------------------------------------------------------------------
# Tests: full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_with_gk_bike():
    """Full pipeline should work with gk_bike for event detection."""
    from myogait import normalize, compute_angles, detect_events, segment_cycles
    data = make_walking_data(300, fps=30.0)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data, method="gk_bike")
    assert "events" in data
    assert data["events"]["method"] == "gk_bike"
    # Should still be able to segment cycles
    cycles = segment_cycles(data)
    assert isinstance(cycles, list)
