"""Tests for enhanced event detection functions."""

import pytest
import numpy as np
from conftest import make_walking_data, walking_data_with_angles


def test_event_consensus_runs():
    """event_consensus should run without error on walking data."""
    from myogait.events import event_consensus
    data = make_walking_data(300, fps=30.0)
    result = event_consensus(data)
    assert "events" in result
    assert result["events"]["method"] == "consensus"


def test_event_consensus_format():
    """Consensus events should have standard format (frame, time, confidence)."""
    from myogait.events import event_consensus
    data = make_walking_data(300, fps=30.0)
    event_consensus(data)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert "frame" in ev
            assert "time" in ev
            assert "confidence" in ev
            assert isinstance(ev["frame"], int)
            assert 0 <= ev["confidence"] <= 1.0


def test_event_consensus_confidence():
    """Consensus events should have confidence reflecting agreement."""
    from myogait.events import event_consensus
    data = make_walking_data(300, fps=30.0)
    event_consensus(data, methods=["zeni", "oconnor", "crossing"])
    # Confidence should be fraction of methods that agreed
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            # With 3 methods, majority = 2, so min confidence = 2/3
            assert ev["confidence"] >= 2 / 3 - 0.01


def test_event_consensus_custom_methods():
    """Should accept custom method list."""
    from myogait.events import event_consensus
    data = make_walking_data(300, fps=30.0)
    event_consensus(data, methods=["zeni", "velocity"])
    assert data["events"]["methods_used"] == ["zeni", "velocity"]
    assert data["events"]["n_methods"] == 2


def test_validate_events_valid_data():
    """Walking data should produce mostly valid events."""
    from myogait.events import detect_events, validate_events
    data = make_walking_data(300, fps=30.0)
    detect_events(data)
    report = validate_events(data)
    # Walking data should have at least some valid cycles
    assert report["n_valid_cycles_left"] >= 1 or report["n_valid_cycles_right"] >= 1


def test_validate_events_structure():
    """validate_events should return dict with valid, issues keys."""
    from myogait.events import detect_events, validate_events
    data = make_walking_data(300, fps=30.0)
    detect_events(data)
    report = validate_events(data)
    assert "valid" in report
    assert "issues" in report
    assert "n_valid_cycles_left" in report
    assert "n_valid_cycles_right" in report
    assert isinstance(report["valid"], bool)
    assert isinstance(report["issues"], list)
    assert isinstance(report["n_valid_cycles_left"], int)
    assert isinstance(report["n_valid_cycles_right"], int)


def test_validate_events_empty():
    """Data with no events should report issues."""
    from myogait.events import validate_events
    data = make_walking_data(300, fps=30.0)
    # No events detected — events key is None
    report = validate_events(data)
    assert report["valid"] is False
    assert len(report["issues"]) > 0
    assert "No events detected" in report["issues"]


def test_detect_events_adaptive():
    """detect_events with adaptive=True should work."""
    from myogait.events import detect_events
    data = make_walking_data(300, fps=30.0)
    detect_events(data, adaptive=True)
    assert "events" in data
    assert data["events"]["method"] == "zeni"
    # Should still detect events
    n_events = sum(
        len(data["events"][k])
        for k in ["left_hs", "right_hs", "left_to", "right_to"]
    )
    assert n_events > 0


def test_detect_events_adaptive_adjusts_params():
    """Adaptive mode should adjust parameters based on speed."""
    from myogait.events import _adaptive_params
    # make_walking_data has stationary hips (hip_x = 0.50 constant)
    # So displacement rate should be very low → slow walk parameters
    data = make_walking_data(300, fps=30.0)
    frames = data["frames"]
    fps = 30.0
    min_cycle, cutoff = _adaptive_params(frames, fps)
    # Stationary hips = very low displacement rate → slow walk
    assert min_cycle == 0.6
    assert cutoff == 4.0


def test_detect_events_backward_compat():
    """adaptive=False (default) should not change behavior."""
    from myogait.events import detect_events
    import copy
    data1 = make_walking_data(300, fps=30.0)
    data2 = copy.deepcopy(data1)
    detect_events(data1, method="zeni", adaptive=False)
    detect_events(data2, method="zeni")
    # Results should be identical
    assert len(data1["events"]["left_hs"]) == len(data2["events"]["left_hs"])
    assert len(data1["events"]["right_hs"]) == len(data2["events"]["right_hs"])
    for i in range(len(data1["events"]["left_hs"])):
        assert data1["events"]["left_hs"][i]["frame"] == data2["events"]["left_hs"][i]["frame"]
