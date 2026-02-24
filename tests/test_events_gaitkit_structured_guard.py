"""Guards for gaitkit structured detector return type."""

import pytest


def test_detect_gaitkit_structured_raises_clear_error_on_none(monkeypatch):
    from myogait import events

    class DummyGK:
        @staticmethod
        def detect_events_structured(method, data, fps=30.0):
            return None

    monkeypatch.setattr(events, "_import_gaitkit", lambda: DummyGK)

    with pytest.raises(RuntimeError, match="returned None"):
        events._detect_gaitkit_structured({"frames": [{}]}, 30.0, method="bayesian_bis")


def test_detect_gaitkit_structured_raises_clear_error_on_non_dict(monkeypatch):
    from myogait import events

    class DummyGK:
        @staticmethod
        def detect_events_structured(method, data, fps=30.0):
            return []

    monkeypatch.setattr(events, "_import_gaitkit", lambda: DummyGK)

    with pytest.raises(TypeError, match="expected dict"):
        events._detect_gaitkit_structured({"frames": [{}]}, 30.0, method="bayesian_bis")
