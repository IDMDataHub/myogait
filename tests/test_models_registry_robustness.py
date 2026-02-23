"""Robustness tests for model registry and lazy loader behavior."""

import types

import pytest

from myogait.models import get_extractor, list_models


class _DummyExtractor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_list_models_is_sorted_and_contains_expected_entries():
    models = list_models()
    assert models == sorted(models)
    assert "mediapipe" in models
    assert "rtmw" in models
    assert "vitpose-huge" in models


def test_get_extractor_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_extractor("does-not-exist")


def test_get_extractor_wraps_import_error_with_install_hint(monkeypatch):
    import importlib

    def _raise_import_error(_name):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError, match="Install with: pip install myogait\\[yolo\\]"):
        get_extractor("yolo")


def test_get_extractor_passes_vitpose_variant_as_model_size(monkeypatch):
    import importlib

    module = types.SimpleNamespace(ViTPosePoseExtractor=_DummyExtractor)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)

    extractor = get_extractor("vitpose-large", custom_flag=True)

    assert isinstance(extractor, _DummyExtractor)
    assert extractor.kwargs["model_size"] == "large"
    assert extractor.kwargs["custom_flag"] is True
