"""Error-path tests for optional model dependencies and setup failures."""

import builtins
import sys

import pytest

from myogait.models.hrnet import HRNETPoseExtractor
from myogait.models.mediapipe import MediaPipePoseExtractor
from myogait.models.mmpose import MMPosePoseExtractor
from myogait.models.rtmw import RTMWPoseExtractor
from myogait.models.vitpose import ViTPosePoseExtractor
from myogait.models.yolo import YOLOPoseExtractor


def _force_import_error(monkeypatch, blocked_roots):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_vitpose_setup_raises_clear_import_error(monkeypatch):
    _force_import_error(monkeypatch, {"transformers"})
    sys.modules.pop("transformers", None)

    with pytest.raises(ImportError, match=r"myogait\[vitpose\]"):
        ViTPosePoseExtractor().setup()


def test_yolo_setup_raises_clear_import_error(monkeypatch):
    _force_import_error(monkeypatch, {"ultralytics"})
    sys.modules.pop("ultralytics", None)

    with pytest.raises(ImportError, match=r"myogait\[yolo\]"):
        YOLOPoseExtractor().setup()


def test_hrnet_setup_wraps_missing_dependency_error(monkeypatch):
    monkeypatch.setattr(HRNETPoseExtractor, "_ensure_mmcv", lambda self: None)
    _force_import_error(monkeypatch, {"mmpose"})
    sys.modules.pop("mmpose", None)

    with pytest.raises(ImportError, match=r"myogait\[mmpose,yolo\]"):
        HRNETPoseExtractor().setup()


def test_mmpose_setup_wraps_missing_dependency_error(monkeypatch):
    monkeypatch.setattr(MMPosePoseExtractor, "_ensure_mmcv", lambda self: None)
    _force_import_error(monkeypatch, {"mmpose"})
    sys.modules.pop("mmpose", None)

    with pytest.raises(ImportError, match=r"myogait\[mmpose,yolo\]"):
        MMPosePoseExtractor().setup()


def test_rtmw_setup_raises_clear_import_error(monkeypatch):
    _force_import_error(monkeypatch, {"rtmlib"})
    sys.modules.pop("rtmlib", None)

    with pytest.raises(ImportError, match=r"myogait\[rtmw\]"):
        RTMWPoseExtractor().setup()


def test_mediapipe_setup_fallback_and_error_message(monkeypatch):
    _force_import_error(monkeypatch, {"mediapipe"})
    sys.modules.pop("mediapipe", None)

    with pytest.raises(ImportError, match=r"myogait\[mediapipe\]"):
        MediaPipePoseExtractor().setup()
