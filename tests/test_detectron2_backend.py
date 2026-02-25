"""Tests for the Detectron2 / Keypoint R-CNN backend."""

import importlib

import pytest

from myogait.models import get_extractor, list_models


# ── Registry ──────────────────────────────────────────────────────────────

def test_list_models_includes_detectron2():
    assert "detectron2" in list_models()


def test_list_models_still_sorted():
    models = list_models()
    assert models == sorted(models)


def test_get_extractor_detectron2_import_hint(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(ImportError, match="pip install myogait\\[detectron2\\]"):
        get_extractor("detectron2")


# ── Class attributes ─────────────────────────────────────────────────────

def test_detectron2_class_attributes():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor
    from myogait.constants import COCO_LANDMARK_NAMES

    ext = Detectron2PoseExtractor()
    assert ext.name == "detectron2"
    assert ext.n_landmarks == 17
    assert ext.is_coco_format is True
    assert ext.landmark_names == COCO_LANDMARK_NAMES


# ── Constructor defaults ─────────────────────────────────────────────────

def test_detectron2_default_params():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()
    assert ext.config_name == "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    assert ext.threshold == pytest.approx(0.7)
    assert ext.device_name == "auto"
    assert ext._predictor is None


def test_detectron2_custom_params():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor(
        config="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        threshold=0.5,
        device="cpu",
    )
    assert "R_101" in ext.config_name
    assert ext.threshold == pytest.approx(0.5)
    assert ext.device_name == "cpu"


# ── Setup error handling ─────────────────────────────────────────────────

def test_setup_without_detectron2_raises():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    import builtins
    real_import = builtins.__import__

    def block_detectron2(name, *args, **kwargs):
        if name.startswith("detectron2"):
            raise ImportError("no detectron2")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = block_detectron2
    try:
        with pytest.raises(ImportError, match="Detectron2 is required"):
            ext.setup()
    finally:
        builtins.__import__ = real_import


def test_teardown_clears_predictor():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()
    ext._predictor = "placeholder"
    ext.teardown()
    assert ext._predictor is None
