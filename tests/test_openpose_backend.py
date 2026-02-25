"""Tests for the OpenPose backend (bottom-up, OpenCV DNN)."""

import importlib
import types

import numpy as np
import pytest

from myogait.models import get_extractor, list_models


# ── Registry ──────────────────────────────────────────────────────────────

def test_list_models_includes_openpose():
    assert "openpose" in list_models()


def test_list_models_still_sorted():
    models = list_models()
    assert models == sorted(models)


def test_get_extractor_openpose_import_hint(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(ImportError, match="pip install myogait\\[openpose\\]"):
        get_extractor("openpose")


# ── Class attributes ─────────────────────────────────────────────────────

def test_openpose_class_attributes():
    from myogait.models.openpose import OpenPosePoseExtractor
    from myogait.constants import COCO_LANDMARK_NAMES

    ext = OpenPosePoseExtractor()
    assert ext.name == "openpose"
    assert ext.n_landmarks == 17
    assert ext.is_coco_format is True
    assert ext.landmark_names == COCO_LANDMARK_NAMES


# ── Mapping table ────────────────────────────────────────────────────────

def test_openpose_to_coco17_mapping_covers_all_17():
    from myogait.models.openpose import _OPENPOSE_TO_COCO17

    coco_indices = set(_OPENPOSE_TO_COCO17.values())
    assert coco_indices == set(range(17)), (
        f"Mapping covers {coco_indices}, expected {{0..16}}"
    )


def test_openpose_to_coco17_mapping_no_duplicate_targets():
    from myogait.models.openpose import _OPENPOSE_TO_COCO17

    targets = list(_OPENPOSE_TO_COCO17.values())
    assert len(targets) == len(set(targets)), "Duplicate COCO target indices"


# ── Heatmap peak detection ───────────────────────────────────────────────

def test_heatmap_peak_detection_basic():
    from myogait.models.openpose import _find_keypoints_from_heatmaps

    heatmaps = np.zeros((18, 46, 46), dtype=np.float32)
    heatmaps[0, 23, 23] = 0.9  # Nose at center
    heatmaps[5, 10, 30] = 0.7  # LShoulder
    kps = _find_keypoints_from_heatmaps(heatmaps, threshold=0.1)

    assert kps[0] == (23, 23, pytest.approx(0.9))
    assert kps[5] == (30, 10, pytest.approx(0.7))


def test_heatmap_peak_below_threshold():
    from myogait.models.openpose import _find_keypoints_from_heatmaps

    heatmaps = np.full((18, 46, 46), 0.05, dtype=np.float32)
    kps = _find_keypoints_from_heatmaps(heatmaps, threshold=0.1)

    for kp in kps:
        assert kp[0] is None
        assert kp[1] is None
        assert kp[2] == 0.0


def test_heatmap_peak_returns_correct_count():
    from myogait.models.openpose import _find_keypoints_from_heatmaps

    heatmaps = np.zeros((18, 20, 20), dtype=np.float32)
    kps = _find_keypoints_from_heatmaps(heatmaps, threshold=0.1)
    assert len(kps) == 18


# ── Default constructor values ───────────────────────────────────────────

def test_openpose_default_params():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()
    assert ext.input_width == 368
    assert ext.input_height == 368
    assert ext.confidence_threshold == pytest.approx(0.1)
    assert ext._net is None


def test_openpose_custom_params():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor(
        input_width=256, input_height=256, confidence_threshold=0.2,
    )
    assert ext.input_width == 256
    assert ext.confidence_threshold == pytest.approx(0.2)
