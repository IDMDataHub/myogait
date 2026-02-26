"""Tests for the OpenPose backend (bottom-up, OpenCV DNN)."""

import importlib
import os
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


# ── Atomic download ─────────────────────────────────────────────────────

def test_safe_download_cleans_up_on_failure(tmp_path):
    from myogait.models.openpose import _safe_download

    dest = str(tmp_path / "model.bin")
    with pytest.raises(Exception):
        _safe_download("http://0.0.0.0:1/missing", dest)

    assert not os.path.exists(dest), "Partial file should be cleaned up"
    # No temp files left either
    assert len(list(tmp_path.iterdir())) == 0


# ── process_frame with synthetic heatmaps ────────────────────────────────

def test_process_frame_returns_none_when_no_keypoints():
    """All heatmap channels below threshold → None."""
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()

    # Mock the DNN network
    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return np.zeros((1, 57, 46, 46), dtype=np.float32)

    ext._net = _FakeNet()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ext.process_frame(frame)
    assert result is None


def test_process_frame_returns_array_with_enough_peaks():
    """Enough heatmap peaks → returns (17, 3) array."""
    from myogait.models.openpose import OpenPosePoseExtractor, _OPENPOSE_TO_COCO17

    ext = OpenPosePoseExtractor(confidence_threshold=0.05)

    heatmaps = np.zeros((1, 57, 46, 46), dtype=np.float32)
    # Set 5 strong peaks (more than the 3-keypoint minimum)
    for op_idx in list(_OPENPOSE_TO_COCO17.keys())[:5]:
        heatmaps[0, op_idx, 20 + op_idx, 20] = 0.8

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return heatmaps

    ext._net = _FakeNet()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ext.process_frame(frame)

    assert result is not None
    assert result.shape == (17, 3)
    assert np.sum(result[:, 2] > 0) >= 5
    # All coordinates should be in [0, 1]
    visible = result[:, 2] > 0
    assert np.all(result[visible, 0] >= 0)
    assert np.all(result[visible, 0] <= 1)
    assert np.all(result[visible, 1] >= 0)
    assert np.all(result[visible, 1] <= 1)


def test_process_frame_exactly_two_peaks_returns_none():
    """Only 2 peaks (below the 3-keypoint minimum) → None."""
    from myogait.models.openpose import OpenPosePoseExtractor, _OPENPOSE_TO_COCO17

    ext = OpenPosePoseExtractor(confidence_threshold=0.05)

    heatmaps = np.zeros((1, 57, 46, 46), dtype=np.float32)
    for op_idx in list(_OPENPOSE_TO_COCO17.keys())[:2]:
        heatmaps[0, op_idx, 23, 23] = 0.8

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return heatmaps

    ext._net = _FakeNet()
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


def test_teardown_resets_net():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()
    ext._net = "placeholder"
    ext.teardown()
    assert ext._net is None
