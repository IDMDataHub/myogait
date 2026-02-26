"""Tests for the Detectron2 / Keypoint R-CNN backend."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from myogait.models import get_extractor, list_models


# -- Registry ----------------------------------------------------------------

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


# -- Class attributes -------------------------------------------------------

def test_detectron2_class_attributes():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor
    from myogait.constants import COCO_LANDMARK_NAMES

    ext = Detectron2PoseExtractor()
    assert ext.name == "detectron2"
    assert ext.n_landmarks == 17
    assert ext.is_coco_format is True
    assert ext.landmark_names == COCO_LANDMARK_NAMES


# -- Constructor defaults ---------------------------------------------------

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


# -- Setup error handling ---------------------------------------------------

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


# -- Mock helpers -----------------------------------------------------------

class _MockBoxes:
    """Mimics Detectron2 Boxes with subscript + .tensor access."""

    def __init__(self, data):
        self.tensor = _T(data)
        self._data = np.asarray(data)

    def __getitem__(self, idx):
        return _MockBoxes(self._data[idx])


class _MockInstances:
    """Mimics Detectron2 Instances output."""

    def __init__(self, pred_classes, pred_keypoints, scores, boxes):
        self.pred_classes = _T(pred_classes)
        self.pred_keypoints = _T(pred_keypoints)
        self.scores = _T(scores)
        self.pred_boxes = _MockBoxes(boxes)
        self._fields = {"pred_keypoints": True}

    def has(self, field):
        return field in self._fields

    def __len__(self):
        return len(self.pred_classes._data)

    def __getitem__(self, idx):
        return _MockInstances(
            self.pred_classes._data[idx],
            self.pred_keypoints._data[idx],
            self.scores._data[idx],
            self.pred_boxes._data[idx],
        )


class _T:
    """Mimics a torch tensor with .cpu().numpy() chain."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return _T(self._data[idx])


# -- process_frame with mock predictor --------------------------------------

def test_process_frame_single_person():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    kps = np.zeros((1, 17, 3), dtype=np.float32)
    kps[0, :, 0] = np.linspace(100, 500, 17)   # x pixels
    kps[0, :, 1] = np.linspace(50, 400, 17)    # y pixels
    kps[0, :, 2] = 0.9                          # confidence

    instances = _MockInstances(
        pred_classes=np.array([0]),
        pred_keypoints=kps,
        scores=np.array([0.95]),
        boxes=np.array([[50, 30, 550, 430]]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ext.process_frame(frame)

    assert result is not None
    assert result.shape == (17, 3)
    # All x should be in [0, 1]
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 1)
    assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1)
    assert np.all(result[:, 2] == pytest.approx(0.9))


def test_process_frame_no_person_returns_none():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()
    # Non-person class (e.g., car = 2)
    instances = _MockInstances(
        pred_classes=np.array([2]),
        pred_keypoints=np.zeros((1, 17, 3)),
        scores=np.array([0.9]),
        boxes=np.array([[0, 0, 100, 100]]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_empty_instances_returns_none():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    instances = _MockInstances(
        pred_classes=np.array([]).reshape(0),
        pred_keypoints=np.zeros((0, 17, 3)),
        scores=np.array([]).reshape(0),
        boxes=np.zeros((0, 4)),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_selects_largest_person():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    kps = np.zeros((2, 17, 3), dtype=np.float32)
    # Person 0: small box, 5 keypoints
    kps[0, 0, :] = [100, 100, 0.9]
    kps[0, 1, :] = [95, 90, 0.9]
    kps[0, 2, :] = [105, 90, 0.9]
    kps[0, 5, :] = [90, 120, 0.9]
    kps[0, 6, :] = [110, 120, 0.9]
    # Person 1: big box, 5 keypoints with nose at (320, 240)
    kps[1, 0, :] = [320, 240, 0.9]
    kps[1, 1, :] = [310, 220, 0.9]
    kps[1, 2, :] = [330, 220, 0.9]
    kps[1, 5, :] = [280, 300, 0.9]
    kps[1, 6, :] = [360, 300, 0.9]

    instances = _MockInstances(
        pred_classes=np.array([0, 0]),
        pred_keypoints=kps,
        scores=np.array([0.8, 0.9]),
        boxes=np.array([
            [90, 90, 110, 110],    # 20x20 = 400 * 0.8 = 320
            [100, 50, 540, 430],   # 440x380 = 167200 * 0.9 = 150480
        ]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is not None
    # Should pick person 1 (largest area*score), nose at (320/640, 240/480)
    assert result[0, 0] == pytest.approx(320 / 640)
    assert result[0, 1] == pytest.approx(240 / 480)


def test_process_frame_min_keypoints_filter():
    """Person with all-zero-confidence keypoints should be rejected."""
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    kps = np.zeros((1, 17, 3), dtype=np.float32)
    kps[0, :, 0] = np.linspace(100, 500, 17)
    kps[0, :, 1] = np.linspace(50, 400, 17)
    kps[0, :, 2] = 0.0  # all zero confidence

    instances = _MockInstances(
        pred_classes=np.array([0]),
        pred_keypoints=kps,
        scores=np.array([0.95]),
        boxes=np.array([[50, 30, 550, 430]]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_coordinates_clamped():
    """Keypoints outside image bounds should be clamped to [0, 1]."""
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    kps = np.zeros((1, 17, 3), dtype=np.float32)
    kps[0, 0, :] = [-50, -30, 0.9]    # outside top-left
    kps[0, 1, :] = [700, 500, 0.9]    # outside bottom-right
    kps[0, 2, :] = [320, 240, 0.9]    # normal
    kps[0, 3:, 2] = 0.0               # rest invisible

    instances = _MockInstances(
        pred_classes=np.array([0]),
        pred_keypoints=kps,
        scores=np.array([0.95]),
        boxes=np.array([[0, 0, 640, 480]]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is not None
    assert result[0, 0] == 0.0  # clamped from negative
    assert result[0, 1] == 0.0
    assert result[1, 0] == 1.0  # clamped from >1
    assert result[1, 1] == 1.0
    assert result[2, 0] == pytest.approx(320 / 640)


def test_process_frame_no_pred_keypoints_field():
    """Instance without pred_keypoints field should return None."""
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    instances = _MockInstances(
        pred_classes=np.array([0]),
        pred_keypoints=np.zeros((1, 17, 3)),
        scores=np.array([0.95]),
        boxes=np.array([[50, 30, 550, 430]]),
    )
    instances._fields = {}  # no pred_keypoints

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


# -- Input validation -------------------------------------------------------

def test_process_frame_none_returns_none():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()
    ext._predictor = lambda f: None
    result = ext.process_frame(None)
    assert result is None


def test_process_frame_empty_frame_returns_none():
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()
    ext._predictor = lambda f: None
    result = ext.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_grayscale_accepted():
    """Grayscale should be auto-converted."""
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    instances = _MockInstances(
        pred_classes=np.array([]).reshape(0),
        pred_keypoints=np.zeros((0, 17, 3)),
        scores=np.array([]).reshape(0),
        boxes=np.zeros((0, 4)),
    )

    ext._predictor = lambda frame: {"instances": instances}
    gray = np.zeros((480, 640), dtype=np.uint8)
    result = ext.process_frame(gray)
    assert result is None  # no crash, no detections


def test_process_frame_nan_scores_handled():
    """NaN in scores should not crash argmax."""
    from myogait.models.keypoint_rcnn import Detectron2PoseExtractor

    ext = Detectron2PoseExtractor()

    kps = np.zeros((2, 17, 3), dtype=np.float32)
    kps[0, :, :2] = 320
    kps[0, :, 2] = 0.9
    kps[1, :, :2] = 320
    kps[1, :, 2] = 0.9

    instances = _MockInstances(
        pred_classes=np.array([0, 0]),
        pred_keypoints=kps,
        scores=np.array([np.nan, 0.9]),
        boxes=np.array([[0, 0, 640, 480], [0, 0, 640, 480]]),
    )

    ext._predictor = lambda frame: {"instances": instances}
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is not None  # should not crash
