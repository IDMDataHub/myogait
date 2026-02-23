"""Runtime branch tests for lightweight model extractor behavior."""

import types

import numpy as np

from myogait.models.rtmw import RTMWPoseExtractor
from myogait.models.sapiens import _crop_and_pad, _remap_landmarks
from myogait.models.yolo import YOLOPoseExtractor


class _DummyTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, item):
        return _DummyTensor(self._arr[item])


class _DummyKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = xy
        self.conf = conf

    def __len__(self):
        if self.xy is None:
            return 0
        return len(self.xy)


class _DummyYOLOResult:
    def __init__(self, keypoints):
        self.keypoints = keypoints


def test_yolo_process_frame_returns_none_when_no_keypoints(monkeypatch):
    extractor = YOLOPoseExtractor()
    extractor._model = lambda _frame, verbose=False: [_DummyYOLOResult(_DummyKeypoints(None))]

    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    assert extractor.process_frame(frame) is None


def test_yolo_process_frame_normalizes_pixel_coordinates():
    xy = _DummyTensor([[[10.0, 5.0]] * 17])
    conf = _DummyTensor([[0.75] * 17])
    extractor = YOLOPoseExtractor()
    extractor._model = lambda _frame, verbose=False: [_DummyYOLOResult(_DummyKeypoints(xy=xy, conf=conf))]

    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    out = extractor.process_frame(frame)

    assert out.shape == (17, 3)
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(out[:, 1], 0.5)
    assert np.allclose(out[:, 2], 0.75)


def test_rtmw_process_frame_returns_none_on_empty_detection():
    extractor = RTMWPoseExtractor(mode="balanced")
    extractor._wholebody = lambda _frame: (None, None)

    frame = np.zeros((40, 80, 3), dtype=np.uint8)

    assert extractor.process_frame(frame) is None


def test_rtmw_process_frame_selects_best_person_and_returns_auxiliary():
    kp = np.zeros((2, 20, 2), dtype=float)
    sc = np.zeros((2, 20), dtype=float)

    # Person 0: low confidence
    kp[0, :17, :] = [20.0, 10.0]
    sc[0, :17] = 0.2

    # Person 1: high confidence -> should be chosen
    kp[1, :17, :] = [40.0, 20.0]
    sc[1, :17] = 0.9

    extractor = RTMWPoseExtractor(mode="balanced")
    extractor._wholebody = lambda _frame: (kp, sc)

    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    out = extractor.process_frame(frame)

    assert out is not None
    assert out["landmarks"].shape == (17, 3)
    assert out["auxiliary_wholebody133"].shape == (133, 3)
    assert np.allclose(out["landmarks"][:, 0], 0.5)
    assert np.allclose(out["landmarks"][:, 1], 0.5)
    assert np.allclose(out["landmarks"][:, 2], 0.9)
    assert np.isnan(out["auxiliary_wholebody133"][132, 0])


def test_crop_and_pad_clamps_bbox_inside_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop, x_off, y_off, crop_w, crop_h = _crop_and_pad(frame, bbox=(-10, -20, 30, 40), pad_ratio=0.5)

    assert crop.ndim == 3
    assert 0 <= x_off < 200
    assert 0 <= y_off < 100
    assert crop_w > 0
    assert crop_h > 0


def test_remap_landmarks_leaves_nan_and_maps_valid_points():
    lm = np.array([[0.5, 0.5, 0.9], [np.nan, np.nan, np.nan]], dtype=float)

    remapped = _remap_landmarks(
        lm,
        x_off=10,
        y_off=20,
        crop_w=50,
        crop_h=40,
        frame_w=100,
        frame_h=100,
    )

    assert np.allclose(remapped[0, :2], [0.35, 0.4])
    assert np.isnan(remapped[1, 0])


def test_rtmw_setup_without_onnxruntime_falls_back_to_opencv(monkeypatch):
    # Force import error only for onnxruntime.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    class DummyWholebody:
        def __init__(self, mode, backend, device):
            self.mode = mode
            self.backend = backend
            self.device = device

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(__import__("sys").modules, "rtmlib", types.SimpleNamespace(Wholebody=DummyWholebody))

    extractor = RTMWPoseExtractor(mode="lightweight")
    extractor.setup()

    assert extractor._wholebody is not None
    assert extractor._wholebody.backend == "opencv"
    assert extractor._wholebody.mode == "lightweight"
