"""Tests for the OpenPose backend (bottom-up, OpenCV DNN)."""

import importlib
import os

import numpy as np
import pytest

from myogait.models import get_extractor, list_models


# -- Registry ----------------------------------------------------------------

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


# -- Class attributes --------------------------------------------------------

def test_openpose_class_attributes():
    from myogait.models.openpose import OpenPosePoseExtractor
    from myogait.constants import COCO_LANDMARK_NAMES

    ext = OpenPosePoseExtractor()
    assert ext.name == "openpose"
    assert ext.n_landmarks == 17
    assert ext.is_coco_format is True
    assert ext.landmark_names == COCO_LANDMARK_NAMES


# -- Mapping table -----------------------------------------------------------

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


# -- Heatmap peak detection --------------------------------------------------

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


def test_heatmap_peak_empty_spatial():
    """Heatmap with 0-size spatial dim should not crash."""
    from myogait.models.openpose import _find_keypoints_from_heatmaps

    heatmaps = np.zeros((18, 0, 0), dtype=np.float32)
    kps = _find_keypoints_from_heatmaps(heatmaps, threshold=0.1)
    assert len(kps) == 18
    for kp in kps:
        assert kp[0] is None


# -- Default constructor values ----------------------------------------------

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


# -- Atomic download --------------------------------------------------------

def test_safe_download_cleans_up_on_failure(tmp_path):
    from myogait.models.openpose import _safe_download

    dest = str(tmp_path / "model.bin")
    with pytest.raises(Exception):
        _safe_download("http://0.0.0.0:1/missing", dest)

    assert not os.path.exists(dest), "Partial file should be cleaned up"
    # No temp files left either
    assert len(list(tmp_path.iterdir())) == 0


def test_safe_download_rejects_too_small(tmp_path):
    """File smaller than min_bytes should be rejected."""
    from myogait.models.openpose import _safe_download
    import http.server
    import threading

    # Serve a tiny file
    content = b"tiny"
    handler_class = type(
        "_H", (http.server.BaseHTTPRequestHandler,),
        {
            "do_GET": lambda self: (
                self.send_response(200),
                self.send_header("Content-Length", str(len(content))),
                self.end_headers(),
                self.wfile.write(content),
            ),
            "log_message": lambda *a: None,
        },
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    dest = str(tmp_path / "model.bin")
    with pytest.raises(RuntimeError, match="too small"):
        _safe_download(f"http://127.0.0.1:{port}/model", dest, min_bytes=1000)

    assert not os.path.exists(dest)
    server.server_close()


def test_safe_download_rejects_bad_sha256(tmp_path):
    """Wrong SHA-256 should be rejected."""
    from myogait.models.openpose import _safe_download
    import http.server
    import threading

    content = b"valid content here"
    handler_class = type(
        "_H", (http.server.BaseHTTPRequestHandler,),
        {
            "do_GET": lambda self: (
                self.send_response(200),
                self.send_header("Content-Length", str(len(content))),
                self.end_headers(),
                self.wfile.write(content),
            ),
            "log_message": lambda *a: None,
        },
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    dest = str(tmp_path / "model.bin")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _safe_download(
            f"http://127.0.0.1:{port}/model", dest,
            expected_sha256="0" * 64,
        )

    assert not os.path.exists(dest)
    server.server_close()


def test_is_valid_caffemodel_removes_truncated(tmp_path, monkeypatch):
    """A truncated caffemodel should be detected and removed."""
    from myogait.models import openpose as mod

    fake_model = tmp_path / "pose_iter_440000.caffemodel"
    fake_model.write_bytes(b"truncated")
    assert not mod._is_valid_caffemodel(str(fake_model))
    assert not fake_model.exists()  # should have been removed


def test_is_valid_caffemodel_nonexistent():
    from myogait.models.openpose import _is_valid_caffemodel
    assert not _is_valid_caffemodel("/nonexistent/path/model.bin")


# -- process_frame with synthetic heatmaps -----------------------------------

def test_process_frame_returns_none_when_no_keypoints():
    """All heatmap channels below threshold -> None."""
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
    """Enough heatmap peaks -> returns (17, 3) array."""
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
    """Only 2 peaks (below the 3-keypoint minimum) -> None."""
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


def test_process_frame_exactly_three_peaks_returns_result():
    """Exactly 3 peaks (the minimum) -> should return (17, 3)."""
    from myogait.models.openpose import OpenPosePoseExtractor, _OPENPOSE_TO_COCO17

    ext = OpenPosePoseExtractor(confidence_threshold=0.05)

    heatmaps = np.zeros((1, 57, 46, 46), dtype=np.float32)
    for op_idx in list(_OPENPOSE_TO_COCO17.keys())[:3]:
        heatmaps[0, op_idx, 20 + op_idx, 20] = 0.8

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return heatmaps

    ext._net = _FakeNet()
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is not None
    assert result.shape == (17, 3)
    assert np.sum(result[:, 2] > 0) == 3


# -- Input validation -------------------------------------------------------

def test_process_frame_none_returns_none():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()
    ext._net = lambda: None  # dummy so setup() isn't called
    result = ext.process_frame(None)
    assert result is None


def test_process_frame_empty_frame_returns_none():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()
    ext._net = lambda: None
    result = ext.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_grayscale_accepted():
    """Grayscale input should be auto-converted, not crash."""
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return np.zeros((1, 57, 46, 46), dtype=np.float32)

    ext._net = _FakeNet()
    gray = np.zeros((480, 640), dtype=np.uint8)
    result = ext.process_frame(gray)
    # No crash; result is None because all heatmaps are zero
    assert result is None


def test_process_frame_rgba_accepted():
    """RGBA input should be auto-converted to RGB."""
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return np.zeros((1, 57, 46, 46), dtype=np.float32)

    ext._net = _FakeNet()
    rgba = np.zeros((480, 640, 4), dtype=np.uint8)
    result = ext.process_frame(rgba)
    assert result is None  # no crash


def test_process_frame_float_input_accepted():
    """Float [0,1] input should be handled."""
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return np.zeros((1, 57, 46, 46), dtype=np.float32)

    ext._net = _FakeNet()
    float_frame = np.zeros((480, 640, 3), dtype=np.float32)
    result = ext.process_frame(float_frame)
    assert result is None  # no crash


def test_process_frame_coordinates_clamped():
    """Coordinates at heatmap edges should be clamped to [0, 1]."""
    from myogait.models.openpose import OpenPosePoseExtractor, _OPENPOSE_TO_COCO17

    ext = OpenPosePoseExtractor(confidence_threshold=0.05)

    heatmaps = np.zeros((1, 57, 46, 46), dtype=np.float32)
    # Place peaks at corners
    keys = list(_OPENPOSE_TO_COCO17.keys())
    heatmaps[0, keys[0], 0, 0] = 0.9     # top-left
    heatmaps[0, keys[1], 45, 45] = 0.9   # bottom-right
    heatmaps[0, keys[2], 0, 45] = 0.9    # top-right

    class _FakeNet:
        def setInput(self, blob):
            pass
        def forward(self):
            return heatmaps

    ext._net = _FakeNet()
    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is not None
    visible = result[:, 2] > 0
    assert np.all(result[visible, 0] >= 0.0)
    assert np.all(result[visible, 0] <= 1.0)
    assert np.all(result[visible, 1] >= 0.0)
    assert np.all(result[visible, 1] <= 1.0)


# -- Teardown ----------------------------------------------------------------

def test_teardown_resets_net():
    from myogait.models.openpose import OpenPosePoseExtractor

    ext = OpenPosePoseExtractor()
    ext._net = "placeholder"
    ext.teardown()
    assert ext._net is None


# -- SHA-256 utility ---------------------------------------------------------

def test_sha256_computes_correctly(tmp_path):
    from myogait.models.openpose import _sha256

    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert _sha256(str(f)) == expected
