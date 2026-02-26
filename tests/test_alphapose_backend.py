"""Tests for the AlphaPose backend (top-down, dual-path loading)."""

import importlib
import os

import numpy as np
import pytest

from myogait.models import get_extractor, list_models


# ── Registry ──────────────────────────────────────────────────────────────

def test_list_models_includes_alphapose():
    assert "alphapose" in list_models()


def test_list_models_still_sorted():
    models = list_models()
    assert models == sorted(models)


def test_get_extractor_alphapose_import_hint(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(ImportError, match="pip install myogait\\[alphapose\\]"):
        get_extractor("alphapose")


# ── Class attributes ─────────────────────────────────────────────────────

def test_alphapose_class_attributes():
    from myogait.models.alphapose import AlphaPosePoseExtractor
    from myogait.constants import COCO_LANDMARK_NAMES

    ext = AlphaPosePoseExtractor()
    assert ext.name == "alphapose"
    assert ext.n_landmarks == 17
    assert ext.is_coco_format is True
    assert ext.landmark_names == COCO_LANDMARK_NAMES


# ── Constructor defaults ─────────────────────────────────────────────────

def test_alphapose_default_params():
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor()
    assert ext.device_name == "auto"
    assert ext.checkpoint is None
    assert ext._model is None
    assert ext._detector is None
    assert ext._device is None


def test_alphapose_custom_params():
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor(device="cpu", checkpoint="/tmp/custom.pth")
    assert ext.device_name == "cpu"
    assert ext.checkpoint == "/tmp/custom.pth"


# ── Setup error handling ─────────────────────────────────────────────────

def test_setup_without_torch_raises():
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor()

    import builtins
    real_import = builtins.__import__

    def block_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = block_torch
    try:
        with pytest.raises(ImportError, match="PyTorch is required"):
            ext.setup()
    finally:
        builtins.__import__ = real_import


def test_teardown_clears_state():
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor()
    ext._model = "placeholder"
    ext._detector = "placeholder"
    ext._device = "placeholder"
    ext.teardown()
    assert ext._model is None
    assert ext._detector is None
    assert ext._device is None


# ── Fallback model builder ───────────────────────────────────────────────

def test_simple_fastpose_builds():
    """Verify the fallback model can be instantiated (requires torch)."""
    torch = pytest.importorskip("torch")
    from myogait.models.alphapose import _build_simple_fastpose

    model = _build_simple_fastpose(num_joints=17)
    # Feed a dummy input
    dummy = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[0] == 1
    assert out.shape[1] == 17
    # heatmap spatial dims should be input / 4 (three stride-2 deconvs
    # undo the stride-32 backbone, leaving overall stride 4)


def test_simple_fastpose_output_spatial_shape():
    """Heatmaps should be input_size / 4."""
    torch = pytest.importorskip("torch")
    from myogait.models.alphapose import _build_simple_fastpose

    model = _build_simple_fastpose(num_joints=17)
    dummy = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        out = model(dummy)
    # ResNet stride=32, 3 deconv stride=2 → net stride=4
    # 256/4=64, 192/4=48
    assert out.shape == (1, 17, 64, 48)


# ── Atomic download ─────────────────────────────────────────────────────

def test_ensure_checkpoint_cleans_up_on_failure(tmp_path, monkeypatch):
    from myogait.models import alphapose as mod

    monkeypatch.setattr(mod, "_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "_FASTPOSE_URL", "http://0.0.0.0:1/bad")

    with pytest.raises(Exception):
        mod._ensure_checkpoint(custom_path=None)

    # No partial file should remain
    files = [f for f in tmp_path.iterdir() if f.name != _FASTPOSE_FILE]
    leftover = [f for f in files if f.stat().st_size > 0]
    assert len(leftover) == 0


_FASTPOSE_FILE = "fast_res50_256x192.pth"


# ── process_frame with mock model + detector ─────────────────────────────

def test_process_frame_no_detections_returns_none():
    """YOLO detects no person → None."""
    torch = pytest.importorskip("torch")
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor()
    ext._device = torch.device("cpu")

    # Mock detector returns empty boxes
    class _FakeDetector:
        def __call__(self, img, verbose=False, classes=None):
            from types import SimpleNamespace
            return [SimpleNamespace(boxes=None)]

    ext._detector = _FakeDetector()
    ext._model = lambda x: torch.zeros(1, 17, 64, 48)

    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None


def test_process_frame_with_mock_returns_landmarks():
    """Full mock: YOLO detects person, model returns heatmaps → (17,3)."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from myogait.models.alphapose import AlphaPosePoseExtractor

    ext = AlphaPosePoseExtractor()
    ext._device = torch.device("cpu")

    # Mock YOLO detector returning one box
    class _FakeBox:
        conf = torch.tensor([0.9])
        xyxy = torch.tensor([[50.0, 30.0, 550.0, 430.0]])

    class _FakeDetector:
        def __call__(self, img, verbose=False, classes=None):
            return [SimpleNamespace(boxes=[_FakeBox()])]

    ext._detector = _FakeDetector()

    # Mock pose model returning heatmaps with clear peaks
    def _fake_model(inp):
        hm = torch.zeros(1, 17, 64, 48)
        for j in range(17):
            hm[0, j, 30 + j % 10, 20 + j % 10] = 5.0  # strong peaks
        return hm

    ext._model = _fake_model

    result = ext.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is not None
    assert result.shape == (17, 3)
    # All visible landmarks should be in [0, 1]
    visible = result[:, 2] > 0
    assert np.sum(visible) >= 3
    assert np.all(result[visible, 0] >= 0) and np.all(result[visible, 0] <= 1)
    assert np.all(result[visible, 1] >= 0) and np.all(result[visible, 1] <= 1)
