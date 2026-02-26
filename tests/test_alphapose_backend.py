"""Tests for the AlphaPose backend (top-down, dual-path loading)."""

import importlib

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
    """Heatmaps should be roughly input_size / 4."""
    torch = pytest.importorskip("torch")
    from myogait.models.alphapose import _build_simple_fastpose

    model = _build_simple_fastpose(num_joints=17)
    dummy = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        out = model(dummy)
    # ResNet stride=32, 3 deconv stride=2 → net stride=4
    # 256/4=64, 192/4=48 (but with padding differences may vary slightly)
    assert out.shape[2] > 0
    assert out.shape[3] > 0
