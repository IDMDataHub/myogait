"""Additional coverage for Sapiens depth/seg auxiliary estimators."""

import builtins
import sys

import numpy as np
import pytest

from myogait.models.sapiens_depth import SapiensDepthEstimator, _find_depth_model
from myogait.models.sapiens_seg import SapiensSegEstimator, _find_seg_model


def _block_hf_import(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "huggingface_hub":
            raise ImportError("blocked huggingface_hub")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("huggingface_hub", None)


def test_find_depth_model_unknown_size_raises():
    with pytest.raises(ValueError, match="Unknown depth model size"):
        _find_depth_model("9b")


def test_find_seg_model_unknown_size_raises():
    with pytest.raises(ValueError, match="Unknown seg model size"):
        _find_seg_model("9b")


def test_find_depth_model_missing_hub_raises_file_not_found(monkeypatch, tmp_path):
    from myogait.models import sapiens_depth as mod

    monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
    _block_hf_import(monkeypatch)

    with pytest.raises(FileNotFoundError, match="huggingface-hub"):
        _find_depth_model("0.3b")


def test_find_seg_model_missing_hub_raises_file_not_found(monkeypatch, tmp_path):
    from myogait.models import sapiens_seg as mod

    monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
    _block_hf_import(monkeypatch)

    with pytest.raises(FileNotFoundError, match="huggingface-hub"):
        _find_seg_model("0.3b")


def test_depth_sample_at_landmarks_flipped_changes_sample_column():
    est = SapiensDepthEstimator()
    depth = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=float,
    )
    landmarks = np.array([[0.25, 0.5, 1.0]], dtype=float)

    normal = est.sample_at_landmarks(depth, landmarks, flipped=False)
    flipped = est.sample_at_landmarks(depth, landmarks, flipped=True)

    assert normal.shape == (1,)
    assert flipped.shape == (1,)
    assert normal[0] != flipped[0]


def test_seg_get_body_mask_and_unknown_class_sampling():
    est = SapiensSegEstimator()

    seg_mask = np.array(
        [
            [0, 255],
            [1, 2],
        ],
        dtype=np.uint8,
    )
    body_mask = est.get_body_mask(seg_mask)
    assert body_mask.shape == seg_mask.shape
    assert set(np.unique(body_mask)).issubset({0, 1})

    landmarks = np.array([[0.75, 0.0, 1.0], [np.nan, np.nan, 0.0]], dtype=float)
    parts = est.sample_at_landmarks(seg_mask, landmarks, flipped=False)
    assert parts[0] == "Unknown"
    assert parts[1] is None
