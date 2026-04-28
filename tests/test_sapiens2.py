"""Tests for Sapiens 2 pose, depth, and segmentation backends.

Covers model registry, loading paths, heatmap conversion, depth/seg
estimators, Intel XPU device selection, and integration with extract().
"""

import builtins
import sys
import types

import numpy as np
import pytest


# ── Pose model registry ─────────────────────────────────────────────

class TestSapiens2PoseRegistry:
    """Verify model definitions and file naming conventions."""

    def test_all_sizes_registered(self):
        from myogait.models.sapiens2 import _MODELS
        assert set(_MODELS.keys()) == {"0.4b", "0.8b", "1b", "5b"}

    def test_filenames_are_safetensors(self):
        from myogait.models.sapiens2 import _MODELS
        for size, (filename, _repo) in _MODELS.items():
            assert filename.endswith(".safetensors"), f"{size}: {filename}"

    def test_repos_follow_naming_convention(self):
        from myogait.models.sapiens2 import _MODELS
        for size, (_filename, repo_id) in _MODELS.items():
            assert repo_id.startswith("facebook/sapiens2-pose-"), (
                f"{size}: {repo_id}"
            )

    def test_model_filenames_backcompat_alias(self):
        from myogait.models.sapiens2 import _MODEL_FILENAMES, _MODELS
        for size in _MODELS:
            assert _MODEL_FILENAMES[size] == _MODELS[size][0]


# ── Pose model download / find ───────────────────────────────────────

class TestSapiens2Download:

    def test_download_unknown_size_rejected(self, monkeypatch):
        fake_hub = types.SimpleNamespace(
            hf_hub_download=lambda **_kw: "/tmp/model.safetensors"
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        from myogait.models import sapiens2 as mod
        with pytest.raises(ValueError, match="Unknown model size"):
            mod.download_model("9b")

    def test_download_success_verifies_integrity(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2 as mod
        calls = {}

        def _fake_hf(**kwargs):
            calls["hub_kwargs"] = kwargs
            return str(tmp_path / "a.safetensors")

        fake_hub = types.SimpleNamespace(hf_hub_download=_fake_hf)
        monkeypatch.setattr(
            mod, "_MODEL_SHA256",
            {"0.4b": "abc", "0.8b": None, "1b": None, "5b": None},
        )
        monkeypatch.setattr(
            mod, "_MODEL_REVISIONS",
            {"0.4b": "rev1", "0.8b": "main", "1b": "main", "5b": "main"},
        )
        monkeypatch.setattr(
            mod, "_verify_model_integrity",
            lambda path, expected, label: calls.setdefault(
                "verify", (path, expected, label)
            ),
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        path = mod.download_model("0.4b", dest=str(tmp_path))
        assert path == str(tmp_path / "a.safetensors")
        assert calls["hub_kwargs"]["revision"] == "rev1"
        assert calls["verify"][1] == "abc"
        assert calls["verify"][2] == "sapiens2-0.4b"


class TestSapiens2FindModel:

    def test_explicit_path_returned(self, tmp_path):
        from myogait.models.sapiens2 import _find_model
        p = tmp_path / "model.safetensors"
        p.write_bytes(b"x")
        assert _find_model("0.4b", model_path=str(p)) == str(p)

    def test_unknown_size_raises(self):
        from myogait.models.sapiens2 import _find_model
        with pytest.raises(ValueError, match="Unknown Sapiens 2 model size"):
            _find_model("9b")

    def test_local_safetensors_found_and_verified(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2 as mod
        filename = mod._MODELS["0.4b"][0]
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / filename).write_bytes(b"payload")

        verify_calls = []
        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [model_dir])
        monkeypatch.setattr(
            mod, "_MODEL_SHA256",
            {"0.4b": "abcd", "0.8b": None, "1b": None, "5b": None},
        )
        monkeypatch.setattr(
            mod, "_verify_model_integrity",
            lambda path, expected, label: verify_calls.append(
                (path, expected, label)
            ),
        )

        resolved = mod._find_model("0.4b")
        assert resolved == str(model_dir / filename)
        assert verify_calls == [(str(model_dir / filename), "abcd", "sapiens2-0.4b")]

    def test_torchscript_fallback_found(self, monkeypatch, tmp_path):
        """If only a .pt2 file exists, it is found as fallback."""
        from myogait.models import sapiens2 as mod
        ts_filename = mod._MODELS["0.4b"][0].replace(".safetensors", ".pt2")
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / ts_filename).write_bytes(b"ts-payload")

        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [model_dir])
        monkeypatch.setattr(
            mod, "_MODEL_SHA256",
            {"0.4b": None, "0.8b": None, "1b": None, "5b": None},
        )

        resolved = mod._find_model("0.4b")
        assert resolved == str(model_dir / ts_filename)

    def test_missing_hub_raises_file_not_found(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2 as mod
        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
        monkeypatch.setattr(
            mod, "download_model",
            lambda *a, **kw: (_ for _ in ()).throw(ImportError("no hub")),
        )
        with pytest.raises(FileNotFoundError, match="huggingface-hub"):
            mod._find_model("0.4b")

    def test_generic_download_error_wrapped(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2 as mod
        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
        monkeypatch.setattr(
            mod, "download_model",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network")),
        )
        with pytest.raises(FileNotFoundError, match="Auto-download failed"):
            mod._find_model("0.4b")


# ── Model loading ────────────────────────────────────────────────────

class TestSapiens2LoadModel:

    def test_torchscript_path_uses_jit_load(self, monkeypatch, tmp_path):
        """A .pt2 path should trigger torch.jit.load()."""
        torch = pytest.importorskip("torch")
        from myogait.models import sapiens2 as mod

        called = {}
        fake_model = types.SimpleNamespace(eval=lambda: None)

        def _fake_jit_load(path, map_location=None):
            called["path"] = path
            return fake_model

        monkeypatch.setattr(torch.jit, "load", _fake_jit_load)

        mod._load_model(str(tmp_path / "model.pt2"), "cpu")
        assert called["path"] == str(tmp_path / "model.pt2")

    def test_safetensors_without_package_raises(self, monkeypatch, tmp_path):
        """SafeTensors path without sapiens2 package should give clear error."""
        pytest.importorskip("torch")
        from myogait.models import sapiens2 as mod

        # Mock safetensors as available
        fake_safetensors = types.ModuleType("safetensors")
        fake_torch_st = types.ModuleType("safetensors.torch")
        fake_torch_st.load_file = lambda f: {}
        fake_safetensors.torch = fake_torch_st
        monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
        monkeypatch.setitem(sys.modules, "safetensors.torch", fake_torch_st)

        # Block sapiens package
        real_import = builtins.__import__

        def _block_sapiens(name, *args, **kwargs):
            if name.startswith("sapiens"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_sapiens)

        with pytest.raises(ImportError, match="sapiens2 package"):
            mod._load_model(str(tmp_path / "model.safetensors"), "cpu")


# ── Heatmap processing (shared with v1) ─────────────────────────────

class TestSapiens2HeatmapProcessing:
    """Heatmap → keypoints uses v1 functions, same Goliath 308 layout."""

    def test_heatmaps_to_coco_produces_17_landmarks(self):
        from myogait.models.sapiens2 import _heatmaps_to_coco
        heatmaps = np.random.rand(308, 64, 48).astype(np.float32)
        pad_info = (0, 0, 768, 1024)
        result = _heatmaps_to_coco(heatmaps, pad_info)
        assert result.shape == (17, 3)
        # At least some landmarks should be valid
        assert np.any(~np.isnan(result[:, 0]))

    def test_heatmaps_to_all_produces_308_landmarks(self):
        from myogait.models.sapiens2 import _heatmaps_to_all
        heatmaps = np.random.rand(308, 64, 48).astype(np.float32)
        pad_info = (0, 0, 768, 1024)
        result = _heatmaps_to_all(heatmaps, pad_info)
        assert result.shape == (308, 3)


# ── Device selection (Intel XPU compatibility) ───────────────────────

class TestSapiens2IntelXPU:
    """Verify Intel Arc / XPU device selection works correctly."""

    def test_get_device_returns_valid(self):
        pytest.importorskip("torch")
        from myogait.models.sapiens2 import _get_device
        device = _get_device()
        assert device.type in ("cpu", "cuda", "xpu")

    def test_extractor_setup_calls_ensure_xpu(self, monkeypatch):
        """Extractor.setup() must call ensure_xpu_torch() for Intel compat."""
        pytest.importorskip("torch")
        from myogait.models.sapiens2 import Sapiens2QuickExtractor
        called = {"xpu": False, "ipex": False}

        monkeypatch.setattr(
            "myogait.models.sapiens2.ensure_xpu_torch",
            lambda: called.__setitem__("xpu", True),
        )
        # Block actual model loading
        monkeypatch.setattr(
            "myogait.models.sapiens2._find_model",
            lambda *a, **kw: "/fake/model.pt2",
        )
        monkeypatch.setattr(
            "myogait.models.sapiens2._load_model",
            lambda *a, **kw: types.SimpleNamespace(eval=lambda: None),
        )
        monkeypatch.setattr(
            "myogait.models.sapiens2._get_device",
            lambda: __import__("torch").device("cpu"),
        )
        monkeypatch.setattr(
            "myogait.models.sapiens2._person_detector",
            types.SimpleNamespace(setup=lambda: None),
        )

        ext = Sapiens2QuickExtractor()
        ext.setup()
        assert called["xpu"] is True


# ── Extractor registration ───────────────────────────────────────────

class TestSapiens2Registration:

    def test_models_registered_in_registry(self):
        from myogait.models import list_models
        models = list_models()
        assert "sapiens2-quick" in models
        assert "sapiens2-mid" in models
        assert "sapiens2-top" in models
        assert "sapiens2-ultra" in models

    def test_extractor_class_names(self):
        from myogait.models.sapiens2 import (
            Sapiens2QuickExtractor,
            Sapiens2MidExtractor,
            Sapiens2TopExtractor,
            Sapiens2UltraExtractor,
        )
        assert Sapiens2QuickExtractor.__name__ == "Sapiens20.4BExtractor"
        assert Sapiens2MidExtractor.__name__ == "Sapiens20.8BExtractor"
        assert Sapiens2TopExtractor.__name__ == "Sapiens21BExtractor"
        assert Sapiens2UltraExtractor.__name__ == "Sapiens25BExtractor"

    def test_extractors_are_coco_format(self):
        from myogait.models.sapiens2 import (
            Sapiens2QuickExtractor,
            Sapiens2MidExtractor,
            Sapiens2TopExtractor,
            Sapiens2UltraExtractor,
        )
        for cls in (
            Sapiens2QuickExtractor,
            Sapiens2MidExtractor,
            Sapiens2TopExtractor,
            Sapiens2UltraExtractor,
        ):
            assert cls.is_coco_format is True
            assert cls.n_landmarks == 17

    def test_extractors_inherit_base(self):
        from myogait.models.base import BasePoseExtractor
        from myogait.models.sapiens2 import Sapiens2QuickExtractor
        assert issubclass(Sapiens2QuickExtractor, BasePoseExtractor)


# ── Depth estimator ──────────────────────────────────────────────────

class TestSapiens2Depth:

    def test_depth_registry_sizes(self):
        from myogait.models.sapiens2_depth import _DEPTH_MODELS
        assert set(_DEPTH_MODELS.keys()) == {"0.4b", "0.8b", "1b", "5b"}
        for size, (filename, repo_id) in _DEPTH_MODELS.items():
            assert filename.endswith(".safetensors")
            assert repo_id.startswith("facebook/sapiens2-depth-")

    def test_find_depth_model_unknown_size_raises(self):
        from myogait.models.sapiens2_depth import _find_depth_model
        with pytest.raises(ValueError, match="Unknown Sapiens 2 depth model size"):
            _find_depth_model("9b")

    def test_find_depth_model_missing_hub(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2_depth as mod

        real_import = builtins.__import__

        def _block_hub(name, *args, **kwargs):
            if name.split(".", 1)[0] == "huggingface_hub":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
        monkeypatch.setattr(builtins, "__import__", _block_hub)
        sys.modules.pop("huggingface_hub", None)

        with pytest.raises(FileNotFoundError, match="huggingface-hub"):
            mod._find_depth_model("0.4b")

    def test_sample_at_landmarks(self):
        from myogait.models.sapiens2_depth import Sapiens2DepthEstimator
        est = Sapiens2DepthEstimator()

        # Gradient depth map
        depth = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        landmarks = np.full((3, 3), np.nan)
        landmarks[0] = [0.5, 0.5, 1.0]
        landmarks[1] = [0.0, 0.0, 1.0]

        depths = est.sample_at_landmarks(depth, landmarks, flipped=False)
        assert depths.shape == (3,)
        assert not np.isnan(depths[0])
        assert not np.isnan(depths[1])
        assert np.isnan(depths[2])
        assert depths[0] > depths[1]

    def test_sample_at_landmarks_flipped(self):
        from myogait.models.sapiens2_depth import Sapiens2DepthEstimator
        est = Sapiens2DepthEstimator()
        depth = np.array(
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8]],
            dtype=float,
        )
        landmarks = np.array([[0.25, 0.5, 1.0]], dtype=float)
        normal = est.sample_at_landmarks(depth, landmarks, flipped=False)
        flipped = est.sample_at_landmarks(depth, landmarks, flipped=True)
        assert normal[0] != flipped[0]


# ── Segmentation estimator ───────────────────────────────────────────

class TestSapiens2Seg:

    def test_seg_registry_sizes(self):
        from myogait.models.sapiens2_seg import _SEG_MODELS
        assert set(_SEG_MODELS.keys()) == {"0.4b", "0.8b", "1b", "5b"}
        for size, (filename, repo_id) in _SEG_MODELS.items():
            assert filename.endswith(".safetensors")
            assert repo_id.startswith("facebook/sapiens2-seg-")

    def test_find_seg_model_unknown_size_raises(self):
        from myogait.models.sapiens2_seg import _find_seg_model
        with pytest.raises(ValueError, match="Unknown Sapiens 2 seg model size"):
            _find_seg_model("9b")

    def test_find_seg_model_missing_hub(self, monkeypatch, tmp_path):
        from myogait.models import sapiens2_seg as mod

        real_import = builtins.__import__

        def _block_hub(name, *args, **kwargs):
            if name.split(".", 1)[0] == "huggingface_hub":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
        monkeypatch.setattr(builtins, "__import__", _block_hub)
        sys.modules.pop("huggingface_hub", None)

        with pytest.raises(FileNotFoundError, match="huggingface-hub"):
            mod._find_seg_model("0.4b")

    def test_29_classes(self):
        from myogait.models.sapiens2_seg import Sapiens2SegEstimator
        assert Sapiens2SegEstimator.n_classes == 29
        assert len(Sapiens2SegEstimator.classes) == 29

    def test_eyeglass_class_at_index_2(self):
        from myogait.constants import SAPIENS2_SEG_CLASSES
        assert SAPIENS2_SEG_CLASSES[2] == "Eyeglass"
        assert "Eyeglass" not in __import__(
            "myogait.constants", fromlist=["GOLIATH_SEG_CLASSES"]
        ).GOLIATH_SEG_CLASSES

    def test_body_mask_and_sampling(self):
        from myogait.models.sapiens2_seg import Sapiens2SegEstimator
        est = Sapiens2SegEstimator()

        seg_mask = np.array([[0, 255], [1, 3]], dtype=np.uint8)
        body_mask = est.get_body_mask(seg_mask)
        assert body_mask.shape == seg_mask.shape
        assert set(np.unique(body_mask)).issubset({0, 1})

        landmarks = np.array(
            [[0.75, 0.0, 1.0], [np.nan, np.nan, 0.0]], dtype=float
        )
        parts = est.sample_at_landmarks(seg_mask, landmarks, flipped=False)
        assert parts[0] == "Unknown"  # 255 is out of range
        assert parts[1] is None


# ── Constants ────────────────────────────────────────────────────────

class TestSapiens2Constants:

    def test_sapiens2_seg_classes_length(self):
        from myogait.constants import SAPIENS2_SEG_CLASSES
        assert len(SAPIENS2_SEG_CLASSES) == 29

    def test_sapiens2_seg_classes_start_end(self):
        from myogait.constants import SAPIENS2_SEG_CLASSES
        assert SAPIENS2_SEG_CLASSES[0] == "Background"
        assert SAPIENS2_SEG_CLASSES[-1] == "Tongue"

    def test_sapiens2_body_indices_exclude_background_clothing(self):
        from myogait.constants import SAPIENS2_SEG_CLASSES, SAPIENS2_SEG_BODY_INDICES
        for idx in SAPIENS2_SEG_BODY_INDICES:
            cls = SAPIENS2_SEG_CLASSES[idx]
            assert cls not in ("Background", "Apparel", "Eyeglass",
                               "Lower_Clothing", "Upper_Clothing",
                               "Left_Shoe", "Right_Shoe",
                               "Left_Sock", "Right_Sock")

    def test_sapiens2_vs_v1_classes_diff(self):
        """Sapiens 2 adds Eyeglass — rest of classes are identical."""
        from myogait.constants import GOLIATH_SEG_CLASSES, SAPIENS2_SEG_CLASSES
        assert len(SAPIENS2_SEG_CLASSES) == len(GOLIATH_SEG_CLASSES) + 1
        v2_without_eyeglass = [
            c for c in SAPIENS2_SEG_CLASSES if c != "Eyeglass"
        ]
        assert v2_without_eyeglass == GOLIATH_SEG_CLASSES


# ── Extract integration ──────────────────────────────────────────────

class TestSapiens2ExtractIntegration:

    def test_sapiens2_size_from_model(self):
        from myogait.extract import _sapiens2_size_from_model
        assert _sapiens2_size_from_model("sapiens2-quick") == "0.4b"
        assert _sapiens2_size_from_model("sapiens2-mid") == "0.8b"
        assert _sapiens2_size_from_model("sapiens2-top") == "1b"
        assert _sapiens2_size_from_model("sapiens2-ultra") == "5b"
        # Default for non-sapiens2 models
        assert _sapiens2_size_from_model("mediapipe") == "0.4b"

    def test_model_size_map_has_all_sapiens2_variants(self):
        from myogait.extract import _MODEL_TO_SAPIENS2_SIZE
        assert "sapiens2-quick" in _MODEL_TO_SAPIENS2_SIZE
        assert "sapiens2-mid" in _MODEL_TO_SAPIENS2_SIZE
        assert "sapiens2-top" in _MODEL_TO_SAPIENS2_SIZE
        assert "sapiens2-ultra" in _MODEL_TO_SAPIENS2_SIZE
