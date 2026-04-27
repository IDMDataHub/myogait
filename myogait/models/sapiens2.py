"""Sapiens 2 pose extractors (Meta AI, ICLR 2026).

Loads Sapiens 2 models from SafeTensors checkpoints via the ``sapiens``
package (from the ``facebookresearch/sapiens2`` repository), or from
TorchScript (``.pt2``) files when available.  Models are downloaded
automatically from Facebook's official HuggingFace repos on first use.

Model output: ``(1, 308, 256, 192)`` heatmaps for 308 Goliath keypoints
(identical keypoint layout to Sapiens v1 — "Sociopticon" = "Goliath").
Converted to 17 COCO keypoints via argmax on selected heatmaps.
Full 308 Goliath keypoints are also returned as auxiliary data.

Available models
----------------
=========  ======  ======================================
 Name       Params  HuggingFace repo
=========  ======  ======================================
 0.4b       ~400 M  facebook/sapiens2-pose-0.4b
 0.8b       ~800 M  facebook/sapiens2-pose-0.8b
 1b         ~1.1 B  facebook/sapiens2-pose-1b
 5b         ~5 B    facebook/sapiens2-pose-5b
=========  ======  ======================================

Intel XPU compatibility
-----------------------
Sapiens 2 uses standard PyTorch ops (``F.scaled_dot_product_attention``,
``nn.RMSNorm``, ``nn.Linear``, ``F.interpolate``).  No custom CUDA
kernels are required, so inference runs on Intel Arc / Xe GPUs via
Intel Extension for PyTorch (IPEX).  Avoid ``torch.compile`` on XPU;
eager mode is used automatically.

References
----------
- Paper: Rawal et al., "Sapiens 2: A Human Foundation Model",
  ICLR 2026. https://arxiv.org/abs/2604.21681
- Code:  https://github.com/facebookresearch/sapiens2
- Models: https://huggingface.co/facebook/sapiens2
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BasePoseExtractor, ensure_xpu_torch, letterbox_resize
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

# Input resolution expected by the model (same as Sapiens v1)
_INPUT_H, _INPUT_W = 1024, 768

# Padding ratio around detected bounding box (20% on each side)
_BBOX_PAD_RATIO = 0.20

# ImageNet normalization (identical to Sapiens v1)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Local paths searched before downloading
_DEFAULT_MODEL_PATHS = [
    Path.home() / ".myogait" / "models",
    Path.cwd() / "models",
]

# ── Model registry ───────────────────────────────────────────────────
# Each entry: (filename, HuggingFace repo_id)

_MODELS = {
    "0.4b": (
        "sapiens2_0.4b_pose.safetensors",
        "facebook/sapiens2-pose-0.4b",
    ),
    "0.8b": (
        "sapiens2_0.8b_pose.safetensors",
        "facebook/sapiens2-pose-0.8b",
    ),
    "1b": (
        "sapiens2_1b_pose.safetensors",
        "facebook/sapiens2-pose-1b",
    ),
    "5b": (
        "sapiens2_5b_pose.safetensors",
        "facebook/sapiens2-pose-5b",
    ),
}

# Optional integrity metadata from environment variables.
_MODEL_SHA256 = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_04B_SHA256"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_08B_SHA256"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_1B_SHA256"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_5B_SHA256"),
}
_MODEL_REVISIONS = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_04B_REVISION", "main"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_08B_REVISION", "main"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_1B_REVISION", "main"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_5B_REVISION", "main"),
}
_STRICT_MODEL_CHECKSUM = os.getenv(
    "MYOGAIT_STRICT_MODEL_CHECKSUM", ""
).strip().lower() in {"1", "true", "yes", "on"}

# Back-compat alias used in tests
_MODEL_FILENAMES = {k: v[0] for k, v in _MODELS.items()}


# ── Shared utilities (reused from sapiens v1) ───────────────────────
# Import common helpers rather than duplicating them.
from .sapiens import (  # noqa: E402
    _verify_model_integrity,
    _get_device,
    _heatmaps_to_coco,
    _heatmaps_to_all,
    _crop_and_pad,
    _remap_landmarks,
    _person_detector,
)


def download_model(model_size: str = "0.4b", dest: Optional[str] = None) -> str:
    """Download a Sapiens 2 model from Facebook's HuggingFace repo.

    Models are hosted publicly by Meta at
    ``huggingface.co/facebook/sapiens2-pose-*``.

    Parameters
    ----------
    model_size : str
        Model variant: ``"0.4b"``, ``"0.8b"``, ``"1b"``, or ``"5b"``.
    dest : str, optional
        Destination directory (default ``~/.myogait/models/``).

    Returns
    -------
    str
        Path to the downloaded model file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for model download. "
            "Install with: pip install huggingface-hub"
        )

    if model_size not in _MODELS:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Available: {', '.join(_MODELS.keys())}"
        )

    filename, repo_id = _MODELS[model_size]
    revision = _MODEL_REVISIONS[model_size]
    expected_sha256 = _MODEL_SHA256[model_size]
    dest_dir = dest or str(Path.home() / ".myogait" / "models")

    if revision in {"main", "master"}:
        logger.warning(
            "Sapiens 2 %s is using mutable HuggingFace revision '%s'. "
            "Set MYOGAIT_SAPIENS2_%s_REVISION to a commit hash for strict pinning.",
            model_size,
            revision,
            model_size.replace(".", "").upper(),
        )

    logger.info(
        f"Downloading Sapiens 2 {model_size} from {repo_id} "
        f"(revision={revision}) to {dest_dir}..."
    )
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        revision=revision,
    )
    _verify_model_integrity(path, expected_sha256, f"sapiens2-{model_size}")
    logger.info(f"Downloaded: {path}")
    return path


def _find_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Find model file on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _MODELS:
        raise ValueError(
            f"Unknown Sapiens 2 model size '{model_size}'. "
            f"Available: {', '.join(_MODELS.keys())}"
        )

    filename = _MODELS[model_size][0]
    expected_sha256 = _MODEL_SHA256[model_size]

    # Search local paths (also look for TorchScript variants)
    ts_filename = filename.replace(".safetensors", ".pt2")
    for d in _DEFAULT_MODEL_PATHS:
        for fn in (filename, ts_filename):
            p = d / fn
            if p.exists():
                if fn == filename:
                    _verify_model_integrity(
                        str(p), expected_sha256, f"sapiens2-{model_size}"
                    )
                return str(p)

    # Auto-download from HuggingFace
    logger.info(
        f"Sapiens 2 {model_size} not found locally — "
        f"downloading from HuggingFace (first use only)..."
    )
    try:
        return download_model(model_size)
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens 2 {model_size} model not found locally and "
            f"huggingface-hub is not installed for auto-download. "
            f"Install with: pip install huggingface-hub\n"
            f"Or download manually:\n"
            f"  myogait download sapiens2-{model_size}\n"
            f"Searched: {[str(d / filename) for d in _DEFAULT_MODEL_PATHS]}"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Sapiens 2 {model_size} model not found. "
            f"Auto-download failed: {e}\n"
            f"Download manually with: myogait download sapiens2-{model_size}\n"
            f"Searched: {[str(d / filename) for d in _DEFAULT_MODEL_PATHS]}"
        )


def _load_model(path: str, device):
    """Load a Sapiens 2 model from either TorchScript or SafeTensors.

    TorchScript (.pt2) files are loaded directly with ``torch.jit.load()``.
    SafeTensors files require the ``sapiens`` package from the
    ``facebookresearch/sapiens2`` repository to reconstruct the model
    architecture before loading weights.
    """
    import torch

    # TorchScript: self-contained, no extra dependencies
    if path.endswith((".pt2", ".pt", ".pth")):
        logger.info("Loading Sapiens 2 from TorchScript: %s", path)
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model

    # SafeTensors: needs sapiens2 package for model architecture
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file  # noqa: F401
        except ImportError:
            raise ImportError(
                "safetensors is required for Sapiens 2 SafeTensors checkpoints. "
                "Install with: pip install safetensors"
            )
        try:
            from sapiens.pose.models import init_model as _sapiens_init_model
        except ImportError:
            raise ImportError(
                "Sapiens 2 SafeTensors checkpoints require the sapiens2 package "
                "to reconstruct the model architecture.\n"
                "Install from source:\n"
                "  git clone https://github.com/facebookresearch/sapiens2\n"
                "  cd sapiens2 && pip install -e .\n\n"
                "Alternatively, convert checkpoints to TorchScript (.pt2) and "
                "place them in ~/.myogait/models/\n"
                "See: https://github.com/facebookresearch/sapiens2#export"
            )
        # The sapiens2 init_model handles config + weight loading
        # We pass device as string since init_model expects that
        device_str = str(device)
        # Use XPU-safe device: avoid torch.compile on Intel GPUs
        if "xpu" in device_str:
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        logger.info(
            "Loading Sapiens 2 from SafeTensors via sapiens2 package: %s", path
        )
        model = _sapiens_init_model(path, device=device_str)
        return model

    # Unknown format — try TorchScript as fallback
    logger.warning(
        "Unknown checkpoint format for %s — trying TorchScript loader.", path
    )
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


def _preprocess(frame_bgr: np.ndarray):
    """Letterbox-resize, normalize, and convert to tensor.

    Identical preprocessing to Sapiens v1 (1024x768, ImageNet norm).

    Returns
    -------
    tensor : torch.Tensor
        (1, 3, H, W) normalized input tensor.
    pad_info : tuple
        (pad_left, pad_top, content_w, content_h) describing where the
        actual image content sits inside the letterboxed canvas.
    """
    import torch

    img, pad_left, pad_top, content_w, content_h = letterbox_resize(
        frame_bgr, _INPUT_W, _INPUT_H,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, (pad_left, pad_top, content_w, content_h)


# ── Extractors ───────────────────────────────────────────────────────

def _make_sapiens2_extractor(name, model_size, label):
    """Factory to avoid duplicating the extractor class body."""

    class _Sapiens2Extractor(BasePoseExtractor):
        __doc__ = (
            f"Sapiens 2 {label} — {model_size.upper()} parameters.\n\n"
            f"Downloads ``{_MODELS[model_size][0]}`` from\n"
            f"``{_MODELS[model_size][1]}`` (auto-downloaded on first use).\n\n"
            f"Supports CUDA, Intel XPU (via IPEX), and CPU backends.\n"
            f"SafeTensors loading requires the sapiens2 package; "
            f"TorchScript (.pt2) files load without extra dependencies.\n\n"
            f"Reference: Rawal et al., *Sapiens 2: A Human Foundation "
            f"Model*, ICLR 2026.\n"
            f"https://arxiv.org/abs/2604.21681"
        )

        landmark_names = COCO_LANDMARK_NAMES
        n_landmarks = 17
        is_coco_format = True

        def __init__(self, model_path: Optional[str] = None, **kwargs):
            self.name = name
            self.model_path = model_path
            self._model = None
            self._device = None
            self._model_size = model_size

        def setup(self):
            try:
                import torch  # noqa: F401
            except ImportError:
                raise ImportError(
                    "PyTorch >= 2.7 is required for Sapiens 2. Install with:\n"
                    "  pip install myogait[sapiens2]\n"
                    "For Intel Arc / Xe GPU acceleration:\n"
                    "  pip install torch --index-url "
                    "https://download.pytorch.org/whl/xpu"
                )
            # Auto-upgrade CPU torch to XPU on Windows (Intel Arc)
            ensure_xpu_torch()
            # Load IPEX if available (enables XPU device for Intel Arc)
            try:
                import intel_extension_for_pytorch  # noqa: F401
            except ImportError:
                pass
            path = _find_model(self._model_size, self.model_path)
            self._device = _get_device()
            logger.info(
                f"Loading Sapiens 2 {label} from {path} on {self._device}..."
            )
            self._model = _load_model(path, self._device)
            # Initialize person detector for top-down crop
            _person_detector.setup()
            logger.info(f"Sapiens 2 {label} ready.")

        def teardown(self):
            self._model = None
            self._device = None

        def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
            if self._model is None:
                self.setup()

            import torch

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h_frame, w_frame = frame_bgr.shape[:2]

            # Top-down: detect person and crop before Sapiens 2
            bbox = _person_detector.detect(frame_rgb)
            if bbox is None:
                return None
            crop, x_off, y_off, crop_w, crop_h = _crop_and_pad(frame_bgr, bbox)
            tensor, pad_info = _preprocess(crop)

            tensor = tensor.to(self._device)

            with torch.no_grad():
                out = self._model(tensor)

            heatmaps = out[0].cpu().numpy()
            coco_lm = _heatmaps_to_coco(heatmaps, pad_info)
            all_lm = _heatmaps_to_all(heatmaps, pad_info)

            # Remap from crop-normalized to frame-normalized coordinates
            coco_lm = _remap_landmarks(
                coco_lm, x_off, y_off, crop_w, crop_h, w_frame, h_frame
            )
            all_lm = _remap_landmarks(
                all_lm, x_off, y_off, crop_w, crop_h, w_frame, h_frame
            )

            return {
                "landmarks": coco_lm,
                "auxiliary_goliath308": all_lm,
            }

    _Sapiens2Extractor.__name__ = f"Sapiens2{label.replace(' ', '')}Extractor"
    _Sapiens2Extractor.__qualname__ = _Sapiens2Extractor.__name__
    return _Sapiens2Extractor


Sapiens2QuickExtractor = _make_sapiens2_extractor(
    "sapiens2-quick", "0.4b", "0.4B"
)
Sapiens2MidExtractor = _make_sapiens2_extractor(
    "sapiens2-mid", "0.8b", "0.8B"
)
Sapiens2TopExtractor = _make_sapiens2_extractor(
    "sapiens2-top", "1b", "1B"
)
Sapiens2UltraExtractor = _make_sapiens2_extractor(
    "sapiens2-ultra", "5b", "5B"
)
