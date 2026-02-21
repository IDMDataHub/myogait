"""Sapiens pose extractors (Meta AI, ECCV 2024).

Loads Sapiens TorchScript (.pt2) models directly with ``torch.jit.load()``.
No external ``sapiens_inference`` dependency required.  Models are
downloaded automatically from Facebook's official HuggingFace repos
on first use.

Model output: ``(1, 308, 256, 192)`` heatmaps for 308 Goliath keypoints.
Converted to 17 COCO keypoints via argmax on selected heatmaps.
Full 308 Goliath keypoints are also returned as auxiliary data.

Available models
----------------
=========  ======  ============  ======================================
 Name       Params  AP (Goliath)  HuggingFace repo
=========  ======  ============  ======================================
 0.3b       336 M   57.3          facebook/sapiens-pose-0.3b-torchscript
 0.6b       664 M   60.9          facebook/sapiens-pose-0.6b-torchscript
 1b         1.1 B   63.9          facebook/sapiens-pose-1b-torchscript
=========  ======  ============  ======================================

References
----------
- Paper: Rawal et al., "Sapiens: Foundation for Human Vision Models",
  ECCV 2024. https://arxiv.org/abs/2408.12569
- Code:  https://github.com/facebookresearch/sapiens
- Models: https://huggingface.co/collections/facebook/sapiens-66d22047daa6402d565cb2fc
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BasePoseExtractor, ensure_xpu_torch
from ..constants import COCO_LANDMARK_NAMES, GOLIATH_TO_COCO

logger = logging.getLogger(__name__)

# Input resolution expected by the model
_INPUT_H, _INPUT_W = 1024, 768

# ImageNet normalization
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
    "0.3b": (
        "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2",
        "facebook/sapiens-pose-0.3b-torchscript",
    ),
    "0.6b": (
        "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2",
        "facebook/sapiens-pose-0.6b-torchscript",
    ),
    "1b": (
        "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
        "facebook/sapiens-pose-1b-torchscript",
    ),
}

# Back-compat alias used in tests
_MODEL_FILENAMES = {k: v[0] for k, v in _MODELS.items()}


def download_model(model_size: str = "0.3b", dest: Optional[str] = None) -> str:
    """Download a Sapiens model from Facebook's HuggingFace repo.

    Models are hosted publicly by Meta at
    ``huggingface.co/facebook/sapiens-pose-*-torchscript``.

    Parameters
    ----------
    model_size : str
        Model variant: ``"0.3b"``, ``"0.6b"``, or ``"1b"``.
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
    dest_dir = dest or str(Path.home() / ".myogait" / "models")

    logger.info(
        f"Downloading Sapiens {model_size} from {repo_id} "
        f"to {dest_dir}..."
    )
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
    )
    logger.info(f"Downloaded: {path}")
    return path


def _find_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Find model file on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _MODELS:
        raise ValueError(
            f"Unknown Sapiens model size '{model_size}'. "
            f"Available: {', '.join(_MODELS.keys())}"
        )

    filename = _MODELS[model_size][0]

    # Search local paths
    for d in _DEFAULT_MODEL_PATHS:
        p = d / filename
        if p.exists():
            return str(p)

    # Auto-download from HuggingFace
    logger.info(
        f"Sapiens {model_size} not found locally — "
        f"downloading from HuggingFace (first use only)..."
    )
    try:
        return download_model(model_size)
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens {model_size} model not found locally and "
            f"huggingface-hub is not installed for auto-download. "
            f"Install with: pip install huggingface-hub\n"
            f"Or download manually:\n"
            f"  myogait download sapiens-{model_size}\n"
            f"Searched: {[str(d / filename) for d in _DEFAULT_MODEL_PATHS]}"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Sapiens {model_size} model not found. "
            f"Auto-download failed: {e}\n"
            f"Download manually with: myogait download sapiens-{model_size}\n"
            f"Searched: {[str(d / filename) for d in _DEFAULT_MODEL_PATHS]}"
        )


def _get_device():
    """Select the best available device: CUDA > XPU (Intel Arc) > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _preprocess(frame_bgr: np.ndarray):
    """Resize, normalize, and convert to tensor."""
    import torch

    img = cv2.resize(frame_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _heatmaps_to_coco(heatmaps: np.ndarray) -> np.ndarray:
    """Convert selected Goliath heatmaps to (17, 3) COCO keypoints."""
    n_kp, hm_h, hm_w = heatmaps.shape
    landmarks = np.full((17, 3), np.nan)

    for goliath_idx, coco_idx in GOLIATH_TO_COCO.items():
        if goliath_idx >= n_kp:
            continue
        hm = heatmaps[goliath_idx]
        flat_idx = np.argmax(hm)
        y_hm, x_hm = np.unravel_index(flat_idx, (hm_h, hm_w))
        conf = float(hm[y_hm, x_hm])
        landmarks[coco_idx] = [x_hm / hm_w, y_hm / hm_h, max(0.0, min(1.0, conf))]

    return landmarks


def _heatmaps_to_all(heatmaps: np.ndarray) -> np.ndarray:
    """Convert all (N, hm_h, hm_w) heatmaps to (N, 3) keypoints."""
    n_kp, hm_h, hm_w = heatmaps.shape
    landmarks = np.full((n_kp, 3), np.nan)

    for i in range(n_kp):
        hm = heatmaps[i]
        flat_idx = np.argmax(hm)
        y_hm, x_hm = np.unravel_index(flat_idx, (hm_h, hm_w))
        conf = float(hm[y_hm, x_hm])
        landmarks[i] = [x_hm / hm_w, y_hm / hm_h, max(0.0, min(1.0, conf))]

    return landmarks


# ── Extractors ───────────────────────────────────────────────────────

def _make_sapiens_extractor(name, model_size, label):
    """Factory to avoid duplicating the extractor class body."""

    class _SapiensExtractor(BasePoseExtractor):
        __doc__ = (
            f"Sapiens {label} — {model_size.upper()} parameters.\n\n"
            f"Loads ``{_MODELS[model_size][0]}`` from\n"
            f"``{_MODELS[model_size][1]}`` (auto-downloaded on first use).\n\n"
            f"Reference: Rawal et al., *Sapiens: Foundation for Human "
            f"Vision Models*, ECCV 2024.\n"
            f"https://arxiv.org/abs/2408.12569"
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
                import torch
            except ImportError:
                raise ImportError(
                    "PyTorch is required for Sapiens. Install with:\n"
                    "  pip install myogait[sapiens]\n"
                    "For Intel Arc / Xe GPU acceleration on Windows:\n"
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
                f"Loading Sapiens {label} from {path} on {self._device}..."
            )
            self._model = torch.jit.load(path, map_location=self._device)
            self._model.eval()
            logger.info(f"Sapiens {label} ready.")

        def teardown(self):
            self._model = None
            self._device = None

        def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
            if self._model is None:
                self.setup()

            import torch

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            tensor = _preprocess(frame_bgr).to(self._device)

            with torch.no_grad():
                out = self._model(tensor)

            heatmaps = out[0].cpu().numpy()
            return {
                "landmarks": _heatmaps_to_coco(heatmaps),
                "auxiliary_goliath308": _heatmaps_to_all(heatmaps),
            }

    _SapiensExtractor.__name__ = f"Sapiens{label.replace(' ', '')}Extractor"
    _SapiensExtractor.__qualname__ = _SapiensExtractor.__name__
    return _SapiensExtractor


SapiensQuickExtractor = _make_sapiens_extractor(
    "sapiens-quick", "0.3b", "0.3B"
)
SapiensMidExtractor = _make_sapiens_extractor(
    "sapiens-mid", "0.6b", "0.6B"
)
SapiensTopExtractor = _make_sapiens_extractor(
    "sapiens-top", "1b", "1B"
)
