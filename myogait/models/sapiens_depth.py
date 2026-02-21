"""Sapiens depth estimation (Meta AI, ECCV 2024).

Loads Sapiens TorchScript (.pt2) depth models from Facebook's HuggingFace
repos.  Same input pipeline as Sapiens pose (1024x768, ImageNet norm).

Output: (1, 1, 1024, 768) relative depth map.  Higher raw values = farther.
Normalised to [0, 1] with min-max on foreground pixels so that
**closer = higher value** (matching the paper's convention).

Available models
----------------
=========  ======================================
 Size       HuggingFace repo
=========  ======================================
 0.3b       facebook/sapiens-depth-0.3b-torchscript
 0.6b       facebook/sapiens-depth-0.6b-torchscript
 1b         facebook/sapiens-depth-1b-torchscript
 2b         facebook/sapiens-depth-2b-torchscript
=========  ======================================

References
----------
- Paper: Rawal et al., "Sapiens: Foundation for Human Vision Models",
  ECCV 2024.  https://arxiv.org/abs/2408.12569
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import ensure_xpu_torch
from .sapiens import (
    _DEFAULT_MODEL_PATHS,
    _get_device, _preprocess,
)

logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────
_DEPTH_MODELS = {
    "0.3b": (
        "sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        "facebook/sapiens-depth-0.3b-torchscript",
    ),
    "0.6b": (
        "sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        "facebook/sapiens-depth-0.6b-torchscript",
    ),
    "1b": (
        "sapiens_1b_render_people_epoch_88_torchscript.pt2",
        "facebook/sapiens-depth-1b-torchscript",
    ),
    "2b": (
        "sapiens_2b_render_people_epoch_25_torchscript.pt2",
        "facebook/sapiens-depth-2b-torchscript",
    ),
}


def _find_depth_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Locate depth model on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _DEPTH_MODELS:
        raise ValueError(
            f"Unknown depth model size '{model_size}'. "
            f"Available: {', '.join(_DEPTH_MODELS.keys())}"
        )

    filename, repo_id = _DEPTH_MODELS[model_size]

    for d in _DEFAULT_MODEL_PATHS:
        p = d / filename
        if p.exists():
            return str(p)

    # Auto-download
    logger.info(f"Sapiens depth {model_size} not found locally — downloading...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens depth {model_size} model not found and huggingface-hub "
            f"is not installed.  Install with: pip install huggingface-hub"
        )
    dest_dir = str(Path.home() / ".myogait" / "models")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
    logger.info(f"Downloaded: {path}")
    return path


class SapiensDepthEstimator:
    """Sapiens monocular relative-depth estimator.

    Not a pose extractor — used as an auxiliary processor alongside
    a Sapiens pose model to add per-landmark depth values.
    """

    def __init__(self, model_size: str = "0.3b", model_path: Optional[str] = None):
        self.model_size = model_size
        self.model_path = model_path
        self._model = None
        self._device = None

    def setup(self):
        import torch
        ensure_xpu_torch()
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            pass
        path = _find_depth_model(self.model_size, self.model_path)
        self._device = _get_device()
        logger.info(f"Loading Sapiens depth {self.model_size} on {self._device}...")
        self._model = torch.jit.load(path, map_location=self._device)
        self._model.eval()
        logger.info("Sapiens depth ready.")

    def teardown(self):
        self._model = None
        self._device = None

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return (H, W) relative depth map normalised to [0, 1].

        Closer pixels have higher values.
        """
        if self._model is None:
            self.setup()

        import torch

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tensor = _preprocess(frame_bgr).to(self._device)

        with torch.no_grad():
            out = self._model(tensor)

        # out shape: (1, 1, 1024, 768)
        depth = out[0, 0].cpu().numpy()

        # Normalise: closer = higher (invert raw values)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = 1.0 - (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth

    def sample_at_landmarks(
        self,
        depth_map: np.ndarray,
        landmarks: np.ndarray,
        flipped: bool = False,
    ) -> np.ndarray:
        """Sample depth values at landmark positions.

        Parameters
        ----------
        depth_map : (H, W) array from ``process_frame``.
        landmarks : (N, 3) array with columns [x_norm, y_norm, vis].
        flipped : bool
            If True, the landmarks have been horizontally flipped
            (x was mirrored).  The depth map is still in original
            orientation, so we invert x before sampling.

        Returns
        -------
        (N,) array of depth values (NaN where landmark is missing).
        """
        dh, dw = depth_map.shape
        n = landmarks.shape[0]
        depths = np.full(n, np.nan)

        for i in range(n):
            x, y = landmarks[i, 0], landmarks[i, 1]
            if np.isnan(x) or np.isnan(y):
                continue
            sx = (1.0 - x) if flipped else x
            px = int(np.clip(sx * dw, 0, dw - 1))
            py = int(np.clip(y * dh, 0, dh - 1))
            depths[i] = float(depth_map[py, px])

        return depths
