"""Sapiens 2 depth estimation (Meta AI, ICLR 2026).

Loads Sapiens 2 depth models from Facebook's HuggingFace repos.
Same input pipeline as Sapiens 2 pose (1024x768, ImageNet norm).

Output: (1, 1, 1024, 768) relative depth map.  Higher raw values = farther.
Normalised to [0, 1] with min-max on foreground pixels so that
**closer = higher value** (matching the paper's convention).

Available models
----------------
=========  ======================================
 Size       HuggingFace repo
=========  ======================================
 0.4b       facebook/sapiens2-depth-0.4b
 0.8b       facebook/sapiens2-depth-0.8b
 1b         facebook/sapiens2-depth-1b
 5b         facebook/sapiens2-depth-5b
=========  ======================================

Intel XPU
---------
Same compatibility as Sapiens 2 pose — see ``sapiens2.py`` docstring.

References
----------
- Paper: Rawal et al., "Sapiens 2: A Human Foundation Model",
  ICLR 2026.  https://arxiv.org/abs/2604.21681
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import ensure_xpu_torch
from .sapiens import (
    _DEFAULT_MODEL_PATHS,
    _verify_model_integrity,
    _get_device,
)
from .sapiens2 import _preprocess, _load_model

logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────
_DEPTH_MODELS = {
    "0.4b": (
        "sapiens2_0.4b_depth.safetensors",
        "facebook/sapiens2-depth-0.4b",
    ),
    "0.8b": (
        "sapiens2_0.8b_depth.safetensors",
        "facebook/sapiens2-depth-0.8b",
    ),
    "1b": (
        "sapiens2_1b_depth.safetensors",
        "facebook/sapiens2-depth-1b",
    ),
    "5b": (
        "sapiens2_5b_depth.safetensors",
        "facebook/sapiens2-depth-5b",
    ),
}
_DEPTH_SHA256 = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_04B_SHA256"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_08B_SHA256"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_1B_SHA256"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_5B_SHA256"),
}
_DEPTH_REVISIONS = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_04B_REVISION", "main"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_08B_REVISION", "main"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_1B_REVISION", "main"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_DEPTH_5B_REVISION", "main"),
}


def _find_depth_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Locate depth model on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _DEPTH_MODELS:
        raise ValueError(
            f"Unknown Sapiens 2 depth model size '{model_size}'. "
            f"Available: {', '.join(_DEPTH_MODELS.keys())}"
        )

    filename, repo_id = _DEPTH_MODELS[model_size]
    expected_sha256 = _DEPTH_SHA256[model_size]
    revision = _DEPTH_REVISIONS[model_size]

    # Search local paths (SafeTensors and TorchScript)
    ts_filename = filename.replace(".safetensors", ".pt2")
    for d in _DEFAULT_MODEL_PATHS:
        for fn in (filename, ts_filename):
            p = d / fn
            if p.exists():
                if fn == filename:
                    _verify_model_integrity(
                        str(p), expected_sha256,
                        f"sapiens2-depth-{model_size}",
                    )
                return str(p)

    # Auto-download
    logger.info(
        f"Sapiens 2 depth {model_size} not found locally — downloading..."
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens 2 depth {model_size} model not found and huggingface-hub "
            f"is not installed.  Install with: pip install huggingface-hub"
        )
    if revision in {"main", "master"}:
        logger.warning(
            "Sapiens 2 depth %s uses mutable revision '%s'. "
            "Set MYOGAIT_SAPIENS2_DEPTH_%s_REVISION to a commit hash "
            "for strict pinning.",
            model_size,
            revision,
            model_size.replace(".", "").upper(),
        )
    dest_dir = str(Path.home() / ".myogait" / "models")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        revision=revision,
    )
    _verify_model_integrity(path, expected_sha256, f"sapiens2-depth-{model_size}")
    logger.info(f"Downloaded: {path}")
    return path


class Sapiens2DepthEstimator:
    """Sapiens 2 monocular relative-depth estimator.

    Not a pose extractor — used as an auxiliary processor alongside
    a Sapiens 2 pose model to add per-landmark depth values.

    Supports CUDA, Intel XPU (via IPEX), and CPU backends.
    """

    def __init__(self, model_size: str = "0.4b", model_path: Optional[str] = None):
        self.model_size = model_size
        self.model_path = model_path
        self._model = None
        self._device = None

    def setup(self):
        import torch  # noqa: F401
        ensure_xpu_torch()
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            pass
        path = _find_depth_model(self.model_size, self.model_path)
        self._device = _get_device()
        logger.info(
            f"Loading Sapiens 2 depth {self.model_size} on {self._device}..."
        )
        self._model = _load_model(path, self._device)
        logger.info("Sapiens 2 depth ready.")

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
        tensor, pad_info = _preprocess(frame_bgr)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)

        # out shape: (1, 1, 1024, 768) — crop to content (remove letterbox)
        depth = out[0, 0].cpu().numpy()
        pl, pt, cw, ch = pad_info
        depth = depth[pt:pt + ch, pl:pl + cw]

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
