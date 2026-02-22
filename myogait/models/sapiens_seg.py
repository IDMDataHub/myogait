"""Sapiens body-part segmentation (Meta AI, ECCV 2024).

Loads Sapiens TorchScript (.pt2) segmentation models from Facebook's
HuggingFace repos.  Same input pipeline as Sapiens pose (1024x768).

Output: (1, 28, 1024, 768) class logits for 28 Goliath body-part classes.
Post-processed via argmax to a per-pixel class mask.

Available models
----------------
=========  =====  ======================================
 Size       mIoU   HuggingFace repo
=========  =====  ======================================
 0.3b       76.7   facebook/sapiens-seg-0.3b-torchscript
 0.6b       77.8   facebook/sapiens-seg-0.6b-torchscript
 1b         79.9   facebook/sapiens-seg-1b-torchscript
=========  =====  ======================================

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
from ..constants import GOLIATH_SEG_CLASSES, GOLIATH_SEG_BODY_INDICES

logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────
_SEG_MODELS = {
    "0.3b": (
        "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
        "facebook/sapiens-seg-0.3b-torchscript",
    ),
    "0.6b": (
        "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
        "facebook/sapiens-seg-0.6b-torchscript",
    ),
    "1b": (
        "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
        "facebook/sapiens-seg-1b-torchscript",
    ),
}


def _find_seg_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Locate segmentation model on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _SEG_MODELS:
        raise ValueError(
            f"Unknown seg model size '{model_size}'. "
            f"Available: {', '.join(_SEG_MODELS.keys())}"
        )

    filename, repo_id = _SEG_MODELS[model_size]

    for d in _DEFAULT_MODEL_PATHS:
        p = d / filename
        if p.exists():
            return str(p)

    logger.info(f"Sapiens seg {model_size} not found locally — downloading...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens seg {model_size} model not found and huggingface-hub "
            f"is not installed.  Install with: pip install huggingface-hub"
        )
    dest_dir = str(Path.home() / ".myogait" / "models")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
    logger.info(f"Downloaded: {path}")
    return path


class SapiensSegEstimator:
    """Sapiens 28-class body-part segmentation estimator.

    Not a pose extractor — used as an auxiliary processor to provide
    body masks and per-landmark body-part labels.
    """

    classes = GOLIATH_SEG_CLASSES
    n_classes = 28

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
        path = _find_seg_model(self.model_size, self.model_path)
        self._device = _get_device()
        logger.info(f"Loading Sapiens seg {self.model_size} on {self._device}...")
        self._model = torch.jit.load(path, map_location=self._device)
        self._model.eval()
        logger.info("Sapiens seg ready.")

    def teardown(self):
        self._model = None
        self._device = None

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return (H, W) segmentation mask with class indices 0-27."""
        if self._model is None:
            self.setup()

        import torch

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tensor, pad_info = _preprocess(frame_bgr)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)

        # out shape: (1, 28, 1024, 768) — crop to content (remove letterbox)
        seg_mask = out[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        pl, pt, cw, ch = pad_info
        seg_mask = seg_mask[pt:pt + ch, pl:pl + cw]
        return seg_mask

    def get_body_mask(self, seg_mask: np.ndarray) -> np.ndarray:
        """Return binary mask: 1 where body is detected, 0 elsewhere."""
        mask = np.isin(seg_mask, GOLIATH_SEG_BODY_INDICES).astype(np.uint8)
        return mask

    def sample_at_landmarks(
        self,
        seg_mask: np.ndarray,
        landmarks: np.ndarray,
        flipped: bool = False,
    ) -> list:
        """Sample body-part class at landmark positions.

        Returns list of class names (str) or None where landmark is missing.
        """
        sh, sw = seg_mask.shape
        n = landmarks.shape[0]
        parts = []

        for i in range(n):
            x, y = landmarks[i, 0], landmarks[i, 1]
            if np.isnan(x) or np.isnan(y):
                parts.append(None)
                continue
            sx = (1.0 - x) if flipped else x
            px = int(np.clip(sx * sw, 0, sw - 1))
            py = int(np.clip(y * sh, 0, sh - 1))
            cls_idx = int(seg_mask[py, px])
            parts.append(GOLIATH_SEG_CLASSES[cls_idx] if cls_idx < len(GOLIATH_SEG_CLASSES) else "Unknown")

        return parts
