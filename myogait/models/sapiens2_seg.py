"""Sapiens 2 body-part segmentation (Meta AI, ICLR 2026).

Loads Sapiens 2 segmentation models from Facebook's HuggingFace repos.
Same input pipeline as Sapiens 2 pose (1024x768).

Output: (1, 29, 1024, 768) class logits for 29 body-part classes.
Post-processed via argmax to a per-pixel class mask.

Note: Sapiens 2 uses 29 classes (adds ``Eyeglass`` at index 2 compared
to Sapiens v1's 28 classes).  Use ``SAPIENS2_SEG_CLASSES`` from
``myogait.constants`` for correct label lookup.

Available models
----------------
=========  ======================================
 Size       HuggingFace repo
=========  ======================================
 0.4b       facebook/sapiens2-seg-0.4b
 0.8b       facebook/sapiens2-seg-0.8b
 1b         facebook/sapiens2-seg-1b
 5b         facebook/sapiens2-seg-5b
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
from ..constants import SAPIENS2_SEG_CLASSES, SAPIENS2_SEG_BODY_INDICES

logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────
_SEG_MODELS = {
    "0.4b": (
        "sapiens2_0.4b_seg.safetensors",
        "facebook/sapiens2-seg-0.4b",
    ),
    "0.8b": (
        "sapiens2_0.8b_seg.safetensors",
        "facebook/sapiens2-seg-0.8b",
    ),
    "1b": (
        "sapiens2_1b_seg.safetensors",
        "facebook/sapiens2-seg-1b",
    ),
    "5b": (
        "sapiens2_5b_seg.safetensors",
        "facebook/sapiens2-seg-5b",
    ),
}
_SEG_SHA256 = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_SEG_04B_SHA256"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_SEG_08B_SHA256"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_SEG_1B_SHA256"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_SEG_5B_SHA256"),
}
_SEG_REVISIONS = {
    "0.4b": os.getenv("MYOGAIT_SAPIENS2_SEG_04B_REVISION", "main"),
    "0.8b": os.getenv("MYOGAIT_SAPIENS2_SEG_08B_REVISION", "main"),
    "1b": os.getenv("MYOGAIT_SAPIENS2_SEG_1B_REVISION", "main"),
    "5b": os.getenv("MYOGAIT_SAPIENS2_SEG_5B_REVISION", "main"),
}


def _find_seg_model(model_size: str, model_path: Optional[str] = None) -> str:
    """Locate segmentation model on disk; auto-download on first use."""
    if model_path and Path(model_path).exists():
        return model_path

    if model_size not in _SEG_MODELS:
        raise ValueError(
            f"Unknown Sapiens 2 seg model size '{model_size}'. "
            f"Available: {', '.join(_SEG_MODELS.keys())}"
        )

    filename, repo_id = _SEG_MODELS[model_size]
    expected_sha256 = _SEG_SHA256[model_size]
    revision = _SEG_REVISIONS[model_size]

    # Search local paths (SafeTensors and TorchScript)
    ts_filename = filename.replace(".safetensors", ".pt2")
    for d in _DEFAULT_MODEL_PATHS:
        for fn in (filename, ts_filename):
            p = d / fn
            if p.exists():
                if fn == filename:
                    _verify_model_integrity(
                        str(p), expected_sha256,
                        f"sapiens2-seg-{model_size}",
                    )
                return str(p)

    logger.info(
        f"Sapiens 2 seg {model_size} not found locally — downloading..."
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Sapiens 2 seg {model_size} model not found and huggingface-hub "
            f"is not installed.  Install with: pip install huggingface-hub"
        )
    if revision in {"main", "master"}:
        logger.warning(
            "Sapiens 2 seg %s uses mutable revision '%s'. "
            "Set MYOGAIT_SAPIENS2_SEG_%s_REVISION to a commit hash "
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
    _verify_model_integrity(path, expected_sha256, f"sapiens2-seg-{model_size}")
    logger.info(f"Downloaded: {path}")
    return path


class Sapiens2SegEstimator:
    """Sapiens 2 29-class body-part segmentation estimator.

    Not a pose extractor — used as an auxiliary processor to provide
    body masks and per-landmark body-part labels.

    Sapiens 2 adds an ``Eyeglass`` class (index 2) compared to v1's
    28-class layout.  Use :data:`myogait.constants.SAPIENS2_SEG_CLASSES`
    for correct class-name lookup.

    Supports CUDA, Intel XPU (via IPEX), and CPU backends.
    """

    classes = SAPIENS2_SEG_CLASSES
    n_classes = 29

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
        path = _find_seg_model(self.model_size, self.model_path)
        self._device = _get_device()
        logger.info(
            f"Loading Sapiens 2 seg {self.model_size} on {self._device}..."
        )
        self._model = _load_model(path, self._device)
        logger.info("Sapiens 2 seg ready.")

    def teardown(self):
        self._model = None
        self._device = None

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return (H, W) segmentation mask with class indices 0-28."""
        if self._model is None:
            self.setup()

        import torch

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tensor, pad_info = _preprocess(frame_bgr)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)

        # out shape: (1, 29, 1024, 768) — crop to content (remove letterbox)
        seg_mask = out[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        pl, pt, cw, ch = pad_info
        seg_mask = seg_mask[pt:pt + ch, pl:pl + cw]
        return seg_mask

    def get_body_mask(self, seg_mask: np.ndarray) -> np.ndarray:
        """Return binary mask: 1 where body is detected, 0 elsewhere."""
        mask = np.isin(seg_mask, SAPIENS2_SEG_BODY_INDICES).astype(np.uint8)
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
            parts.append(
                SAPIENS2_SEG_CLASSES[cls_idx]
                if cls_idx < len(SAPIENS2_SEG_CLASSES)
                else "Unknown"
            )

        return parts
