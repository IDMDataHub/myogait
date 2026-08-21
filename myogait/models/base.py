"""Base class for pose extractors."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def letterbox_resize(img, target_w, target_h):
    """Resize image preserving aspect ratio, pad remaining area with black.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W, 3), any dtype.
    target_w, target_h : int
        Desired output dimensions.

    Returns
    -------
    canvas : np.ndarray
        Resized+padded image at (target_h, target_w, 3).
    pad_left : int
        Horizontal padding (left side) in pixels.
    pad_top : int
        Vertical padding (top side) in pixels.
    content_w : int
        Width of the actual image content within the canvas.
    content_h : int
        Height of the actual image content within the canvas.
    """
    import cv2

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, pad_left, pad_top, new_w, new_h


def ensure_xpu_torch(auto_upgrade: bool = False):
    """On Windows, detect a CPU-only PyTorch when an Intel XPU build is
    needed — and (only on explicit opt-in) upgrade and restart.

    PyPI distributes a CPU-only ``torch`` wheel for Windows; Intel
    Arc / Xe GPUs require the XPU build from PyTorch's dedicated index.

    By default this function only **detects and warns** with the manual
    install command.  The automatic path — a synchronous
    ``pip install --force-reinstall`` followed by ``os.execv`` that
    **replaces the current process** — is destructive inside any
    long-lived host (a web app, a notebook kernel, a job runner: every
    concurrent session dies), so it never runs implicitly.  Opt in
    with ``auto_upgrade=True`` or the environment variable
    ``MYOGAIT_AUTO_XPU=1`` (intended for the ``myogait setup-sapiens2``
    CLI and other single-purpose processes).

    On Linux/macOS, or when CUDA/XPU is already available, this is a
    no-op.
    """
    import os
    import platform

    if platform.system() != "Windows":
        return

    try:
        import torch
    except ImportError:
        return  # torch not installed yet — nothing to upgrade

    if torch.cuda.is_available():
        return
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return

    _is_cpu_build = "+cpu" in torch.__version__ or not hasattr(torch, "xpu")
    if not _is_cpu_build:
        return

    if not (auto_upgrade or os.environ.get("MYOGAIT_AUTO_XPU") == "1"):
        logger.warning(
            "Detected CPU-only PyTorch (%s) on Windows — Intel Arc/Xe GPUs "
            "need the XPU build. Install it manually:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/xpu\n"
            "(or opt in to automatic upgrade+restart with "
            "ensure_xpu_torch(auto_upgrade=True) / MYOGAIT_AUTO_XPU=1 — "
            "never do this inside a shared or long-lived process).",
            torch.__version__,
        )
        return

    logger.warning(
        "Detected CPU-only PyTorch (%s) on Windows. "
        "Upgrading to XPU build for Intel Arc GPU support...",
        torch.__version__,
    )
    import subprocess
    import sys

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "torch",
            "--index-url", "https://download.pytorch.org/whl/xpu",
        ])
        logger.warning(
            "PyTorch XPU installed successfully. "
            "Restarting process to load the new build..."
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except subprocess.CalledProcessError:
        logger.warning(
            "Could not auto-install PyTorch XPU. Install manually:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/xpu"
        )


@dataclass
class PoseFrame:
    """Pose detection result for a single video frame."""
    frame_index: int
    landmarks: np.ndarray  # Shape: (N, 3) - x, y, visibility
    landmark_confidences: np.ndarray
    overall_confidence: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    inverted: bool = False


class BasePoseExtractor(ABC):
    """Abstract base class for all pose extractors.

    Subclasses must implement process_frame() which takes an RGB frame
    and returns landmarks as a numpy array.
    """

    name: str = "Base"
    landmark_names: List[str] = []
    n_landmarks: int = 0
    is_coco_format: bool = False

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Process a single RGB frame and return landmarks.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3).

        Returns:
            Array of shape (N, 3) with [x_normalized, y_normalized, visibility]
            where x and y are in [0, 1] relative to image dimensions.
            Returns None if no pose detected.

            May also return a dict with keys ``"landmarks"`` (the primary
            array) and optional auxiliary keys like
            ``"auxiliary_goliath308"`` for dense keypoint sets (Sapiens).
        """
        pass

    def setup(self):
        """Initialize the model. Called before processing starts."""
        pass

    def teardown(self):
        """Release model resources. Called after processing ends."""
        pass

    @staticmethod
    def release_gpu_memory():
        """Explicitly return freed tensors to the CUDA/XPU allocator.

        Dropping the last reference to a model (``self._model = None``)
        releases tensors to *PyTorch's caching allocator*, not to the
        device: on a long-lived process that cycles through heavy
        models (Sapiens 2 up to 5B parameters), VRAM stays reserved and
        fragments across sessions.  GPU extractors should call this at
        the end of ``teardown()``.  Safe no-op when torch or the device
        is absent.
        """
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:  # never let cleanup raise
            pass
