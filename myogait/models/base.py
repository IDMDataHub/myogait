"""Base class for pose extractors."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def ensure_xpu_torch():
    """On Windows, auto-upgrade CPU-only PyTorch to the XPU build.

    PyPI distributes a CPU-only ``torch`` wheel for Windows.  Intel Arc / Xe
    GPUs require the XPU build from PyTorch's dedicated index.  This function
    detects the situation, upgrades ``torch`` automatically, and **restarts
    the current Python process** so the new build is loaded transparently.

    Call this in every extractor's ``setup()`` before touching the GPU.
    On Linux/macOS or when CUDA/XPU is already available the function is a
    no-op.
    """
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

    logger.warning(
        "Detected CPU-only PyTorch (%s) on Windows. "
        "Upgrading to XPU build for Intel Arc GPU support...",
        torch.__version__,
    )
    import os
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
