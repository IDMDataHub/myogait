"""Base class for pose extractors."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


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
