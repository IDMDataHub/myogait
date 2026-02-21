"""RTMW whole-body 133-keypoint extractor via rtmlib.

RTMW (Real-Time Multi-person Whole-body) outputs 133 keypoints:
body (17) + feet (6) + face (68) + left hand (21) + right hand (21).

Uses rtmlib for lightweight ONNX-based inference (no MMPose/mmcv
needed).  The first 17 keypoints are COCO-format body keypoints
used by the gait pipeline; all 133 are stored as auxiliary data.

Available modes
---------------
==============  ====================  =============================
 Mode            Detector              Pose model
==============  ====================  =============================
 performance     YOLOX-m (HumanArt)    RTMW-x-l 384x288
 balanced        YOLOX-m (HumanArt)    RTMW-x-l 256x192
 lightweight     YOLOX-tiny            RTMW-l-m 256x192
==============  ====================  =============================

References
----------
- Paper: Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation
  based on MMPose", 2023.
- RTMW: https://github.com/open-mmlab/mmpose/tree/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14
- rtmlib: https://github.com/Tau-J/rtmlib
"""

import logging
from typing import Optional

import numpy as np

from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES, WHOLEBODY_LANDMARK_NAMES

logger = logging.getLogger(__name__)


class RTMWPoseExtractor(BasePoseExtractor):
    """RTMW — whole-body 133-keypoint pose estimator.

    Returns 17 COCO keypoints for the gait pipeline, plus all 133
    whole-body keypoints as ``auxiliary_wholebody133``.

    Install: ``pip install myogait[rtmw]``
    """

    name = "rtmw"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, mode: str = "balanced", **kwargs):
        self.mode = mode
        self._wholebody = None

    def setup(self):
        try:
            from rtmlib import Wholebody
        except ImportError:
            raise ImportError(
                "rtmlib is required for RTMW. "
                "Install with: pip install myogait[rtmw]"
            )

        backend = "onnxruntime"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            backend = "opencv"
            logger.warning(
                "onnxruntime not found, falling back to opencv backend. "
                "Install onnxruntime for better performance."
            )

        device = "cpu"
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                device = "cuda"
        except Exception:
            pass

        logger.info(f"Loading RTMW ({self.mode}) with {backend} on {device}...")
        self._wholebody = Wholebody(
            mode=self.mode,
            backend=backend,
            device=device,
        )
        logger.info("RTMW ready.")

    def teardown(self):
        self._wholebody = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
        if self._wholebody is None:
            self.setup()

        import cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        keypoints, scores = self._wholebody(frame_bgr)

        if keypoints is None or len(keypoints) == 0:
            return None

        # keypoints: (N_persons, 133, 2), scores: (N_persons, 133)
        # Use the first / largest person
        if keypoints.ndim == 3:
            # Pick person with highest mean score
            best = int(np.argmax(scores.mean(axis=1)))
            kp = keypoints[best]    # (133, 2)
            sc = scores[best]       # (133,)
        else:
            kp = keypoints  # (133, 2)
            sc = scores     # (133,)

        # Build 17-keypoint COCO array (normalised)
        coco_17 = np.zeros((17, 3))
        coco_17[:, 0] = kp[:17, 0] / w
        coco_17[:, 1] = kp[:17, 1] / h
        coco_17[:, 2] = sc[:17]

        # Build full 133-keypoint array (normalised)
        n_total = min(kp.shape[0], 133)
        all_133 = np.full((133, 3), np.nan)
        all_133[:n_total, 0] = kp[:n_total, 0] / w
        all_133[:n_total, 1] = kp[:n_total, 1] / h
        all_133[:n_total, 2] = sc[:n_total]

        return {
            "landmarks": coco_17,
            "auxiliary_wholebody133": all_133,
        }
