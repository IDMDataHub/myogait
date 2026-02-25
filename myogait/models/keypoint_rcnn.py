"""Detectron2 Keypoint R-CNN pose extractor (top-down).

Uses Meta's Detectron2 DefaultPredictor with Keypoint R-CNN (R50-FPN)
for simultaneous person detection and 17 COCO keypoint estimation.

Reference: Wu et al., "Detectron2", 2019.
           He et al., "Mask R-CNN", ICCV 2017.
"""

import logging
import numpy as np
from typing import Optional
from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
_DEFAULT_THRESHOLD = 0.7


class Detectron2PoseExtractor(BasePoseExtractor):
    """Detectron2 Keypoint R-CNN -- 17 COCO keypoints.

    Uses Meta's Keypoint R-CNN (ResNet-50-FPN, 3x schedule) for
    simultaneous person detection and pose estimation in a single pass.
    Model weights are auto-downloaded from the Detectron2 model zoo on
    first use.

    Install: ``pip install myogait[detectron2]``
    """

    name = "detectron2"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, config: str = None, threshold: float = _DEFAULT_THRESHOLD,
                 device: str = "auto"):
        self.config_name = config or _DEFAULT_CONFIG
        self.threshold = threshold
        self.device_name = device
        self._predictor = None

    def setup(self):
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
        except ImportError:
            raise ImportError(
                "Detectron2 is required. "
                "Install with: pip install myogait[detectron2]"
            )

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_name)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold

        if self.device_name == "auto":
            import torch
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            cfg.MODEL.DEVICE = self.device_name

        logger.info(
            "Loading Keypoint R-CNN (%s) on %s...",
            self.config_name, cfg.MODEL.DEVICE,
        )
        self._predictor = DefaultPredictor(cfg)
        logger.info("Detectron2 Keypoint R-CNN ready.")

    def teardown(self):
        self._predictor = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._predictor is None:
            self.setup()

        import cv2

        h, w = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        outputs = self._predictor(frame_bgr)
        instances = outputs["instances"]

        if len(instances) == 0:
            return None

        # Filter for person class (class 0 in COCO)
        classes = instances.pred_classes.cpu().numpy()
        person_mask = classes == 0

        if not person_mask.any():
            return None

        if not instances.has("pred_keypoints"):
            return None

        keypoints = instances.pred_keypoints[person_mask].cpu().numpy()
        scores = instances.scores[person_mask].cpu().numpy()
        boxes = instances.pred_boxes[person_mask].tensor.cpu().numpy()

        if len(keypoints) == 0:
            return None

        # Select the largest person (area * confidence)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(np.argmax(areas * scores))

        kps = keypoints[best_idx]  # (17, 3) [x_pixel, y_pixel, confidence]

        landmarks = np.column_stack([
            kps[:, 0] / w,
            kps[:, 1] / h,
            kps[:, 2],
        ])

        return landmarks
