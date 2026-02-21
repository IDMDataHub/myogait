"""HRNet pose extractor via MMPose.

HRNet-W48 with YOLO person detector.
Output: 17 COCO keypoints in pixel coordinates.
"""

import numpy as np
import logging
from typing import Optional
from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

# Default config/checkpoint paths (can be overridden)
HRNET_CONFIG = (
    "configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
)
HRNET_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
)


class HRNETPoseExtractor(BasePoseExtractor):
    """High-Resolution Network pose extractor."""

    name = "hrnet"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, device: str = "cpu", config: str = None, checkpoint: str = None):
        self.device = device
        self.config = config or HRNET_CONFIG
        self.checkpoint = checkpoint or HRNET_CHECKPOINT
        self._pose_model = None
        self._detector = None

    def setup(self):
        try:
            from mmpose.apis import init_model
            from ultralytics import YOLO

            logger.info("Loading HRNet-W48...")
            self._pose_model = init_model(self.config, self.checkpoint, device=self.device)
            self._detector = YOLO("yolov8n.pt")
            logger.info("HRNet loaded.")
        except ImportError as e:
            raise ImportError(
                "MMPose/YOLO not installed. Install with: pip install myogait[mmpose,yolo]\n"
                f"Original error: {e}"
            )

    def teardown(self):
        self._pose_model = None
        self._detector = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._pose_model is None:
            self.setup()

        import cv2
        from mmpose.apis import inference_topdown

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = frame_rgb.shape[:2]

        # Detect persons
        det_results = self._detector(frame_bgr, verbose=False, classes=[0])
        bboxes = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                if box.conf > 0.5:
                    bboxes.append(box.xyxy[0].cpu().numpy())

        if not bboxes:
            return None

        pose_results = inference_topdown(self._pose_model, frame_bgr, bboxes)
        if not pose_results:
            return None

        pred = pose_results[0].pred_instances
        if not hasattr(pred, "keypoints") or len(pred.keypoints) == 0:
            return None

        kps = pred.keypoints[0]  # (17, 2) pixels
        scores = pred.keypoint_scores[0] if hasattr(pred, "keypoint_scores") else np.ones(17)

        landmarks = np.column_stack([
            kps[:, 0] / w,
            kps[:, 1] / h,
            scores[:17],
        ])
        return landmarks
