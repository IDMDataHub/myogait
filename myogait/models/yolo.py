"""YOLOv8 Pose extractor (17 COCO landmarks)."""

import numpy as np
from typing import Optional
from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES


class YOLOPoseExtractor(BasePoseExtractor):
    """Ultralytics YOLOv8-Pose - 17 COCO keypoints."""

    name = "yolo"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self.model_path = model_path
        self._model = None

    def setup(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install myogait[yolo]"
            )

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            self.setup()
        import cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self._model(frame_bgr, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return None
        kps = results[0].keypoints
        if kps.xy is None or len(kps.xy) == 0:
            return None
        # Take first detected person
        xy = kps.xy[0].cpu().numpy()  # (17, 2) in pixels
        conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17)
        h, w = frame_rgb.shape[:2]
        landmarks = np.column_stack([
            xy[:, 0] / w,  # normalize x
            xy[:, 1] / h,  # normalize y
            conf,
        ])
        return landmarks
