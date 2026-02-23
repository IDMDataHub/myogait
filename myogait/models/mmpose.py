"""RTMPose extractor via MMPose.

RTMPose-m with YOLO person detector.
Output: 17 COCO keypoints in pixel coordinates.
"""

import numpy as np
import logging
from typing import Optional
from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

RTMPOSE_CONFIG = (
    "configs/body_2d_keypoint/rtmpose/coco/"
    "rtmpose-m_8xb256-420e_aic-coco-256x192.py"
)
RTMPOSE_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
)


class MMPosePoseExtractor(BasePoseExtractor):
    """RTMPose pose extractor via MMPose."""

    name = "mmpose"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, device: str = "cpu", config: str = None, checkpoint: str = None):
        self.device = device
        self.config = config or RTMPOSE_CONFIG
        self.checkpoint = checkpoint or RTMPOSE_CHECKPOINT
        self._pose_model = None
        self._detector = None

    def setup(self):
        self._ensure_mmcv()
        try:
            from mmpose.apis import init_model
            from ultralytics import YOLO

            logger.info("Loading RTMPose-m...")
            self._pose_model = init_model(self.config, self.checkpoint, device=self.device)
            self._detector = YOLO("yolov8n.pt")
            logger.info("RTMPose loaded.")
        except ImportError as e:
            raise ImportError(
                "MMPose/YOLO not installed. Install with: pip install myogait[mmpose,yolo]\n"
                f"Original error: {e}"
            )

    @staticmethod
    def _ensure_mmcv():
        """Auto-install mmcv via openmim if missing (PyPI has no Windows wheels)."""
        try:
            import mmcv  # noqa: F401
        except ImportError:
            logger.info("mmcv not found — installing via openmim (first run only)...")
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "mim", "install", "mmcv"],
                stdout=subprocess.DEVNULL,
            )
            logger.info("mmcv installed successfully.")

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

        kps = pred.keypoints[0]
        scores = pred.keypoint_scores[0] if hasattr(pred, "keypoint_scores") else np.ones(17)

        landmarks = np.column_stack([
            kps[:, 0] / w,
            kps[:, 1] / h,
            scores[:17],
        ])
        return landmarks
