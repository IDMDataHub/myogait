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
        self._ensure_mmcv()
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

    @staticmethod
    def _ensure_mmcv():
        """Auto-install mmcv prebuilt wheel from OpenMMLab if missing."""
        try:
            import mmcv  # noqa: F401
        except ImportError:
            import subprocess
            import sys
            # Detect torch version and CUDA to pick the right prebuilt wheel.
            try:
                import torch
                major, minor = torch.__version__.split(".")[:2]
                cu = torch.version.cuda
                cu_tag = f"cu{cu.replace('.', '')}" if cu else "cpu"
            except Exception:
                major, minor, cu_tag = "2", "4", "cpu"
            # OpenMMLab may not publish wheels for every torch release;
            # try the exact version first, then fall back to older ones.
            candidates = [f"{major}.{int(minor) - i}"
                          for i in range(int(minor))]
            base = "https://download.openmmlab.com/mmcv/dist"
            cmd_base = [sys.executable, "-m", "pip", "install", "mmcv==2.2.0",
                        "--no-cache-dir", "--only-binary", "mmcv"]
            for tv in candidates:
                index = f"{base}/{cu_tag}/torch{tv}/index.html"
                logger.info("mmcv not found — trying %s …", index)
                try:
                    subprocess.check_call(
                        cmd_base + ["-f", index],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info("mmcv installed successfully (torch%s index).", tv)
                    return
                except subprocess.CalledProcessError:
                    continue
            raise ImportError(
                "Could not install mmcv: no prebuilt wheel found for your "
                f"platform (torch {major}.{minor}, {cu_tag}). "
                "See https://mmcv.readthedocs.io/en/latest/get_started/installation.html"
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
