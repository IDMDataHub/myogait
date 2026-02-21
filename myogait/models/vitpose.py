"""ViTPose extractor via HuggingFace Transformers.

Uses RT-DETR for person detection, then ViTPose for pose estimation.
Fully pip-installable (``transformers + torch``), supports CUDA, XPU
(Intel Arc), and CPU.

Available models
----------------
====================  ======  =======================================
 Name                  Size    HuggingFace repo
====================  ======  =======================================
 vitpose-base           90 M   usyd-community/vitpose-base-simple
 vitpose-large         400 M   usyd-community/vitpose-plus-large
 vitpose-huge          900 M   usyd-community/vitpose-plus-huge
====================  ======  =======================================

References
----------
- Paper: Xu et al., "ViTPose: Simple Vision Transformer Baselines for
  Human Pose Estimation", NeurIPS 2022.
  https://arxiv.org/abs/2204.12484
- Paper: Xu et al., "ViTPose++: Vision Transformer for Generic Body
  Pose Estimation", TPAMI 2024.
  https://arxiv.org/abs/2212.04246
"""

import logging
from typing import Optional

import numpy as np

from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

_VITPOSE_MODELS = {
    "base": "usyd-community/vitpose-base-simple",
    "large": "usyd-community/vitpose-plus-large",
    "huge": "usyd-community/vitpose-plus-huge",
}

_DETECTOR_REPO = "PekingU/rtdetr_r50vd_coco_o365"


def _get_torch_device():
    """Select best device: CUDA > XPU > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class ViTPosePoseExtractor(BasePoseExtractor):
    """ViTPose — Vision Transformer for pose estimation.

    Top-down architecture: RT-DETR detects people, ViTPose estimates
    17 COCO keypoints per person.  Uses the largest detected person
    (single-person gait analysis).

    Install: ``pip install myogait[vitpose]``
    """

    name = "vitpose"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, model_size: str = "base", **kwargs):
        self.model_size = model_size
        self._device = None
        self._detector = None
        self._detector_processor = None
        self._pose_model = None
        self._pose_processor = None

    def setup(self):
        try:
            from transformers import (
                AutoProcessor,
                RTDetrForObjectDetection,
                VitPoseForPoseEstimation,
            )
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers is required for ViTPose. "
                "Install with: pip install myogait[vitpose]"
            )

        import torch
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            pass

        self._device = _get_torch_device()
        repo = _VITPOSE_MODELS.get(self.model_size)
        if repo is None:
            raise ValueError(
                f"Unknown ViTPose size '{self.model_size}'. "
                f"Available: {', '.join(_VITPOSE_MODELS.keys())}"
            )

        logger.info(f"Loading RT-DETR person detector on {self._device}...")
        self._detector_processor = AutoProcessor.from_pretrained(_DETECTOR_REPO)
        self._detector = RTDetrForObjectDetection.from_pretrained(_DETECTOR_REPO)
        self._detector.to(self._device).eval()

        logger.info(f"Loading ViTPose {self.model_size} on {self._device}...")
        self._pose_processor = AutoProcessor.from_pretrained(repo)
        self._pose_model = VitPoseForPoseEstimation.from_pretrained(repo)
        self._pose_model.to(self._device).eval()
        logger.info("ViTPose ready.")

    def teardown(self):
        self._detector = None
        self._detector_processor = None
        self._pose_model = None
        self._pose_processor = None
        self._device = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._pose_model is None:
            self.setup()

        import torch
        from PIL import Image

        h, w = frame_rgb.shape[:2]
        image = Image.fromarray(frame_rgb)

        # ── Step 1: detect people ────────────────────────────────────
        det_inputs = self._detector_processor(
            images=image, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            det_outputs = self._detector(**det_inputs)

        results = self._detector_processor.post_process_object_detection(
            det_outputs,
            target_sizes=torch.tensor([(h, w)], device=self._device),
            threshold=0.3,
        )
        result = results[0]

        # Filter for "person" class (label 0 in COCO)
        person_mask = result["labels"] == 0
        if not person_mask.any():
            return None

        person_boxes = result["boxes"][person_mask].cpu().numpy()
        person_scores = result["scores"][person_mask].cpu().numpy()

        # Use the largest / most confident person
        areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (
            person_boxes[:, 3] - person_boxes[:, 1]
        )
        best_idx = int(np.argmax(areas * person_scores))
        box = person_boxes[best_idx]  # x1, y1, x2, y2

        # Convert to COCO format (x, y, w, h)
        coco_box = np.array([[
            box[0], box[1], box[2] - box[0], box[3] - box[1]
        ]])

        # ── Step 2: estimate pose ────────────────────────────────────
        pose_inputs = self._pose_processor(
            image, boxes=[coco_box], return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            pose_outputs = self._pose_model(**pose_inputs)

        pose_results = self._pose_processor.post_process_pose_estimation(
            pose_outputs, boxes=[coco_box]
        )

        if not pose_results or not pose_results[0]:
            return None

        person = pose_results[0][0]
        keypoints = person["keypoints"].cpu().numpy()  # (17, 2) pixel coords
        scores = person["scores"].cpu().numpy()         # (17,)

        # Build (17, 3) array: [x_norm, y_norm, confidence]
        landmarks = np.zeros((17, 3))
        landmarks[:, 0] = keypoints[:, 0] / w
        landmarks[:, 1] = keypoints[:, 1] / h
        landmarks[:, 2] = scores

        return landmarks
