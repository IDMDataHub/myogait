"""AlphaPose pose extractor (top-down).

Uses YOLO person detector + AlphaPose FastPose (ResNet-50) for 17 COCO
keypoint estimation.  Supports both the official AlphaPose library and
direct weight loading via a minimal PyTorch reimplementation.

Reference: Fang et al., "AlphaPose: Whole-Body Regional Multi-Person Pose
Estimation and Tracking in Real-Time", TPAMI 2022.
"""

import hashlib
import logging
import os
import numpy as np
from typing import Optional
from .base import BasePoseExtractor, ensure_xpu_torch
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".myogait", "models", "alphapose")

_FASTPOSE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn&export=download&confirm=t"
)
_FASTPOSE_FILE = "fast_res50_256x192.pth"
_FASTPOSE_MIN_BYTES = 100_000_000  # ~155 MB expected

_INPUT_H = 256
_INPUT_W = 192
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_MIN_KEYPOINTS = 3


def _ensure_checkpoint(custom_path=None):
    """Download FastPose checkpoint if not already present.

    Returns the path to the checkpoint file.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    dest = os.path.join(_MODEL_DIR, _FASTPOSE_FILE)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size >= _FASTPOSE_MIN_BYTES:
            return dest
        logger.warning(
            "Checkpoint %s is too small (%d bytes), re-downloading.", dest, size,
        )
        os.remove(dest)

    os.makedirs(_MODEL_DIR, exist_ok=True)
    logger.info("Downloading AlphaPose FastPose (~155 MB) to %s ...", dest)
    import shutil
    import tempfile
    import urllib.request

    tmp_fd, tmp_path = tempfile.mkstemp(dir=_MODEL_DIR)
    try:
        os.close(tmp_fd)
        resp = urllib.request.urlopen(_FASTPOSE_URL, timeout=300)
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        size = os.path.getsize(tmp_path)
        if size < _FASTPOSE_MIN_BYTES:
            raise RuntimeError(
                f"Downloaded file too small ({size} bytes, "
                f"expected >= {_FASTPOSE_MIN_BYTES}). URL may be invalid."
            )

        os.replace(tmp_path, dest)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    logger.info(
        "Downloaded (%d MB).", os.path.getsize(dest) // (1024 * 1024),
    )
    return dest


def _build_simple_fastpose(num_joints=17):
    """Build a minimal FastPose network (ResNet-50 + deconv head).

    This is used as a fallback when the official AlphaPose library is not
    installed.  The architecture matches FastPose-ResNet-50 so that
    pretrained weights can be loaded with ``strict=False``.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    class _SimpleFastPose(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = resnet50(weights=None)
            # Remove avgpool and fc layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(2048, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.final_layer = nn.Conv2d(256, num_joints, 1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.deconv(x)
            return self.final_layer(x)

    return _SimpleFastPose()


def _safe_torch_load(torch, path, device):
    """Load checkpoint preferring ``weights_only=True`` for safety."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch < 1.13 has no weights_only parameter
        return torch.load(path, map_location=device)
    except Exception:
        # Fallback for checkpoints with non-tensor data
        logger.warning("weights_only=True failed, falling back to unsafe load.")
        return torch.load(path, map_location=device, weights_only=False)


class AlphaPosePoseExtractor(BasePoseExtractor):
    """AlphaPose FastPose (ResNet-50) -- 17 COCO keypoints.

    Top-down pipeline: YOLO person detection followed by AlphaPose pose
    estimation on the cropped bounding box.

    Two loading paths are tried in order:

    1. Official ``alphapose`` library (if installed from source).
    2. Fallback: minimal PyTorch reimplementation that loads the same
       pretrained weights.

    Model weights are auto-downloaded on first use to
    ``~/.myogait/models/alphapose/``.

    Install: ``pip install myogait[alphapose]``
    """

    name = "alphapose"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, device: str = "auto", checkpoint: str = None,
                 confidence_threshold: float = 0.1):
        self.device_name = device
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._detector = None
        self._device = None

    def setup(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for AlphaPose. "
                "Install with: pip install myogait[alphapose]"
            )

        ensure_xpu_torch()

        # Device selection
        if self.device_name == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self._device = torch.device("xpu")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.device_name)

        # Person detector
        try:
            from ultralytics import YOLO
            self._detector = YOLO("yolov8n.pt")
        except ImportError:
            raise ImportError(
                "Ultralytics is required for AlphaPose person detection. "
                "Install with: pip install myogait[alphapose]"
            )

        # Pose model
        checkpoint_path = _ensure_checkpoint(self.checkpoint)
        logger.info("Loading AlphaPose FastPose on %s...", self._device)
        self._model = self._load_model(checkpoint_path, torch)
        logger.info("AlphaPose ready.")

    def _load_model(self, checkpoint_path, torch):
        """Load FastPose model, trying official library first."""
        # Path 1: official AlphaPose library
        try:
            from alphapose.models import builder as model_builder
            from easydict import EasyDict

            cfg = EasyDict({
                "MODEL": {
                    "TYPE": "FastPose",
                    "PRETRAINED": "",
                    "NUM_DECONV_FILTERS": [256, 256, 256],
                    "NUM_DECONV_KERNELS": [4, 4, 4],
                    "NUM_JOINTS": 17,
                    "EXTRA": {"PRESET": "simple_base"},
                },
                "DATA_PRESET": {
                    "NUM_JOINTS": 17,
                    "IMAGE_SIZE": [_INPUT_W, _INPUT_H],
                    "HEATMAP_SIZE": [_INPUT_W // 4, _INPUT_H // 4],
                },
            })
            model = model_builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
            state = _safe_torch_load(torch, checkpoint_path, self._device)
            result = model.load_state_dict(state, strict=False)
            if result.missing_keys:
                logger.warning(
                    "AlphaPose (official): %d missing keys in checkpoint.",
                    len(result.missing_keys),
                )
            model.to(self._device).eval()
            logger.info("Loaded via official AlphaPose library.")
            return model
        except ImportError as exc:
            logger.info(
                "AlphaPose library not available (%s), using fallback loader.", exc,
            )
        except Exception as exc:
            logger.warning(
                "AlphaPose official loader failed (%s), trying fallback.", exc,
            )

        # Path 2: minimal PyTorch reimplementation
        model = _build_simple_fastpose(num_joints=17)
        state = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            logger.warning(
                "AlphaPose (fallback): %d missing keys — model may "
                "produce degraded results. Missing: %s",
                len(result.missing_keys),
                ", ".join(result.missing_keys[:5]),
            )
        model.to(self._device).eval()
        logger.info("Loaded via fallback FastPose reimplementation.")
        return model

    def teardown(self):
        self._model = None
        self._detector = None
        self._device = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            self.setup()

        import cv2
        import torch

        if frame_rgb is None or frame_rgb.ndim < 2:
            return None

        # Ensure 3-channel uint8
        if frame_rgb.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[2] == 4:
            frame_rgb = frame_rgb[:, :, :3]

        h, w = frame_rgb.shape[:2]
        if h == 0 or w == 0:
            return None

        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb * 255 if frame_rgb.max() <= 1.0
                                else frame_rgb, 0, 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # --- Person detection ---
        det_results = self._detector(frame_bgr, verbose=False, classes=[0])
        bboxes = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                if conf > 0.5:
                    bboxes.append(box.xyxy[0].cpu().numpy())

        if not bboxes:
            return None

        # Take the largest bbox by area
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best_bbox = bboxes[int(np.argmax(areas))]

        x1, y1, x2, y2 = [int(v) for v in best_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # --- Preprocess crop ---
        inp = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, scale
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1))
        inp = torch.from_numpy(inp).unsqueeze(0).to(self._device)

        # --- Inference ---
        with torch.no_grad():
            heatmaps = self._model(inp)  # (1, 17, H/4, W/4)

        heatmaps = heatmaps[0].cpu().numpy()  # (17, hm_h, hm_w)
        hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

        if hm_h == 0 or hm_w == 0:
            return None

        # --- Decode heatmaps ---
        landmarks = np.zeros((17, 3))
        for j in range(17):
            hm = heatmaps[j]
            max_val = float(np.max(hm))
            if max_val < self.confidence_threshold:
                continue
            flat_idx = int(np.argmax(hm))
            y_hm, x_hm = np.unravel_index(flat_idx, (hm_h, hm_w))

            # Heatmap -> crop -> original frame -> normalized [0, 1]
            x_crop = x_hm / hm_w * (x2 - x1)
            y_crop = y_hm / hm_h * (y2 - y1)
            landmarks[j] = [
                np.clip((x_crop + x1) / w, 0.0, 1.0),
                np.clip((y_crop + y1) / h, 0.0, 1.0),
                max_val,
            ]

        if np.sum(landmarks[:, 2] > 0) < _MIN_KEYPOINTS:
            return None

        return landmarks
