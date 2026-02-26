"""MediaPipe Pose extractor (33 landmarks).

Uses the new MediaPipe Tasks API (PoseLandmarker).
Falls back to legacy mp.solutions.pose if available.
"""

import os
import logging
import numpy as np
from typing import Optional
from .base import BasePoseExtractor
from ..constants import MP_LANDMARK_NAMES

logger = logging.getLogger(__name__)

# Default model path (downloaded on first use)
_DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".myogait", "models")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


def _ensure_model(model_path: str = None) -> str:
    """Download the pose landmarker model if needed."""
    if model_path and os.path.exists(model_path):
        return model_path

    default_path = os.path.join(_DEFAULT_MODEL_DIR, "pose_landmarker_heavy.task")
    if os.path.exists(default_path):
        return default_path

    os.makedirs(_DEFAULT_MODEL_DIR, exist_ok=True)
    logger.info(f"Downloading MediaPipe pose model to {default_path}...")
    import shutil
    import tempfile
    import urllib.request

    tmp_fd, tmp_path = tempfile.mkstemp(dir=_DEFAULT_MODEL_DIR)
    try:
        os.close(tmp_fd)
        resp = urllib.request.urlopen(_MODEL_URL, timeout=300)
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp_path, default_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    logger.info(f"Downloaded ({os.path.getsize(default_path)} bytes)")
    return default_path


class MediaPipePoseExtractor(BasePoseExtractor):
    """Google MediaPipe Pose - 33 full-body landmarks.

    Uses the Tasks API (PoseLandmarker) with VIDEO running mode.
    The heavy model is downloaded automatically on first use.
    """

    name = "mediapipe"
    landmark_names = MP_LANDMARK_NAMES
    n_landmarks = 33
    is_coco_format = False

    # Maximum dimension to avoid poor detection on very large frames
    MAX_DIMENSION = 1080

    def __init__(self, model_path: str = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 fps: float = 30.0):
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.fps = fps
        self._landmarker = None
        self._use_legacy = False
        self._legacy_pose = None
        self._frame_counter = 0

    def setup(self):
        # Try new Tasks API first
        try:
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode,
            )
            from mediapipe.tasks.python import BaseOptions

            resolved_path = _ensure_model(self.model_path)
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=resolved_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._frame_counter = 0
            logger.info("MediaPipe PoseLandmarker (Tasks API) initialized")
            return
        except (ImportError, Exception) as e:
            logger.warning(f"Tasks API unavailable ({e}), trying legacy API...")

        # Fallback to legacy solutions API
        try:
            import mediapipe as mp
            self._legacy_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._use_legacy = True
            logger.info("MediaPipe legacy Pose initialized")
        except (ImportError, AttributeError):
            raise ImportError(
                "MediaPipe not installed. Install with: pip install myogait[mediapipe]"
            )

    def teardown(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy_pose is not None:
            self._legacy_pose.close()
            self._legacy_pose = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._landmarker is None and self._legacy_pose is None:
            self.setup()

        if self._use_legacy:
            return self._process_legacy(frame_rgb)
        else:
            return self._process_tasks(frame_rgb)

    def _maybe_resize(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Resize frame if larger than MAX_DIMENSION to improve detection."""
        h, w = frame_rgb.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.MAX_DIMENSION:
            return frame_rgb
        import cv2
        scale = self.MAX_DIMENSION / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _process_tasks(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Process using new Tasks API."""
        import mediapipe as mp

        frame_rgb = self._maybe_resize(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamps must be monotonically increasing (in ms)
        timestamp_ms = int(self._frame_counter * 1000 / self.fps)
        self._frame_counter += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        lm_list = result.pose_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.visibility]
            for lm in lm_list
        ])
        return landmarks

    def _process_legacy(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Process using legacy solutions API."""
        frame_rgb = self._maybe_resize(frame_rgb)
        results = self._legacy_pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = np.array([
            [lm.x, lm.y, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])
        return landmarks
