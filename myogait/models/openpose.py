"""OpenPose pose extractor via OpenCV DNN (bottom-up).

Uses the CMU Caffe model for 18-keypoint detection, mapped to 17 COCO keypoints.
No additional dependencies beyond OpenCV (included in core myogait install).

Reference: Cao et al., "Realtime Multi-Person 2D Pose Estimation
using Part Affinity Fields", CVPR 2017.
"""

import hashlib
import logging
import os
import numpy as np
from typing import Optional
from .base import BasePoseExtractor
from ..constants import COCO_LANDMARK_NAMES

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".myogait", "models", "openpose")

_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/"
    "master/models/pose/coco/pose_deploy_linevec.prototxt"
)

_CAFFEMODEL_URLS = [
    "https://huggingface.co/camenduru/openpose/resolve/main/"
    "models/pose/coco/pose_iter_440000.caffemodel",
    "https://huggingface.co/gaijingeek/openpose-models/resolve/main/"
    "models/pose/coco/pose_iter_440000.caffemodel",
]

_PROTOTXT_FILE = "pose_deploy_linevec.prototxt"
_CAFFEMODEL_FILE = "pose_iter_440000.caffemodel"
_CAFFEMODEL_SHA256 = "b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549"
_CAFFEMODEL_MIN_BYTES = 100_000_000  # ~200 MB expected

_DOWNLOAD_TIMEOUT = 300  # seconds

# OpenPose COCO-18 keypoint order:
#  0:Nose  1:Neck  2:RShoulder  3:RElbow  4:RWrist  5:LShoulder  6:LElbow
#  7:LWrist  8:RHip  9:RKnee  10:RAnkle  11:LHip  12:LKnee  13:LAnkle
#  14:REye  15:LEye  16:REar  17:LEar
#
# Standard COCO-17 order:
#  0:Nose  1:LEye  2:REye  3:LEar  4:REar  5:LShoulder  6:RShoulder
#  7:LElbow  8:RElbow  9:LWrist  10:RWrist  11:LHip  12:RHip
#  13:LKnee  14:RKnee  15:LAnkle  16:RAnkle

_OPENPOSE_TO_COCO17 = {
    0: 0,    # Nose -> Nose
    15: 1,   # LEye -> LEFT_EYE
    14: 2,   # REye -> RIGHT_EYE
    17: 3,   # LEar -> LEFT_EAR
    16: 4,   # REar -> RIGHT_EAR
    5: 5,    # LShoulder -> LEFT_SHOULDER
    2: 6,    # RShoulder -> RIGHT_SHOULDER
    6: 7,    # LElbow -> LEFT_ELBOW
    3: 8,    # RElbow -> RIGHT_ELBOW
    7: 9,    # LWrist -> LEFT_WRIST
    4: 10,   # RWrist -> RIGHT_WRIST
    11: 11,  # LHip -> LEFT_HIP
    8: 12,   # RHip -> RIGHT_HIP
    12: 13,  # LKnee -> LEFT_KNEE
    9: 14,   # RKnee -> RIGHT_KNEE
    13: 15,  # LAnkle -> LEFT_ANKLE
    10: 16,  # RAnkle -> RIGHT_ANKLE
}


def _sha256(path):
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_download(url, dest, expected_sha256=None, min_bytes=0,
                   timeout=_DOWNLOAD_TIMEOUT):
    """Download *url* to *dest* atomically with integrity checks.

    Uses temp file + ``os.replace`` to prevent corrupt partial files.
    Optionally verifies SHA-256 hash and minimum file size.
    """
    import shutil
    import tempfile
    import urllib.request

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest))
    try:
        os.close(tmp_fd)
        resp = urllib.request.urlopen(url, timeout=timeout)
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        size = os.path.getsize(tmp_path)
        if min_bytes and size < min_bytes:
            raise RuntimeError(
                f"Downloaded file too small ({size} bytes, "
                f"expected >= {min_bytes}). URL may be invalid: {url}"
            )

        if expected_sha256:
            actual = _sha256(tmp_path)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {os.path.basename(dest)}: "
                    f"expected {expected_sha256[:16]}..., got {actual[:16]}... "
                    f"File may be corrupt or tampered with."
                )

        os.replace(tmp_path, dest)  # atomic on same filesystem
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _is_valid_caffemodel(path):
    """Check if an existing caffemodel file is valid (not truncated/corrupt)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < _CAFFEMODEL_MIN_BYTES:
        logger.warning(
            "Caffemodel %s is too small (%d bytes), re-downloading.", path, size,
        )
        os.remove(path)
        return False
    return True


def _ensure_models():
    """Download OpenPose Caffe model files if not already present.

    Returns (prototxt_path, caffemodel_path).
    """
    os.makedirs(_MODEL_DIR, exist_ok=True)

    prototxt_path = os.path.join(_MODEL_DIR, _PROTOTXT_FILE)
    caffemodel_path = os.path.join(_MODEL_DIR, _CAFFEMODEL_FILE)

    if not os.path.exists(prototxt_path):
        logger.info("Downloading OpenPose prototxt to %s ...", prototxt_path)
        _safe_download(_PROTOTXT_URL, prototxt_path)
        logger.info("Prototxt downloaded.")

    if not _is_valid_caffemodel(caffemodel_path):
        last_err = None
        for url in _CAFFEMODEL_URLS:
            try:
                logger.info(
                    "Downloading OpenPose caffemodel (~200 MB) from %s ...",
                    url.split("/")[2],
                )
                _safe_download(
                    url, caffemodel_path,
                    expected_sha256=_CAFFEMODEL_SHA256,
                    min_bytes=_CAFFEMODEL_MIN_BYTES,
                )
                logger.info(
                    "Caffemodel downloaded (%d MB).",
                    os.path.getsize(caffemodel_path) // (1024 * 1024),
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                logger.warning("Mirror failed (%s), trying next...", exc)
        if last_err is not None:
            raise RuntimeError(
                "Could not download OpenPose caffemodel from any mirror. "
                "Download manually to: " + caffemodel_path
            ) from last_err

    return prototxt_path, caffemodel_path


def _find_keypoints_from_heatmaps(heatmaps, threshold=0.1):
    """Extract keypoint locations from OpenPose heatmaps.

    Parameters
    ----------
    heatmaps : np.ndarray
        Shape ``(n_parts, hm_h, hm_w)`` confidence maps.
    threshold : float
        Minimum confidence to accept a peak.

    Returns
    -------
    list[tuple]
        ``(x, y, confidence)`` per part.  ``(None, None, 0.0)`` when the
        peak falls below *threshold*.
    """
    keypoints = []
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i]
        if hm.size == 0:
            keypoints.append((None, None, 0.0))
            continue
        max_val = float(np.max(hm))
        if max_val < threshold:
            keypoints.append((None, None, 0.0))
            continue
        flat_idx = int(np.argmax(hm))
        y, x = np.unravel_index(flat_idx, hm.shape)
        keypoints.append((int(x), int(y), max_val))
    return keypoints


class OpenPosePoseExtractor(BasePoseExtractor):
    """OpenPose via OpenCV DNN -- 17 COCO keypoints (bottom-up).

    The CMU Caffe model is auto-downloaded on first use to
    ``~/.myogait/models/openpose/``.  No extra ``pip install`` needed.
    """

    name = "openpose"
    landmark_names = COCO_LANDMARK_NAMES
    n_landmarks = 17
    is_coco_format = True

    def __init__(self, input_width: int = 368, input_height: int = 368,
                 confidence_threshold: float = 0.1):
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self._net = None

    def setup(self):
        import cv2

        prototxt, caffemodel = _ensure_models()
        logger.info("Loading OpenPose COCO model via OpenCV DNN...")
        try:
            self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        except cv2.error as exc:
            raise RuntimeError(
                f"Failed to load OpenPose model. Files may be corrupt — "
                f"delete {_MODEL_DIR} and retry. Original error: {exc}"
            ) from exc

        # Try CUDA, fall back to CPU
        try:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Validate CUDA actually works with a tiny forward pass
            test_blob = cv2.dnn.blobFromImage(
                np.zeros((16, 16, 3), dtype=np.uint8), 1.0 / 255,
                (16, 16), (0, 0, 0), swapRB=False, crop=False,
            )
            self._net.setInput(test_blob)
            self._net.forward()
            logger.info("OpenPose using CUDA backend.")
        except Exception:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("OpenPose using CPU backend.")

        logger.info("OpenPose ready.")

    def teardown(self):
        self._net = None

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._net is None:
            self.setup()

        import cv2

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

        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1.0 / 255,
            (self.input_width, self.input_height),
            (0, 0, 0), swapRB=False, crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()  # (1, 57, H_out, W_out)

        # First 19 channels: 18 body-part heatmaps + 1 background
        heatmaps = output[0, :19, :, :]
        hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

        if hm_h == 0 or hm_w == 0:
            return None

        # Find peaks for each of the 18 body parts (skip background channel)
        op_keypoints = _find_keypoints_from_heatmaps(
            heatmaps[:18], self.confidence_threshold,
        )

        # Map OpenPose-18 to standard COCO-17
        landmarks = np.zeros((17, 3))
        for op_idx, coco_idx in _OPENPOSE_TO_COCO17.items():
            if op_idx >= len(op_keypoints):
                continue
            x, y, conf = op_keypoints[op_idx]
            if x is None:
                continue
            landmarks[coco_idx] = [
                np.clip(x / hm_w, 0.0, 1.0),
                np.clip(y / hm_h, 0.0, 1.0),
                conf,
            ]

        if np.sum(landmarks[:, 2] > 0) < 3:
            return None

        return landmarks
