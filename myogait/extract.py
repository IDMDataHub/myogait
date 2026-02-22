"""Video pose extraction: video file to JSON pivot.

Extracts 2D pose landmarks from video using a configurable model
backend (MediaPipe, YOLO, Sapiens, ViTPose, RTMW, HRNet, MMPose). Handles
direction detection, landmark flipping, and label inversion
correction automatically.

Functions
---------
extract
    Extract pose landmarks from a video file (main entry point).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .constants import (
    MP_LANDMARK_NAMES, COCO_LANDMARK_NAMES, COCO_TO_MP, MP_NAME_TO_INDEX,
    GOLIATH_LANDMARK_NAMES, GOLIATH_SEG_CLASSES, WHOLEBODY_LANDMARK_NAMES,
    GOLIATH_FOOT_INDICES, RTMW_FOOT_INDICES,
)
from .models import get_extractor
from .schema import create_empty
from .experimental import (
    build_video_degradation_config,
    is_video_degradation_active,
    compute_fps_sampling,
    degraded_resolution,
    apply_video_degradation,
)

logger = logging.getLogger(__name__)

# Mapping from pose model names to Sapiens model sizes (for depth/seg defaults)
_MODEL_TO_SAPIENS_SIZE = {
    "sapiens-quick": "0.3b",
    "sapiens-mid": "0.6b",
    "sapiens-top": "1b",
}


def _sapiens_size_from_model(model: str) -> str:
    """Infer Sapiens model size from pose model name."""
    return _MODEL_TO_SAPIENS_SIZE.get(model, "0.3b")


def _coco_to_mediapipe(landmarks_17: np.ndarray) -> np.ndarray:
    """Convert 17 COCO landmarks to 33 MediaPipe format.

    Fills missing landmarks (face details, hands, feet) with NaN.

    Args:
        landmarks_17: Array of shape (17, 3) with [x, y, visibility].

    Returns:
        Array of shape (33, 3) in MediaPipe order.
    """
    mp33 = np.full((33, 3), np.nan)
    for coco_idx, coco_name in enumerate(COCO_LANDMARK_NAMES):
        mp_name = COCO_TO_MP.get(coco_name)
        if mp_name and mp_name in MP_NAME_TO_INDEX:
            mp_idx = MP_NAME_TO_INDEX[mp_name]
            mp33[mp_idx] = landmarks_17[coco_idx]
    return mp33


def _enrich_foot_landmarks(frame: dict) -> None:
    """Inject detected foot landmarks from auxiliary data into frame landmarks.

    When Sapiens (Goliath 308) or RTMW (WholeBody 133) auxiliary keypoints
    are available, extracts the real foot landmarks (big_toe, small_toe, heel)
    and writes them into the frame's landmarks dict.  This replaces the
    geometric estimation that would otherwise be done during angle
    computation.

    Also computes LEFT_FOOT_INDEX and RIGHT_FOOT_INDEX as the midpoint
    of big_toe and small_toe, which is more accurate than the geometric
    estimate from ankle position.

    Modifies *frame* in place.  Sets ``frame["foot_landmarks_source"]``
    to ``"detected"`` when real foot landmarks are injected.
    """
    lm = frame.get("landmarks")
    if not lm:
        return

    # Determine which auxiliary format is present and select the mapping
    aux_data = None
    foot_map = None
    if "goliath308" in frame:
        aux_data = frame["goliath308"]
        foot_map = GOLIATH_FOOT_INDICES
    elif "wholebody133" in frame:
        aux_data = frame["wholebody133"]
        foot_map = RTMW_FOOT_INDICES

    if aux_data is None or foot_map is None:
        return

    # Extract foot landmarks from auxiliary data
    injected = False
    for aux_idx, lm_name in foot_map.items():
        if aux_idx >= len(aux_data):
            continue
        point = aux_data[aux_idx]
        # Each point is [x, y, confidence]
        if isinstance(point, (list, tuple)) and len(point) >= 3:
            x, y, conf = float(point[0]), float(point[1]), float(point[2])
        else:
            continue

        if np.isnan(x) or np.isnan(y) or conf < 0.1:
            continue

        lm[lm_name] = {"x": x, "y": y, "visibility": conf}
        injected = True

    if not injected:
        return

    # Compute FOOT_INDEX as midpoint of big_toe and small_toe (more
    # accurate than the geometric estimate from ankle/knee).
    for side in ("LEFT", "RIGHT"):
        big_toe = lm.get(f"{side}_BIG_TOE")
        small_toe = lm.get(f"{side}_SMALL_TOE")
        if (big_toe is not None and small_toe is not None
                and not np.isnan(big_toe["x"]) and not np.isnan(small_toe["x"])):
            mid_x = (big_toe["x"] + small_toe["x"]) / 2.0
            mid_y = (big_toe["y"] + small_toe["y"]) / 2.0
            mid_vis = min(big_toe["visibility"], small_toe["visibility"])
            lm[f"{side}_FOOT_INDEX"] = {
                "x": mid_x, "y": mid_y, "visibility": mid_vis,
            }

    frame["foot_landmarks_source"] = "detected"


def _detect_direction(frames_landmarks: list) -> str:
    """Detect walking direction (left or right) from landmarks.

    Uses nose position relative to ear midpoint.

    Returns:
        'left' or 'right'.
    """
    nose_idx = MP_NAME_TO_INDEX.get("NOSE", 0)
    left_ear_idx = MP_NAME_TO_INDEX.get("LEFT_EAR", 7)
    right_ear_idx = MP_NAME_TO_INDEX.get("RIGHT_EAR", 8)

    diffs = []
    for lm in frames_landmarks:
        if lm is None:
            continue
        if lm.shape[0] < 33:
            continue
        nose_x = lm[nose_idx, 0]
        ear_mid_x = (lm[left_ear_idx, 0] + lm[right_ear_idx, 0]) / 2
        if np.isnan(nose_x) or np.isnan(ear_mid_x):
            continue
        diffs.append(nose_x - ear_mid_x)

    if not diffs:
        return "left"  # default

    return "left" if np.median(diffs) < 0 else "right"


def _flip_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mirror landmarks horizontally and swap left/right labels.

    Args:
        landmarks: (33, 3) array in MediaPipe format.

    Returns:
        Flipped array.
    """
    flipped = landmarks.copy()
    # Mirror x coordinate
    flipped[:, 0] = 1.0 - flipped[:, 0]

    # Swap left/right pairs
    for name in MP_LANDMARK_NAMES:
        if not name.startswith("LEFT_"):
            continue
        right_name = name.replace("LEFT_", "RIGHT_")
        if right_name not in MP_NAME_TO_INDEX:
            continue
        li = MP_NAME_TO_INDEX[name]
        ri = MP_NAME_TO_INDEX[right_name]
        flipped[li], flipped[ri] = flipped[ri].copy(), flipped[li].copy()

    return flipped


def _flip_auxiliary(aux: np.ndarray, landmark_names: list) -> np.ndarray:
    """Mirror auxiliary landmarks horizontally and swap left/right indices.

    Args:
        aux: (N, 3) array of auxiliary landmarks (x, y, confidence).
        landmark_names: List of landmark name strings of length N, used to
            identify left/right pairs for swapping.

    Returns:
        Flipped array with mirrored x and swapped left/right pairs.
    """
    flipped = aux.copy()
    # Mirror x coordinate
    flipped[:, 0] = 1.0 - flipped[:, 0]

    # Build left/right swap pairs from landmark names
    n = len(landmark_names)
    visited = set()
    for i, name in enumerate(landmark_names):
        if i in visited:
            continue
        # Try multiple left/right naming conventions used in Goliath/WholeBody
        partner_name = None
        if "left_" in name:
            partner_name = name.replace("left_", "right_")
        elif "right_" in name:
            partner_name = name.replace("right_", "left_")
        elif name.startswith("l_"):
            partner_name = "r_" + name[2:]
        elif name.startswith("r_"):
            partner_name = "l_" + name[2:]

        if partner_name is None:
            continue

        # Find partner index
        for j in range(n):
            if j != i and landmark_names[j] == partner_name:
                flipped[i], flipped[j] = flipped[j].copy(), flipped[i].copy()
                visited.add(i)
                visited.add(j)
                break

    return flipped


def extract(
    video_path: str,
    model: str = "mediapipe",
    max_frames: Optional[int] = None,
    flip_if_right: bool = True,
    correct_inversions: bool = True,
    with_depth: bool = False,
    with_seg: bool = False,
    depth_model_size: Optional[str] = None,
    seg_model_size: Optional[str] = None,
    experimental: Optional[dict] = None,
    progress_callback=None,
    **kwargs,
) -> dict:
    """Extract pose landmarks from a video.

    Parameters
    ----------
    video_path : str
        Path to video file (mp4, mov, avi).
    model : str, optional
        Pose model name (default ``"mediapipe"``).
    max_frames : int, optional
        Process at most N frames (default: all).
    flip_if_right : bool, optional
        Auto-detect walking direction and flip if right (default True).
    correct_inversions : bool, optional
        Detect and correct left/right label swaps (default True).
    with_depth : bool, optional
        Run Sapiens depth estimation alongside pose (default False).
        Adds per-landmark depth values to each frame.
    with_seg : bool, optional
        Run Sapiens body-part segmentation alongside pose (default False).
        Adds per-landmark body-part labels to each frame.
    depth_model_size : str, optional
        Sapiens depth model size (default: matches pose model).
    seg_model_size : str, optional
        Sapiens seg model size (default: matches pose model).
    experimental : dict, optional
        Experimental input degradation controls (AIM benchmark only).
        Supported keys: ``target_fps``, ``downscale``, ``contrast``,
        ``aspect_ratio``, ``perspective_x``, ``perspective_y``, ``enabled``.
        Defaults keep the input unchanged.
    progress_callback : callable, optional
        Callback ``fn(float)`` receiving progress from 0.0 to 1.0.
    **kwargs
        Extra arguments passed to the model extractor.

    Returns
    -------
    dict
        Pivot JSON dict with ``extraction`` and ``frames`` populated.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If the video cannot be opened by OpenCV.
    """
    video_path = str(video_path)
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Open video for metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    exp_cfg = build_video_degradation_config(experimental)
    exp_active = is_video_degradation_active(exp_cfg)
    frame_stride, fps = compute_fps_sampling(source_fps, exp_cfg.get("target_fps"))
    width, height = degraded_resolution(source_width, source_height, exp_cfg)

    estimated_total = source_total_frames
    if estimated_total > 0 and frame_stride > 1:
        estimated_total = int(np.ceil(estimated_total / frame_stride))
    if max_frames is not None:
        estimated_total = min(estimated_total, max_frames)

    # Create pivot structure with effective extraction metadata.
    data = create_empty(video_path, fps, width, height, estimated_total)

    # Get extractor — pass fps for models that need it (mediapipe)
    if model == "mediapipe" and "fps" not in kwargs:
        kwargs["fps"] = fps
    extractor = get_extractor(model, **kwargs)
    extractor.setup()

    is_coco = extractor.is_coco_format

    # ── Optional auxiliary Sapiens models (depth / seg) ──────────
    depth_estimator = None
    seg_estimator = None

    if with_depth:
        from .models.sapiens_depth import SapiensDepthEstimator
        _ds = depth_model_size or _sapiens_size_from_model(model)
        depth_estimator = SapiensDepthEstimator(model_size=_ds)
        depth_estimator.setup()
        logger.info(f"Depth estimation enabled (sapiens-depth-{_ds})")

    if with_seg:
        from .models.sapiens_seg import SapiensSegEstimator
        _ss = seg_model_size or _sapiens_size_from_model(model)
        seg_estimator = SapiensSegEstimator(model_size=_ss)
        seg_estimator.setup()
        logger.info(f"Segmentation enabled (sapiens-seg-{_ss})")

    logger.info(
        f"Extracting ~{estimated_total} frames with {model} "
        f"({extractor.n_landmarks} landmarks, {width}x{height} @ {fps:.1f}fps)"
    )

    # Process frames
    raw_landmarks = []  # list of np.ndarray or None
    auxiliary_list = []  # list of np.ndarray or None (e.g. Goliath 308)
    depth_maps = []     # list of np.ndarray or None
    seg_masks = []      # list of np.ndarray or None
    frame_idx = 0
    source_idx = 0
    detected_count = 0
    log_interval = max(1, max(estimated_total, 1) // 10)  # log every ~10%

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if source_idx % frame_stride != 0:
                source_idx += 1
                continue
            if max_frames is not None and frame_idx >= max_frames:
                break

            if exp_active:
                frame = apply_video_degradation(frame, exp_cfg)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = extractor.process_frame(frame_rgb)

            # process_frame returns np.ndarray or dict with auxiliary data
            auxiliary = None
            if isinstance(result, dict):
                lm = result.get("landmarks")
                # Check for auxiliary keypoints (Goliath 308 or WholeBody 133)
                for aux_key in ("auxiliary_goliath308", "auxiliary_wholebody133"):
                    aux = result.get(aux_key)
                    if aux is not None:
                        auxiliary = aux
                        break
            else:
                lm = result

            if lm is not None:
                detected_count += 1
                if is_coco:
                    lm = lm.copy()
                    # Normalize pixel coords to [0,1] if needed
                    if lm[:, 0].max() > 1.5:  # likely pixel coordinates
                        lm[:, 0] /= width
                        lm[:, 1] /= height
                    lm = _coco_to_mediapipe(lm)

            raw_landmarks.append(lm)
            auxiliary_list.append(auxiliary)

            # Run depth / seg on the same frame
            depth_maps.append(
                depth_estimator.process_frame(frame_rgb) if depth_estimator else None
            )
            seg_masks.append(
                seg_estimator.process_frame(frame_rgb) if seg_estimator else None
            )

            frame_idx += 1
            source_idx += 1

            if frame_idx % log_interval == 0:
                pct = 100 * frame_idx / max(estimated_total, 1)
                det_pct = 100 * detected_count / frame_idx if frame_idx > 0 else 0
                logger.info(f"  {frame_idx}/{estimated_total} ({pct:.0f}%) — {det_pct:.0f}% detected")

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx / max(estimated_total, 1))
    finally:
        cap.release()
        extractor.teardown()
        if depth_estimator:
            depth_estimator.teardown()
        if seg_estimator:
            seg_estimator.teardown()

    det_pct = 100 * detected_count / len(raw_landmarks) if raw_landmarks else 0
    logger.info(f"Extraction done: {detected_count}/{len(raw_landmarks)} frames detected ({det_pct:.0f}%)")

    # Update actual frame count
    data["meta"]["n_frames"] = len(raw_landmarks)
    data["meta"]["duration_s"] = round(len(raw_landmarks) / fps, 3) if fps > 0 else 0.0

    # Direction detection and flip
    was_flipped = False
    if flip_if_right:
        direction = _detect_direction(raw_landmarks)
        logger.info(f"Walking direction: {direction}")
        if direction == "right":
            was_flipped = True
            raw_landmarks = [
                _flip_landmarks(lm) if lm is not None else None
                for lm in raw_landmarks
            ]
            # Flip auxiliary data (goliath308, wholebody133) so foot landmarks
            # injected by _enrich_foot_landmarks are in the correct space.
            for i, aux in enumerate(auxiliary_list):
                if aux is not None:
                    n_aux = aux.shape[0]
                    if n_aux == 308:
                        auxiliary_list[i] = _flip_auxiliary(aux, GOLIATH_LANDMARK_NAMES)
                    elif n_aux == 133:
                        auxiliary_list[i] = _flip_auxiliary(aux, list(WHOLEBODY_LANDMARK_NAMES))
                    else:
                        # Unknown format: just mirror x
                        aux_copy = aux.copy()
                        aux_copy[:, 0] = 1.0 - aux_copy[:, 0]
                        auxiliary_list[i] = aux_copy
    else:
        direction = "unknown"

    # Label inversion correction
    if correct_inversions:
        raw_landmarks, inversion_mask = _correct_label_inversions(raw_landmarks)
        # Also correct auxiliary data (goliath308, wholebody133)
        if any(inversion_mask):
            aux_names = None
            first_aux = next((a for a in auxiliary_list if a is not None), None)
            if first_aux is not None:
                n_aux = first_aux.shape[0]
                if n_aux == 308:
                    aux_names = GOLIATH_LANDMARK_NAMES
                elif n_aux == 133:
                    aux_names = WHOLEBODY_LANDMARK_NAMES
            if aux_names is not None:
                for i, is_inv in enumerate(inversion_mask):
                    if is_inv and i < len(auxiliary_list) and auxiliary_list[i] is not None:
                        auxiliary_list[i] = _swap_auxiliary_lr(
                            auxiliary_list[i], aux_names,
                        )

    # Build frames — trim leading/trailing frames where legs are not
    # fully visible, but keep everything in between (continuous timeline).
    has_auxiliary = any(a is not None for a in auxiliary_list)

    # A frame has "visible legs" when hip, knee and ankle landmarks are
    # detected, not NaN, and not stuck at the image edge.
    _LEG_INDICES = [
        MP_NAME_TO_INDEX.get("LEFT_HIP", 23),
        MP_NAME_TO_INDEX.get("RIGHT_HIP", 24),
        MP_NAME_TO_INDEX.get("LEFT_KNEE", 25),
        MP_NAME_TO_INDEX.get("RIGHT_KNEE", 26),
        MP_NAME_TO_INDEX.get("LEFT_ANKLE", 27),
        MP_NAME_TO_INDEX.get("RIGHT_ANKLE", 28),
    ]
    _EDGE = 0.02  # 2% margin from image border

    _MIN_LEG_VIS = 0.3  # minimum visibility for each leg landmark

    def _legs_visible(lm):
        """Return True if all leg landmarks are detected with good confidence
        and not stuck at the image edge."""
        if lm is None:
            return False
        for idx in _LEG_INDICES:
            x, y, vis = lm[idx, 0], lm[idx, 1], lm[idx, 2]
            if np.isnan(x) or np.isnan(y):
                return False
            if x <= _EDGE or x >= 1.0 - _EDGE or y <= _EDGE or y >= 1.0 - _EDGE:
                return False
            if np.isnan(vis) or vis < _MIN_LEG_VIS:
                return False
        return True

    # Find first and last frames where legs are fully visible
    first_det = next((i for i in range(len(raw_landmarks)) if _legs_visible(raw_landmarks[i])), 0)
    last_det = next((i for i in range(len(raw_landmarks) - 1, -1, -1) if _legs_visible(raw_landmarks[i])),
                    len(raw_landmarks) - 1)
    if first_det > 0 or last_det < len(raw_landmarks) - 1:
        logger.info("Trimming frames: keeping %d-%d (dropped %d leading, %d trailing)",
                     first_det, last_det,
                     first_det, len(raw_landmarks) - 1 - last_det)

    frames = []
    for idx in range(first_det, last_det + 1):
        lm = raw_landmarks[idx]
        frame_data = {
            "frame_idx": idx,
            "time_s": round(idx / fps, 4) if fps > 0 else 0.0,
            "landmarks": {},
            "confidence": 0.0,
        }
        if lm is not None:
            valid_vis = lm[:, 2][~np.isnan(lm[:, 2])]
            frame_data["confidence"] = float(np.mean(valid_vis)) if len(valid_vis) > 0 else 0.0
            for i, name in enumerate(MP_LANDMARK_NAMES):
                frame_data["landmarks"][name] = {
                    "x": float(lm[i, 0]),
                    "y": float(lm[i, 1]),
                    "visibility": float(lm[i, 2]) if not np.isnan(lm[i, 2]) else 0.0,
                }

        # Store auxiliary keypoints (e.g. Goliath 308 from Sapiens)
        aux = auxiliary_list[idx] if idx < len(auxiliary_list) else None
        if aux is not None:
            # Determine auxiliary format by shape
            n_aux = aux.shape[0]
            aux_key = "goliath308" if n_aux == 308 else f"wholebody{n_aux}"
            frame_data[aux_key] = [
                [round(float(aux[i, 0]), 5), round(float(aux[i, 1]), 5),
                 round(float(aux[i, 2]), 4)]
                for i in range(n_aux)
            ]

        # Depth: sample at final landmark positions
        dmap = depth_maps[idx] if idx < len(depth_maps) else None
        if dmap is not None and lm is not None:
            dh, dw = dmap.shape
            landmark_depths = {}
            for i, name in enumerate(MP_LANDMARK_NAMES):
                x, y = lm[i, 0], lm[i, 1]
                if np.isnan(x) or np.isnan(y):
                    continue
                sx = (1.0 - x) if was_flipped else x
                px = int(np.clip(sx * dw, 0, dw - 1))
                py = int(np.clip(y * dh, 0, dh - 1))
                landmark_depths[name] = round(float(dmap[py, px]), 4)
            frame_data["landmark_depths"] = landmark_depths

        # Segmentation: sample body-part class at landmark positions
        smask = seg_masks[idx] if idx < len(seg_masks) else None
        if smask is not None and lm is not None:
            sh, sw = smask.shape
            landmark_parts = {}
            for i, name in enumerate(MP_LANDMARK_NAMES):
                x, y = lm[i, 0], lm[i, 1]
                if np.isnan(x) or np.isnan(y):
                    continue
                sx = (1.0 - x) if was_flipped else x
                px = int(np.clip(sx * sw, 0, sw - 1))
                py = int(np.clip(y * sh, 0, sh - 1))
                cls_idx = int(smask[py, px])
                if cls_idx < len(GOLIATH_SEG_CLASSES):
                    landmark_parts[name] = GOLIATH_SEG_CLASSES[cls_idx]
            frame_data["landmark_body_parts"] = landmark_parts

        frames.append(frame_data)

    # Enrich foot landmarks from auxiliary data (Sapiens/RTMW) before
    # angle computation.  This injects real detected toe/heel positions
    # into the landmarks dict, replacing later geometric estimates.
    for frame_data in frames:
        _enrich_foot_landmarks(frame_data)

    data["frames"] = frames
    extraction_meta = {
        "model": model,
        "model_detail": extractor.name,
        "keypoint_format": "mediapipe33",
        "n_landmarks": 33,
        "landmark_names": MP_LANDMARK_NAMES,
        "direction_detected": direction,
        "inversions_corrected": correct_inversions,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if exp_active:
        extraction_meta["experimental"] = {
            "family": "video_degradation",
            "scope": "AIM benchmark only",
            "source_fps": round(float(source_fps), 4),
            "source_resolution": [int(source_width), int(source_height)],
            "frame_stride": int(frame_stride),
            "effective_fps": round(float(fps), 4),
            "config": exp_cfg,
        }
    if has_auxiliary:
        # Detect auxiliary type from first non-None entry
        first_aux = next((a for a in auxiliary_list if a is not None), None)
        if first_aux is not None:
            n_aux = first_aux.shape[0]
            if n_aux == 308:
                extraction_meta["auxiliary_format"] = "goliath308"
                extraction_meta["auxiliary_n_landmarks"] = 308
                extraction_meta["auxiliary_landmark_names"] = GOLIATH_LANDMARK_NAMES
            elif n_aux == 133:
                extraction_meta["auxiliary_format"] = "wholebody133"
                extraction_meta["auxiliary_n_landmarks"] = 133
                extraction_meta["auxiliary_landmark_names"] = WHOLEBODY_LANDMARK_NAMES

    if with_depth:
        extraction_meta["depth_model"] = f"sapiens-depth-{depth_model_size or _sapiens_size_from_model(model)}"
    if with_seg:
        extraction_meta["seg_model"] = f"sapiens-seg-{seg_model_size or _sapiens_size_from_model(model)}"
        extraction_meta["seg_classes"] = GOLIATH_SEG_CLASSES

    data["extraction"] = extraction_meta

    return data


def detect_treadmill(data: dict) -> dict:
    """Detect if the subject is walking on a treadmill.

    Uses hip center displacement analysis. On a treadmill, the subject
    stays roughly in the same horizontal position throughout the video.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with frames populated.

    Returns
    -------
    dict
        Modified data with extraction.treadmill and
        extraction.treadmill_confidence added.
    """
    frames = data.get("frames", [])

    if not frames:
        if data.get("extraction") is None:
            data["extraction"] = {}
        data["extraction"]["treadmill"] = False
        data["extraction"]["treadmill_confidence"] = 0.0
        return data

    # Extract hip center x position across all frames
    hip_x_values = []
    for f in frames:
        lm = f.get("landmarks", {})
        left_hip = lm.get("LEFT_HIP")
        right_hip = lm.get("RIGHT_HIP")
        if left_hip is not None and right_hip is not None:
            lh_x = left_hip.get("x")
            rh_x = right_hip.get("x")
            if lh_x is not None and rh_x is not None:
                if not (np.isnan(lh_x) or np.isnan(rh_x)):
                    hip_x_values.append((lh_x + rh_x) / 2)

    if len(hip_x_values) < 2:
        if data.get("extraction") is None:
            data["extraction"] = {}
        data["extraction"]["treadmill"] = False
        data["extraction"]["treadmill_confidence"] = 0.0
        return data

    hip_x = np.array(hip_x_values)

    # Total displacement as fraction of frame width (normalized coords)
    total_displacement = float(np.max(hip_x) - np.min(hip_x))

    # Variance of hip x position
    hip_x_var = float(np.var(hip_x))

    # Thresholds (in normalized coordinates, 0-1)
    # If total displacement < 10% of frame width → treadmill
    displacement_threshold = 0.10
    variance_threshold = 0.005  # low variance suggests treadmill

    is_treadmill = (total_displacement < displacement_threshold)
    low_variance = (hip_x_var < variance_threshold)

    # Confidence based on how strongly the indicators point
    if is_treadmill and low_variance:
        confidence = min(1.0, 1.0 - total_displacement / displacement_threshold)
        confidence = max(confidence, 0.5)
    elif is_treadmill or low_variance:
        confidence = 0.5
    else:
        confidence = max(0.0, 1.0 - (total_displacement - displacement_threshold) /
                         displacement_threshold)

    # Final decision: both criteria or strong displacement criterion
    detected = is_treadmill

    if data.get("extraction") is None:
        data["extraction"] = {}
    data["extraction"]["treadmill"] = bool(detected)
    data["extraction"]["treadmill_confidence"] = round(float(confidence), 3)

    logger.info(
        f"Treadmill detection: {'treadmill' if detected else 'overground'} "
        f"(displacement={total_displacement:.3f}, variance={hip_x_var:.5f}, "
        f"confidence={confidence:.3f})"
    )

    return data


def detect_multi_person(data: dict) -> dict:
    """Detect potential multi-person interference in pose data.

    Identifies frames where landmark positions jump anomalously,
    suggesting the pose estimator may have switched between people.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with frames populated.

    Returns
    -------
    dict
        Modified data with extraction.multi_person_warning and
        extraction.suspicious_frames added.
    """
    frames = data.get("frames", [])

    if data.get("extraction") is None:
        data["extraction"] = {}

    if len(frames) < 2:
        data["extraction"]["multi_person_warning"] = False
        data["extraction"]["suspicious_frames"] = []
        return data

    # Track key landmark positions across frames
    key_landmarks = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"]

    suspicious_frames = []
    jump_threshold = 0.30  # 30% of frame width in normalized coordinates

    for i in range(1, len(frames)):
        prev_lm = frames[i - 1].get("landmarks", {})
        curr_lm = frames[i].get("landmarks", {})

        jumps = []
        for lm_name in key_landmarks:
            prev = prev_lm.get(lm_name)
            curr = curr_lm.get(lm_name)

            if prev is None or curr is None:
                continue

            prev_x = prev.get("x")
            prev_y = prev.get("y")
            curr_x = curr.get("x")
            curr_y = curr.get("y")

            if (prev_x is None or prev_y is None or
                    curr_x is None or curr_y is None):
                continue

            if np.isnan(prev_x) or np.isnan(prev_y) or \
               np.isnan(curr_x) or np.isnan(curr_y):
                continue

            distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            jumps.append(distance)

        if jumps:
            max_jump = max(jumps)
            mean_jump = np.mean(jumps)

            # Flag if maximum jump or mean jump across landmarks exceeds threshold
            if max_jump > jump_threshold or mean_jump > jump_threshold * 0.5:
                suspicious_frames.append(i)

    # Also check for sudden visibility drops followed by different positions
    for i in range(1, len(frames) - 1):
        curr_conf = frames[i].get("confidence", 1.0)
        prev_conf = frames[i - 1].get("confidence", 1.0)
        next_conf = frames[i + 1].get("confidence", 1.0)

        # Sudden confidence drop and recovery suggests person switch
        if curr_conf < 0.3 and prev_conf > 0.7 and next_conf > 0.7:
            if i not in suspicious_frames:
                suspicious_frames.append(i)

    suspicious_frames.sort()
    has_warning = len(suspicious_frames) > 0

    data["extraction"]["multi_person_warning"] = bool(has_warning)
    data["extraction"]["suspicious_frames"] = suspicious_frames

    if has_warning:
        logger.warning(
            f"Multi-person interference detected at {len(suspicious_frames)} "
            f"frames: {suspicious_frames[:10]}{'...' if len(suspicious_frames) > 10 else ''}"
        )

    return data


def _correct_label_inversions(landmarks_list: list) -> tuple:
    """Detect and correct left/right label swaps across frames.

    Uses **position-proximity tracking** between consecutive raw
    frames.  For each pair (i, i+1), the total squared displacement
    of keeping L/R assignment vs swapping is compared across four
    landmark pairs (hip, knee, ankle, shoulder).  When swapping is
    cheaper a swap-transition is recorded and the inversion state is
    toggled.

    This works in sagittal view where left/right landmarks overlap in x
    and simple x-ordering heuristics fail, because it tracks positional
    continuity rather than static ordering.

    A final majority-vote pass ensures the initial reference was correct:
    if more than half the frames end up marked as inverted, the polarity
    is flipped.

    Returns
    -------
    tuple
        ``(corrected_landmarks, inversion_mask)`` where *inversion_mask*
        is a list of booleans indicating which frames had their
        left/right labels swapped.
    """
    # Landmark pairs used for proximity cost
    _TRACK_PAIRS = [
        (MP_NAME_TO_INDEX.get("LEFT_HIP", 23),
         MP_NAME_TO_INDEX.get("RIGHT_HIP", 24)),
        (MP_NAME_TO_INDEX.get("LEFT_KNEE", 25),
         MP_NAME_TO_INDEX.get("RIGHT_KNEE", 26)),
        (MP_NAME_TO_INDEX.get("LEFT_ANKLE", 27),
         MP_NAME_TO_INDEX.get("RIGHT_ANKLE", 28)),
        (MP_NAME_TO_INDEX.get("LEFT_SHOULDER", 11),
         MP_NAME_TO_INDEX.get("RIGHT_SHOULDER", 12)),
    ]

    n = len(landmarks_list)
    inversion_mask = [False] * n
    if n < 2:
        return landmarks_list, inversion_mask

    # ── Step 1: Detect swap transitions between consecutive frames ───
    transitions = set()
    prev = None
    for i in range(n):
        curr = landmarks_list[i]
        if curr is None or np.all(np.isnan(curr[:, 0])):
            continue
        if prev is None:
            prev = curr
            continue

        cost_keep = 0.0
        cost_swap = 0.0
        n_valid = 0
        for li, ri in _TRACK_PAIRS:
            if (np.any(np.isnan(prev[[li, ri], :2]))
                    or np.any(np.isnan(curr[[li, ri], :2]))):
                continue
            cost_keep += (np.sum((prev[li, :2] - curr[li, :2]) ** 2)
                          + np.sum((prev[ri, :2] - curr[ri, :2]) ** 2))
            cost_swap += (np.sum((prev[li, :2] - curr[ri, :2]) ** 2)
                          + np.sum((prev[ri, :2] - curr[li, :2]) ** 2))
            n_valid += 1

        if n_valid > 0 and cost_keep > 0:
            # Only flag when swap is SIGNIFICANTLY cheaper (ratio < 0.1).
            # At natural leg crossings cost_swap ≈ cost_keep (ratio ~0.2-0.5);
            # at true label inversions the ratio is near zero.
            if cost_swap / cost_keep < 0.1:
                transitions.add(i)

        prev = curr

    # ── Step 2: Toggle inversion state at each transition ────────────
    in_inversion = False
    for i in range(n):
        if i in transitions:
            in_inversion = not in_inversion
        inversion_mask[i] = in_inversion

    # ── Step 3: Majority-vote polarity check ─────────────────────────
    n_inv = sum(inversion_mask)
    if n_inv > n / 2:
        inversion_mask = [not m for m in inversion_mask]
        n_inv = sum(inversion_mask)

    if n_inv == 0:
        return landmarks_list, inversion_mask

    logger.info(f"Detected {n_inv} label inversions, correcting...")

    # Find all left/right index pairs
    pairs = []
    for name in MP_LANDMARK_NAMES:
        if name.startswith("LEFT_"):
            right_name = name.replace("LEFT_", "RIGHT_")
            if right_name in MP_NAME_TO_INDEX:
                pairs.append((MP_NAME_TO_INDEX[name], MP_NAME_TO_INDEX[right_name]))

    result = [lm.copy() if lm is not None else None for lm in landmarks_list]
    for i, is_inv in enumerate(inversion_mask):
        if is_inv and result[i] is not None:
            for li, ri in pairs:
                result[i][li], result[i][ri] = result[i][ri].copy(), result[i][li].copy()

    return result, inversion_mask


def _swap_auxiliary_lr(aux: np.ndarray, landmark_names: list) -> np.ndarray:
    """Swap left/right landmark pairs in auxiliary data (no x mirror).

    Used to correct label inversions in auxiliary keypoints (Goliath 308,
    WholeBody 133) when the main landmarks have been corrected.
    """
    swapped = aux.copy()
    n = len(landmark_names)
    visited = set()
    for i, name in enumerate(landmark_names):
        if i >= n or i in visited:
            continue
        partner_name = None
        if "left_" in name:
            partner_name = name.replace("left_", "right_")
        elif "right_" in name:
            partner_name = name.replace("right_", "left_")
        elif name.startswith("l_"):
            partner_name = "r_" + name[2:]
        elif name.startswith("r_"):
            partner_name = "l_" + name[2:]
        if partner_name is None:
            continue
        for j in range(n):
            if j != i and landmark_names[j] == partner_name:
                swapped[i], swapped[j] = aux[j].copy(), aux[i].copy()
                visited.add(i)
                visited.add(j)
                break
    return swapped


# ── Sagittal alignment detection ─────────────────────────────────


def detect_sagittal_alignment(data: dict, threshold_deg: float = 15.0) -> dict:
    """Detect whether the camera view is close to pure sagittal.

    Computes the ratio of hip width (LEFT_HIP to RIGHT_HIP distance)
    to mean femur length (hip-to-knee distance). In a pure sagittal
    view both hips overlap, so the ratio is small (~0.0-0.4). As the
    camera deviates toward frontal/oblique, the ratio increases.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    threshold_deg : float
        Maximum acceptable deviation angle (degrees) to be considered
        sagittal (default 15.0).

    Returns
    -------
    dict
        Keys:

        - ``deviation_angle_deg`` (float): estimated camera deviation.
        - ``is_sagittal`` (bool): True if deviation < *threshold_deg*.
        - ``hip_width_ratio`` (float): mean hip-width / femur-length.
        - ``confidence`` (float): 0-1, higher when more frames contribute.
        - ``warning`` (str or None): human-readable warning if oblique.
    """
    frames = data.get("frames", [])

    ratios = []
    for f in frames:
        lm = f.get("landmarks", {})
        lh = lm.get("LEFT_HIP")
        rh = lm.get("RIGHT_HIP")
        lk = lm.get("LEFT_KNEE")
        rk = lm.get("RIGHT_KNEE")

        if not all([lh, rh, lk, rk]):
            continue

        coords = {}
        skip = False
        for name, val in [("lh", lh), ("rh", rh), ("lk", lk), ("rk", rk)]:
            x, y = val.get("x"), val.get("y")
            if x is None or y is None or np.isnan(x) or np.isnan(y):
                skip = True
                break
            coords[name] = np.array([x, y])
        if skip:
            continue

        hip_width = np.linalg.norm(coords["lh"] - coords["rh"])
        femur_left = np.linalg.norm(coords["lh"] - coords["lk"])
        femur_right = np.linalg.norm(coords["rh"] - coords["rk"])
        femur_mean = (femur_left + femur_right) / 2.0

        if femur_mean < 1e-6:
            continue

        ratios.append(hip_width / femur_mean)

    if not ratios:
        return {
            "deviation_angle_deg": 0.0,
            "is_sagittal": True,
            "hip_width_ratio": 0.0,
            "confidence": 0.0,
            "warning": "No valid frames to assess sagittal alignment.",
        }

    ratio = float(np.mean(ratios))
    deviation_deg = float(np.degrees(np.arcsin(min(ratio, 1.0))))
    is_sagittal = deviation_deg < threshold_deg
    confidence = float(min(1.0, len(ratios) / max(1, len(frames))))

    warning = None
    if not is_sagittal:
        warning = (
            f"Camera appears oblique (deviation ~{deviation_deg:.1f} deg). "
            f"Sagittal-plane angles may be inaccurate."
        )

    return {
        "deviation_angle_deg": round(deviation_deg, 2),
        "is_sagittal": is_sagittal,
        "hip_width_ratio": round(ratio, 4),
        "confidence": round(confidence, 3),
        "warning": warning,
    }


# ── Auto-crop ROI ────────────────────────────────────────────────


def auto_crop_roi(
    video_path: str,
    data: Optional[dict] = None,
    padding: float = 0.15,
    output_path: Optional[str] = None,
) -> dict:
    """Compute a bounding box around the subject and optionally crop the video.

    When *data* is provided, the bounding box is derived from the global
    extent of all landmarks across all frames. Otherwise, the full frame
    is returned.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    data : dict, optional
        Pivot JSON dict with ``frames`` populated.
    padding : float
        Fractional padding to add around the bounding box (default 0.15).
    output_path : str, optional
        If given, write a cropped video to this path.

    Returns
    -------
    dict
        Keys:

        - ``bbox`` (tuple): ``(x1, y1, x2, y2)`` in pixel coordinates.
        - ``output_path`` (str or None): path to cropped video if written.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    if data is not None and data.get("frames"):
        xs, ys = [], []
        for f in data["frames"]:
            lm = f.get("landmarks", {})
            for name, val in lm.items():
                x, y = val.get("x"), val.get("y")
                if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                    xs.append(x)
                    ys.append(y)

        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        else:
            x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
    else:
        x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0

    # Add padding
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0.0, x_min - bw * padding)
    y_min = max(0.0, y_min - bh * padding)
    x_max = min(1.0, x_max + bw * padding)
    y_max = min(1.0, y_max + bh * padding)

    # Convert to pixel coordinates
    x1 = int(x_min * width)
    y1 = int(y_min * height)
    x2 = int(x_max * width)
    y2 = int(y_max * height)

    written_path = None
    if output_path is not None:
        crop_w = x2 - x1
        crop_h = y2 - y1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (crop_w, crop_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y1:y2, x1:x2]
            writer.write(cropped)
        writer.release()
        written_path = output_path
        logger.info(f"Cropped video saved to {output_path}")

    cap.release()

    return {
        "bbox": (x1, y1, x2, y2),
        "output_path": written_path,
    }


# ── Person selection (placeholder) ───────────────────────────────


def select_person(
    data: dict,
    strategy: str = "largest",
    bbox: Optional[tuple] = None,
) -> dict:
    """Select or validate a single person in the pose data.

    Currently the extraction pipeline supports only one person per
    frame, so this function mainly validates the data and adds
    selection metadata. When multi-person support is added, it will
    filter landmarks to the chosen subject.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    strategy : str
        Selection strategy: ``'largest'`` (biggest bounding box) or
        ``'center'`` (closest to image centre). Default ``'largest'``.
    bbox : tuple, optional
        If provided ``(x1, y1, x2, y2)`` in normalised coordinates,
        keep only landmarks within this box.

    Returns
    -------
    dict
        Keys:

        - ``selected`` (bool): True if a valid person was found.
        - ``strategy`` (str): the strategy used.
        - ``n_frames_with_landmarks`` (int): frame count with data.
        - ``bbox`` (tuple or None): bounding box of the selected person.
        - ``multi_person_warning`` (bool): from extraction metadata.
    """
    frames = data.get("frames", [])
    extraction = data.get("extraction", {}) or {}

    n_with_lm = 0
    all_xs, all_ys = [], []

    for f in frames:
        lm = f.get("landmarks", {})
        if not lm:
            continue
        has_valid = False
        for name, val in lm.items():
            x, y = val.get("x"), val.get("y")
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                # If bbox filter is given, skip landmarks outside it
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    if not (bx1 <= x <= bx2 and by1 <= y <= by2):
                        continue
                all_xs.append(x)
                all_ys.append(y)
                has_valid = True
        if has_valid:
            n_with_lm += 1

    person_bbox = None
    if all_xs and all_ys:
        person_bbox = (
            round(min(all_xs), 4),
            round(min(all_ys), 4),
            round(max(all_xs), 4),
            round(max(all_ys), 4),
        )

    return {
        "selected": n_with_lm > 0,
        "strategy": strategy,
        "n_frames_with_landmarks": n_with_lm,
        "bbox": person_bbox,
        "multi_person_warning": bool(extraction.get("multi_person_warning", False)),
    }
