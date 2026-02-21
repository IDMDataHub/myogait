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

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    # Create pivot structure
    data = create_empty(video_path, fps, width, height, total_frames)

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
        f"Extracting {total_frames} frames with {model} "
        f"({extractor.n_landmarks} landmarks, {width}x{height} @ {fps:.1f}fps)"
    )

    # Process frames
    raw_landmarks = []  # list of np.ndarray or None
    auxiliary_list = []  # list of np.ndarray or None (e.g. Goliath 308)
    depth_maps = []     # list of np.ndarray or None
    seg_masks = []      # list of np.ndarray or None
    frame_idx = 0
    detected_count = 0
    log_interval = max(1, total_frames // 10)  # log every ~10%

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

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

            if frame_idx % log_interval == 0:
                pct = 100 * frame_idx / total_frames
                det_pct = 100 * detected_count / frame_idx if frame_idx > 0 else 0
                logger.info(f"  {frame_idx}/{total_frames} ({pct:.0f}%) — {det_pct:.0f}% detected")

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx / total_frames)
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
        raw_landmarks = _correct_label_inversions(raw_landmarks)

    # Build frames
    has_auxiliary = any(a is not None for a in auxiliary_list)
    frames = []
    for idx, lm in enumerate(raw_landmarks):
        frame_data = {
            "frame_idx": idx,
            "time_s": round(idx / fps, 4) if fps > 0 else 0.0,
            "landmarks": {},
            "confidence": 0.0,
        }
        if lm is not None:
            for i, name in enumerate(MP_LANDMARK_NAMES):
                frame_data["landmarks"][name] = {
                    "x": float(lm[i, 0]),
                    "y": float(lm[i, 1]),
                    "visibility": float(lm[i, 2]) if not np.isnan(lm[i, 2]) else 0.0,
                }
            valid_vis = lm[:, 2][~np.isnan(lm[:, 2])]
            frame_data["confidence"] = float(np.mean(valid_vis)) if len(valid_vis) > 0 else 0.0

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


def _correct_label_inversions(landmarks_list: list) -> list:
    """Detect and correct left/right label swaps across frames.

    Uses hip and knee x-coordinate ordering to detect inversions.
    """
    lh = MP_NAME_TO_INDEX.get("LEFT_HIP", 23)
    rh = MP_NAME_TO_INDEX.get("RIGHT_HIP", 24)
    lk = MP_NAME_TO_INDEX.get("LEFT_KNEE", 25)
    rk = MP_NAME_TO_INDEX.get("RIGHT_KNEE", 26)

    inversions = []
    for i in range(1, len(landmarks_list)):
        prev = landmarks_list[i - 1]
        curr = landmarks_list[i]
        if prev is None or curr is None:
            continue

        if np.any(np.isnan(prev[[lh, rh], 0])) or np.any(np.isnan(curr[[lh, rh], 0])):
            continue

        prev_hip_order = prev[lh, 0] < prev[rh, 0]
        curr_hip_order = curr[lh, 0] < curr[rh, 0]

        if prev_hip_order != curr_hip_order:
            # Confirm with knees
            if np.any(np.isnan(prev[[lk, rk], 0])) or np.any(np.isnan(curr[[lk, rk], 0])):
                continue
            prev_knee_order = prev[lk, 0] < prev[rk, 0]
            curr_knee_order = curr[lk, 0] < curr[rk, 0]
            if prev_knee_order != curr_knee_order:
                inversions.append(i)

    if not inversions:
        return landmarks_list

    logger.info(f"Detected {len(inversions)} label inversions, correcting...")

    # Find left/right index pairs
    pairs = []
    for name in MP_LANDMARK_NAMES:
        if name.startswith("LEFT_"):
            right_name = name.replace("LEFT_", "RIGHT_")
            if right_name in MP_NAME_TO_INDEX:
                pairs.append((MP_NAME_TO_INDEX[name], MP_NAME_TO_INDEX[right_name]))

    result = [lm.copy() if lm is not None else None for lm in landmarks_list]
    inversions_set = set(inversions)
    in_inversion = False

    for i in range(len(result)):
        if i in inversions_set:
            in_inversion = not in_inversion
        if in_inversion and result[i] is not None:
            for li, ri in pairs:
                result[i][li], result[i][ri] = result[i][ri].copy(), result[i][li].copy()

    return result


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
