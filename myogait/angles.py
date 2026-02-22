"""Joint angle computation from pose landmarks.

Supports multiple computation methods via a registry:
    - "sagittal_vertical_axis" (default): vertical-axis reference method.
      Uses atan2 of the thigh vector relative to the trunk vector,
      providing natural 0° in standing posture and avoiding ±180°
      discontinuities common in 3-point methods.
      Ref: Davis RB, Ounpuu S, Tyburski D, Gage JR. A gait analysis
      data collection and reduction technique. Hum Mov Sci.
      1991;10(5):575-587. doi:10.1016/0167-9457(91)90046-Z

    - "sagittal_classic": classical 3-point interior angle method.
      Ref: Kadaba MP, Ramakrishnan HK, Wootten ME. Measurement of
      lower extremity kinematics during level walking. J Orthop Res.
      1990;8(3):383-392. doi:10.1002/jor.1100080310

Each method is a function(frame, model) -> dict of angles for one frame.
New methods can be registered via register_angle_method().

Angles convention (ISB recommendations):
    - Hip: positive = flexion, negative = extension
    - Knee: 0 = full extension, positive = flexion
    - Ankle: positive = dorsiflexion, negative = plantarflexion
    - Trunk: positive = forward lean
    - Pelvis tilt: positive = right side up

References for conventions:
    Wu G, Siegler S, Allard P, et al. ISB recommendation on definitions
    of joint coordinate system of various joints for the reporting of
    human joint motion — part I: ankle, hip, and spine. J Biomech.
    2002;35(4):543-548. doi:10.1016/S0021-9290(01)00222-6

Correction factor:
    The default correction_factor=0.8 compensates for 2D projection
    overestimation of sagittal-plane joint ROM.
    Ref: Mündermann L, Corazza S, Andriacchi TP. The evolution of
    methods for the capture of human movement leading to markerless
    motion capture for biomechanical applications. J Neuroeng Rehabil.
    2006;3:6. doi:10.1186/1743-0003-3-6
"""

import copy
import logging
from typing import Callable, Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ── Geometry helpers ─────────────────────────────────────────────────


def _get_xy(frame: dict, name: str) -> Optional[np.ndarray]:
    """Extract [x, y] array from a frame's landmarks dict."""
    lm = frame.get("landmarks", {}).get(name)
    if lm is None:
        return None
    x, y = lm.get("x"), lm.get("y")
    if x is None or y is None:
        return None
    try:
        if np.isnan(float(x)) or np.isnan(float(y)):
            return None
    except (TypeError, ValueError):
        pass
    return np.array([x, y])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees [0, 180]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _trunk_angle(left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                 left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """Trunk lean relative to vertical. Positive = forward lean."""
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    trunk = shoulder_center - hip_center
    vertical = np.array([0.0, -1.0])  # up in image coords
    angle = _angle_between(vertical, trunk)
    return angle if trunk[0] > 0 else -angle


def _pelvis_tilt(left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """Pelvis tilt. Positive = right side up.

    Returns NaN in lateral view (hips overlap < 2%).
    """
    pelvis = right_hip - left_hip
    dist = np.linalg.norm(pelvis)
    if dist < 0.02:
        return np.nan
    horizontal = np.array([1.0, 0.0])
    angle = _angle_between(horizontal, pelvis)
    return angle if pelvis[1] < 0 else -angle


def _has_detected_foot(lm: dict, side: str) -> bool:
    """Return True if the heel for *side* is already a real detected landmark.

    A detected heel will have visibility > 0.3.  When the enrichment step
    in extract.py has injected real foot landmarks from Sapiens or RTMW
    auxiliary data, those landmarks carry the model's confidence score
    (typically > 0.8).  Geometric estimates are assigned visibility = 0.5
    but should NOT prevent re-estimation when they come from a prior
    pipeline run; the check here is intentionally > 0.3 so that both
    detected and estimated landmarks are accepted, but missing/NaN ones
    are not.
    """
    heel = lm.get(f"{side}_HEEL")
    fi = lm.get(f"{side}_FOOT_INDEX")
    if heel is None or fi is None:
        return False
    hx = heel.get("x")
    fx = fi.get("x")
    hv = heel.get("visibility", 0.0)
    fv = fi.get("visibility", 0.0)
    if hx is None or fx is None:
        return False
    if isinstance(hx, float) and np.isnan(hx):
        return False
    if isinstance(fx, float) and np.isnan(fx):
        return False
    return hv > 0.3 and fv > 0.3


def _estimate_foot_landmarks(frame: dict) -> dict:
    """Estimate HEEL and FOOT_INDEX from ANKLE for COCO models.

    If real detected foot landmarks are already present (injected by
    ``_enrich_foot_landmarks`` in extract.py from Sapiens / RTMW
    auxiliary data), those landmarks are kept and estimation is skipped
    for that side.
    """
    lm = frame.get("landmarks", {})

    # Check if any side actually needs estimation before deep-copying
    needs_estimation = False
    for side in ("LEFT", "RIGHT"):
        if _has_detected_foot(lm, side):
            continue
        ankle = _get_xy(frame, f"{side}_ANKLE")
        knee = _get_xy(frame, f"{side}_KNEE")
        if ankle is None or knee is None:
            continue
        needs_estimation = True
        break

    if not needs_estimation:
        return frame

    frame = copy.deepcopy(frame)
    lm = frame.get("landmarks", {})

    for side in ("LEFT", "RIGHT"):
        # Skip estimation when real detected foot landmarks exist
        if _has_detected_foot(lm, side):
            continue

        ankle = _get_xy(frame, f"{side}_ANKLE")
        knee = _get_xy(frame, f"{side}_KNEE")
        if ankle is None or knee is None:
            continue

        dx = ankle[0] - knee[0]
        dy = ankle[1] - knee[1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            continue
        foot_len = length * 0.25

        heel_name = f"{side}_HEEL"
        heel = lm.get(heel_name)
        heel_is_nan = False
        try:
            heel_is_nan = heel is not None and heel.get("x") is not None and np.isnan(float(heel["x"]))
        except (TypeError, ValueError):
            pass
        if heel is None or heel.get("x") is None or heel_is_nan:
            lm[heel_name] = {
                "x": float(ankle[0] + dx / length * foot_len * 0.3),
                "y": float(ankle[1] + dy / length * foot_len * 0.3 + foot_len * 0.1),
                "visibility": 0.5,
            }

        fi_name = f"{side}_FOOT_INDEX"
        fi = lm.get(fi_name)
        fi_is_nan = False
        try:
            fi_is_nan = fi is not None and fi.get("x") is not None and np.isnan(float(fi["x"]))
        except (TypeError, ValueError):
            pass
        if fi is None or fi.get("x") is None or fi_is_nan:
            lm[fi_name] = {
                "x": float(ankle[0] - dx / length * foot_len * 0.5),
                "y": float(ankle[1] + foot_len * 0.15),
                "visibility": 0.5,
            }

    frame["landmarks"] = lm
    return frame


def _get_foot_index_from_toes(frame: dict, side: str) -> Optional[np.ndarray]:
    """Compute foot index as midpoint of big_toe and small_toe when available.

    Falls back to the existing FOOT_INDEX landmark if toe data is not
    present.  This gives a more accurate foot reference point when
    real toe landmarks have been injected from Sapiens or RTMW
    auxiliary data.
    """
    big_toe = _get_xy(frame, f"{side}_BIG_TOE")
    small_toe = _get_xy(frame, f"{side}_SMALL_TOE")
    if big_toe is not None and small_toe is not None:
        return (big_toe + small_toe) / 2.0
    # Fallback to standard FOOT_INDEX
    return _get_xy(frame, f"{side}_FOOT_INDEX")


def _unwrap_angles(values: list) -> list:
    """Unwrap angle series to remove ±180 discontinuities.

    Does NOT recenter on median -- the absolute anatomical reference
    must be preserved.  Recentering, if needed, is handled solely by
    neutral calibration (which by default only touches ankle joints).
    """
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return values
    unwrapped = np.degrees(np.unwrap(np.radians(arr[mask])))
    result = arr.copy()
    result[mask] = unwrapped
    return result.tolist()


def _detect_walking_direction(data: dict) -> str:
    """Detect walking direction from hip center horizontal displacement.

    Analyses the mean horizontal displacement of the hip center
    (average of LEFT_HIP and RIGHT_HIP x-coordinates) across all
    frames.  If x increases over time the subject walks left-to-right
    (in image coordinates); if it decreases the subject walks
    right-to-left.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.

    Returns
    -------
    str
        ``"left_to_right"`` or ``"right_to_left"``.
    """
    frames = data.get("frames", [])
    if len(frames) < 2:
        return "left_to_right"

    xs = []
    for frame in frames:
        if frame.get("confidence", 1.0) < 0.1:
            continue
        lm = frame.get("landmarks", {})
        lh = lm.get("LEFT_HIP")
        rh = lm.get("RIGHT_HIP")
        if (lh is not None and rh is not None
                and lh.get("x") is not None and rh.get("x") is not None
                and not np.isnan(lh["x"]) and not np.isnan(rh["x"])):
            xs.append((lh["x"] + rh["x"]) / 2.0)

    if len(xs) < 2:
        return "left_to_right"

    # Use the overall displacement (last quarter vs first quarter) to be
    # robust to noise.
    n = len(xs)
    q = max(1, n // 4)
    mean_start = float(np.mean(xs[:q]))
    mean_end = float(np.mean(xs[-q:]))

    return "left_to_right" if mean_end >= mean_start else "right_to_left"


def _extract_landmark_positions(frame: dict) -> dict:
    """Extract landmark positions for gaitkit compatibility."""
    lp = {}
    for name in ["left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle"]:
        mp_name = name.upper()
        pt = _get_xy(frame, mp_name)
        vis = frame.get("landmarks", {}).get(mp_name, {}).get("visibility", 0.0)
        if pt is not None:
            lp[name] = [float(pt[0]), float(pt[1]), float(vis)]
    return lp


# ── Method: sagittal_vertical_axis (default) ────────────────────────


def _method_sagittal_vertical_axis(frame: dict, model: str) -> dict:
    """Vertical-axis reference method (gaitopensim).

    Hip angle = atan2(thigh_from_vertical) - atan2(trunk_from_vertical).
    Natural 0° in standing posture, no discontinuities.
    """
    needs_foot = model != "mediapipe"
    f = _estimate_foot_landmarks(frame) if needs_foot else frame
    result = {"frame_idx": frame["frame_idx"]}

    l_shoulder = _get_xy(f, "LEFT_SHOULDER")
    r_shoulder = _get_xy(f, "RIGHT_SHOULDER")
    l_hip = _get_xy(f, "LEFT_HIP")
    r_hip = _get_xy(f, "RIGHT_HIP")

    if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
        hip_center = (l_hip + r_hip) / 2
        shoulder_center = (l_shoulder + r_shoulder) / 2
        trunk_vec = hip_center - shoulder_center
        result["trunk_angle"] = _trunk_angle(l_shoulder, r_shoulder, l_hip, r_hip)
        result["pelvis_tilt"] = _pelvis_tilt(l_hip, r_hip)
    else:
        trunk_vec = None
        result["trunk_angle"] = np.nan
        result["pelvis_tilt"] = np.nan

    result["landmark_positions"] = _extract_landmark_positions(f)

    for side, prefix in [("L", "LEFT"), ("R", "RIGHT")]:
        hip = _get_xy(f, f"{prefix}_HIP")
        knee = _get_xy(f, f"{prefix}_KNEE")
        ankle = _get_xy(f, f"{prefix}_ANKLE")
        # Use real toe midpoint when available, else standard FOOT_INDEX
        foot = _get_foot_index_from_toes(f, prefix)

        # Hip: vertical axis method
        if hip is not None and knee is not None and trunk_vec is not None:
            thigh_vector = knee - hip
            thigh_angle = np.arctan2(thigh_vector[0], thigh_vector[1])
            trunk_angle = np.arctan2(trunk_vec[0], trunk_vec[1])
            raw = np.degrees(thigh_angle - trunk_angle)
            # Normalize to [-180, 180] to avoid wrap-around artifacts
            result[f"hip_{side}"] = float(((raw + 180) % 360) - 180)
        else:
            result[f"hip_{side}"] = np.nan

        # Knee: 180 - interior angle
        if hip is not None and knee is not None and ankle is not None:
            thigh = hip - knee
            shank = ankle - knee
            result[f"knee_{side}"] = 180.0 - _angle_between(thigh, shank)
        else:
            result[f"knee_{side}"] = np.nan

        # Ankle: 90 - angle(shank, foot)
        if knee is not None and ankle is not None and foot is not None:
            shank = ankle - knee
            foot_vec = foot - ankle
            result[f"ankle_{side}"] = 90.0 - _angle_between(shank, foot_vec)
        else:
            result[f"ankle_{side}"] = np.nan

    return result


# ── Method: sagittal_classic ─────────────────────────────────────────


def _method_sagittal_classic(frame: dict, model: str) -> dict:
    """Classical 3-point interior angle method.

    Hip = 180 - angle(shoulder-hip-knee). 0° = aligned, positive = flexion.
    Knee = 180 - angle(hip-knee-ankle). 0° = full extension.
    Ankle = 90 - angle(knee-ankle-foot). dorsi/plantarflexion.

    Simpler than vertical-axis but has a non-zero offset in standing.
    """
    needs_foot = model != "mediapipe"
    f = _estimate_foot_landmarks(frame) if needs_foot else frame
    result = {"frame_idx": frame["frame_idx"]}

    l_shoulder = _get_xy(f, "LEFT_SHOULDER")
    r_shoulder = _get_xy(f, "RIGHT_SHOULDER")
    l_hip = _get_xy(f, "LEFT_HIP")
    r_hip = _get_xy(f, "RIGHT_HIP")

    if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
        result["trunk_angle"] = _trunk_angle(l_shoulder, r_shoulder, l_hip, r_hip)
        result["pelvis_tilt"] = _pelvis_tilt(l_hip, r_hip)
    else:
        result["trunk_angle"] = np.nan
        result["pelvis_tilt"] = np.nan

    result["landmark_positions"] = _extract_landmark_positions(f)

    for side, prefix in [("L", "LEFT"), ("R", "RIGHT")]:
        shoulder = _get_xy(f, f"{prefix}_SHOULDER")
        hip = _get_xy(f, f"{prefix}_HIP")
        knee = _get_xy(f, f"{prefix}_KNEE")
        ankle = _get_xy(f, f"{prefix}_ANKLE")
        # Use real toe midpoint when available, else standard FOOT_INDEX
        foot = _get_foot_index_from_toes(f, prefix)

        # Hip: 180 - angle(shoulder, hip, knee)
        if shoulder is not None and hip is not None and knee is not None:
            v1 = shoulder - hip
            v2 = knee - hip
            result[f"hip_{side}"] = 180.0 - _angle_between(v1, v2)
        else:
            result[f"hip_{side}"] = np.nan

        # Knee: 180 - angle(hip, knee, ankle)
        if hip is not None and knee is not None and ankle is not None:
            thigh = hip - knee
            shank = ankle - knee
            result[f"knee_{side}"] = 180.0 - _angle_between(thigh, shank)
        else:
            result[f"knee_{side}"] = np.nan

        # Ankle: 90 - angle(knee, ankle, foot)
        if knee is not None and ankle is not None and foot is not None:
            shank = ankle - knee
            foot_vec = foot - ankle
            result[f"ankle_{side}"] = 90.0 - _angle_between(shank, foot_vec)
        else:
            result[f"ankle_{side}"] = np.nan

    return result


# ── Method registry ──────────────────────────────────────────────────

ANGLE_METHODS: Dict[str, Callable] = {
    "sagittal_vertical_axis": _method_sagittal_vertical_axis,
    "sagittal_classic": _method_sagittal_classic,
}


def register_angle_method(name: str, func: Callable):
    """Register a custom angle computation method.

    The function must accept (frame: dict, model: str) -> dict
    and return a dict with keys: frame_idx, hip_L, hip_R, knee_L, knee_R,
    ankle_L, ankle_R, trunk_angle, pelvis_tilt, landmark_positions.
    """
    ANGLE_METHODS[name] = func


def list_angle_methods() -> list:
    """Return available angle computation method names."""
    return list(ANGLE_METHODS.keys())


# ── Public API ───────────────────────────────────────────────────────


def compute_angles(
    data: dict,
    method: str = "sagittal_vertical_axis",
    correction_factor: float = 0.8,
    calibrate: bool = True,
    calibration_frames: int = 30,
    calibration_joints: Optional[list] = None,
    min_confidence: float = 0.0,
) -> dict:
    """Compute joint angles and add to pivot JSON.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    method : str, optional
        Angle computation method (default ``"sagittal_vertical_axis"``).
        See ``list_angle_methods()`` for available methods.
    correction_factor : float, optional
        Scale factor for 2D ROM overestimation (default 0.8 for
        MediaPipe, use 1.0 for 3D models).
    calibrate : bool, optional
        Subtract neutral angle from the first N frames (default True).
    calibration_frames : int, optional
        Number of frames for neutral calibration (default 30).
    calibration_joints : list of str, optional
        Joints to calibrate (default: ``["ankle_L", "ankle_R"]``).
    min_confidence : float, optional
        Skip angle computation on frames with confidence below this
        threshold (default 0.0, i.e. compute on all frames).  Skipped
        frames still appear in the output with all joint values set to
        ``None`` so that frame indices stay aligned.

    Returns
    -------
    dict
        Modified *data* dict with ``angles`` populated.

    Raises
    ------
    ValueError
        If *data* has no frames or *method* is unknown.
    """
    if not data.get("frames"):
        raise ValueError("No frames in data. Run extract() first.")

    if method not in ANGLE_METHODS:
        available = ", ".join(ANGLE_METHODS.keys())
        raise ValueError(f"Unknown angle method '{method}'. Available: {available}")

    if calibration_joints is None:
        calibration_joints = ["ankle_L", "ankle_R"]

    method_func = ANGLE_METHODS[method]
    model = data.get("extraction", {}).get("model", "mediapipe")

    # Detect walking direction (only relevant for sagittal_vertical_axis)
    walking_direction = _detect_walking_direction(data)
    logger.info("Detected walking direction: %s", walking_direction)

    # Compute per-frame angles (skip low-confidence frames)
    joint_keys = ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]
    angle_frames = []
    n_skipped = 0
    for frame in data["frames"]:
        conf = frame.get("confidence", 0.0)
        if conf < min_confidence:
            # Placeholder with None values so frame indices stay aligned
            af = {"frame_idx": frame["frame_idx"]}
            for k in joint_keys:
                af[k] = np.nan
            af["trunk_angle"] = np.nan
            af["pelvis_tilt"] = np.nan
            af["landmark_positions"] = {}
            angle_frames.append(af)
            n_skipped += 1
        else:
            angle_frames.append(method_func(frame, model))
    if n_skipped:
        logger.info("Skipped %d/%d low-confidence frames (< %.2f)",
                     n_skipped, len(data["frames"]), min_confidence)

    # If the subject walks right-to-left the sagittal_vertical_axis
    # hip sign is mirrored.  Invert so that flexion stays positive.
    if walking_direction == "right_to_left" and method == "sagittal_vertical_axis":
        for af in angle_frames:
            for side in ("L", "R"):
                key = f"hip_{side}"
                v = af.get(key)
                if v is not None and not np.isnan(v):
                    af[key] = -v

    # Trunk angle also needs direction correction for right-to-left walking.
    if walking_direction == "right_to_left":
        for af in angle_frames:
            v = af.get("trunk_angle")
            if v is not None and not np.isnan(v):
                af["trunk_angle"] = -v

    # Apply correction factor
    if correction_factor != 1.0:
        joint_keys = ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]
        for af in angle_frames:
            for key in joint_keys:
                v = af.get(key)
                if v is not None and not np.isnan(v):
                    af[key] = v * correction_factor

    # Normalize hip angles to [-180, 180] (no unwrap — gait is cyclic)
    for side in ("L", "R"):
        key = f"hip_{side}"
        for af in angle_frames:
            v = af.get(key)
            if v is not None and not np.isnan(v):
                af[key] = float(((v + 180) % 360) - 180)

    # Neutral calibration
    if calibrate and len(angle_frames) >= calibration_frames:
        for key in calibration_joints:
            vals = [af[key] for af in angle_frames[:calibration_frames]
                    if not np.isnan(af.get(key, np.nan))]
            if vals:
                offset = float(np.median(vals))
                for af in angle_frames:
                    v = af.get(key)
                    if v is not None and not np.isnan(v):
                        af[key] = v - offset

    # Convert NaN to None for JSON serialization
    for af in angle_frames:
        for key in list(af.keys()):
            if key == "landmark_positions":
                continue
            if isinstance(af[key], float) and np.isnan(af[key]):
                af[key] = None

    data["angles"] = {
        "method": method,
        "correction_factor": correction_factor,
        "calibrated": calibrate,
        "calibration_frames": calibration_frames if calibrate else None,
        "calibration_joints": calibration_joints if calibrate else [],
        "walking_direction": walking_direction,
        "joints": ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"],
        "extra": ["trunk_angle", "pelvis_tilt"],
        "frames": angle_frames,
    }

    return data


# ── Extended angle functions ─────────────────────────────────────────


def _head_angle(frame: dict) -> float:
    """Compute head posture angle (head-trunk angle in the sagittal plane).

    Measures the angle between the head vector (ear_midpoint to nose) and
    the trunk vector (hip_center to shoulder_center). A positive value
    indicates forward head posture.

    Uses landmarks: NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER,
    RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP.

    Args:
        frame: Single frame dict with ``landmarks``.

    Returns:
        Head angle in degrees. Positive = forward head posture.
        Returns NaN if required landmarks are missing.
    """
    nose = _get_xy(frame, "NOSE")
    l_ear = _get_xy(frame, "LEFT_EAR")
    r_ear = _get_xy(frame, "RIGHT_EAR")
    l_shoulder = _get_xy(frame, "LEFT_SHOULDER")
    r_shoulder = _get_xy(frame, "RIGHT_SHOULDER")
    l_hip = _get_xy(frame, "LEFT_HIP")
    r_hip = _get_xy(frame, "RIGHT_HIP")

    if any(v is None for v in [nose, l_ear, r_ear, l_shoulder, r_shoulder, l_hip, r_hip]):
        return np.nan

    ear_mid = (l_ear + r_ear) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2

    # Head vector: ear midpoint to nose
    head_vec = nose - ear_mid
    # Trunk vector: hip center to shoulder center (points upward)
    trunk_vec = shoulder_center - hip_center

    angle = _angle_between(head_vec, trunk_vec)

    # Sign convention: positive = nose forward of trunk line
    # Use cross product sign to determine direction
    cross = head_vec[0] * trunk_vec[1] - head_vec[1] * trunk_vec[0]
    return angle if cross >= 0 else -angle


def _arm_angles(frame: dict) -> dict:
    """Compute shoulder flexion and elbow flexion angles for both sides.

    Shoulder flexion: angle of the upper arm vector relative to trunk
    vertical (hip_center to shoulder_center direction). 0 = arm hanging
    straight down.

    Elbow flexion: 180 minus the interior angle at the elbow
    (shoulder-elbow-wrist). 0 = full extension, positive = flexion.

    Uses landmarks: LEFT/RIGHT SHOULDER, ELBOW, WRIST, HIP.

    Args:
        frame: Single frame dict with ``landmarks``.

    Returns:
        Dict with keys: shoulder_flex_L, shoulder_flex_R,
        elbow_flex_L, elbow_flex_R (all in degrees, NaN if missing).
    """
    l_shoulder = _get_xy(frame, "LEFT_SHOULDER")
    r_shoulder = _get_xy(frame, "RIGHT_SHOULDER")
    l_hip = _get_xy(frame, "LEFT_HIP")
    r_hip = _get_xy(frame, "RIGHT_HIP")

    result = {
        "shoulder_flex_L": np.nan,
        "shoulder_flex_R": np.nan,
        "elbow_flex_L": np.nan,
        "elbow_flex_R": np.nan,
    }

    # Trunk vertical direction (used for shoulder flexion reference)
    trunk_vertical = None
    if (l_shoulder is not None and r_shoulder is not None
            and l_hip is not None and r_hip is not None):
        shoulder_center = (l_shoulder + r_shoulder) / 2
        hip_center = (l_hip + r_hip) / 2
        # Points downward in image coords (hip below shoulder)
        trunk_vertical = hip_center - shoulder_center

    for side, prefix in [("L", "LEFT"), ("R", "RIGHT")]:
        shoulder = _get_xy(frame, f"{prefix}_SHOULDER")
        elbow = _get_xy(frame, f"{prefix}_ELBOW")
        wrist = _get_xy(frame, f"{prefix}_WRIST")

        # Shoulder flexion
        if shoulder is not None and elbow is not None and trunk_vertical is not None:
            upper_arm = elbow - shoulder
            result[f"shoulder_flex_{side}"] = _angle_between(upper_arm, trunk_vertical)

        # Elbow flexion
        if shoulder is not None and elbow is not None and wrist is not None:
            v1 = shoulder - elbow
            v2 = wrist - elbow
            result[f"elbow_flex_{side}"] = 180.0 - _angle_between(v1, v2)

    return result


def _pelvis_sagittal_tilt(frame: dict) -> float:
    """Compute pelvis anterior/posterior tilt in the sagittal plane.

    Measures the angle between the hip_center-to-shoulder_center vector
    and the vertical axis. This differs from the existing ``_pelvis_tilt``
    which measures frontal-plane (lateral) tilt.

    Positive = anterior tilt (trunk leaning forward relative to hips).

    Uses landmarks: LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER.

    Args:
        frame: Single frame dict with ``landmarks``.

    Returns:
        Sagittal pelvis tilt in degrees. Positive = anterior tilt.
        Returns NaN if required landmarks are missing.
    """
    l_hip = _get_xy(frame, "LEFT_HIP")
    r_hip = _get_xy(frame, "RIGHT_HIP")
    l_shoulder = _get_xy(frame, "LEFT_SHOULDER")
    r_shoulder = _get_xy(frame, "RIGHT_SHOULDER")

    if any(v is None for v in [l_hip, r_hip, l_shoulder, r_shoulder]):
        return np.nan

    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2

    # Vector from hip center to shoulder center
    trunk_vec = shoulder_center - hip_center
    # Vertical (pointing up in image = negative y)
    vertical = np.array([0.0, -1.0])

    angle = _angle_between(trunk_vec, vertical)

    # Sign: positive when trunk tilts forward (shoulder x > hip x)
    return angle if trunk_vec[0] >= 0 else -angle


def _depth_enhanced_angles(frame: dict) -> Optional[dict]:
    """Correct 2D joint angles using depth data to compensate for foreshortening.

    When a limb segment is angled toward or away from the camera, its
    projected 2D length is shorter than the true 3D length. If depth data
    is available (from Sapiens depth estimation), we can estimate the
    foreshortening factor and correct the 2D angles.

    The correction uses the depth difference between proximal and distal
    joints to estimate the out-of-plane angle, then scales the 2D angle
    by the inverse cosine of that angle.

    Only works when the frame has a ``landmark_depths`` key.

    Args:
        frame: Single frame dict with ``landmarks`` and optionally
            ``landmark_depths`` mapping landmark names to depth values.

    Returns:
        Dict with corrected angles (hip_L, hip_R, knee_L, knee_R,
        ankle_L, ankle_R) or None if no depth data is available.
    """
    depths = frame.get("landmark_depths")
    if depths is None:
        return None

    result = {}

    for side, prefix in [("L", "LEFT"), ("R", "RIGHT")]:
        # Hip angle correction: use hip-knee depth difference
        hip = _get_xy(frame, f"{prefix}_HIP")
        knee = _get_xy(frame, f"{prefix}_KNEE")
        ankle = _get_xy(frame, f"{prefix}_ANKLE")

        d_hip = depths.get(f"{prefix}_HIP")
        d_knee = depths.get(f"{prefix}_KNEE")
        d_ankle = depths.get(f"{prefix}_ANKLE")

        # Hip correction
        if (hip is not None and knee is not None
                and d_hip is not None and d_knee is not None):
            seg_2d = np.linalg.norm(knee - hip)
            depth_diff = abs(d_knee - d_hip)
            if seg_2d > 1e-6:
                # Estimate true 3D length
                seg_3d = np.sqrt(seg_2d ** 2 + depth_diff ** 2)
                correction = seg_3d / seg_2d if seg_2d > 0 else 1.0
                # Simple scaling: multiply 2D angle by correction factor
                # (capped to avoid extreme corrections)
                correction = min(correction, 2.0)
            else:
                correction = 1.0
            result[f"hip_{side}_correction"] = float(correction)
        else:
            result[f"hip_{side}_correction"] = 1.0

        # Knee correction
        if (knee is not None and ankle is not None
                and d_knee is not None and d_ankle is not None):
            seg_2d = np.linalg.norm(ankle - knee)
            depth_diff = abs(d_ankle - d_knee)
            if seg_2d > 1e-6:
                seg_3d = np.sqrt(seg_2d ** 2 + depth_diff ** 2)
                correction = min(seg_3d / seg_2d, 2.0)
            else:
                correction = 1.0
            result[f"knee_{side}_correction"] = float(correction)
        else:
            result[f"knee_{side}_correction"] = 1.0

        # Ankle correction
        if (ankle is not None and d_ankle is not None):
            foot = _get_xy(frame, f"{prefix}_FOOT_INDEX")
            d_foot = depths.get(f"{prefix}_FOOT_INDEX")
            if foot is not None and d_foot is not None:
                seg_2d = np.linalg.norm(foot - ankle)
                depth_diff = abs(d_foot - d_ankle)
                if seg_2d > 1e-6:
                    seg_3d = np.sqrt(seg_2d ** 2 + depth_diff ** 2)
                    correction = min(seg_3d / seg_2d, 2.0)
                else:
                    correction = 1.0
                result[f"ankle_{side}_correction"] = float(correction)
            else:
                result[f"ankle_{side}_correction"] = 1.0
        else:
            result[f"ankle_{side}_correction"] = 1.0

    return result


def compute_frontal_angles(data: dict) -> dict:
    """Compute frontal-plane angles from depth-enhanced pose data.

    Requires ``landmark_depths`` in frame data (from Sapiens depth
    estimation). Computes hip abduction and knee valgus/varus for
    each side.

    Hip abduction: angle of the thigh in the frontal plane relative
    to vertical, using the mediolateral (x) and depth (z) components.

    Knee valgus: frontal-plane angle at the knee using hip, knee, ankle
    positions in the x-z plane.

    Results are stored in ``data["angles_frontal"]``.

    Args:
        data: Pivot JSON dict with ``frames`` populated, ideally with
            ``landmark_depths`` in each frame.

    Returns:
        Modified *data* dict with ``angles_frontal`` populated.
        If no depth data is available, ``data["angles_frontal"]`` is set
        to None.
    """
    frames = data.get("frames", [])
    if not frames:
        data["angles_frontal"] = None
        return data

    # Check if any frame has depth data
    has_depth = any(f.get("landmark_depths") is not None for f in frames)
    if not has_depth:
        data["angles_frontal"] = None
        return data

    frontal_frames = []
    for frame in frames:
        depths = frame.get("landmark_depths", {})
        if not depths:
            frontal_frames.append({
                "frame_idx": frame["frame_idx"],
                "hip_abduction_L": None,
                "hip_abduction_R": None,
                "knee_valgus_L": None,
                "knee_valgus_R": None,
            })
            continue

        result = {"frame_idx": frame["frame_idx"]}

        for side, prefix in [("L", "LEFT"), ("R", "RIGHT")]:
            hip = _get_xy(frame, f"{prefix}_HIP")
            knee = _get_xy(frame, f"{prefix}_KNEE")
            ankle = _get_xy(frame, f"{prefix}_ANKLE")

            d_hip = depths.get(f"{prefix}_HIP")
            d_knee = depths.get(f"{prefix}_KNEE")
            d_ankle = depths.get(f"{prefix}_ANKLE")

            # Hip abduction: frontal plane angle of thigh
            if (hip is not None and knee is not None
                    and d_hip is not None and d_knee is not None):
                # Frontal plane: x (mediolateral) and z (depth)
                dx = knee[0] - hip[0]
                dz = d_knee - d_hip
                # Vertical in frontal plane is pure z-depth direction
                frontal_vec = np.array([dx, dz])
                vertical = np.array([0.0, 1.0])
                angle = _angle_between(frontal_vec, vertical)
                result[f"hip_abduction_{side}"] = float(angle)
            else:
                result[f"hip_abduction_{side}"] = None

            # Knee valgus: frontal plane angle at knee
            if (hip is not None and knee is not None and ankle is not None
                    and d_hip is not None and d_knee is not None and d_ankle is not None):
                # Thigh in frontal plane
                v1_x = hip[0] - knee[0]
                v1_z = d_hip - d_knee
                # Shank in frontal plane
                v2_x = ankle[0] - knee[0]
                v2_z = d_ankle - d_knee
                v1 = np.array([v1_x, v1_z])
                v2 = np.array([v2_x, v2_z])
                angle = 180.0 - _angle_between(v1, v2)
                result[f"knee_valgus_{side}"] = float(angle)
            else:
                result[f"knee_valgus_{side}"] = None

        frontal_frames.append(result)

    data["angles_frontal"] = {
        "frames": frontal_frames,
        "joints": ["hip_abduction_L", "hip_abduction_R", "knee_valgus_L", "knee_valgus_R"],
    }
    return data


def compute_extended_angles(data: dict) -> dict:
    """Compute extended angles (head, arms, pelvis sagittal) and add to angle frames.

    Call AFTER ``compute_angles()``. Adds the following keys to each
    angle frame in ``data["angles"]["frames"]``:

    - ``head_angle``: head posture angle (forward head = positive)
    - ``shoulder_flex_L``, ``shoulder_flex_R``: shoulder flexion
    - ``elbow_flex_L``, ``elbow_flex_R``: elbow flexion
    - ``pelvis_sagittal_tilt``: anterior/posterior pelvic tilt

    Also attempts depth-enhanced angle correction if depth data is
    available in the pose frames, storing correction factors in each
    angle frame.

    Args:
        data: Pivot JSON dict with ``angles`` populated (via
            ``compute_angles()``).

    Returns:
        Modified *data* dict with extended angles added.

    Raises:
        ValueError: If ``data["angles"]`` is not populated.
    """
    if not data.get("angles") or not data["angles"].get("frames"):
        raise ValueError("No angles computed. Run compute_angles() first.")

    angle_frames = data["angles"]["frames"]
    pose_frames = data.get("frames", [])

    for i, af in enumerate(angle_frames):
        if i >= len(pose_frames):
            break
        frame = pose_frames[i]

        # Head angle
        af["head_angle"] = _head_angle(frame)

        # Arm angles
        arm = _arm_angles(frame)
        af["shoulder_flex_L"] = arm["shoulder_flex_L"]
        af["shoulder_flex_R"] = arm["shoulder_flex_R"]
        af["elbow_flex_L"] = arm["elbow_flex_L"]
        af["elbow_flex_R"] = arm["elbow_flex_R"]

        # Pelvis sagittal tilt
        af["pelvis_sagittal_tilt"] = _pelvis_sagittal_tilt(frame)

        # Depth-enhanced correction
        depth_corr = _depth_enhanced_angles(frame)
        if depth_corr is not None:
            af["depth_corrections"] = depth_corr

    # Convert NaN to None for JSON serialization in new keys
    extended_keys = [
        "head_angle", "shoulder_flex_L", "shoulder_flex_R",
        "elbow_flex_L", "elbow_flex_R", "pelvis_sagittal_tilt",
    ]
    for af in angle_frames:
        for key in extended_keys:
            val = af.get(key)
            if isinstance(val, float) and np.isnan(val):
                af[key] = None

    return data


def foot_progression_angle(data: dict) -> dict:
    """Compute foot progression angle (heel-to-toe vs horizontal).

    The foot progression angle (FPA) is the angle between the
    heel-to-toe vector and the horizontal axis.  Positive values
    indicate external rotation (out-toeing), negative values indicate
    internal rotation (in-toeing).

    Uses LEFT_HEEL -> LEFT_FOOT_INDEX and RIGHT_HEEL -> RIGHT_FOOT_INDEX.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.

    Returns
    -------
    dict
        ``{"foot_angle_L": [...], "foot_angle_R": [...]}``, where each
        list has one entry per frame (degrees, or ``None`` if landmarks
        are missing).
    """
    frames = data.get("frames", [])
    foot_angle_L = []
    foot_angle_R = []

    for frame in frames:
        for side, result_list in [("LEFT", foot_angle_L), ("RIGHT", foot_angle_R)]:
            heel = _get_xy(frame, f"{side}_HEEL")
            # Prefer real toe midpoint over estimated FOOT_INDEX
            toe = _get_foot_index_from_toes(frame, side)
            if heel is not None and toe is not None:
                vec = toe - heel
                # atan2(y, x) -- but y is inverted in image coords so
                # we use -y to get the real-world angle.  Positive angle
                # means the toe points outward (external rotation).
                angle = np.degrees(np.arctan2(-vec[1], vec[0]))
                result_list.append(float(angle))
            else:
                result_list.append(None)

    return {"foot_angle_L": foot_angle_L, "foot_angle_R": foot_angle_R}
