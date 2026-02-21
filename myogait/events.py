"""Gait event detection: heel strike (HS) and toe off (TO).

Methods available:
    - "zeni" (default): Ankle AP position relative to pelvis.
      Ref: Zeni JA Jr, Richards JG, Higginson JS. Two simple methods
      for determining gait events during treadmill and overground
      walking using kinematic data. Gait Posture. 2008;27(4):710-714.
      doi:10.1016/j.gaitpost.2007.07.007

    - "velocity": Foot vertical velocity zero-crossings.
      Ref: Hreljac A, Marshall RN. Algorithms to determine event timing
      during normal walking using kinematic data. J Biomech.
      2000;33(6):783-786. doi:10.1016/S0021-9290(00)00014-2

    - "crossing": Knee/ankle X-coordinate crossing detection.
      Based on contralateral limb progression analysis.
      Ref: Desailly E, Daniel Y, Sardain P, Lacouture P. Foot contact
      event detection using kinematic data in cerebral palsy children
      and normal adults gait. Gait Posture. 2009;29(1):76-80.
      doi:10.1016/j.gaitpost.2008.06.009

    - "oconnor": Heel anteroposterior velocity zero-crossings.
      Ref: O'Connor CM, Thorpe SK, O'Malley MJ, Vaughan CL.
      Automatic detection of gait events using kinematic data.
      Gait Posture. 2007;25(3):469-474.
      doi:10.1016/j.gaitpost.2006.05.016

All methods are registered in EVENT_METHODS and can be extended
via register_event_method().
"""

import logging
import math
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

logger = logging.getLogger(__name__)


def _extract_landmark_series(frames: list, name: str, coord: str = "x") -> np.ndarray:
    """Extract a single coordinate time series for a landmark."""
    values = []
    for f in frames:
        lm = f.get("landmarks", {}).get(name)
        if lm is not None and lm.get(coord) is not None:
            values.append(float(lm[coord]))
        else:
            values.append(np.nan)
    return np.array(values)


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaN values."""
    out = arr.copy()
    # Forward fill
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    # Backward fill
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out


def _lowpass_filter(signal_arr: np.ndarray, cutoff: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return signal_arr
    b, a = butter(order, cutoff / nyq, btype="low")
    # Need enough data for filtfilt
    if len(signal_arr) < 3 * max(len(b), len(a)):
        return signal_arr
    return filtfilt(b, a, signal_arr)


def _detect_zeni(
    frames: list,
    fps: float,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> Dict[str, list]:
    """Zeni method: ankle position relative to pelvis.

    HS = ankle most ANTERIOR (peak of ankle_rel)
    TO = ankle most POSTERIOR (trough of ankle_rel)
    """
    min_distance = max(1, int(min_cycle_duration * fps / 2))

    # Extract x-coordinates
    left_ankle_x = _fill_nan(_extract_landmark_series(frames, "LEFT_ANKLE", "x"))
    right_ankle_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_ANKLE", "x"))
    left_hip_x = _fill_nan(_extract_landmark_series(frames, "LEFT_HIP", "x"))
    right_hip_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_HIP", "x"))

    if np.all(np.isnan(left_ankle_x)) or np.all(np.isnan(left_hip_x)):
        logger.warning("Not enough landmark data for event detection")
        return {"left_hs": [], "right_hs": [], "left_to": [], "right_to": []}

    # Pelvis midpoint
    pelvis_x = (left_hip_x + right_hip_x) / 2

    # Relative ankle position
    left_ankle_rel = left_ankle_x - pelvis_x
    right_ankle_rel = right_ankle_x - pelvis_x

    # Low-pass filter
    left_ankle_rel = _lowpass_filter(left_ankle_rel, cutoff_freq, fps)
    right_ankle_rel = _lowpass_filter(right_ankle_rel, cutoff_freq, fps)

    # Detect peaks and troughs
    results = {}
    for side, rel_signal in [("left", left_ankle_rel), ("right", right_ankle_rel)]:
        # HS = peaks (foot most anterior)
        hs_indices, hs_props = find_peaks(
            rel_signal, distance=min_distance, prominence=0.005
        )
        # TO = troughs (foot most posterior)
        to_indices, to_props = find_peaks(
            -rel_signal, distance=min_distance, prominence=0.005
        )

        # Build event lists with confidence from prominence
        hs_proms = hs_props.get("prominences", np.ones(len(hs_indices)))
        max_hs_prom = np.max(hs_proms) if len(hs_proms) > 0 else 1.0
        to_proms = to_props.get("prominences", np.ones(len(to_indices)))
        max_to_prom = np.max(to_proms) if len(to_proms) > 0 else 1.0

        hs_events = []
        for i, idx in enumerate(hs_indices):
            conf = float(hs_proms[i] / max_hs_prom) if max_hs_prom > 0 else 1.0
            hs_events.append({
                "frame": int(idx),
                "time": round(float(idx / fps), 4),
                "confidence": round(conf, 3),
            })

        to_events = []
        for i, idx in enumerate(to_indices):
            conf = float(to_proms[i] / max_to_prom) if max_to_prom > 0 else 1.0
            to_events.append({
                "frame": int(idx),
                "time": round(float(idx / fps), 4),
                "confidence": round(conf, 3),
            })

        results[f"{side}_hs"] = hs_events
        results[f"{side}_to"] = to_events

    return results


def _detect_crossing(
    frames: list,
    fps: float,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> Dict[str, list]:
    """Crossing method: detect gait events from knee/ankle X crossing.

    When left knee X crosses right knee X, it indicates mid-stance/swing
    transitions. HS occurs when the swinging leg passes the stance leg.
    """
    min_distance = max(1, int(min_cycle_duration * fps / 2))

    left_knee_x = _fill_nan(_extract_landmark_series(frames, "LEFT_KNEE", "x"))
    right_knee_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_KNEE", "x"))
    left_ankle_x = _fill_nan(_extract_landmark_series(frames, "LEFT_ANKLE", "x"))
    right_ankle_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_ANKLE", "x"))

    if np.all(np.isnan(left_knee_x)) or np.all(np.isnan(right_knee_x)):
        return {"left_hs": [], "right_hs": [], "left_to": [], "right_to": []}

    # Filter
    left_knee_x = _lowpass_filter(left_knee_x, cutoff_freq, fps)
    right_knee_x = _lowpass_filter(right_knee_x, cutoff_freq, fps)
    left_ankle_x = _lowpass_filter(left_ankle_x, cutoff_freq, fps)
    right_ankle_x = _lowpass_filter(right_ankle_x, cutoff_freq, fps)

    # Compute crossing signal (difference between left and right knee x)
    knee_diff = left_knee_x - right_knee_x

    # Find zero-crossings
    crossings = []
    for i in range(1, len(knee_diff)):
        if knee_diff[i - 1] * knee_diff[i] < 0:
            crossings.append(i)

    # Classify crossings: when left passes right going forward → left HS
    left_hs, right_hs = [], []
    left_to, right_to = [], []

    for idx in crossings:
        if idx < 1 or idx >= len(knee_diff) - 1:
            continue
        # Rising crossing (left moves forward past right) → left heel strike
        if knee_diff[idx] > knee_diff[idx - 1]:
            left_hs.append({"frame": int(idx), "time": round(idx / fps, 4), "confidence": 0.8})
            # Toe off for opposite side typically ~10% before
            to_offset = max(1, int(0.1 * min_cycle_duration * fps))
            to_frame = max(0, idx - to_offset)
            right_to.append({"frame": int(to_frame), "time": round(to_frame / fps, 4), "confidence": 0.7})
        else:
            right_hs.append({"frame": int(idx), "time": round(idx / fps, 4), "confidence": 0.8})
            to_offset = max(1, int(0.1 * min_cycle_duration * fps))
            to_frame = max(0, idx - to_offset)
            left_to.append({"frame": int(to_frame), "time": round(to_frame / fps, 4), "confidence": 0.7})

    # Filter events too close together
    def _filter_close(events_list, min_dist):
        if not events_list:
            return events_list
        filtered = [events_list[0]]
        for ev in events_list[1:]:
            if ev["frame"] - filtered[-1]["frame"] >= min_dist:
                filtered.append(ev)
        return filtered

    return {
        "left_hs": _filter_close(left_hs, min_distance),
        "right_hs": _filter_close(right_hs, min_distance),
        "left_to": _filter_close(left_to, min_distance),
        "right_to": _filter_close(right_to, min_distance),
    }


def _detect_velocity(
    frames: list,
    fps: float,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> Dict[str, list]:
    """Velocity method: foot vertical velocity zero-crossings.

    HS = foot y-velocity changes from downward to upward (foot hits ground).
    TO = foot y-velocity changes from upward to downward (foot lifts).
    """
    min_distance = max(1, int(min_cycle_duration * fps / 2))

    results = {}
    for side, heel_name, toe_name in [
        ("left", "LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("right", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ]:
        heel_y = _fill_nan(_extract_landmark_series(frames, heel_name, "y"))
        toe_y = _fill_nan(_extract_landmark_series(frames, toe_name, "y"))

        # Fall back to ankle if heel/toe not available
        if np.all(np.isnan(heel_y)):
            heel_y = _fill_nan(_extract_landmark_series(frames, f"{side.upper()}_ANKLE", "y"))
        if np.all(np.isnan(toe_y)):
            toe_y = heel_y.copy()

        if np.all(np.isnan(heel_y)):
            results[f"{side}_hs"] = []
            results[f"{side}_to"] = []
            continue

        heel_y = _lowpass_filter(heel_y, cutoff_freq, fps)
        toe_y = _lowpass_filter(toe_y, cutoff_freq, fps)

        # Compute velocity (y increases downward in image coords)
        heel_vy = np.gradient(heel_y, 1.0 / fps)
        toe_vy = np.gradient(toe_y, 1.0 / fps)

        # HS: heel velocity goes from positive (moving down) to negative (bounce up)
        # In image coords: y increases downward, so positive vy = moving down
        hs_events = []
        for i in range(1, len(heel_vy)):
            if heel_vy[i - 1] > 0 and heel_vy[i] <= 0:
                hs_events.append(i)

        # TO: toe velocity goes from ~0 to negative (lifting up)
        to_events = []
        for i in range(1, len(toe_vy)):
            if toe_vy[i - 1] >= 0 and toe_vy[i] < 0:
                to_events.append(i)

        # Filter close events and find prominent ones
        def _to_event_list(indices, min_dist):
            if not indices:
                return []
            filtered = [indices[0]]
            for idx in indices[1:]:
                if idx - filtered[-1] >= min_dist:
                    filtered.append(idx)
            return [
                {"frame": int(idx), "time": round(idx / fps, 4), "confidence": 0.75}
                for idx in filtered
            ]

        results[f"{side}_hs"] = _to_event_list(hs_events, min_distance)
        results[f"{side}_to"] = _to_event_list(to_events, min_distance)

    return results


def _detect_oconnor(
    frames: list,
    fps: float,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> Dict[str, list]:
    """O'Connor method: heel AP velocity zero-crossings.

    Ref: O'Connor et al., Gait Posture 2007;25(3):469-474.
    HS = heel forward velocity crosses zero from positive to negative.
    TO = heel forward velocity crosses zero from negative to positive.
    """
    min_distance = max(1, int(min_cycle_duration * fps / 2))

    left_hip_x = _fill_nan(_extract_landmark_series(frames, "LEFT_HIP", "x"))
    right_hip_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_HIP", "x"))

    if np.all(np.isnan(left_hip_x)):
        return {"left_hs": [], "right_hs": [], "left_to": [], "right_to": []}

    pelvis_x = (left_hip_x + right_hip_x) / 2

    results = {}
    for side, heel_name in [("left", "LEFT_HEEL"), ("right", "RIGHT_HEEL")]:
        heel_x = _fill_nan(_extract_landmark_series(frames, heel_name, "x"))

        # Fall back to ankle
        if np.all(np.isnan(heel_x)):
            heel_x = _fill_nan(_extract_landmark_series(
                frames, f"{side.upper()}_ANKLE", "x"))

        if np.all(np.isnan(heel_x)):
            results[f"{side}_hs"] = []
            results[f"{side}_to"] = []
            continue

        # Relative heel position to pelvis
        heel_rel = heel_x - pelvis_x
        heel_rel = _lowpass_filter(heel_rel, cutoff_freq, fps)

        # Velocity of relative heel position
        heel_vel = np.gradient(heel_rel, 1.0 / fps)
        heel_vel = _lowpass_filter(heel_vel, cutoff_freq, fps)

        # HS: velocity zero-crossing from positive to negative (foot decelerating)
        hs_indices = []
        to_indices = []
        for i in range(1, len(heel_vel)):
            if heel_vel[i - 1] > 0 and heel_vel[i] <= 0:
                hs_indices.append(i)
            elif heel_vel[i - 1] < 0 and heel_vel[i] >= 0:
                to_indices.append(i)

        def _to_events(indices, min_dist):
            if not indices:
                return []
            filtered = [indices[0]]
            for idx in indices[1:]:
                if idx - filtered[-1] >= min_dist:
                    filtered.append(idx)
            return [
                {"frame": int(idx), "time": round(idx / fps, 4), "confidence": 0.8}
                for idx in filtered
            ]

        results[f"{side}_hs"] = _to_events(hs_indices, min_distance)
        results[f"{side}_to"] = _to_events(to_indices, min_distance)

    return results


def event_consensus(
    data: dict,
    methods: list = None,
    tolerance: int = 3,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> dict:
    """Multi-method consensus event detection.

    Runs multiple event detection methods and finds consensus events
    by clustering detections that fall within *tolerance* frames of
    each other and retaining only those detected by a majority of
    methods.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    methods : list, optional
        List of method names to use (default ``["zeni", "oconnor", "crossing"]``).
    tolerance : int, optional
        Maximum frame distance to consider events as the same (default 3).
    min_cycle_duration : float, optional
        Minimum gait cycle duration in seconds (default 0.4).
    cutoff_freq : float, optional
        Low-pass filter cutoff frequency in Hz (default 6.0).

    Returns
    -------
    dict
        Modified *data* dict with ``events`` populated using consensus events.
        Each event dict includes a ``confidence`` field reflecting the fraction
        of methods that agreed on that event.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if not data.get("frames"):
        raise ValueError("No frames in data. Run extract() first.")

    if methods is None:
        methods = ["zeni", "oconnor", "crossing"]

    fps = data.get("meta", {}).get("fps", 30.0)
    frames = data["frames"]

    # Collect events from each method
    all_results = []
    for method_name in methods:
        if method_name not in EVENT_METHODS:
            logger.warning(f"Skipping unknown method: {method_name}")
            continue
        detect_func = EVENT_METHODS[method_name]
        result = detect_func(frames, fps, min_cycle_duration, cutoff_freq)
        all_results.append(result)

    n_methods = len(all_results)
    if n_methods == 0:
        data["events"] = {
            "method": "consensus",
            "fps": fps,
            "min_cycle_duration": min_cycle_duration,
            "left_hs": [], "right_hs": [], "left_to": [], "right_to": [],
        }
        return data

    majority_threshold = math.ceil(n_methods / 2)

    consensus_events = {}
    for event_type in ["left_hs", "right_hs", "left_to", "right_to"]:
        # Collect all event frames from all methods
        all_frames = []
        for result in all_results:
            for ev in result.get(event_type, []):
                all_frames.append(ev["frame"])

        if not all_frames:
            consensus_events[event_type] = []
            continue

        all_frames.sort()

        # Cluster events within tolerance
        clusters: List[List[int]] = []
        current_cluster = [all_frames[0]]
        for frame_idx in all_frames[1:]:
            if frame_idx - current_cluster[-1] <= tolerance:
                current_cluster.append(frame_idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [frame_idx]
        clusters.append(current_cluster)

        # Keep clusters with majority agreement
        events = []
        for cluster in clusters:
            if len(cluster) >= majority_threshold:
                # Use median frame as the consensus frame
                median_frame = int(np.median(cluster))
                confidence = round(len(cluster) / n_methods, 3)
                events.append({
                    "frame": median_frame,
                    "time": round(float(median_frame / fps), 4),
                    "confidence": min(confidence, 1.0),
                })

        consensus_events[event_type] = events

    data["events"] = {
        "method": "consensus",
        "methods_used": methods[:],
        "n_methods": n_methods,
        "tolerance": tolerance,
        "fps": fps,
        "min_cycle_duration": min_cycle_duration,
        **consensus_events,
    }

    n_total = sum(len(v) for k, v in consensus_events.items())
    logger.info(f"Consensus detection: {n_total} events from {n_methods} methods")

    return data


def validate_events(data: dict) -> dict:
    """Biomechanical plausibility check of detected gait events.

    Checks event ordering, cycle durations, stance phase ratios,
    and left/right alternation to assess whether detected events
    are biomechanically plausible.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events`` populated.

    Returns
    -------
    dict
        Validation report with keys:
        - ``valid`` (bool): True if no critical issues found.
        - ``issues`` (list of str): Description of each issue.
        - ``n_valid_cycles_left`` (int): Number of valid left gait cycles.
        - ``n_valid_cycles_right`` (int): Number of valid right gait cycles.
    """
    issues: List[str] = []
    n_valid_left = 0
    n_valid_right = 0

    events = data.get("events")
    if events is None:
        return {
            "valid": False,
            "issues": ["No events detected"],
            "n_valid_cycles_left": 0,
            "n_valid_cycles_right": 0,
        }

    fps = events.get("fps", data.get("meta", {}).get("fps", 30.0))

    for side in ["left", "right"]:
        hs_list = events.get(f"{side}_hs", [])
        to_list = events.get(f"{side}_to", [])

        if not hs_list:
            issues.append(f"No heel strikes detected on {side} side")
            continue
        if not to_list:
            issues.append(f"No toe offs detected on {side} side")
            continue

        # Check HS/TO alternation: HS should precede TO within each cycle
        hs_frames = sorted([e["frame"] for e in hs_list])
        to_frames = sorted([e["frame"] for e in to_list])

        # Check cycle durations (HS to HS)
        valid_cycles = 0
        for i in range(len(hs_frames) - 1):
            cycle_duration = (hs_frames[i + 1] - hs_frames[i]) / fps

            if cycle_duration < 0.4:
                issues.append(
                    f"{side.capitalize()} cycle at frame {hs_frames[i]}: "
                    f"duration {cycle_duration:.2f}s < 0.4s minimum"
                )
                continue
            if cycle_duration > 2.5:
                issues.append(
                    f"{side.capitalize()} cycle at frame {hs_frames[i]}: "
                    f"duration {cycle_duration:.2f}s > 2.5s maximum"
                )
                continue

            # Check stance phase ratio: find TO between consecutive HS
            hs_start = hs_frames[i]
            hs_end = hs_frames[i + 1]
            to_between = [t for t in to_frames if hs_start < t < hs_end]

            if to_between:
                stance_frames = to_between[0] - hs_start
                cycle_frames = hs_end - hs_start
                stance_ratio = stance_frames / cycle_frames if cycle_frames > 0 else 0

                if stance_ratio < 0.30:
                    issues.append(
                        f"{side.capitalize()} cycle at frame {hs_start}: "
                        f"stance ratio {stance_ratio:.1%} < 30%"
                    )
                    continue
                if stance_ratio > 0.80:
                    issues.append(
                        f"{side.capitalize()} cycle at frame {hs_start}: "
                        f"stance ratio {stance_ratio:.1%} > 80%"
                    )
                    continue

            valid_cycles += 1

        if side == "left":
            n_valid_left = valid_cycles
        else:
            n_valid_right = valid_cycles

    # Check left/right alternation
    left_hs_frames = sorted([e["frame"] for e in events.get("left_hs", [])])
    right_hs_frames = sorted([e["frame"] for e in events.get("right_hs", [])])

    if left_hs_frames and right_hs_frames:
        # Merge and check alternation
        all_hs = sorted(
            [("L", f) for f in left_hs_frames] + [("R", f) for f in right_hs_frames],
            key=lambda x: x[1],
        )
        consecutive_same = 0
        for i in range(1, len(all_hs)):
            if all_hs[i][0] == all_hs[i - 1][0]:
                consecutive_same += 1
        if consecutive_same > len(all_hs) * 0.5:
            issues.append(
                "Left and right heel strikes do not alternate well "
                f"({consecutive_same} consecutive same-side events)"
            )

    is_valid = len(issues) == 0

    return {
        "valid": is_valid,
        "issues": issues,
        "n_valid_cycles_left": n_valid_left,
        "n_valid_cycles_right": n_valid_right,
    }


# ── Method registry ──────────────────────────────────────────────────


EVENT_METHODS: Dict[str, Callable] = {
    "zeni": _detect_zeni,
    "crossing": _detect_crossing,
    "velocity": _detect_velocity,
    "oconnor": _detect_oconnor,
}


def register_event_method(name: str, func: Callable):
    """Register a custom event detection method.

    The function must accept (frames, fps, min_cycle_duration, cutoff_freq)
    and return a dict with keys: left_hs, right_hs, left_to, right_to.
    """
    EVENT_METHODS[name] = func


def list_event_methods() -> list:
    """Return available event detection method names."""
    return list(EVENT_METHODS.keys())


def _adaptive_params(frames: list, fps: float) -> tuple:
    """Estimate walking speed and return adapted (min_cycle_duration, cutoff_freq).

    Speed is estimated from the rate of hip x-displacement over time.
    The displacement rate (in normalized coordinates per second) is
    converted to an approximate speed category.

    Parameters
    ----------
    frames : list
        Frame list with landmark data.
    fps : float
        Video frame rate.

    Returns
    -------
    tuple
        (min_cycle_duration, cutoff_freq) adapted for the estimated speed.
    """
    left_hip_x = _fill_nan(_extract_landmark_series(frames, "LEFT_HIP", "x"))
    right_hip_x = _fill_nan(_extract_landmark_series(frames, "RIGHT_HIP", "x"))

    if np.all(np.isnan(left_hip_x)):
        return 0.4, 6.0  # defaults

    pelvis_x = (left_hip_x + right_hip_x) / 2

    # Estimate displacement rate in normalized coordinates per second
    n_frames = len(pelvis_x)
    if n_frames < 2:
        return 0.4, 6.0

    duration_s = n_frames / fps
    if duration_s <= 0:
        return 0.4, 6.0

    # Use standard deviation of frame-to-frame displacement as speed proxy
    # For treadmill: low displacement rate; for overground: higher
    frame_displacements = np.abs(np.diff(pelvis_x))
    displacement_rate = float(np.nanmean(frame_displacements)) * fps

    # Convert normalized displacement rate to approximate m/s
    # Typical camera FOV captures ~3-5m, so norm_rate * ~4 ≈ m/s
    estimated_speed = displacement_rate * 4.0

    if estimated_speed < 0.5:
        # Slow walk (possibly treadmill or very slow)
        return 0.6, 4.0
    elif estimated_speed > 1.5:
        # Fast walk
        return 0.3, 8.0
    else:
        # Normal walking
        return 0.4, 6.0


# ── Public API ───────────────────────────────────────────────────────


def detect_events(
    data: dict,
    method: str = "zeni",
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
    adaptive: bool = False,
) -> dict:
    """Detect gait events (heel strike and toe off) from pose data.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    method : str, optional
        Detection method name (default ``"zeni"``). Use
        ``list_event_methods()`` to see available methods.
    min_cycle_duration : float, optional
        Minimum gait cycle duration in seconds (default 0.4).
    cutoff_freq : float, optional
        Low-pass filter cutoff frequency in Hz (default 6.0).
    adaptive : bool, optional
        When True, estimate walking speed from hip displacement and
        automatically adjust ``min_cycle_duration`` and ``cutoff_freq``
        (default False). The original parameter values are overridden
        based on estimated speed:
        - Slow (< 0.5 m/s equivalent): min_cycle=0.6, cutoff=4.0
        - Normal (0.5-1.5 m/s equivalent): min_cycle=0.4, cutoff=6.0
        - Fast (> 1.5 m/s equivalent): min_cycle=0.3, cutoff=8.0

    Returns
    -------
    dict
        Modified *data* dict with ``events`` populated.

    Raises
    ------
    ValueError
        If *data* has no frames or *method* is unknown.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if not data.get("frames"):
        raise ValueError("No frames in data. Run extract() first.")

    fps = data.get("meta", {}).get("fps", 30.0)
    frames = data["frames"]

    # Adaptive parameter tuning based on estimated walking speed
    if adaptive:
        min_cycle_duration, cutoff_freq = _adaptive_params(frames, fps)
        logger.info(
            f"Adaptive mode: min_cycle={min_cycle_duration:.2f}s, "
            f"cutoff={cutoff_freq:.1f}Hz"
        )

    logger.info(f"Detecting gait events with method={method}, fps={fps:.1f}")

    if method not in EVENT_METHODS:
        available = ", ".join(EVENT_METHODS.keys())
        raise ValueError(f"Unknown method: {method}. Available: {available}")

    detect_func = EVENT_METHODS[method]
    events = detect_func(frames, fps, min_cycle_duration, cutoff_freq)

    n_events = sum(len(v) for v in events.values())
    logger.info(
        f"Detected {n_events} events: "
        f"HS_L={len(events['left_hs'])}, HS_R={len(events['right_hs'])}, "
        f"TO_L={len(events['left_to'])}, TO_R={len(events['right_to'])}"
    )

    data["events"] = {
        "method": method,
        "fps": fps,
        "min_cycle_duration": min_cycle_duration,
        **events,
    }

    return data
