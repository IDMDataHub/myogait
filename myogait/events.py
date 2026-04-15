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

    - "gk_bike", "gk_zeni", "gk_oconnor", etc.: gaitkit backends.
      Requires the optional ``gaitkit`` package (``pip install gaitkit``).
      10 individual methods plus "gk_ensemble" (multi-method voting).

All methods are registered in EVENT_METHODS and can be extended
via register_event_method().
"""

import logging
import math
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

from .axis_utils import detect_walking_direction_from_feet

logger = logging.getLogger(__name__)

# Module-level reference to the current data dict, set by detect_events()
# and event_consensus() before invoking detection methods.  This allows
# gaitkit wrapper functions (which have a fixed signature taking only
# ``frames``) to access the full data dict including angle data.
_current_data: Optional[dict] = None
_current_femur_length_mm: float = 400.0


def _remap_event_frames(events: dict, frames: list, fps: float) -> None:
    """Remap array-index based event frames to actual frame_idx values.

    Detection functions use ``find_peaks`` on arrays indexed 0..N-1 (position
    in the ``frames`` list).  When the subject is not visible at the start of
    the video the first extracted frame may have ``frame_idx >> 0``, causing a
    mismatch with angle frames (which use the original video frame_idx).

    This function converts in-place each event's ``frame`` from an array index
    to the corresponding ``frame_idx`` and recalculates ``time`` accordingly.
    """
    # Build lookup: array position → original frame_idx
    idx_map = [f.get("frame_idx", i) for i, f in enumerate(frames)]
    n = len(idx_map)

    for key in ("left_hs", "right_hs", "left_to", "right_to"):
        for ev in events.get(key, []):
            arr_idx = ev["frame"]
            if 0 <= arr_idx < n:
                real_idx = idx_map[arr_idx]
                ev["frame"] = real_idx
                ev["time"] = round(ev["time"] + (real_idx - arr_idx) / fps, 4)


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

    # Detect walking direction from foot orientation (toe vs heel).
    # This is more robust than displacement-based detection and
    # works on treadmills.
    if _current_data is not None:
        walking_right = detect_walking_direction_from_feet(_current_data) == "right"
    else:
        # Fallback: pelvis displacement
        valid_pelvis = pelvis_x[~np.isnan(pelvis_x)]
        walking_right = (valid_pelvis[-1] > valid_pelvis[0]
                         if len(valid_pelvis) >= 2 else True)

    # Relative ankle position
    left_ankle_rel = left_ankle_x - pelvis_x
    right_ankle_rel = right_ankle_x - pelvis_x

    # Flip signal for right-to-left walking so peaks always = anterior
    if not walking_right:
        left_ankle_rel = -left_ankle_rel
        right_ankle_rel = -right_ankle_rel

    # Low-pass filter
    left_ankle_rel = _lowpass_filter(left_ankle_rel, cutoff_freq, fps)
    right_ankle_rel = _lowpass_filter(right_ankle_rel, cutoff_freq, fps)

    # Detect peaks and troughs
    results = {}
    for side, rel_signal in [("left", left_ankle_rel), ("right", right_ankle_rel)]:
        # Auto-scale prominence to 10% of signal amplitude (more robust
        # than a fixed threshold on videos with varying resolution/scale)
        valid_sig = rel_signal[~np.isnan(rel_signal)]
        if len(valid_sig) > 2:
            prom = max(0.005, float(np.ptp(valid_sig) * 0.10))
        else:
            prom = 0.005
        # HS = peaks (foot most anterior)
        hs_indices, hs_props = find_peaks(
            rel_signal, distance=min_distance, prominence=prom
        )
        # TO = troughs (foot most posterior)
        to_indices, to_props = find_peaks(
            -rel_signal, distance=min_distance, prominence=prom
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

    # Detect walking direction from foot orientation
    if _current_data is not None:
        walking_right = detect_walking_direction_from_feet(_current_data) == "right"
    else:
        valid_la = left_ankle_x[~np.isnan(left_ankle_x)]
        walking_right = len(valid_la) >= 2 and valid_la[-1] > valid_la[0]

    # Compute crossing signal (difference between left and right knee x)
    # Flip for right-to-left walking so rising crossing always = left forward
    knee_diff = left_knee_x - right_knee_x
    if not walking_right:
        knee_diff = -knee_diff

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
        # Use same threshold convention as HS for symmetry
        to_events = []
        for i in range(1, len(toe_vy)):
            if toe_vy[i - 1] > 0 and toe_vy[i] <= 0:
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

    # Detect walking direction from foot orientation
    if _current_data is not None:
        walking_right = detect_walking_direction_from_feet(_current_data) == "right"
    else:
        valid_pelvis = pelvis_x[~np.isnan(pelvis_x)]
        walking_right = len(valid_pelvis) >= 2 and valid_pelvis[-1] > valid_pelvis[0]

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
        # Flip for right-to-left so positive velocity = forward
        if not walking_right:
            heel_rel = -heel_rel
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
    femur_length_mm: float = 400.0,
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
    femur_length_mm : float, optional
        Reference femur length in mm for gaitkit methods (default 400).

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
    global _current_data, _current_femur_length_mm
    _current_data = data
    _current_femur_length_mm = femur_length_mm
    all_results = []
    try:
        for method_name in methods:
            # Lazy-register gaitkit methods if needed
            if method_name.startswith("gk_") and method_name not in EVENT_METHODS:
                if _is_gaitkit_available():
                    gk_name = method_name[3:]
                    if gk_name == "ensemble":
                        EVENT_METHODS[method_name] = _make_gaitkit_ensemble_wrapper()
                    elif gk_name in _GAITKIT_METHODS:
                        EVENT_METHODS[method_name] = _make_gaitkit_wrapper(gk_name)

            if method_name not in EVENT_METHODS:
                logger.warning(f"Skipping unknown method: {method_name}")
                continue
            detect_func = EVENT_METHODS[method_name]
            result = detect_func(frames, fps, min_cycle_duration, cutoff_freq)
            all_results.append(result)
    finally:
        _current_data = None

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
        # Collect all event frames from all methods, tagged with method index
        all_events = []  # list of (frame, method_index)
        for method_idx, result in enumerate(all_results):
            for ev in result.get(event_type, []):
                all_events.append((ev["frame"], method_idx))

        if not all_events:
            consensus_events[event_type] = []
            continue

        all_events.sort(key=lambda x: x[0])

        # Cluster events within tolerance
        clusters: List[List[tuple]] = []
        current_cluster = [all_events[0]]
        for item in all_events[1:]:
            if item[0] - current_cluster[-1][0] <= tolerance:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        clusters.append(current_cluster)

        # Keep clusters with majority agreement (count unique methods)
        events = []
        for cluster in clusters:
            unique_methods = len(set(m_idx for _, m_idx in cluster))
            if unique_methods >= majority_threshold:
                # Use median frame as the consensus frame
                cluster_frames = [f for f, _ in cluster]
                median_frame = int(np.median(cluster_frames))
                confidence = round(unique_methods / n_methods, 3)
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


# ── gaitkit integration (optional dependency) ───────────────────────

# All 10 gaitkit methods that can be registered with the "gk_" prefix.
_GAITKIT_METHODS = [
    "bike", "zeni", "oconnor", "hreljac", "mickelborough",
    "ghoussayni", "vancanneyt", "dgei", "intellevent", "deepevent",
]


def _is_gaitkit_available() -> bool:
    """Check whether the gaitkit package is importable."""
    try:
        import importlib
        importlib.import_module("gaitkit")
        return True
    except ImportError:
        return False


def _import_gaitkit():
    """Import and return the gaitkit module, raising ImportError if absent."""
    try:
        import gaitkit
        return gaitkit
    except ImportError:
        raise ImportError(
            "gaitkit is not installed. Install it with: pip install gaitkit"
        )


def _convert_to_gaitkit_frames(data: dict, femur_length_mm: float = 400.0) -> list:
    """Convert myogait data dict to gaitkit angle frame format.

    Maps myogait angle field names to gaitkit field names:
        - frame_idx -> frame_index
        - hip_L -> left_hip_angle, hip_R -> right_hip_angle
        - knee_L -> left_knee_angle, knee_R -> right_knee_angle
        - ankle_L -> left_ankle_angle, ankle_R -> right_ankle_angle

    Landmark positions are converted from myogait normalised [0, 1]
    coordinates to millimetres using *femur_length_mm* as reference.
    The median hip→knee distance (in normalised coords) is mapped to
    *femur_length_mm* and the same scale factor is applied to every
    landmark.  This ensures that gaitkit velocity features (mm/s) are
    in realistic units rather than normalised coordinates.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` (preferred) or ``frames`` populated.
    femur_length_mm : float, optional
        Reference femur length in millimetres (default 400).  Used to
        scale normalised landmark positions to real-world units.

    Returns
    -------
    list
        List of gaitkit-compatible angle frame dicts.
    """
    # Field mapping from myogait angle keys to gaitkit keys
    _angle_map = {
        "hip_L": "left_hip_angle",
        "hip_R": "right_hip_angle",
        "knee_L": "left_knee_angle",
        "knee_R": "right_knee_angle",
        "ankle_L": "left_ankle_angle",
        "ankle_R": "right_ankle_angle",
    }

    # Landmark name mapping: myogait UPPER_NAME -> gaitkit lower_name
    _landmark_name_map = {
        "LEFT_HIP": "left_hip",
        "RIGHT_HIP": "right_hip",
        "LEFT_KNEE": "left_knee",
        "RIGHT_KNEE": "right_knee",
        "LEFT_ANKLE": "left_ankle",
        "RIGHT_ANKLE": "right_ankle",
        "LEFT_HEEL": "left_heel",
        "RIGHT_HEEL": "right_heel",
        "LEFT_FOOT_INDEX": "left_toe",
        "RIGHT_FOOT_INDEX": "right_toe",
    }

    # Determine source: prefer angles if available, fall back to frames
    angle_frames = None
    if data.get("angles") and data["angles"].get("frames"):
        angle_frames = data["angles"]["frames"]

    raw_frames = data.get("frames", [])

    gaitkit_frames = []
    n_frames = len(angle_frames) if angle_frames else len(raw_frames)

    for i in range(n_frames):
        gk_frame = {}

        # -- frame_index --
        if angle_frames and i < len(angle_frames):
            af = angle_frames[i]
            gk_frame["frame_index"] = af.get("frame_idx", i)

            # Map angle fields
            for mg_key, gk_key in _angle_map.items():
                val = af.get(mg_key)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    gk_frame[gk_key] = float(val)
                else:
                    gk_frame[gk_key] = 0.0

            # Landmark positions from angle frames
            lp = af.get("landmark_positions", {})
            gk_landmarks = {}
            for gk_name, coords in lp.items():
                # coords is [x, y, visibility] in myogait.
                # gaitkit uses vert_axis=2 (z) for vertical features.
                # For 2D video, y is the vertical axis (0=top, 1=bottom),
                # so map to (x, 0.0, 1.0 - y) to give gaitkit a correct
                # upward-positive vertical signal.
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    gk_landmarks[gk_name] = (
                        float(coords[0]), 0.0, 1.0 - float(coords[1])
                    )
            gk_frame["landmark_positions"] = gk_landmarks
        else:
            gk_frame["frame_index"] = i

            # No angle data available: fill with zeros
            for gk_key in _angle_map.values():
                gk_frame[gk_key] = 0.0

            gk_frame["landmark_positions"] = {}

        # Enrich landmark positions from raw frames if available
        if i < len(raw_frames):
            rf = raw_frames[i]
            landmarks = rf.get("landmarks", {})
            existing_lp = gk_frame.get("landmark_positions", {})

            for mp_name, gk_name in _landmark_name_map.items():
                if gk_name not in existing_lp:
                    lm = landmarks.get(mp_name)
                    if lm and lm.get("x") is not None:
                        x_val = lm["x"]
                        y_val = lm.get("y", 0.0)
                        if not (isinstance(x_val, float) and np.isnan(x_val)):
                            existing_lp[gk_name] = (
                                float(x_val), 0.0, 1.0 - float(y_val)
                            )

            gk_frame["landmark_positions"] = existing_lp

        gaitkit_frames.append(gk_frame)

    # ── Scale normalised positions to millimetres ────────────────────
    # Compute median femur length (hip→knee) in normalised coords,
    # then scale all landmark positions so that distance equals
    # femur_length_mm.
    femur_dists = []
    for gk_frame in gaitkit_frames:
        lp = gk_frame.get("landmark_positions", {})
        for hip_k, knee_k in [("left_hip", "left_knee"),
                               ("right_hip", "right_knee")]:
            hip = lp.get(hip_k)
            knee = lp.get(knee_k)
            if hip is not None and knee is not None:
                dx = hip[0] - knee[0]
                dz = hip[2] - knee[2]
                d = np.sqrt(dx * dx + dz * dz)
                if d > 1e-6:
                    femur_dists.append(d)

    if femur_dists:
        femur_norm = float(np.median(femur_dists))
        if femur_norm > 1e-6:
            scale = femur_length_mm / femur_norm
            for gk_frame in gaitkit_frames:
                lp = gk_frame.get("landmark_positions", {})
                scaled_lp = {}
                for name, coords in lp.items():
                    scaled_lp[name] = (
                        coords[0] * scale,
                        coords[1] * scale,
                        coords[2] * scale,
                    )
                gk_frame["landmark_positions"] = scaled_lp

    return gaitkit_frames


def _gaitkit_result_to_myogait(gk_result, fps: float) -> Dict[str, list]:
    """Convert a gaitkit GaitResult to myogait event format.

    Parameters
    ----------
    gk_result : gaitkit.GaitResult
        Result from gaitkit.detect() with .left_hs, .right_hs,
        .left_to, .right_to attributes.
    fps : float
        Frame rate (used only as fallback if time is missing).

    Returns
    -------
    dict
        Dict with keys: left_hs, right_hs, left_to, right_to.
        Each value is a list of event dicts with frame, time, confidence.
    """
    def _event_value(ev, *keys, default=None):
        """Get an event field from dict-like or attribute-like objects."""
        if isinstance(ev, dict):
            for key in keys:
                if key in ev and ev[key] is not None:
                    return ev[key]
            return default

        for key in keys:
            val = getattr(ev, key, None)
            if val is not None:
                return val
        return default

    result = {}
    for event_type in ["left_hs", "right_hs", "left_to", "right_to"]:
        gk_events = getattr(gk_result, event_type, [])
        myogait_events = []
        for ev in gk_events:
            frame_val = _event_value(ev, "frame", "frame_idx", "index", "idx", default=0)
            time_val = _event_value(ev, "time", "time_s", "timestamp", default=None)
            confidence_val = _event_value(
                ev,
                "confidence",
                "conf",
                "score",
                "agreement",
                "vote_ratio",
                default=1.0,
            )
            myogait_events.append({
                "frame": int(frame_val),
                "time": round(float(time_val if time_val is not None else int(frame_val) / fps), 4),
                "confidence": round(float(confidence_val), 3),
            })
        result[event_type] = myogait_events
    return result


def _detect_gaitkit(
    data: dict,
    fps: float,
    method: str = "bike",
    femur_length_mm: float = 400.0,
    **kwargs,
) -> Dict[str, list]:
    """Detect gait events using a gaitkit method.

    Parameters
    ----------
    data : dict
        Full myogait data dict.
    fps : float
        Frame rate.
    method : str
        gaitkit method name (without "gk_" prefix).
    femur_length_mm : float, optional
        Reference femur length in mm for position scaling (default 400).
    **kwargs
        Extra keyword arguments passed to gaitkit.detect().

    Returns
    -------
    dict
        Events in myogait format: {left_hs, right_hs, left_to, right_to}.
    """
    gaitkit = _import_gaitkit()
    gk_frames = _convert_to_gaitkit_frames(data, femur_length_mm=femur_length_mm)
    gk_result = gaitkit.detect(gk_frames, method=method, fps=fps, **kwargs)
    return _gaitkit_result_to_myogait(gk_result, fps)


def _detect_gaitkit_ensemble(
    data: dict,
    fps: float,
    methods: list = None,
    min_votes: int = 2,
    tolerance_ms: float = 50.0,
    weights: str = "benchmark",
    femur_length_mm: float = 400.0,
    **kwargs,
) -> Dict[str, list]:
    """Detect gait events using gaitkit multi-method ensemble voting.

    Parameters
    ----------
    data : dict
        Full myogait data dict.
    fps : float
        Frame rate.
    methods : list, optional
        gaitkit method names to use (default: ["bike", "zeni", "oconnor"]).
    min_votes : int, optional
        Minimum number of methods that must agree (default 2).
    tolerance_ms : float, optional
        Temporal tolerance in ms for clustering (default 50.0).
    weights : str, optional
        Weight scheme for voting (default "benchmark").
    femur_length_mm : float, optional
        Reference femur length in mm for position scaling (default 400).
    **kwargs
        Extra keyword arguments passed to gaitkit.detect_ensemble().

    Returns
    -------
    dict
        Events in myogait format: {left_hs, right_hs, left_to, right_to}.
    """
    gaitkit = _import_gaitkit()
    gk_frames = _convert_to_gaitkit_frames(data, femur_length_mm=femur_length_mm)

    if methods is None:
        methods = ["bike", "zeni", "oconnor"]

    gk_result = gaitkit.detect_ensemble(
        gk_frames,
        methods=methods,
        fps=fps,
        min_votes=min_votes,
        tolerance_ms=tolerance_ms,
        weights=weights,
        **kwargs,
    )
    return _gaitkit_result_to_myogait(gk_result, fps)


def _detect_gaitkit_structured(
    data: dict,
    fps: float,
    method: str = "bike",
) -> Dict[str, list]:
    """Detect gait events using gaitkit.detect_events_structured().

    This uses gaitkit's structured API which can accept myogait data
    directly. The returned dict is converted to myogait event format.

    Parameters
    ----------
    data : dict
        Full myogait data dict.
    fps : float
        Frame rate.
    method : str
        gaitkit method name (without "gk_" prefix).

    Returns
    -------
    dict
        Events in myogait format: {left_hs, right_hs, left_to, right_to}.
    """
    gaitkit = _import_gaitkit()
    result_dict = gaitkit.detect_events_structured(method, data, fps=fps)
    if result_dict is None:
        raise RuntimeError(
            f"gaitkit structured detector '{method}' returned None"
        )
    if not isinstance(result_dict, dict):
        raise TypeError(
            f"gaitkit structured detector '{method}' returned "
            f"{type(result_dict).__name__}, expected dict"
        )

    # Convert structured result to myogait event format
    events = {"left_hs": [], "right_hs": [], "left_to": [], "right_to": []}

    for hs in result_dict.get("heel_strikes", []):
        side = hs.get("side", "left").lower()
        key = f"{side}_hs"
        if key in events:
            events[key].append({
                "frame": int(hs.get("frame", 0)),
                "time": round(float(hs.get("time", hs.get("frame", 0) / fps)), 4),
                "confidence": round(float(hs.get("confidence", 1.0)), 3),
            })

    for to in result_dict.get("toe_offs", []):
        side = to.get("side", "left").lower()
        key = f"{side}_to"
        if key in events:
            events[key].append({
                "frame": int(to.get("frame", 0)),
                "time": round(float(to.get("time", to.get("frame", 0) / fps)), 4),
                "confidence": round(float(to.get("confidence", 1.0)), 3),
            })

    return events


def _make_gaitkit_wrapper(gk_method_name: str) -> Callable:
    """Create a wrapper function for a gaitkit method.

    The returned function has the standard event method signature:
    ``(frames, fps, min_cycle_duration, cutoff_freq) -> dict``

    However, since gaitkit methods need the full data dict (for angle
    and landmark information), this wrapper stores a reference to the
    data dict in a closure via ``detect_events()``. The frames argument
    is used to reconstruct a minimal data dict.

    Parameters
    ----------
    gk_method_name : str
        The gaitkit method name (e.g. "bike", "zeni").

    Returns
    -------
    Callable
        A function with the standard event detection signature.
    """
    def wrapper(frames, fps, min_cycle_duration=0.4, cutoff_freq=6.0):
        # Reconstruct minimal data dict from frames, including angles
        # from the original data dict when available.
        data_proxy = {"frames": frames, "meta": {"fps": fps}}
        if _current_data is not None and _current_data.get("angles"):
            data_proxy["angles"] = _current_data["angles"]
        return _detect_gaitkit(data_proxy, fps, method=gk_method_name,
                               femur_length_mm=_current_femur_length_mm)
    wrapper.__doc__ = f"gaitkit '{gk_method_name}' event detection method."
    wrapper.__name__ = f"_detect_gk_{gk_method_name}"
    return wrapper


def _make_gaitkit_ensemble_wrapper() -> Callable:
    """Create a wrapper for gaitkit ensemble method."""
    def wrapper(frames, fps, min_cycle_duration=0.4, cutoff_freq=6.0):
        data_proxy = {"frames": frames, "meta": {"fps": fps}}
        if _current_data is not None and _current_data.get("angles"):
            data_proxy["angles"] = _current_data["angles"]
        return _detect_gaitkit_ensemble(
            data_proxy, fps, femur_length_mm=_current_femur_length_mm)
    wrapper.__doc__ = "gaitkit ensemble (multi-method voting) event detection."
    wrapper.__name__ = "_detect_gk_ensemble"
    return wrapper


# ── Method registry ──────────────────────────────────────────────────


EVENT_METHODS: Dict[str, Callable] = {
    "zeni": _detect_zeni,
    "crossing": _detect_crossing,
    "velocity": _detect_velocity,
    "oconnor": _detect_oconnor,
}

# Register gaitkit methods if available
if _is_gaitkit_available():
    for _gk_method in _GAITKIT_METHODS:
        EVENT_METHODS[f"gk_{_gk_method}"] = _make_gaitkit_wrapper(_gk_method)
    EVENT_METHODS["gk_ensemble"] = _make_gaitkit_ensemble_wrapper()


def register_event_method(name: str, func: Callable):
    """Register a custom event detection method.

    The function must accept (frames, fps, min_cycle_duration, cutoff_freq)
    and return a dict with keys: left_hs, right_hs, left_to, right_to.
    """
    EVENT_METHODS[name] = func


def list_event_methods() -> list:
    """Return available event detection method names.

    When the optional ``gaitkit`` package is installed, the gaitkit
    methods (prefixed with ``gk_``) are included automatically.
    """
    methods = list(EVENT_METHODS.keys())
    # If gaitkit is available but methods were not yet registered
    # (e.g. gaitkit was installed after module import), register now.
    if _is_gaitkit_available():
        for gk_method in _GAITKIT_METHODS:
            name = f"gk_{gk_method}"
            if name not in EVENT_METHODS:
                EVENT_METHODS[name] = _make_gaitkit_wrapper(gk_method)
        if "gk_ensemble" not in EVENT_METHODS:
            EVENT_METHODS["gk_ensemble"] = _make_gaitkit_ensemble_wrapper()
        methods = list(EVENT_METHODS.keys())
    return methods


def _adaptive_params(frames: list, fps: float) -> tuple:
    """Estimate gait-speed category and return adapted event parameters.

    Uses the rate of hip x-displacement over time in normalized
    coordinates per second as a proxy for progression speed.

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

    # Use frame-to-frame displacement rate as progression-speed proxy.
    frame_displacements = np.abs(np.diff(pelvis_x))
    displacement_rate = float(np.nanmean(frame_displacements)) * fps

    # Classify directly in normalized-coordinates per second.
    # This avoids dataset/camera-dependent conversion assumptions.
    if displacement_rate < 0.02:
        # Slow walk / treadmill-like progression
        return 0.6, 4.0
    elif displacement_rate > 0.08:
        # Fast walk
        return 0.3, 8.0
    else:
        # Normal walking
        return 0.4, 6.0


# ── Public API ───────────────────────────────────────────────────────


def _compute_motion_mask(frames: list, fps: float,
                         window_s: float = 0.5,
                         min_ankle_std_ratio: float = 0.003) -> np.ndarray:
    """Per-frame boolean mask: True where the subject is actually moving.

    A frame is flagged as "motion" when the rolling standard deviation
    of at least one ankle y-coordinate over ``window_s`` exceeds
    ``min_ankle_std_ratio`` times the range of that coordinate over
    the whole clip.  This captures normal gait and excludes standstill
    preludes without requiring fixed absolute thresholds.

    Parameters
    ----------
    frames : list
        Per-frame dicts with ``landmarks``.
    fps : float
        Frame rate, used to size the rolling window.
    window_s : float, optional
        Rolling window length in seconds (default 0.5 s).
    min_ankle_std_ratio : float, optional
        Minimum rolling std, expressed as a fraction of the whole-clip
        y-range of the ankle, required to consider the frame "moving"
        (default 0.3%).

    Returns
    -------
    np.ndarray
        Boolean array, one entry per frame.  ``True`` means "moving",
        ``False`` means "standstill".  When the signal is too short or
        both ankles are missing, returns all-True (fail-safe: never
        over-trim).
    """
    n = len(frames)
    if n == 0:
        return np.zeros(0, dtype=bool)
    w = max(3, int(round(window_s * max(fps, 1e-6))))

    def _ankle_y(side: str) -> np.ndarray:
        out = np.full(n, np.nan)
        for i, f in enumerate(frames):
            lm = f.get("landmarks") or {}
            d = lm.get(f"{side}_ANKLE")
            if isinstance(d, dict) and d.get("y") is not None:
                out[i] = float(d["y"])
        return out

    def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)
        # Simple rolling std with NaN-aware
        out = np.full_like(arr, np.nan)
        half = w // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            seg = arr[lo:hi]
            if np.sum(~np.isnan(seg)) >= 3:
                out[i] = np.nanstd(seg, ddof=0)
        return out

    lay = _ankle_y("LEFT")
    ray = _ankle_y("RIGHT")
    if np.all(np.isnan(lay)) and np.all(np.isnan(ray)):
        return np.ones(n, dtype=bool)

    left_range = float(np.nanmax(lay) - np.nanmin(lay)) if not np.all(np.isnan(lay)) else 0.0
    right_range = float(np.nanmax(ray) - np.nanmin(ray)) if not np.all(np.isnan(ray)) else 0.0
    range_ref = max(left_range, right_range, 1e-6)
    thr = min_ankle_std_ratio * range_ref

    std_L = _rolling_std(lay, w)
    std_R = _rolling_std(ray, w)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        s_l = std_L[i] if not np.isnan(std_L[i]) else 0.0
        s_r = std_R[i] if not np.isnan(std_R[i]) else 0.0
        mask[i] = (s_l > thr) or (s_r > thr)

    # Fail-safe: if the mask would kill more than 95% of frames, something
    # is wrong (very slow gait, small motion) — keep everything
    if mask.sum() / n < 0.05:
        return np.ones(n, dtype=bool)
    return mask


def detect_events(
    data: dict,
    method: str = "zeni",
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
    adaptive: bool = False,
    femur_length_mm: float = 400.0,
    trim_standstill: bool = True,
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
    femur_length_mm : float, optional
        Reference femur length in millimetres (default 400). Used by
        gaitkit methods (``gk_*``) to convert normalised landmark
        positions to real-world units before computing velocity
        features.
    trim_standstill : bool, optional
        When True (default), remove detected events that fall inside
        detected standstill regions.  Fixes a class of false-positive
        heel-strike detections on videos with a stationary prelude
        before the subject starts walking — especially affects
        ``gk_bike`` whose adaptive threshold collapses to noise level
        during standstill.

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

    # Lazy-register gaitkit methods if method looks like gk_* and gaitkit is available
    if method.startswith("gk_") and method not in EVENT_METHODS:
        if _is_gaitkit_available():
            gk_name = method[3:]  # strip "gk_" prefix
            if gk_name == "ensemble":
                EVENT_METHODS[method] = _make_gaitkit_ensemble_wrapper()
            elif gk_name in _GAITKIT_METHODS:
                EVENT_METHODS[method] = _make_gaitkit_wrapper(gk_name)

    if method not in EVENT_METHODS:
        available = ", ".join(EVENT_METHODS.keys())
        raise ValueError(f"Unknown method: {method}. Available: {available}")

    detect_func = EVENT_METHODS[method]
    global _current_data, _current_femur_length_mm
    _current_data = data
    _current_femur_length_mm = femur_length_mm
    try:
        events = detect_func(frames, fps, min_cycle_duration, cutoff_freq)
    finally:
        _current_data = None

    # Filter out events that fall inside a standstill region.  This fixes
    # adaptive detectors (notably gk_bike) that fire spurious heel strikes
    # during stationary preludes because their threshold collapses to the
    # level of landmark micro-noise.
    if trim_standstill:
        mask = _compute_motion_mask(frames, fps)
        trimmed = 0
        for key in ("left_hs", "right_hs", "left_to", "right_to"):
            lst = events.get(key, [])
            if not lst: continue
            kept = []
            for ev_idx in lst:
                idx = int(ev_idx) if not isinstance(ev_idx, dict) else int(
                    ev_idx.get("frame_idx", ev_idx.get("index", 0)))
                if 0 <= idx < len(mask) and mask[idx]:
                    kept.append(ev_idx)
                else:
                    trimmed += 1
            events[key] = kept
        if trimmed:
            logger.info(
                "trim_standstill removed %d events inside standstill regions",
                trimmed,
            )

    # Remap array indices to actual frame_idx values.
    # Detection functions return indices into the frames list (0..N-1),
    # but frames may not start at 0 if the subject is absent at the
    # beginning of the video.  segment_cycles matches events to angle
    # frames by frame_idx, so they must agree.
    _remap_event_frames(events, frames, fps)

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
