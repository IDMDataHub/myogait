"""Gait cycle segmentation, normalization, and averaging.

Segments continuous angle data into individual gait cycles
(heel-strike to heel-strike), normalizes each cycle to 0--100%%
(101 points by default), and computes mean +/- SD curves per side.

The gait cycle is defined as the period from one heel strike (initial
contact) to the subsequent ipsilateral heel strike, following the
convention of Perry & Burnfield:

    Ref: Perry J, Burnfield JM. Gait Analysis: Normal and
    Pathological Function. 2nd ed. SLACK Incorporated; 2010.

Time normalization to 0-100%% of the gait cycle enables inter-subject
and inter-trial comparison:

    Ref: Duhamel A, Bourriez JL, Devos P, et al. Statistical tools
    for clinical gait analysis. Gait Posture. 2004;20(2):204-212.
    doi:10.1016/j.gaitpost.2003.09.010

Stance/swing phase subdivision:
    - Stance phase: 0-60%% (initial contact to toe-off)
    - Swing phase: 60-100%% (toe-off to next initial contact)
    Ref: Levine D, Richards J, Whittle MW. Whittle's Gait Analysis.
    5th ed. Churchill Livingstone (Elsevier); 2012. Chapter 2.

Functions
---------
segment_cycles
    Segment gait data into normalized cycles and compute averages.
ensemble_average
    Compute ensemble averages across multiple trials.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Joints extracted from angle frames
_JOINT_KEYS = ["hip", "knee", "ankle", "trunk"]
# Mapping from joint short name to angle frame keys (L/R)
_SIDE_KEYS = {
    "hip": ("hip_L", "hip_R"),
    "knee": ("knee_L", "knee_R"),
    "ankle": ("ankle_L", "ankle_R"),
    "trunk": ("trunk_angle", "trunk_angle"),  # same for both sides
}

# Frontal-plane angle keys that may be present in angle frames.
# Each entry maps a summary key to the (left_key, right_key) in the raw
# angle frame dict.  Keys where L/R share the same source use the same
# string for both sides (e.g. pelvis_list is not side-specific).
_FRONTAL_KEYS = {
    "pelvis_list": ("pelvis_list", "pelvis_list"),
    "hip_adduction": ("hip_adduction_L", "hip_adduction_R"),
    "knee_valgus": ("knee_valgus_L", "knee_valgus_R"),
}


def _find_to_between(to_events: list, start_frame: int, end_frame: int) -> Optional[int]:
    """Find the first toe-off event between two heel strikes."""
    for ev in to_events:
        f = ev["frame"]
        if start_frame < f < end_frame:
            return f
    return None


def _extract_cycle_angles(
    angle_frames: list,
    start_frame: int,
    end_frame: int,
    side: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract angle values for one cycle."""
    # Get angle frames within the cycle
    cycle_afs = [af for af in angle_frames if start_frame <= af["frame_idx"] <= end_frame]

    if len(cycle_afs) < 10:
        return None

    result = {}

    for joint in _JOINT_KEYS:
        key_l, key_r = _SIDE_KEYS[joint]
        key = key_l if side == "left" else key_r

        values = []
        for af in cycle_afs:
            v = af.get(key)
            if v is None:
                values.append(np.nan)
            else:
                values.append(float(v))

        arr = np.array(values)
        nans = np.isnan(arr)
        if nans.all():
            # Skip this joint but don't reject the whole cycle
            continue
        if nans.any():
            x = np.arange(len(arr))
            arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        result[joint] = arr

    # Extract frontal-plane angles when present
    for frontal_joint, (key_l, key_r) in _FRONTAL_KEYS.items():
        key = key_l if side == "left" else key_r

        values = []
        for af in cycle_afs:
            v = af.get(key)
            if v is None:
                values.append(np.nan)
            else:
                values.append(float(v))

        arr = np.array(values)
        nans = np.isnan(arr)
        if nans.all():
            continue
        if nans.any():
            x = np.arange(len(arr))
            arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        result[frontal_joint] = arr

    if not result:
        return None

    return result


def _normalize_to_percent(values: np.ndarray, n_points: int = 101) -> np.ndarray:
    """Resample a variable-length array to n_points (0-100%)."""
    original = np.linspace(0, 100, len(values))
    target = np.linspace(0, 100, n_points)
    return np.interp(target, original, values)


def segment_cycles(
    data: dict,
    n_points: int = 101,
    min_duration: float = 0.4,
    max_duration: float = 2.5,
) -> dict:
    """Segment gait data into cycles and compute normalized averages.

    Each cycle runs from one heel strike (HS) to the next HS on the
    same side. Cycles are time-normalized to 0--100%% and mean +/- SD
    curves are computed per side.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events`` and ``angles`` populated.
    n_points : int, optional
        Number of points for time normalization (default 101).
    min_duration : float, optional
        Minimum valid cycle duration in seconds (default 0.4).
    max_duration : float, optional
        Maximum valid cycle duration in seconds (default 2.5).

    Returns
    -------
    dict
        Keys: ``cycles`` (list of cycle dicts) and ``summary``
        (per-side mean/std curves).

    Raises
    ------
    ValueError
        If *data* has no events or angles.
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    events = data.get("events")
    angles = data.get("angles")
    if events is None:
        raise ValueError("No events in data. Run detect_events() first.")
    if angles is None:
        raise ValueError("No angles in data. Run compute_angles() first.")

    fps = data.get("meta", {}).get("fps", 30.0)
    angle_frames = angles.get("frames", [])

    cycles = []
    cycle_id = 0

    for side in ("left", "right"):
        hs_list = events.get(f"{side}_hs", [])
        to_list = events.get(f"{side}_to", [])

        if len(hs_list) < 2:
            logger.info(f"Not enough HS events for {side} side ({len(hs_list)})")
            continue

        # Sort by frame
        hs_sorted = sorted(hs_list, key=lambda e: e["frame"])
        to_sorted = sorted(to_list, key=lambda e: e["frame"])

        for i in range(len(hs_sorted) - 1):
            start_frame = hs_sorted[i]["frame"]
            end_frame = hs_sorted[i + 1]["frame"]
            duration = (end_frame - start_frame) / fps

            # Validate duration
            if duration < min_duration or duration > max_duration:
                logger.debug(f"Cycle {side} {start_frame}-{end_frame} rejected: {duration:.2f}s")
                continue

            # Find toe-off within cycle
            to_frame = _find_to_between(to_sorted, start_frame, end_frame)
            if to_frame is not None:
                stance_pct = round((to_frame - start_frame) / (end_frame - start_frame) * 100, 1)
                swing_pct = round(100 - stance_pct, 1)
            else:
                stance_pct = None
                swing_pct = None

            # Extract and normalize angles
            raw_angles = _extract_cycle_angles(angle_frames, start_frame, end_frame, side)
            if raw_angles is None:
                logger.debug(f"Cycle {side} {start_frame}-{end_frame}: not enough angle data")
                continue

            angles_normalized = {}
            for joint, vals in raw_angles.items():
                angles_normalized[joint] = _normalize_to_percent(vals, n_points).tolist()

            cycles.append({
                "cycle_id": cycle_id,
                "side": side,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "toe_off_frame": to_frame,
                "duration": round(duration, 4),
                "stance_pct": stance_pct,
                "swing_pct": swing_pct,
                "angles_normalized": angles_normalized,
            })
            cycle_id += 1

    # Compute summary statistics
    summary = {}
    for side in ("left", "right"):
        side_cycles = [c for c in cycles if c["side"] == side]
        if not side_cycles:
            continue

        side_summary = {"n_cycles": len(side_cycles)}

        # Sagittal joints
        for joint in _JOINT_KEYS:
            arrs = []
            for c in side_cycles:
                vals = c["angles_normalized"].get(joint)
                if vals is not None:
                    arrs.append(np.array(vals))
            if not arrs:
                continue

            stacked = np.stack(arrs)  # (n_cycles, n_points)
            side_summary[f"{joint}_mean"] = np.mean(stacked, axis=0).tolist()
            side_summary[f"{joint}_std"] = np.std(stacked, axis=0).tolist()

        # Frontal-plane joints (only when present in cycle data)
        for frontal_joint in _FRONTAL_KEYS:
            arrs = []
            for c in side_cycles:
                vals = c["angles_normalized"].get(frontal_joint)
                if vals is not None:
                    arrs.append(np.array(vals))
            if not arrs:
                continue

            stacked = np.stack(arrs)
            side_summary[f"{frontal_joint}_mean"] = np.mean(stacked, axis=0).tolist()
            side_summary[f"{frontal_joint}_std"] = np.std(stacked, axis=0).tolist()

        summary[side] = side_summary

    logger.info(
        f"Segmented {len(cycles)} valid cycles: "
        f"L={summary.get('left', {}).get('n_cycles', 0)}, "
        f"R={summary.get('right', {}).get('n_cycles', 0)}"
    )

    data["cycles_data"] = {"cycles": cycles, "summary": summary}
    return {"cycles": cycles, "summary": summary}


def ensemble_average(
    trial_cycles_list: List[dict],
    joints: Optional[List[str]] = None,
    n_points: int = 101,
) -> dict:
    """Compute ensemble averages across multiple trials.

    Aggregates per-trial mean curves produced by :func:`segment_cycles`
    to obtain a grand mean, inter-trial variability (SD of trial means),
    and average intra-trial variability (mean of within-trial SDs).

    This follows the multi-trial averaging procedure recommended by
    OpenCap for markerless motion capture:

        Ref: Uhlrich SD, Falisse A, Kidzinski L, et al. OpenCap:
        Human movement dynamics from smartphone videos. PLoS Comput
        Biol. 2023;19(10):e1011462. doi:10.1371/journal.pcbi.1011462

    Parameters
    ----------
    trial_cycles_list : list of dict
        Each element is the return value of :func:`segment_cycles`,
        containing ``"cycles"`` and ``"summary"`` keys.
    joints : list of str, optional
        Restrict output to these joint names (e.g. ``["hip", "knee"]``).
        When *None* (default), all joints found in trial summaries are
        included.
    n_points : int, optional
        Expected number of points per normalized curve (default 101).
        Used only for validation; curves are not re-interpolated.

    Returns
    -------
    dict
        Nested dict keyed by side (``"left"``, ``"right"``), then by
        joint name.  Each joint entry contains:

        - ``grand_mean`` : np.ndarray of shape (*n_points*,)
        - ``inter_trial_sd`` : np.ndarray of shape (*n_points*,)
        - ``intra_trial_sd`` : np.ndarray of shape (*n_points*,)
        - ``n_trials`` : int -- number of trials contributing
        - ``n_total_cycles`` : int -- total cycles across all trials

    Raises
    ------
    TypeError
        If *trial_cycles_list* is not a list.
    ValueError
        If *trial_cycles_list* is empty.
    """
    if not isinstance(trial_cycles_list, list):
        raise TypeError("trial_cycles_list must be a list of segment_cycles() outputs")
    if len(trial_cycles_list) == 0:
        raise ValueError("trial_cycles_list must contain at least one trial")

    result: Dict[str, Dict[str, dict]] = {}

    for side in ("left", "right"):
        side_result: Dict[str, dict] = {}

        # Discover all joint names available across trials for this side.
        # Summary keys follow the pattern  {joint}_mean / {joint}_std.
        all_joint_names: set = set()
        for trial in trial_cycles_list:
            side_summary = trial.get("summary", {}).get(side, {})
            for key in side_summary:
                if key.endswith("_mean"):
                    jname = key[: -len("_mean")]
                    all_joint_names.add(jname)

        # Optionally restrict to requested joints
        if joints is not None:
            all_joint_names = all_joint_names & set(joints)

        for jname in sorted(all_joint_names):
            mean_key = f"{jname}_mean"
            std_key = f"{jname}_std"

            trial_means: List[np.ndarray] = []
            trial_stds: List[np.ndarray] = []
            total_cycles = 0

            for trial in trial_cycles_list:
                side_summary = trial.get("summary", {}).get(side, {})
                mean_vals = side_summary.get(mean_key)
                std_vals = side_summary.get(std_key)
                n_cyc = side_summary.get("n_cycles", 0)

                if mean_vals is None:
                    # This trial does not have this joint -- skip gracefully
                    continue

                mean_arr = np.asarray(mean_vals, dtype=float)
                trial_means.append(mean_arr)
                total_cycles += n_cyc

                if std_vals is not None:
                    trial_stds.append(np.asarray(std_vals, dtype=float))

            if not trial_means:
                continue

            stacked_means = np.stack(trial_means)  # (n_trials, n_points)
            grand_mean = np.mean(stacked_means, axis=0)
            inter_trial_sd = np.std(stacked_means, axis=0, ddof=0)

            if trial_stds:
                stacked_stds = np.stack(trial_stds)
                intra_trial_sd = np.mean(stacked_stds, axis=0)
            else:
                intra_trial_sd = np.zeros(n_points, dtype=float)

            side_result[jname] = {
                "grand_mean": grand_mean,
                "inter_trial_sd": inter_trial_sd,
                "intra_trial_sd": intra_trial_sd,
                "n_trials": len(trial_means),
                "n_total_cycles": total_cycles,
            }

        if side_result:
            result[side] = side_result

    return result
