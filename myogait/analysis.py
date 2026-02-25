"""Gait analysis: spatio-temporal parameters, symmetry, and variability.

Computes clinical gait metrics from detected events and segmented cycles.

Functions
---------
analyze_gait
    Compute comprehensive gait statistics (main entry point).
regularity_index
    Stride regularity via autocorrelation.
    Ref: Moe-Nilssen R, Helbostad JL. Estimation of gait cycle
    characteristics by trunk accelerometry. J Biomech.
    2004;37(1):121-126. doi:10.1016/S0021-9290(03)00233-1
harmonic_ratio
    Gait smoothness via FFT harmonic analysis.
    Ref: Smidt GL, Arora JS, Johnston RC. Accelerographic analysis
    of several types of walking. Am J Phys Med.
    1971;50(6):285-300.
    Ref: Gage JR. An overview of normal walking. Instr Course Lect.
    1990;39:291-303.
    Ref: Bellanca JL, Lowry KA, VanSwearingen JM, Brach JS,
    Redfern MS. Harmonic ratios: a quantification of step to step
    symmetry. J Biomech. 2013;46(4):828-831.
    doi:10.1016/j.jbiomech.2012.12.008
step_length
    Pixel-based step length estimation with optional calibration.
    Anthropometric femur ratio (24.5%% of height) based on:
    Ref: Drillis R, Contini R, Bluestein M. Body segment parameters:
    a survey of measurement techniques. Artif Limbs. 1964;8(1):44-66.
walking_speed
    Average walking speed estimation (stride length / stride time).
detect_pathologies
    Advanced gait pattern detection.
    References:
    - Trendelenburg: Trendelenburg F. Ueber den Gang bei angeborener
      Hüftgelenksluxation. Dtsch Med Wochenschr. 1895;21:21-24.
      Hardcastle P, Nade S. The significance of the Trendelenburg
      test. J Bone Joint Surg Br. 1985;67(5):741-746.
    - Spastic gait: Gage JR, Novacheck TF. An update on the
      treatment of gait problems in cerebral palsy. J Pediatr
      Orthop B. 2001;10(4):265-274.
    - Steppage/foot drop: Stewart JD. Foot drop: where, why and
      what to do? Pract Neurol. 2008;8(3):158-169.
      doi:10.1136/jnnp.2008.149393
    - Crouch gait: Rodda JM, Graham HK. Classification of gait
      patterns in spastic hemiplegia and spastic diplegia: a basis
      for a management algorithm. Eur J Neurol. 2001;8(Suppl 5):
      98-108. doi:10.1046/j.1468-1331.2001.00042.x
compute_derivatives
    Angular velocity and acceleration via central differences.
    Ref: Winter DA. Biomechanics and Motor Control of Human Movement.
    4th ed. Wiley; 2009. Chapter 2.

Symmetry index formula:
    SI = |L - R| / (0.5 * (L + R)) * 100
    Ref: Robinson RO, Herzog W, Nigg BM. Use of force platform
    variables to quantify the effects of chiropractic manipulation
    on gait symmetry. J Manipulative Physiol Ther.
    1987;10(4):172-176.

Variability metrics:
    Ref: Hausdorff JM, Rios DA, Edelberg HK. Gait variability and
    fall risk in community-living older adults: a 1-year prospective
    study. Arch Phys Med Rehabil. 2001;82(8):1050-1056.
    doi:10.1053/apmr.2001.24893
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _symmetry_index(left: float, right: float) -> float:
    """SI = |L - R| / (0.5 * (L + R)) * 100. Returns 0 if both are 0."""
    denom = 0.5 * (left + right)
    if denom == 0:
        return 0.0
    return abs(left - right) / denom * 100


def _cv(values: list) -> float:
    """Coefficient of variation (%)."""
    if len(values) < 2:
        return 0.0
    m = np.mean(values)
    if m == 0:
        return 0.0
    return float(np.std(values, ddof=1) / m * 100)


def _rom(values: list) -> float:
    """Range of motion (max - min)."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if not valid:
        return 0.0
    return float(np.ptp(valid))


def analyze_gait(
    data: dict,
    cycles: dict,
    height_m: Optional[float] = None,
) -> dict:
    """Compute comprehensive gait statistics.

    Aggregates spatio-temporal parameters, symmetry indices,
    variability metrics, regularity, harmonic ratio, step length,
    walking speed, and pathology detection into a single report.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events`` and ``angles``.
    cycles : dict
        Output of ``segment_cycles()``.
    height_m : float, optional
        Subject height in meters. Enables calibrated step length
        and walking speed in m/s.

    Returns
    -------
    dict
        Analysis report with keys: ``spatiotemporal``, ``symmetry``,
        ``variability``, ``regularity``, ``harmonic_ratio``,
        ``step_length``, ``walking_speed``, ``pathologies``,
        ``pathology_flags``.

    Raises
    ------
    TypeError
        If *data* or *cycles* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if not isinstance(cycles, dict):
        raise TypeError("cycles must be a dict")
    fps = data.get("meta", {}).get("fps", 30.0)
    events = data.get("events", {})
    angles = data.get("angles", {})
    cycle_list = cycles.get("cycles", [])

    stats = {
        "spatiotemporal": _spatiotemporal(cycle_list, events, fps),
        "symmetry": _symmetry(cycle_list, angles),
        "variability": _variability(cycle_list),
        "regularity": regularity_index(data),
        "harmonic_ratio": harmonic_ratio(data),
        "step_length": step_length(data, cycles, height_m),
        "walking_speed": walking_speed(data, cycles, height_m),
        "pathologies": detect_pathologies(data, cycles),
        "pathology_flags": [],
    }

    # Detect pathology flags
    stats["pathology_flags"] = _detect_flags(stats)
    _add_legacy_summary_aliases(stats)

    return stats


def _add_legacy_summary_aliases(stats: dict) -> None:
    """Expose backward-compatible top-level summary keys."""
    st = stats.get("spatiotemporal", {})
    ws = stats.get("walking_speed", {})

    cadence = st.get("cadence_steps_per_min")
    if cadence is not None:
        stats["cadence"] = cadence

    speed = ws.get("speed_mean")
    if speed is not None:
        stats["speed"] = speed

    stance_left = st.get("stance_pct_left")
    stance_right = st.get("stance_pct_right")
    stance_vals = [v for v in (stance_left, stance_right) if v is not None]
    if stance_vals:
        stats["stance_pct"] = round(float(np.mean(stance_vals)), 1)


def _spatiotemporal(cycle_list: list, events: dict, fps: float) -> dict:
    """Compute spatio-temporal parameters."""
    left_cycles = [c for c in cycle_list if c["side"] == "left"]
    right_cycles = [c for c in cycle_list if c["side"] == "right"]

    left_durations = [c["duration"] for c in left_cycles] if left_cycles else []
    right_durations = [c["duration"] for c in right_cycles] if right_cycles else []
    all_durations = left_durations + right_durations

    stride_time_mean = float(np.mean(all_durations)) if all_durations else 0.0
    stride_time_std = float(np.std(all_durations)) if len(all_durations) > 1 else 0.0

    # Cadence: 2 steps per stride
    cadence = (60.0 / stride_time_mean * 2) if stride_time_mean > 0 else 0.0

    # Step time (alternating feet)
    all_hs = []
    for ev in events.get("left_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "left"})
    for ev in events.get("right_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "right"})
    all_hs.sort(key=lambda e: e["frame"])

    step_times = []
    for i in range(len(all_hs) - 1):
        if all_hs[i]["side"] != all_hs[i + 1]["side"]:
            dt = (all_hs[i + 1]["frame"] - all_hs[i]["frame"]) / fps
            step_times.append(dt)

    step_time_mean = float(np.mean(step_times)) if step_times else stride_time_mean / 2
    step_time_std = float(np.std(step_times)) if len(step_times) > 1 else 0.0

    # Stance/swing percentages
    left_stance = [c["stance_pct"] for c in left_cycles if c["stance_pct"] is not None]
    right_stance = [c["stance_pct"] for c in right_cycles if c["stance_pct"] is not None]

    stance_left = float(np.mean(left_stance)) if left_stance else None
    stance_right = float(np.mean(right_stance)) if right_stance else None

    # Double support
    double_support = None
    if stance_left is not None and stance_right is not None:
        double_support = round(stance_left + stance_right - 100, 1)

    return {
        "cadence_steps_per_min": round(cadence, 1),
        "stride_time_mean_s": round(stride_time_mean, 3),
        "stride_time_std_s": round(stride_time_std, 3),
        "stride_time_left_s": round(float(np.mean(left_durations)), 3) if left_durations else None,
        "stride_time_right_s": round(float(np.mean(right_durations)), 3) if right_durations else None,
        "step_time_mean_s": round(step_time_mean, 3),
        "step_time_std_s": round(step_time_std, 3),
        "stance_pct_left": round(stance_left, 1) if stance_left is not None else None,
        "stance_pct_right": round(stance_right, 1) if stance_right is not None else None,
        "swing_pct_left": round(100 - stance_left, 1) if stance_left is not None else None,
        "swing_pct_right": round(100 - stance_right, 1) if stance_right is not None else None,
        "double_support_pct": double_support,
        "n_cycles_left": len(left_cycles),
        "n_cycles_right": len(right_cycles),
        "n_cycles_total": len(left_cycles) + len(right_cycles),
    }


def _symmetry(cycle_list: list, angles: dict) -> dict:
    """Compute symmetry indices."""
    angle_frames = angles.get("frames", [])

    # ROM from full angle series
    joints = ["hip", "knee", "ankle"]
    rom_left = {}
    rom_right = {}
    for j in joints:
        rom_left[j] = _rom([af.get(f"{j}_L") for af in angle_frames])
        rom_right[j] = _rom([af.get(f"{j}_R") for af in angle_frames])

    si = {}
    for j in joints:
        si[f"{j}_rom_si"] = round(_symmetry_index(rom_left[j], rom_right[j]), 1)

    # Temporal symmetry
    left_durations = [c["duration"] for c in cycle_list if c["side"] == "left"]
    right_durations = [c["duration"] for c in cycle_list if c["side"] == "right"]
    if left_durations and right_durations:
        si["step_time_si"] = round(
            _symmetry_index(float(np.mean(left_durations)), float(np.mean(right_durations))), 1
        )

    # Stance symmetry
    left_stance = [c["stance_pct"] for c in cycle_list if c["side"] == "left" and c["stance_pct"] is not None]
    right_stance = [c["stance_pct"] for c in cycle_list if c["side"] == "right" and c["stance_pct"] is not None]
    if left_stance and right_stance:
        si["stance_time_si"] = round(
            _symmetry_index(float(np.mean(left_stance)), float(np.mean(right_stance))), 1
        )

    # Overall
    si_values = [v for k, v in si.items() if k.endswith("_si")]
    si["overall_si"] = round(float(np.mean(si_values)), 1) if si_values else 0.0

    return si


def _variability(cycle_list: list) -> dict:
    """Compute cycle-to-cycle variability."""
    all_durations = [c["duration"] for c in cycle_list]
    all_stance = [c["stance_pct"] for c in cycle_list if c["stance_pct"] is not None]

    # Kinematic variability: ROM per normalized cycle
    rom_by_side_joint = {}
    for side in ("left", "right"):
        side_cycles = [c for c in cycle_list if c["side"] == side]
        for joint in ("hip", "knee", "ankle"):
            roms = []
            for c in side_cycles:
                vals = c.get("angles_normalized", {}).get(joint)
                if vals:
                    roms.append(float(np.ptp(vals)))
            key = f"{side}_{joint}_rom_cv"
            rom_by_side_joint[key] = round(_cv(roms), 1) if roms else 0.0

    return {
        "cycle_duration_cv": round(_cv(all_durations), 1),
        "cycle_duration_sd": round(float(np.std(all_durations)), 3) if len(all_durations) > 1 else 0.0,
        "stance_pct_cv": round(_cv(all_stance), 1) if all_stance else 0.0,
        **rom_by_side_joint,
    }


def _detect_flags(stats: dict) -> List[str]:
    """Detect potential pathology flags."""
    flags = []
    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    var = stats.get("variability", {})

    # Cadence
    cadence = st.get("cadence_steps_per_min", 0)
    if cadence > 0 and cadence < 80:
        flags.append(f"Low cadence: {cadence:.0f} steps/min (normal: 100-120)")
    elif cadence > 140:
        flags.append(f"High cadence: {cadence:.0f} steps/min (normal: 100-120)")

    # Prolonged stance
    for side in ("left", "right"):
        stance = st.get(f"stance_pct_{side}")
        if stance is not None and stance > 70:
            flags.append(f"Prolonged stance {side}: {stance:.1f}% (normal: ~60%)")

    # Asymmetry
    for key, val in sym.items():
        if key.endswith("_si") and key != "overall_si" and val > 20:
            joint = key.replace("_si", "").replace("_rom", "")
            flags.append(f"Asymmetry {joint}: SI={val:.1f}% (>20%)")

    # High variability
    duration_cv = var.get("cycle_duration_cv", 0)
    if duration_cv > 20:
        flags.append(f"High cycle duration variability: CV={duration_cv:.1f}% (>20%)")

    return flags


# ── Regularity index (autocorrelation) ───────────────────────────────


def regularity_index(data: dict, signal_key: str = "LEFT_ANKLE") -> dict:
    """Compute stride regularity using autocorrelation.

    Based on Moe-Nilssen & Helbostad (2004). Uses the vertical
    position signal to compute step and stride regularity
    coefficients via unbiased autocorrelation.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    signal_key : str, optional
        Landmark name for the signal (default ``"LEFT_ANKLE"``).

    Returns
    -------
    dict
        Keys: ``step_regularity``, ``stride_regularity``,
        ``symmetry_ratio``. Values are None if insufficient data.
    """
    frames = data.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)

    if len(frames) < 30:
        return {"step_regularity": None, "stride_regularity": None, "symmetry_ratio": None}

    # Extract vertical position
    y_vals = []
    for f in frames:
        lm = f.get("landmarks", {}).get(signal_key)
        if lm and lm.get("y") is not None:
            y_vals.append(float(lm["y"]))
        else:
            y_vals.append(np.nan)

    y = np.array(y_vals)
    valid = ~np.isnan(y)
    if valid.sum() < 30:
        return {"step_regularity": None, "stride_regularity": None, "symmetry_ratio": None}

    # Interpolate NaN
    x_idx = np.arange(len(y))
    y[~valid] = np.interp(x_idx[~valid], x_idx[valid], y[valid])

    # Detrend
    y = y - np.mean(y)

    # Autocorrelation (unbiased)
    n = len(y)
    autocorr = np.correlate(y, y, mode="full")
    autocorr = autocorr[n - 1:]  # positive lags only
    autocorr = autocorr / autocorr[0]  # normalize

    # Expected step period: ~0.4-0.7s → 12-21 frames at 30fps
    min_lag = max(1, int(0.3 * fps))
    max_lag = min(len(autocorr) - 1, int(1.5 * fps))

    if max_lag <= min_lag:
        return {"step_regularity": None, "stride_regularity": None, "symmetry_ratio": None}

    # First peak = step regularity (Ad1)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(autocorr[min_lag:max_lag])
    if len(peaks) == 0:
        return {"step_regularity": None, "stride_regularity": None, "symmetry_ratio": None}

    step_lag = peaks[0] + min_lag
    step_reg = float(autocorr[step_lag])

    # Second peak = stride regularity (Ad2)
    stride_min = step_lag + min_lag
    stride_max = min(len(autocorr) - 1, step_lag * 3)
    if stride_max > stride_min:
        peaks2, _ = find_peaks(autocorr[stride_min:stride_max])
        if len(peaks2) > 0:
            stride_lag = peaks2[0] + stride_min
            stride_reg = float(autocorr[stride_lag])
        else:
            stride_reg = None
    else:
        stride_reg = None

    # Symmetry ratio
    sym_ratio = None
    if stride_reg is not None and stride_reg > 0:
        sym_ratio = round(step_reg / stride_reg, 3)

    return {
        "step_regularity": round(step_reg, 3) if step_reg is not None else None,
        "stride_regularity": round(stride_reg, 3) if stride_reg is not None else None,
        "symmetry_ratio": sym_ratio,
    }


# ── Harmonic ratio ───────────────────────────────────────────────────


def harmonic_ratio(data: dict, signal_key: str = "LEFT_ANKLE") -> dict:
    """Compute harmonic ratio of gait signal via FFT.

    The harmonic ratio measures gait smoothness as the ratio of
    even to odd harmonics (AP direction). Higher values indicate
    smoother, more symmetric gait.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    signal_key : str, optional
        Landmark name (default ``"LEFT_ANKLE"``).

    Returns
    -------
    dict
        Keys: ``hr_ap`` (anteroposterior), ``hr_vertical``.
        Values are None if insufficient data.

    References
    ----------
    Smidt et al. (1971), Gage (1991).
    """
    frames = data.get("frames", [])

    if len(frames) < 60:
        return {"hr_ap": None, "hr_vertical": None}

    x_vals, y_vals = [], []
    for f in frames:
        lm = f.get("landmarks", {}).get(signal_key)
        if lm:
            x_vals.append(float(lm.get("x", np.nan)))
            y_vals.append(float(lm.get("y", np.nan)))
        else:
            x_vals.append(np.nan)
            y_vals.append(np.nan)

    def _compute_hr(signal, ap=False):
        sig = np.array(signal)
        valid = ~np.isnan(sig)
        if valid.sum() < 30:
            return None
        x_idx = np.arange(len(sig))
        sig[~valid] = np.interp(x_idx[~valid], x_idx[valid], sig[valid])
        sig = sig - np.mean(sig)

        # FFT
        fft_vals = np.abs(np.fft.rfft(sig))
        if len(fft_vals) < 21:
            return None

        # First 20 harmonics
        harmonics = fft_vals[1:21]
        even = harmonics[1::2]  # 2nd, 4th, 6th...
        odd = harmonics[0::2]   # 1st, 3rd, 5th...

        sum_odd = np.sum(odd)
        sum_even = np.sum(even)

        # AP direction: odd/even (Bellanca et al. 2013)
        # Vertical direction: even/odd
        if ap:
            if sum_even == 0:
                return None
            return round(float(sum_odd / sum_even), 3)
        else:
            if sum_odd == 0:
                return None
            return round(float(sum_even / sum_odd), 3)

    return {
        "hr_ap": _compute_hr(x_vals, ap=True),
        "hr_vertical": _compute_hr(y_vals, ap=False),
    }


# ── Step length estimation ───────────────────────────────────────────


def step_length(
    data: dict,
    cycles: dict,
    height_m: Optional[float] = None,
) -> dict:
    """Estimate step and stride length from pose data.

    Uses the horizontal distance between ankle positions at heel
    strike. When subject height is provided, calibrates the
    pixel-to-meter conversion using femur length as an
    anthropometric reference (~24.5%% of body height).

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` and ``events``.
    cycles : dict
        Output of ``segment_cycles()``.
    height_m : float, optional
        Subject height in meters for calibration.

    Returns
    -------
    dict
        Keys: ``step_length_left``, ``step_length_right``,
        ``stride_length_left``, ``stride_length_right``,
        ``unit``, ``calibrated``.
    """
    frames = data.get("frames", [])
    events = data.get("events", {})

    if not frames or not events:
        return {"step_length_left": None, "step_length_right": None,
                "stride_length_left": None, "stride_length_right": None}

    extraction = data.get("extraction", {})
    if isinstance(extraction, dict) and extraction.get("treadmill") is True:
        return {
            "step_length_left": None,
            "step_length_right": None,
            "stride_length_left": None,
            "stride_length_right": None,
            "unit": "m" if height_m else "normalized",
            "calibrated": height_m is not None,
            "valid_for_progression": False,
            "limitation": (
                "Treadmill-like trial detected: image-progression step/stride length "
                "is not reliable."
            ),
        }

    # Estimate pixel-to-meter scale
    # Use femur (hip-knee) length as reference: ~25% of body height
    scale = 1.0  # pixels per meter (default)
    if height_m is not None:
        femur_lengths = []
        for f in frames[:min(60, len(frames))]:
            lm = f.get("landmarks", {})
            hip = lm.get("LEFT_HIP")
            knee = lm.get("LEFT_KNEE")
            if hip and knee and hip.get("x") is not None and knee.get("x") is not None:
                dx = hip["x"] - knee["x"]
                dy = hip["y"] - knee["y"]
                femur_lengths.append(np.sqrt(dx**2 + dy**2))
        if femur_lengths:
            median_femur_px = np.median(femur_lengths)
            femur_m = height_m * 0.245  # femur ≈ 24.5% of height
            if median_femur_px > 0:
                scale = femur_m / median_femur_px

    # Compute step lengths (distance between consecutive HS of opposite feet)
    all_hs = []
    for ev in events.get("left_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "left"})
    for ev in events.get("right_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "right"})
    all_hs.sort(key=lambda e: e["frame"])

    step_lengths = {"left": [], "right": []}
    for i in range(len(all_hs) - 1):
        if all_hs[i]["side"] == all_hs[i + 1]["side"]:
            continue
        f1 = all_hs[i]["frame"]
        f2 = all_hs[i + 1]["frame"]
        if f1 >= len(frames) or f2 >= len(frames):
            continue

        side = all_hs[i + 1]["side"]  # step is named by the leading foot
        ankle_name = f"{side.upper()}_ANKLE"

        lm1 = frames[f1].get("landmarks", {}).get(ankle_name, {})
        lm2 = frames[f2].get("landmarks", {}).get(ankle_name, {})
        x1 = lm1.get("x")
        x2 = lm2.get("x")
        if x1 is not None and x2 is not None:
            dist = abs(x2 - x1) * scale
            step_lengths[side].append(dist)

    # Stride lengths from cycles
    stride_lengths = {"left": [], "right": []}
    for c in cycles.get("cycles", []):
        f1 = c["start_frame"]
        f2 = c["end_frame"]
        if f1 >= len(frames) or f2 >= len(frames):
            continue
        ankle_name = f"{c['side'].upper()}_ANKLE"
        lm1 = frames[f1].get("landmarks", {}).get(ankle_name, {})
        lm2 = frames[f2].get("landmarks", {}).get(ankle_name, {})
        x1 = lm1.get("x")
        x2 = lm2.get("x")
        if x1 is not None and x2 is not None:
            dist = abs(x2 - x1) * scale
            stride_lengths[c["side"]].append(dist)

    def _mean_or_none(vals):
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "step_length_left": _mean_or_none(step_lengths["left"]),
        "step_length_right": _mean_or_none(step_lengths["right"]),
        "stride_length_left": _mean_or_none(stride_lengths["left"]),
        "stride_length_right": _mean_or_none(stride_lengths["right"]),
        "unit": "m" if height_m else "normalized",
        "calibrated": height_m is not None,
        "valid_for_progression": True,
    }


# ── Walking speed ────────────────────────────────────────────────────


def walking_speed(
    data: dict,
    cycles: dict,
    height_m: Optional[float] = None,
) -> dict:
    """Estimate average walking speed.

    Computes speed as stride_length / stride_time. When height
    is provided, the result is in m/s; otherwise in normalized
    units per second.

    Parameters
    ----------
    data : dict
        Pivot JSON dict.
    cycles : dict
        Output of ``segment_cycles()``.
    height_m : float, optional
        Subject height in meters.

    Returns
    -------
    dict
        Keys: ``speed_mean``, ``speed_left``, ``speed_right``,
        ``unit``.
    """
    frames = data.get("frames", [])
    height_m_val = height_m or data.get("subject", {}).get("height_m") if data.get("subject") else height_m

    extraction = data.get("extraction", {})
    if isinstance(extraction, dict) and extraction.get("treadmill") is True:
        return {
            "speed_mean": None,
            "speed_left": None,
            "speed_right": None,
            "unit": "m/s" if height_m_val else "norm/s",
            "valid_for_progression": False,
            "limitation": (
                "Treadmill-like trial detected: walking speed from image progression "
                "is not reliable."
            ),
        }

    # Compute scale factor (same logic as step_length)
    scale = 1.0
    if height_m_val:
        femur_lengths = []
        for f in frames:
            lm = f.get("landmarks", {})
            hip = lm.get("LEFT_HIP")
            knee = lm.get("LEFT_KNEE")
            if hip and knee and hip.get("x") is not None and knee.get("x") is not None:
                dx = hip["x"] - knee["x"]
                dy = hip["y"] - knee["y"]
                femur_lengths.append(np.sqrt(dx**2 + dy**2))
        if femur_lengths:
            median_femur_px = np.median(femur_lengths)
            femur_m = height_m_val * 0.245
            if median_femur_px > 0:
                scale = femur_m / median_femur_px

    cycle_list = cycles.get("cycles", [])

    speeds = {"left": [], "right": []}
    for c in cycle_list:
        side = c["side"]
        f1 = c["start_frame"]
        f2 = c["end_frame"]
        if f1 >= len(frames) or f2 >= len(frames):
            continue
        ankle_name = f"{side.upper()}_ANKLE"
        lm1 = frames[f1].get("landmarks", {}).get(ankle_name, {})
        lm2 = frames[f2].get("landmarks", {}).get(ankle_name, {})
        x1 = lm1.get("x")
        x2 = lm2.get("x")
        if x1 is not None and x2 is not None and c["duration"] > 0:
            stride_l = abs(x2 - x1) * scale
            speeds[side].append(stride_l / c["duration"])

    def _mean_or_none(vals):
        return round(float(np.mean(vals)), 3) if vals else None

    all_speeds = speeds["left"] + speeds["right"]
    return {
        "speed_mean": _mean_or_none(all_speeds),
        "speed_left": _mean_or_none(speeds["left"]),
        "speed_right": _mean_or_none(speeds["right"]),
        "unit": "m/s" if height_m_val else "norm/s",
        "valid_for_progression": True,
    }


# ── Advanced pathology detection ─────────────────────────────────────


def detect_pathologies(data: dict, cycles: dict) -> List[dict]:
    """Detect advanced gait pathology patterns.

    Screens normalized gait cycles for patterns suggestive of
    common gait disorders.

    Detected patterns:

    - **Trendelenburg**: excessive pelvis drop during stance.
    - **Spastic gait**: reduced knee flexion in swing.
    - **Steppage gait**: excessive hip flexion compensating foot drop.
    - **Crouch gait**: persistent knee flexion throughout cycle.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles``.
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    list of dict
        Each dict has keys: ``pattern``, ``side``, ``severity``,
        ``value``, ``description``.
    """
    pathologies = []
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])

    if not angle_frames:
        return pathologies

    cycle_list = cycles.get("cycles", [])

    for side in ("left", "right"):
        side_cycles = [c for c in cycle_list if c["side"] == side]
        if not side_cycles:
            continue

        # Aggregate normalized curves
        hip_curves = [np.array(c["angles_normalized"]["hip"])
                      for c in side_cycles if "hip" in c.get("angles_normalized", {})]
        knee_curves = [np.array(c["angles_normalized"]["knee"])
                       for c in side_cycles if "knee" in c.get("angles_normalized", {})]
        ankle_curves = [np.array(c["angles_normalized"]["ankle"])
                        for c in side_cycles if "ankle" in c.get("angles_normalized", {})]

        # Spastic gait: reduced knee flexion in swing (60-100%)
        if knee_curves:
            knee_mean = np.mean(knee_curves, axis=0)
            swing_knee_max = np.max(knee_mean[60:])  # swing phase
            if swing_knee_max < 40:
                confidence = min(1.0, max(0.0, (40.0 - float(swing_knee_max)) / 20.0))
                pathologies.append({
                    "pattern": "spastic",
                    "side": side,
                    "severity": "moderate" if swing_knee_max < 30 else "mild",
                    "value": round(float(swing_knee_max), 1),
                    "confidence": round(confidence, 2),
                    "description": f"Reduced swing knee flexion ({swing_knee_max:.1f} deg, normal: 60-70)",
                })

        # Steppage: excessive hip flexion in swing (compensating for foot drop)
        if hip_curves and ankle_curves:
            hip_mean = np.mean(hip_curves, axis=0)
            ankle_mean = np.mean(ankle_curves, axis=0)
            swing_hip_max = np.max(hip_mean[60:])
            ankle_rom = np.ptp(ankle_mean)
            if swing_hip_max > 45 and ankle_rom < 15:
                conf_hip = max(0.0, (float(swing_hip_max) - 45.0) / 25.0)
                conf_ankle = max(0.0, (15.0 - float(ankle_rom)) / 10.0)
                confidence = min(1.0, max(0.0, 0.5 * (conf_hip + conf_ankle)))
                pathologies.append({
                    "pattern": "steppage",
                    "side": side,
                    "severity": "moderate" if ankle_rom < 10 else "mild",
                    "value": round(float(ankle_rom), 1),
                    "confidence": round(confidence, 2),
                    "description": f"Suspected foot drop: ankle ROM={ankle_rom:.1f} deg, hip overflexion={swing_hip_max:.1f} deg",
                })

        # Crouch gait: knee never fully extends (min knee angle > 15)
        if knee_curves:
            knee_mean = np.mean(knee_curves, axis=0)
            min_knee = np.min(knee_mean)
            if min_knee > 15:
                confidence = min(1.0, max(0.0, (float(min_knee) - 15.0) / 20.0))
                pathologies.append({
                    "pattern": "crouch",
                    "side": side,
                    "severity": "severe" if min_knee > 25 else "moderate",
                    "value": round(float(min_knee), 1),
                    "confidence": round(confidence, 2),
                    "description": f"Persistent knee flexion (min={min_knee:.1f} deg, normal: ~0)",
                })

    # Trendelenburg: check pelvis tilt during stance (0-60% of cycle)
    # Runs once (not per-side) since pelvis_tilt is a global measurement.
    pelvis_vals = [af.get("pelvis_tilt") for af in angle_frames
                   if af.get("pelvis_tilt") is not None]
    if pelvis_vals:
        pelvis_range = np.ptp([v for v in pelvis_vals if not np.isnan(v)])
        if pelvis_range > 10:
            confidence = min(1.0, max(0.0, (float(pelvis_range) - 10.0) / 10.0))
            pathologies.append({
                "pattern": "trendelenburg",
                "side": "bilateral",
                "severity": "moderate" if pelvis_range > 15 else "mild",
                "value": round(float(pelvis_range), 1),
                "confidence": round(confidence, 2),
                "description": f"Excessive pelvis drop ({pelvis_range:.1f} deg range)",
            })

    return pathologies


# ── Single support time ──────────────────────────────────────────────


def single_support_time(data: dict, cycles: dict) -> dict:
    """Compute single support time per side.

    Single support time (SST) is the period during stance when only
    one foot is on the ground. It corresponds to the swing phase of
    the contralateral limb.

    Normally SST ≈ 40% of the gait cycle. Reduced SST on one side
    may indicate pain avoidance or instability.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events``.
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``sst_left_s``, ``sst_right_s``, ``sst_left_pct``,
        ``sst_right_pct``, ``sst_symmetry_index``.

    References
    ----------
    Perry J, Burnfield JM. Gait Analysis: Normal and Pathological
    Function. 2nd ed. SLACK; 2010:9-16.
    """
    events = data.get("events", {})
    fps = data.get("meta", {}).get("fps", 30.0)
    cycle_list = cycles.get("cycles", [])

    # Collect toe-off events per side
    to_frames = {
        "left": sorted(ev["frame"] for ev in events.get("left_to", [])),
        "right": sorted(ev["frame"] for ev in events.get("right_to", [])),
    }

    sst = {"left": [], "right": []}

    for c in cycle_list:
        side = c["side"]
        contra = "right" if side == "left" else "left"
        start = c["start_frame"]
        end = c["end_frame"]

        # Find contralateral TO within this cycle
        contra_tos = [f for f in to_frames[contra] if start <= f <= end]
        if not contra_tos:
            continue

        # SST: from contralateral TO to end of cycle
        contra_to = contra_tos[0]
        sst_frames = end - contra_to
        sst_s = sst_frames / fps
        sst_pct = (sst_frames / (end - start)) * 100 if (end - start) > 0 else 0
        sst[side].append({"s": sst_s, "pct": sst_pct})

    result = {}
    for side in ("left", "right"):
        if sst[side]:
            result[f"sst_{side}_s"] = round(float(np.mean([v["s"] for v in sst[side]])), 3)
            result[f"sst_{side}_pct"] = round(float(np.mean([v["pct"] for v in sst[side]])), 1)
        else:
            result[f"sst_{side}_s"] = None
            result[f"sst_{side}_pct"] = None

    if result["sst_left_s"] is not None and result["sst_right_s"] is not None:
        result["sst_symmetry_index"] = round(
            _symmetry_index(result["sst_left_s"], result["sst_right_s"]), 1
        )
    else:
        result["sst_symmetry_index"] = None

    return result


# ── Toe clearance ────────────────────────────────────────────────────


def toe_clearance(data: dict, cycles: dict) -> dict:
    """Compute minimum toe clearance during swing phase.

    Minimum toe clearance (MTC) is the smallest distance between
    the foot and the ground during mid-swing. Low MTC is a risk
    factor for tripping and falls.

    Normal MTC ≈ 1-2 cm (10-20 px in normalized coordinates).

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames``.
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``mtc_left``, ``mtc_right``, ``mtc_left_cv``,
        ``mtc_right_cv``, ``unit``.

    References
    ----------
    Winter DA. Foot trajectory in human gait: a precise and
    multifactorial motor control task. Phys Ther.
    1992;72(1):45-53. doi:10.1093/ptj/72.1.45

    Begg R, Best R, Dell'Oro L, Taylor S. Minimum foot clearance
    during walking: strategies for the minimisation of trip-related
    falls. Gait Posture. 2007;25(2):191-198.
    doi:10.1016/j.gaitpost.2006.03.008
    """
    frames = data.get("frames", [])
    cycle_list = cycles.get("cycles", [])

    if not frames or not cycle_list:
        return {"mtc_left": None, "mtc_right": None,
                "mtc_left_cv": None, "mtc_right_cv": None, "unit": "normalized"}

    # Find ground level from heel positions during stance
    heel_y_all = []
    for f in frames:
        for heel_name in ("LEFT_HEEL", "RIGHT_HEEL"):
            lm = f.get("landmarks", {}).get(heel_name)
            if lm and lm.get("y") is not None and not np.isnan(lm["y"]):
                heel_y_all.append(lm["y"])
    ground_y = np.percentile(heel_y_all, 95) if heel_y_all else 0.82

    mtc = {"left": [], "right": []}
    for c in cycle_list:
        side = c["side"]
        to_frame = c.get("to_frame")
        end_frame = c["end_frame"]
        if to_frame is None:
            continue

        # Swing phase: TO to end of cycle
        foot_name = f"{side.upper()}_FOOT_INDEX"
        min_clearance = float("inf")
        for fi in range(to_frame, min(end_frame, len(frames))):
            lm = frames[fi].get("landmarks", {}).get(foot_name)
            if lm and lm.get("y") is not None and not np.isnan(lm["y"]):
                clearance = ground_y - lm["y"]
                if clearance < min_clearance:
                    min_clearance = clearance

        if min_clearance < float("inf"):
            mtc[side].append(min_clearance)

    result = {}
    for side in ("left", "right"):
        if mtc[side]:
            result[f"mtc_{side}"] = round(float(np.mean(mtc[side])), 4)
            result[f"mtc_{side}_cv"] = round(_cv(mtc[side]), 1)
        else:
            result[f"mtc_{side}"] = None
            result[f"mtc_{side}_cv"] = None
    result["unit"] = "normalized"
    return result


# ── Stride variability (extended) ────────────────────────────────────


def stride_variability(data: dict, cycles: dict) -> dict:
    """Compute extended stride variability metrics.

    High gait variability is associated with increased fall risk
    and neurodegenerative conditions. This function computes the
    coefficient of variation (CV) for multiple gait parameters.

    Parameters
    ----------
    data : dict
        Pivot JSON dict.
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``stride_time_cv``, ``step_time_cv``,
        ``step_length_cv_left``, ``step_length_cv_right``,
        ``rom_cv_hip_left``, ``rom_cv_hip_right``,
        ``rom_cv_knee_left``, ``rom_cv_knee_right``,
        ``rom_cv_ankle_left``, ``rom_cv_ankle_right``.

    References
    ----------
    Hausdorff JM, et al. Gait variability and fall risk in
    community-living older adults. Arch Phys Med Rehabil.
    2001;82(8):1050-1056. doi:10.1053/apmr.2001.24893
    """
    events = data.get("events", {})
    fps = data.get("meta", {}).get("fps", 30.0)
    cycle_list = cycles.get("cycles", [])

    # Stride time CV
    durations = [c["duration"] for c in cycle_list]
    stride_time_cv = round(_cv(durations), 1)

    # Step time CV
    all_hs = []
    for ev in events.get("left_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "left"})
    for ev in events.get("right_hs", []):
        all_hs.append({"frame": ev["frame"], "side": "right"})
    all_hs.sort(key=lambda e: e["frame"])

    step_times = []
    for i in range(len(all_hs) - 1):
        if all_hs[i]["side"] != all_hs[i + 1]["side"]:
            dt = (all_hs[i + 1]["frame"] - all_hs[i]["frame"]) / fps
            step_times.append(dt)
    step_time_cv = round(_cv(step_times), 1)

    # Step length CV (use ankle x displacement at HS)
    frames = data.get("frames", [])
    step_lengths = {"left": [], "right": []}
    for i in range(len(all_hs) - 1):
        if all_hs[i]["side"] == all_hs[i + 1]["side"]:
            continue
        f1, f2 = all_hs[i]["frame"], all_hs[i + 1]["frame"]
        if f1 >= len(frames) or f2 >= len(frames):
            continue
        side = all_hs[i + 1]["side"]
        ankle = f"{side.upper()}_ANKLE"
        lm1 = frames[f1].get("landmarks", {}).get(ankle, {})
        lm2 = frames[f2].get("landmarks", {}).get(ankle, {})
        x1, x2 = lm1.get("x"), lm2.get("x")
        if x1 is not None and x2 is not None:
            step_lengths[side].append(abs(x2 - x1))

    # ROM CV per joint per side
    rom_cv = {}
    for side in ("left", "right"):
        side_cycles = [c for c in cycle_list if c["side"] == side]
        for joint in ("hip", "knee", "ankle"):
            roms = []
            for c in side_cycles:
                vals = c.get("angles_normalized", {}).get(joint)
                if vals:
                    roms.append(float(np.ptp(vals)))
            rom_cv[f"rom_cv_{joint}_{side}"] = round(_cv(roms), 1) if roms else 0.0

    return {
        "stride_time_cv": stride_time_cv,
        "step_time_cv": step_time_cv,
        "step_length_cv_left": round(_cv(step_lengths["left"]), 1),
        "step_length_cv_right": round(_cv(step_lengths["right"]), 1),
        **rom_cv,
    }


# ── Arm swing analysis ───────────────────────────────────────────────


def arm_swing_analysis(data: dict, cycles: dict) -> dict:
    """Analyze arm swing during gait.

    Measures shoulder flexion amplitude, bilateral asymmetry, and
    arm-leg coordination. Reduced arm swing is an early indicator
    of Parkinson's disease and neurological conditions.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` (including extended angles
        with ``shoulder_flex_L`` / ``shoulder_flex_R``).
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``amplitude_left``, ``amplitude_right``,
        ``asymmetry_index``, ``coordination_score``,
        ``reduced_swing``.

    References
    ----------
    Meyns P, Bruijn SM, Duysens J. The how and why of arm swing
    during human walking. Gait Posture. 2013;38(4):555-562.
    doi:10.1016/j.gaitpost.2013.02.006

    Mirelman A, et al. Arm swing as a potential new prodromal
    marker of Parkinson's disease. Mov Disord. 2016;31(10):
    1527-1534. doi:10.1002/mds.26720
    """
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])

    if not angle_frames:
        return {
            "amplitude_left": None, "amplitude_right": None,
            "asymmetry_index": None, "coordination_score": None,
            "reduced_swing": None,
        }

    # Extract shoulder flexion series
    shoulder_l = [af.get("shoulder_flex_L") for af in angle_frames]
    shoulder_r = [af.get("shoulder_flex_R") for af in angle_frames]

    # Filter out None/NaN
    shoulder_l_clean = [v for v in shoulder_l if v is not None and not np.isnan(v)]
    shoulder_r_clean = [v for v in shoulder_r if v is not None and not np.isnan(v)]

    if not shoulder_l_clean or not shoulder_r_clean:
        # Fall back to wrist x displacement as arm swing proxy
        frames = data.get("frames", [])
        wrist_l_x = [f.get("landmarks", {}).get("LEFT_WRIST", {}).get("x")
                      for f in frames]
        wrist_r_x = [f.get("landmarks", {}).get("RIGHT_WRIST", {}).get("x")
                      for f in frames]
        wrist_l_clean = [v for v in wrist_l_x if v is not None and not np.isnan(v)]
        wrist_r_clean = [v for v in wrist_r_x if v is not None and not np.isnan(v)]

        amp_l = float(np.ptp(wrist_l_clean)) * 100 if wrist_l_clean else None
        amp_r = float(np.ptp(wrist_r_clean)) * 100 if wrist_r_clean else None
    else:
        amp_l = float(np.ptp(shoulder_l_clean))
        amp_r = float(np.ptp(shoulder_r_clean))

    result = {
        "amplitude_left": round(amp_l, 1) if amp_l is not None else None,
        "amplitude_right": round(amp_r, 1) if amp_r is not None else None,
    }

    # Asymmetry
    if amp_l is not None and amp_r is not None:
        result["asymmetry_index"] = round(_symmetry_index(amp_l, amp_r), 1)
    else:
        result["asymmetry_index"] = None

    # Coordination: correlation between contralateral arm and leg
    ankle_l = [af.get("hip_L") for af in angle_frames]
    ankle_l_clean = [v for v in ankle_l if v is not None and not np.isnan(v)]

    if shoulder_r_clean and ankle_l_clean and len(shoulder_r_clean) == len(ankle_l_clean):
        corr = np.corrcoef(shoulder_r_clean, ankle_l_clean)[0, 1]
        result["coordination_score"] = round(max(0, -corr * 100), 1)
    else:
        result["coordination_score"] = None

    # Reduced swing flag
    result["reduced_swing"] = None
    if amp_l is not None and amp_r is not None:
        avg_amp = (amp_l + amp_r) / 2
        result["reduced_swing"] = avg_amp < 10

    return result


# ── Speed-normalized parameters ──────────────────────────────────────


def speed_normalized_params(
    data: dict,
    cycles: dict,
    height_m: float,
) -> dict:
    """Compute dimensionless speed-normalized gait parameters.

    Uses Froude number normalization (Hof 1996) to allow
    speed-independent comparison between individuals of different
    heights. The Froude number is:

        Fr = v^2 / (g * L)

    where v is walking speed, g is gravity, and L is leg length
    (estimated as 53% of body height).

    Parameters
    ----------
    data : dict
        Pivot JSON dict.
    cycles : dict
        Output of ``segment_cycles()``.
    height_m : float
        Subject height in meters (required).

    Returns
    -------
    dict
        Keys: ``froude_number``, ``dimensionless_speed``,
        ``dimensionless_cadence``, ``dimensionless_stride_length``,
        ``leg_length_m``.

    References
    ----------
    Hof AL. Scaling gait data to body size. Gait Posture.
    1996;4(3):222-223. doi:10.1016/0966-6362(95)01057-2
    """
    g = 9.81
    leg_length = height_m * 0.53

    ws = walking_speed(data, cycles, height_m)
    speed = ws.get("speed_mean")

    cycle_list = cycles.get("cycles", [])
    durations = [c["duration"] for c in cycle_list]
    stride_time = float(np.mean(durations)) if durations else None

    sl = step_length(data, cycles, height_m)

    result = {"leg_length_m": round(leg_length, 3)}

    if speed is not None and speed > 0:
        froude = speed ** 2 / (g * leg_length)
        result["froude_number"] = round(froude, 3)
        result["dimensionless_speed"] = round(speed / np.sqrt(g * leg_length), 3)
    else:
        result["froude_number"] = None
        result["dimensionless_speed"] = None

    if stride_time is not None and stride_time > 0:
        result["dimensionless_cadence"] = round(
            (1 / stride_time) * np.sqrt(leg_length / g), 3
        )
    else:
        result["dimensionless_cadence"] = None

    stride_l = sl.get("stride_length_left")
    stride_r = sl.get("stride_length_right")
    if stride_l is not None and stride_r is not None:
        avg_stride = (stride_l + stride_r) / 2
        result["dimensionless_stride_length"] = round(avg_stride / leg_length, 3)
    else:
        result["dimensionless_stride_length"] = None

    return result


# ── Clinical pattern detectors ───────────────────────────────────────


def detect_equinus(cycles: dict) -> dict:
    """Detect equinus gait pattern.

    Equinus is diagnosed when peak ankle dorsiflexion during
    stance phase (0-60% of gait cycle) is <= 0 deg. This indicates
    the ankle never reaches neutral, typical in spastic diplegic
    cerebral palsy or post-stroke.

    Parameters
    ----------
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``detected``, ``details`` (list of per-cycle dicts
        with ``side``, ``peak_dorsiflexion``, ``severity``).

    References
    ----------
    Rodda JM, Graham HK. Classification of gait patterns in
    spastic hemiplegia and spastic diplegia. Eur J Neurol.
    2001;8(Suppl 5):98-108. doi:10.1046/j.1468-1331.2001.00042.x
    """
    cycle_list = cycles.get("cycles", [])
    details = []

    for side in ("left", "right"):
        side_cycles = [c for c in cycle_list if c["side"] == side]
        ankle_peaks = []
        for c in side_cycles:
            vals = c.get("angles_normalized", {}).get("ankle")
            if vals:
                stance_vals = np.array(vals[:61])
                peak_df = float(np.max(stance_vals))
                ankle_peaks.append(peak_df)

        if ankle_peaks:
            mean_peak = float(np.mean(ankle_peaks))
            if mean_peak <= 0:
                severity = "severe" if mean_peak <= -10 else "moderate" if mean_peak <= -5 else "mild"
                details.append({
                    "side": side,
                    "peak_dorsiflexion": round(mean_peak, 1),
                    "severity": severity,
                })

    return {
        "detected": len(details) > 0,
        "details": details,
    }


def detect_antalgic(cycles: dict) -> dict:
    """Detect antalgic (pain-avoidance) gait pattern.

    Antalgic gait is characterized by asymmetric stance duration,
    with reduced stance time on the painful limb (< 55% stance
    on one side vs > 65% on the other).

    Parameters
    ----------
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``detected``, ``details`` (dict with ``short_side``,
        ``stance_left_pct``, ``stance_right_pct``, ``asymmetry``).

    References
    ----------
    Perry J, Burnfield JM. Gait Analysis: Normal and Pathological
    Function. 2nd ed. SLACK; 2010:163-177.
    """
    cycle_list = cycles.get("cycles", [])

    stance_l = [c["stance_pct"] for c in cycle_list
                if c["side"] == "left" and c["stance_pct"] is not None]
    stance_r = [c["stance_pct"] for c in cycle_list
                if c["side"] == "right" and c["stance_pct"] is not None]

    if not stance_l or not stance_r:
        return {"detected": False, "details": {}}

    mean_l = float(np.mean(stance_l))
    mean_r = float(np.mean(stance_r))
    asymmetry = abs(mean_l - mean_r)

    detected = False
    short_side = None
    if mean_l < 55 and mean_r > 60:
        detected = True
        short_side = "left"
    elif mean_r < 55 and mean_l > 60:
        detected = True
        short_side = "right"

    return {
        "detected": detected,
        "details": {
            "short_side": short_side,
            "stance_left_pct": round(mean_l, 1),
            "stance_right_pct": round(mean_r, 1),
            "asymmetry": round(asymmetry, 1),
        },
    }


def detect_parkinsonian(data: dict, cycles: dict) -> dict:
    """Detect parkinsonian gait features.

    Screens for a combination of:
    - Short stride length (reduced ankle excursion)
    - Reduced arm swing
    - Elevated cadence (festination)

    Two or more features -> suspected parkinsonian pattern.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and ``frames``.
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Keys: ``detected``, ``features`` (list of feature names),
        ``details`` (per-feature values).

    References
    ----------
    Morris ME, Iansek R, Matyas TA, Summers JJ. Stride length
    regulation in Parkinson's disease. Brain. 1996;119(Pt 2):
    551-568. doi:10.1093/brain/119.2.551

    Mirelman A, et al. Arm swing as a potential new prodromal
    marker of Parkinson's disease. Mov Disord. 2016;31(10):
    1527-1534.
    """
    features = []
    details = {}

    # 1. Short stride: reduced ankle x excursion
    frames = data.get("frames", [])
    ankle_x = [f.get("landmarks", {}).get("LEFT_ANKLE", {}).get("x")
               for f in frames]
    ankle_x_clean = [v for v in ankle_x if v is not None and not np.isnan(v)]
    if ankle_x_clean:
        ankle_excursion = float(np.ptp(ankle_x_clean))
        details["ankle_excursion"] = round(ankle_excursion, 4)
        if ankle_excursion < 0.08:
            features.append("short_stride")

    # 2. Reduced arm swing
    arm = arm_swing_analysis(data, cycles)
    if arm["amplitude_left"] is not None and arm["amplitude_right"] is not None:
        avg_amp = (arm["amplitude_left"] + arm["amplitude_right"]) / 2
        details["arm_swing_amplitude"] = round(avg_amp, 1)
        if avg_amp < 10:
            features.append("reduced_arm_swing")

    # 3. High cadence (festination)
    cycle_list = cycles.get("cycles", [])
    durations = [c["duration"] for c in cycle_list]
    if durations:
        stride_time = float(np.mean(durations))
        cadence = 60.0 / stride_time * 2 if stride_time > 0 else 0
        details["cadence"] = round(cadence, 1)
        if cadence > 130:
            features.append("festination")

    return {
        "detected": len(features) >= 2,
        "features": features,
        "details": details,
    }


# ── Segment lengths ──────────────────────────────────────────────────


DEFAULT_SEGMENTS = [
    ("LEFT_HIP", "LEFT_KNEE", "femur_L"),
    ("RIGHT_HIP", "RIGHT_KNEE", "femur_R"),
    ("LEFT_KNEE", "LEFT_ANKLE", "tibia_L"),
    ("RIGHT_KNEE", "RIGHT_ANKLE", "tibia_R"),
    ("LEFT_SHOULDER", "LEFT_ELBOW", "upper_arm_L"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "upper_arm_R"),
    ("LEFT_ELBOW", "LEFT_WRIST", "forearm_L"),
    ("RIGHT_ELBOW", "RIGHT_WRIST", "forearm_R"),
    ("LEFT_SHOULDER", "LEFT_HIP", "trunk_L"),
    ("RIGHT_SHOULDER", "RIGHT_HIP", "trunk_R"),
]


def segment_lengths(
    data: dict,
    segments: Optional[List] = None,
    unit: str = "normalized",
    height_m: Optional[float] = None,
) -> dict:
    """Compute segment lengths across all frames.

    For each defined segment (proximal-distal landmark pair), computes
    the Euclidean 2D distance at each frame, then reports summary
    statistics (mean, std, CV) and the full time series.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    segments : list of tuple, optional
        List of (proximal, distal, name) tuples. Defaults to
        ``DEFAULT_SEGMENTS``.
    unit : str, optional
        ``"normalized"`` (default) or ``"m"`` (requires *height_m*).
    height_m : float, optional
        Subject height in meters. When provided and unit is ``"m"``,
        lengths are scaled assuming the subject's extent in the image
        approximates *height_m*.

    Returns
    -------
    dict
        Per-segment dict with ``mean``, ``std``, ``cv``,
        ``time_series`` keys, plus a top-level ``quality_flags`` list
        of segment names with CV > 15%.
    """
    frames = data.get("frames", [])
    if segments is None:
        segments = DEFAULT_SEGMENTS

    result = {}
    quality_flags = []

    for proximal, distal, seg_name in segments:
        distances = []
        for f in frames:
            lm = f.get("landmarks", {})
            p = lm.get(proximal, {})
            d = lm.get(distal, {})
            px, py = p.get("x"), p.get("y")
            dx, dy = d.get("x"), d.get("y")
            if (px is not None and py is not None
                    and dx is not None and dy is not None
                    and not np.isnan(px) and not np.isnan(py)
                    and not np.isnan(dx) and not np.isnan(dy)):
                dist = np.sqrt((px - dx) ** 2 + (py - dy) ** 2)
                if height_m is not None and unit == "m":
                    dist = dist * height_m
                distances.append(float(dist))
            else:
                distances.append(float("nan"))

        valid = [v for v in distances if not np.isnan(v)]
        if valid:
            mean_val = float(np.mean(valid))
            std_val = float(np.std(valid))
            cv_val = float(std_val / mean_val * 100) if mean_val > 0 else 0.0
        else:
            mean_val = 0.0
            std_val = 0.0
            cv_val = 0.0

        result[seg_name] = {
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "cv": round(cv_val, 2),
            "time_series": distances,
        }

        if cv_val > 15.0:
            quality_flags.append(seg_name)

    result["quality_flags"] = quality_flags
    return result


# ── Instantaneous cadence ────────────────────────────────────────────


def instantaneous_cadence(data: dict) -> dict:
    """Compute instantaneous cadence from heel-strike intervals.

    Collects all heel-strike events (left and right), sorts them by
    frame, and computes the step time between consecutive heel strikes
    regardless of side. Instantaneous cadence is 60 / step_time for
    each pair.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``events`` containing ``left_hs`` and
        ``right_hs``.

    Returns
    -------
    dict
        Keys: ``times`` (list of midpoint times), ``cadence`` (list of
        instantaneous cadence values in steps/min), ``mean``, ``std``,
        ``cv``, ``trend_slope`` (linear regression slope).
    """
    events = data.get("events", {})
    fps = data.get("meta", {}).get("fps", 30.0)

    all_hs = []
    for ev in events.get("left_hs", []):
        all_hs.append(ev["frame"])
    for ev in events.get("right_hs", []):
        all_hs.append(ev["frame"])
    all_hs.sort()

    times = []
    cadences = []
    for i in range(len(all_hs) - 1):
        dt_frames = all_hs[i + 1] - all_hs[i]
        if dt_frames <= 0:
            continue
        step_time = dt_frames / fps
        cad = 60.0 / step_time
        mid_time = (all_hs[i] + all_hs[i + 1]) / 2.0 / fps
        times.append(float(mid_time))
        cadences.append(float(cad))

    if cadences:
        mean_cad = float(np.mean(cadences))
        std_cad = float(np.std(cadences))
        cv_cad = float(std_cad / mean_cad * 100) if mean_cad > 0 else 0.0
        # Linear regression for trend
        if len(times) >= 2:
            coeffs = np.polyfit(times, cadences, 1)
            trend_slope = float(coeffs[0])
        else:
            trend_slope = 0.0
    else:
        mean_cad = 0.0
        std_cad = 0.0
        cv_cad = 0.0
        trend_slope = 0.0

    return {
        "times": times,
        "cadence": cadences,
        "mean": round(mean_cad, 2),
        "std": round(std_cad, 2),
        "cv": round(cv_cad, 2),
        "trend_slope": round(trend_slope, 4),
    }


# ── ROM summary per cycle ───────────────────────────────────────────


def compute_rom_summary(data: dict, cycles: dict) -> dict:
    """Compute range-of-motion summary per joint per side per cycle.

    For each joint (hip, knee, ankle) and each side (L, R), extracts
    the normalized angle curve from each gait cycle and computes the
    ROM (max - min). Summary statistics (mean, std, CV) are computed
    across cycles.

    Parameters
    ----------
    data : dict
        Pivot JSON dict (used for context, not directly needed).
    cycles : dict
        Output of ``segment_cycles()``.

    Returns
    -------
    dict
        Per-joint/side dict, e.g. ``{"hip_L": {"rom_per_cycle": [...],
        "rom_mean": float, "rom_std": float, "rom_cv": float}, ...}``.
    """
    cycle_list = cycles.get("cycles", [])
    result = {}

    for joint in ("hip", "knee", "ankle"):
        for side_label, side_name in (("L", "left"), ("R", "right")):
            side_cycles = [c for c in cycle_list if c["side"] == side_name]
            roms = []
            for c in side_cycles:
                vals = c.get("angles_normalized", {}).get(joint)
                if vals:
                    roms.append(float(np.ptp(vals)))

            key = f"{joint}_{side_label}"
            if roms:
                rom_mean = float(np.mean(roms))
                rom_std = float(np.std(roms))
                rom_cv = float(rom_std / rom_mean * 100) if rom_mean > 0 else 0.0
            else:
                rom_mean = 0.0
                rom_std = 0.0
                rom_cv = 0.0

            result[key] = {
                "rom_per_cycle": [round(r, 2) for r in roms],
                "rom_mean": round(rom_mean, 2),
                "rom_std": round(rom_std, 2),
                "rom_cv": round(rom_cv, 2),
            }

    return result


# ── Center of mass estimation ────────────────────────────────────────


SEGMENT_COM_RATIOS = {  # ratio from proximal end
    "head": 0.5,  # approximation
    "trunk": 0.50,
    "upper_arm": 0.436,
    "forearm": 0.430,
    "thigh": 0.433,
    "shank": 0.433,
}

SEGMENT_MASS_RATIOS = {  # fraction of total body mass
    "head": 0.081,
    "trunk": 0.497,
    "upper_arm": 0.028,  # x2 for bilateral
    "forearm": 0.022,    # x2
    "thigh": 0.100,      # x2
    "shank": 0.047,      # x2
}


def estimate_center_of_mass(data: dict, model: str = "winter") -> dict:
    """Estimate whole-body center of mass using Winter's segment ratios.

    Uses the segmental analysis method with Winter (2009) body segment
    parameter tables. For each frame, computes each segment's CoM
    position from its proximal and distal landmarks, then computes a
    mass-weighted average.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    model : str, optional
        Body segment parameter model. Currently only ``"winter"``
        is supported.

    Returns
    -------
    dict
        Keys: ``com_x`` (list), ``com_y`` (list),
        ``vertical_excursion`` (peak-to-peak of com_y),
        ``smoothness`` (inverse of normalized jerk).
    """
    frames = data.get("frames", [])

    # Define segments: (proximal_landmark, distal_landmark, segment_name, bilateral_factor)
    segment_defs = [
        ("LEFT_SHOULDER", "LEFT_HIP", "trunk", 0.5),  # half for each side
        ("RIGHT_SHOULDER", "RIGHT_HIP", "trunk", 0.5),
        ("LEFT_SHOULDER", "LEFT_ELBOW", "upper_arm", 1.0),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW", "upper_arm", 1.0),
        ("LEFT_ELBOW", "LEFT_WRIST", "forearm", 1.0),
        ("RIGHT_ELBOW", "RIGHT_WRIST", "forearm", 1.0),
        ("LEFT_HIP", "LEFT_KNEE", "thigh", 1.0),
        ("RIGHT_HIP", "RIGHT_KNEE", "thigh", 1.0),
        ("LEFT_KNEE", "LEFT_ANKLE", "shank", 1.0),
        ("RIGHT_KNEE", "RIGHT_ANKLE", "shank", 1.0),
    ]

    com_x_list = []
    com_y_list = []

    for f in frames:
        lm = f.get("landmarks", {})
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for prox_name, dist_name, seg_type, bilateral_factor in segment_defs:
            prox = lm.get(prox_name, {})
            dist = lm.get(dist_name, {})
            px, py = prox.get("x"), prox.get("y")
            dx, dy = dist.get("x"), dist.get("y")

            if (px is None or py is None or dx is None or dy is None
                    or np.isnan(px) or np.isnan(py) or np.isnan(dx) or np.isnan(dy)):
                continue

            com_ratio = SEGMENT_COM_RATIOS[seg_type]
            mass_ratio = SEGMENT_MASS_RATIOS[seg_type] * bilateral_factor

            seg_com_x = px + com_ratio * (dx - px)
            seg_com_y = py + com_ratio * (dy - py)

            weighted_x += mass_ratio * seg_com_x
            weighted_y += mass_ratio * seg_com_y
            total_mass += mass_ratio

        # Add head approximation (midpoint of shoulders up to nose)
        nose = lm.get("NOSE", {})
        ls = lm.get("LEFT_SHOULDER", {})
        rs = lm.get("RIGHT_SHOULDER", {})
        if (nose.get("x") is not None and ls.get("x") is not None
                and rs.get("x") is not None):
            head_prox_x = (ls["x"] + rs["x"]) / 2
            head_prox_y = (ls["y"] + rs["y"]) / 2
            head_com_x = head_prox_x + SEGMENT_COM_RATIOS["head"] * (nose["x"] - head_prox_x)
            head_com_y = head_prox_y + SEGMENT_COM_RATIOS["head"] * (nose["y"] - head_prox_y)
            head_mass = SEGMENT_MASS_RATIOS["head"]
            weighted_x += head_mass * head_com_x
            weighted_y += head_mass * head_com_y
            total_mass += head_mass

        if total_mass > 0:
            com_x_list.append(float(weighted_x / total_mass))
            com_y_list.append(float(weighted_y / total_mass))
        else:
            com_x_list.append(float("nan"))
            com_y_list.append(float("nan"))

    # Vertical excursion
    valid_y = [v for v in com_y_list if not np.isnan(v)]
    vertical_excursion = float(np.ptp(valid_y)) if valid_y else 0.0

    # Smoothness: inverse of normalized jerk
    fps = data.get("meta", {}).get("fps", 30.0)
    smoothness = 0.0
    if len(valid_y) > 3:
        y_arr = np.array(com_y_list)
        valid_mask = ~np.isnan(y_arr)
        if valid_mask.sum() > 3:
            y_clean = y_arr[valid_mask]
            # Compute jerk (third derivative)
            vel = np.diff(y_clean) * fps
            acc = np.diff(vel) * fps
            jerk = np.diff(acc) * fps
            jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
            smoothness = float(1.0 / (1.0 + jerk_rms)) if jerk_rms >= 0 else 0.0

    return {
        "com_x": com_x_list,
        "com_y": com_y_list,
        "vertical_excursion": round(vertical_excursion, 6),
        "smoothness": round(smoothness, 6),
    }


# ── Postural sway ───────────────────────────────────────────────────


def postural_sway(
    data: dict,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> dict:
    """Compute postural sway metrics from ankle midpoint (COP approximation).

    Uses the midpoint of the two ankles as an approximation of the
    center of pressure (COP). Computes the 95% confidence ellipse area
    via eigenvalue decomposition of the 2D covariance matrix, mean
    sway velocity, and mediolateral (ML) and anteroposterior (AP)
    ranges.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    start_frame : int, optional
        First frame to include (default: 0).
    end_frame : int, optional
        Last frame to include (default: all frames).

    Returns
    -------
    dict
        Keys: ``cop_x``, ``cop_y`` (lists), ``ellipse_area``,
        ``sway_velocity``, ``ml_range``, ``ap_range``.
    """
    frames = data.get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(frames)

    cop_x = []
    cop_y = []

    for f in frames[start_frame:end_frame]:
        lm = f.get("landmarks", {})
        la = lm.get("LEFT_ANKLE", {})
        ra = lm.get("RIGHT_ANKLE", {})
        lx, ly = la.get("x"), la.get("y")
        rx, ry = ra.get("x"), ra.get("y")

        if (lx is not None and ly is not None
                and rx is not None and ry is not None
                and not np.isnan(lx) and not np.isnan(ly)
                and not np.isnan(rx) and not np.isnan(ry)):
            cop_x.append(float((lx + rx) / 2.0))
            cop_y.append(float((ly + ry) / 2.0))
        else:
            cop_x.append(float("nan"))
            cop_y.append(float("nan"))

    # Filter valid points
    valid_mask = [not (np.isnan(x) or np.isnan(y)) for x, y in zip(cop_x, cop_y)]
    vx = np.array([cop_x[i] for i in range(len(cop_x)) if valid_mask[i]])
    vy = np.array([cop_y[i] for i in range(len(cop_y)) if valid_mask[i]])

    if len(vx) < 3:
        return {
            "cop_x": cop_x, "cop_y": cop_y,
            "ellipse_area": 0.0, "sway_velocity": 0.0,
            "ml_range": 0.0, "ap_range": 0.0,
        }

    # ML = x direction, AP = y direction
    ml_range = float(np.ptp(vx))
    ap_range = float(np.ptp(vy))

    # 95% confidence ellipse via eigenvalue decomposition
    cov_matrix = np.cov(vx, vy)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # ensure non-negative
    # 95% confidence: chi-square with 2 dof, p=0.05 => 5.991
    ellipse_area = float(np.pi * 5.991 * np.sqrt(eigenvalues[0] * eigenvalues[1]))

    # Mean sway velocity
    displacements = np.sqrt(np.diff(vx) ** 2 + np.diff(vy) ** 2)
    total_path = float(np.sum(displacements))
    duration = len(vx) / fps
    sway_velocity = total_path / duration if duration > 0 else 0.0

    return {
        "cop_x": cop_x,
        "cop_y": cop_y,
        "ellipse_area": round(ellipse_area, 8),
        "sway_velocity": round(sway_velocity, 6),
        "ml_range": round(ml_range, 6),
        "ap_range": round(ap_range, 6),
    }


# ── PCA waveform analysis ───────────────────────────────────────────


def pca_waveform_analysis(
    cycles: dict,
    joints: list = None,
    n_components: int = 3,
    n_points: int = 101,
) -> dict:
    """Principal component analysis of gait waveforms.

    Performs PCA on time-normalized joint angle waveforms across gait
    cycles. Extracts principal movement patterns (eigenvectors) and
    scores that quantify each cycle's deviation along those patterns.

    Parameters
    ----------
    cycles : dict
        Output of segment_cycles(), containing ``cycles["cycles"]``
        list where each cycle has ``angles`` dict with joint arrays.
    joints : list of str, optional
        Joint names to analyze. Defaults to ["hip_L", "knee_L", "ankle_L"].
    n_components : int
        Number of principal components to retain. Default 3.
    n_points : int
        Number of points for time normalization. Default 101 (0-100% gait cycle).

    Returns
    -------
    dict
        Per-joint results: {joint_name: {
            "mean": np.ndarray (n_points,),
            "components": np.ndarray (n_components, n_points),
            "explained_variance_ratio": np.ndarray (n_components,),
            "scores": np.ndarray (n_cycles, n_components),
            "n_cycles_used": int
        }}

    Raises
    ------
    ValueError
        If fewer than 3 valid cycles are available for any requested joint.

    References
    ----------
    Deluzio KJ, Astephen JL. Biomechanical features of gait waveform
    data associated with knee osteoarthritis. Gait Posture.
    2007;25(1):86-93. doi:10.1016/j.gaitpost.2006.01.007
    Federolf P, Boyer K, Andriacchi TP. Application of principal
    component analysis in clinical gait research. J Biomech.
    2013;46(15):2549-2555. doi:10.1016/j.jbiomech.2013.07.014
    """
    if joints is None:
        joints = ["hip_L", "knee_L", "ankle_L"]

    cycle_list = cycles.get("cycles", [])
    results = {}

    for joint in joints:
        # Collect valid waveforms for this joint
        waveforms = []
        for c in cycle_list:
            angles = c.get("angles", {})
            waveform = angles.get(joint)
            if waveform is None:
                continue
            # Skip all-None waveforms
            if all(v is None for v in waveform):
                continue
            waveforms.append(waveform)

        if len(waveforms) < 3:
            raise ValueError(
                f"Need at least 3 valid cycles for PCA on '{joint}', "
                f"got {len(waveforms)}."
            )

        # Time-normalize each waveform to n_points using linear interpolation
        normalized = np.empty((len(waveforms), n_points))
        x_out = np.linspace(0, 1, n_points)
        for i, wf in enumerate(waveforms):
            x_in = np.linspace(0, 1, len(wf))
            normalized[i, :] = np.interp(x_out, x_in, wf)

        # Compute mean waveform and center the data
        mean_waveform = np.mean(normalized, axis=0)
        centered = normalized - mean_waveform

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Clamp n_components to available singular values
        k = min(n_components, len(S))

        # Principal components (rows of Vt)
        components = Vt[:k, :]

        # Scores: project centered data onto principal components
        scores = centered @ components.T

        # Explained variance ratio
        total_var = np.sum(S ** 2)
        if total_var > np.finfo(float).eps:
            explained_variance_ratio = S[:k] ** 2 / total_var
        else:
            explained_variance_ratio = np.zeros(k)

        results[joint] = {
            "mean": mean_waveform,
            "components": components,
            "explained_variance_ratio": explained_variance_ratio,
            "scores": scores,
            "n_cycles_used": len(waveforms),
        }

    return results


# ── Angular derivatives ─────────────────────────────────────────────

_DEFAULT_DERIVATIVE_JOINTS = [
    "hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R",
]


def compute_derivatives(
    data: dict,
    joints: list = None,
    max_order: int = 2,
) -> dict:
    """Compute angular velocity and acceleration via central differences.

    Calculates time derivatives of joint angle waveforms using
    finite central differences. First derivative gives angular velocity
    (deg/s), second derivative gives angular acceleration (deg/s²).

    Parameters
    ----------
    data : dict
        myogait data dict with ``data["angles"]["frames"]`` populated
        and ``data["meta"]["fps"]`` available.
    joints : list of str, optional
        Joint names to compute derivatives for.
        Defaults to ["hip_L", "hip_R", "knee_L", "knee_R",
                      "ankle_L", "ankle_R"].
    max_order : int
        Maximum derivative order (1 or 2). Default 2.

    Returns
    -------
    dict
        ``data["derivatives"]``: dict with keys per joint, each containing
        "velocity" (np.ndarray) and "acceleration" (np.ndarray, if max_order>=2).
        Units: deg/s and deg/s².

    Raises
    ------
    ValueError
        If ``data["angles"]["frames"]`` is missing or empty.

    References
    ----------
    Winter DA. Biomechanics and Motor Control of Human Movement.
    4th ed. Wiley; 2009. Chapter 2.
    """
    # ── validate input ────────────────────────────────────────────
    angles = data.get("angles")
    if angles is None or not angles.get("frames"):
        raise ValueError(
            "data['angles']['frames'] must be populated before "
            "computing derivatives."
        )
    angle_frames = angles["frames"]

    fps = data.get("meta", {}).get("fps", 30.0)
    dt = 1.0 / fps

    if joints is None:
        joints = list(_DEFAULT_DERIVATIVE_JOINTS)

    # ── compute derivatives per joint ─────────────────────────────
    derivatives: Dict[str, dict] = {}

    for joint in joints:
        # Extract angle time series, preserving None/NaN as NaN
        raw = []
        for af in angle_frames:
            val = af.get(joint)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raw.append(np.nan)
            else:
                raw.append(float(val))

        angles_arr = np.array(raw, dtype=np.float64)

        # 1st derivative: angular velocity (deg/s)
        velocity = np.gradient(angles_arr, dt)

        joint_result: Dict[str, Any] = {"velocity": velocity}

        # 2nd derivative: angular acceleration (deg/s²)
        if max_order >= 2:
            acceleration = np.gradient(velocity, dt)
            joint_result["acceleration"] = acceleration

        derivatives[joint] = joint_result

    # ── store and return ──────────────────────────────────────────
    data["derivatives"] = derivatives
    return data["derivatives"]


# ── Time-frequency analysis ──────────────────────────────────────────


def time_frequency_analysis(
    data: dict,
    joints: list = None,
    method: str = "cwt",
    freq_range: tuple = (0.5, 15.0),
    n_freqs: int = 50,
) -> dict:
    """Time-frequency analysis of gait angle signals.

    Computes the time-frequency representation of joint angle
    waveforms using continuous wavelet transform (CWT) or
    short-time Fourier transform (STFT).

    Parameters
    ----------
    data : dict
        myogait data dict with ``data["angles"]["frames"]`` and
        ``data["meta"]["fps"]``.
    joints : list of str, optional
        Joint names. Defaults to ["hip_L", "knee_L", "ankle_L"].
    method : str
        "cwt" (continuous wavelet transform using Morlet wavelet)
        or "stft" (short-time Fourier transform). Default "cwt".
    freq_range : tuple
        (min_freq, max_freq) in Hz. Default (0.5, 15.0).
    n_freqs : int
        Number of frequency bins. Default 50.

    Returns
    -------
    dict
        Per-joint results: {joint_name: {
            "power": np.ndarray (n_freqs, n_times),
            "frequencies": np.ndarray (n_freqs,),
            "times": np.ndarray (n_times,),
            "dominant_frequency": float,
            "method": str,
        }}

    References
    ----------
    Ismail AR, Asfour SS. Continuous wavelet transform application
    to EMG signals during human gait. Conf Rec IEEE Eng Med Biol Soc.
    1999.
    """
    if joints is None:
        joints = ["hip_L", "knee_L", "ankle_L"]

    angle_frames = data.get("angles", {}).get("frames", [])
    fps = data.get("meta", {}).get("fps", 30.0)
    n_frames = len(angle_frames)

    results = {}

    for joint in joints:
        # Extract angle values for this joint
        raw = []
        for af in angle_frames:
            val = af.get(joint)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raw.append(np.nan)
            else:
                raw.append(float(val))

        signal = np.array(raw, dtype=np.float64)

        # Interpolate NaN values
        valid_mask = ~np.isnan(signal)
        if valid_mask.sum() > 1:
            x_idx = np.arange(len(signal))
            signal[~valid_mask] = np.interp(
                x_idx[~valid_mask], x_idx[valid_mask], signal[valid_mask]
            )
        elif valid_mask.sum() <= 1:
            # Not enough valid data; fill with zeros
            signal = np.zeros_like(signal)

        # Remove mean to focus on oscillatory content
        signal = signal - np.mean(signal)

        times = np.arange(n_frames) / fps
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freqs)

        if method == "cwt":
            power = _cwt_morlet(signal, fs=fps, frequencies=frequencies)
        elif method == "stft":
            power, frequencies, times = _stft_analysis(
                signal, fs=fps, freq_range=freq_range, n_freqs=n_freqs,
                n_frames=n_frames,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cwt' or 'stft'.")

        # Dominant frequency: frequency with highest total power
        total_power_per_freq = np.sum(power, axis=1)
        dominant_idx = int(np.argmax(total_power_per_freq))
        dominant_frequency = float(frequencies[dominant_idx])

        results[joint] = {
            "power": power,
            "frequencies": frequencies,
            "times": times,
            "dominant_frequency": dominant_frequency,
            "method": method,
        }

    return results


def _cwt_morlet(
    signal: np.ndarray,
    fs: float,
    frequencies: np.ndarray,
    w0: float = 5.0,
) -> np.ndarray:
    """Compute CWT using Morlet wavelet via FFT convolution.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    frequencies : np.ndarray
        Array of frequencies at which to evaluate the CWT.
    w0 : float
        Central frequency parameter of the Morlet wavelet (default 5.0).

    Returns
    -------
    np.ndarray
        Power matrix of shape (len(frequencies), len(signal)).
    """
    n = len(signal)
    # Zero-pad for efficient FFT convolution
    N = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    signal_fft = np.fft.fft(signal, N)
    angular_freqs = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / fs)

    power = np.empty((len(frequencies), n))
    for i, freq in enumerate(frequencies):
        scale = w0 * fs / (2 * np.pi * freq)
        # Morlet wavelet in frequency domain (analytic)
        norm = (np.pi ** -0.25) * np.sqrt(2 * np.pi * scale / fs)
        wavelet_fft = norm * np.exp(
            -0.5 * (scale * angular_freqs / fs - w0) ** 2
        )
        # Keep only positive frequencies for analytic wavelet
        wavelet_fft[angular_freqs < 0] = 0
        wavelet_fft *= 2
        # Convolution in frequency domain
        coeff_fft = signal_fft * np.conj(wavelet_fft)
        coeff = np.fft.ifft(coeff_fft)[:n]
        power[i, :] = np.abs(coeff) ** 2

    return power


def _stft_analysis(
    signal: np.ndarray,
    fs: float,
    freq_range: tuple,
    n_freqs: int,
    n_frames: int,
) -> tuple:
    """Compute STFT-based time-frequency representation.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    freq_range : tuple
        (min_freq, max_freq) in Hz.
    n_freqs : int
        Desired number of frequency bins.
    n_frames : int
        Original number of frames (for time axis reference).

    Returns
    -------
    tuple
        (power, frequencies, times) where power has shape
        (n_freq_bins, n_time_bins).
    """
    from scipy.signal import stft as scipy_stft

    # Choose nperseg: try to get reasonable time-frequency resolution
    nperseg = min(len(signal), max(16, 2 ** int(np.ceil(np.log2(len(signal) // 4)))))
    noverlap = nperseg // 2

    f_stft, t_stft, Zxx = scipy_stft(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
    )

    # Compute power from complex STFT
    stft_power = np.abs(Zxx) ** 2

    # Restrict to freq_range
    freq_mask = (f_stft >= freq_range[0]) & (f_stft <= freq_range[1])
    if not np.any(freq_mask):
        # If no frequencies in range, return all
        freq_mask = np.ones(len(f_stft), dtype=bool)

    frequencies = f_stft[freq_mask]
    power = stft_power[freq_mask, :]
    times = t_stft

    return power, frequencies, times
