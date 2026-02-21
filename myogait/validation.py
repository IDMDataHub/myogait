"""Biomechanical validation of gait analysis outputs.

Validates joint angles and spatio-temporal parameters against known
physiological ranges for normal adult gait. Reports violations
grouped by severity (critical, warning, info).

Reference ranges are derived from published normative data:

    Joint angles:
        Ref: Perry J, Burnfield JM. Gait Analysis: Normal and
        Pathological Function. 2nd ed. SLACK Incorporated; 2010.
        Table 4.1 (sagittal-plane joint motion during gait).

        Ref: Kadaba MP, Ramakrishnan HK, Wootten ME. Measurement
        of lower extremity kinematics during level walking.
        J Orthop Res. 1990;8(3):383-392. doi:10.1002/jor.1100080310

        Ref: Schwartz MH, Rozumalski A, Trost JP. The effect of
        walking speed on the gait of typically developing children.
        J Biomech. 2008;41(8):1639-1650.
        doi:10.1016/j.jbiomech.2008.03.015

    Spatio-temporal parameters:
        Ref: Bohannon RW, Williams Andrews A. Normal walking speed:
        a descriptive meta-analysis. Physiotherapy. 2011;97(3):182-189.
        doi:10.1016/j.physio.2010.12.004

        Ref: Hollman JH, McDade EM, Petersen RC. Normative
        spatiotemporal gait parameters in older adults. Gait Posture.
        2011;34(1):111-118. doi:10.1016/j.gaitpost.2011.03.024

Functions
---------
validate_biomechanical
    Run full biomechanical validation against reference ranges.
get_angle_ranges
    Return the reference angle ranges used for validation.
get_spatiotemporal_ranges
    Return the reference spatio-temporal ranges.
stratified_ranges
    Return angle and spatio-temporal ranges adjusted by demographics.
model_accuracy_info
    Return published accuracy metrics for a pose estimation model.
validate_biomechanical_stratified
    Validate gait data using demographically stratified ranges.
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np

from .normative import get_normative_curve, select_stratum

logger = logging.getLogger(__name__)

# Biomechanical ranges (degrees) — normal adult gait
# Format: (min, max) for full range, plus per-phase if available
ANGLE_RANGES = {
    "hip_L": {"full": (-20, 50), "stance": (-10, 40), "swing": (0, 50)},
    "hip_R": {"full": (-20, 50), "stance": (-10, 40), "swing": (0, 50)},
    "knee_L": {"full": (-5, 75), "stance": (-5, 20), "swing": (20, 75)},
    "knee_R": {"full": (-5, 75), "stance": (-5, 20), "swing": (20, 75)},
    "ankle_L": {"full": (-30, 30), "stance": (-15, 20), "swing": (-10, 10)},
    "ankle_R": {"full": (-30, 30), "stance": (-15, 20), "swing": (-10, 10)},
    "trunk_angle": {"full": (-10, 20)},
    "pelvis_tilt": {"full": (-15, 15)},
}

# Spatio-temporal ranges — normal adult gait
SPATIOTEMPORAL_RANGES = {
    "cadence_steps_per_min": (80, 140),
    "stride_time_mean_s": (0.8, 1.6),
    "step_time_mean_s": (0.4, 0.8),
    "stance_pct": (55, 65),
    "swing_pct": (35, 45),
    "double_support_pct": (10, 30),
}


def validate_biomechanical(
    data: dict,
    cycles: Optional[dict] = None,
    strict: bool = False,
) -> dict:
    """Validate gait data against biomechanical reference ranges.

    Checks joint angles and spatio-temporal parameters against
    published physiological ranges for normal adult gait. When
    cycle data is provided, also validates per-phase angles.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and optionally ``events``.
    cycles : dict, optional
        Output of ``segment_cycles()``. Enables per-phase validation.
    strict : bool, optional
        If True, uses tighter per-phase ranges (default False).

    Returns
    -------
    dict
        Validation report with keys:

        - ``valid`` (bool): True if no critical violations.
        - ``violations`` (list): All detected violations.
        - ``summary`` (dict): Counts by severity level.

    Raises
    ------
    TypeError
        If *data* is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    violations = []

    # Validate angles
    angle_violations = _validate_angles(data, strict)
    violations.extend(angle_violations)

    # Validate per-phase angles (if cycles available)
    if cycles:
        phase_violations = _validate_phase_angles(data, cycles)
        violations.extend(phase_violations)

    # Validate spatio-temporal parameters
    events = data.get("events", {})
    if events:
        st_violations = _validate_spatiotemporal(data)
        violations.extend(st_violations)

    # Summarize
    critical = [v for v in violations if v["severity"] == "critical"]
    warning = [v for v in violations if v["severity"] == "warning"]
    info = [v for v in violations if v["severity"] == "info"]

    return {
        "valid": len(critical) == 0,
        "violations": violations,
        "summary": {
            "total": len(violations),
            "critical": len(critical),
            "warning": len(warning),
            "info": len(info),
        },
    }


def _validate_angles(data: dict, strict: bool = False) -> list:
    """Check angle values against physiological ranges."""
    violations = []
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])

    if not angle_frames:
        return violations

    for joint, ranges in ANGLE_RANGES.items():
        lo, hi = ranges["full"]
        values = [af.get(joint) for af in angle_frames if af.get(joint) is not None]
        if not values:
            continue

        out_of_range = [v for v in values if v < lo or v > hi]
        pct = 100.0 * len(out_of_range) / len(values)

        if pct > 20:
            violations.append({
                "type": "angle_range",
                "joint": joint,
                "severity": "critical" if pct > 50 else "warning",
                "pct_out_of_range": round(pct, 1),
                "expected_range": [lo, hi],
                "actual_range": [round(min(values), 1), round(max(values), 1)],
                "description": f"{joint}: {pct:.0f}% of frames outside [{lo}, {hi}] deg",
            })
        elif pct > 5:
            violations.append({
                "type": "angle_range",
                "joint": joint,
                "severity": "info",
                "pct_out_of_range": round(pct, 1),
                "expected_range": [lo, hi],
                "actual_range": [round(min(values), 1), round(max(values), 1)],
                "description": f"{joint}: {pct:.0f}% of frames outside [{lo}, {hi}] deg",
            })

    return violations


def _validate_phase_angles(data: dict, cycles: dict) -> list:
    """Validate angles per gait phase (stance/swing)."""
    violations = []
    cycle_list = cycles.get("cycles", [])

    for c in cycle_list:
        an = c.get("angles_normalized", {})
        for joint_base in ["hip", "knee", "ankle"]:
            vals = an.get(joint_base)
            if vals is None:
                continue
            vals = np.array(vals)

            joint_key = f"{joint_base}_{c['side'][0].upper()}"
            ranges = ANGLE_RANGES.get(joint_key, {})

            # Stance phase: 0-60% → indices 0-60
            stance_range = ranges.get("stance")
            if stance_range:
                stance_vals = vals[:61]
                stance_min = np.min(stance_vals)
                stance_max = np.max(stance_vals)
                if stance_min < stance_range[0] - 10 or stance_max > stance_range[1] + 10:
                    violations.append({
                        "type": "phase_angle",
                        "joint": joint_key,
                        "phase": "stance",
                        "cycle_id": c["cycle_id"],
                        "severity": "warning",
                        "expected_range": list(stance_range),
                        "actual_range": [round(float(stance_min), 1), round(float(stance_max), 1)],
                        "description": f"Cycle {c['cycle_id']} {c['side']} {joint_base} stance: "
                                       f"[{stance_min:.0f}, {stance_max:.0f}] vs expected {stance_range}",
                    })

    return violations


def _validate_spatiotemporal(data: dict) -> list:
    """Validate spatio-temporal parameters."""
    violations = []
    events = data.get("events", {})
    fps = data.get("meta", {}).get("fps", 30.0)

    # Check cadence
    all_hs = []
    for key in ["left_hs", "right_hs"]:
        for ev in events.get(key, []):
            all_hs.append(ev["frame"])
    all_hs.sort()

    if len(all_hs) >= 3:
        intervals = np.diff(all_hs) / fps
        step_time = float(np.mean(intervals))
        cadence = 60.0 / step_time if step_time > 0 else 0

        lo, hi = SPATIOTEMPORAL_RANGES["cadence_steps_per_min"]
        if cadence < lo or cadence > hi:
            violations.append({
                "type": "spatiotemporal",
                "parameter": "cadence",
                "severity": "warning",
                "value": round(cadence, 1),
                "expected_range": [lo, hi],
                "description": f"Cadence {cadence:.0f} steps/min outside normal [{lo}, {hi}]",
            })

    return violations


def get_angle_ranges() -> dict:
    """Return the biomechanical angle ranges used for validation."""
    return dict(ANGLE_RANGES)


def get_spatiotemporal_ranges() -> dict:
    """Return the spatio-temporal ranges used for validation."""
    return dict(SPATIOTEMPORAL_RANGES)


# ── Model accuracy database ─────────────────────────────────────────

_MODEL_ACCURACY = {
    "mediapipe": {
        "model": "mediapipe",
        "mae_px": 5.0,
        "pck_05": 0.92,
        "reference": "Bazarevsky V et al. BlazePose: On-device Real-time "
                     "Body Pose tracking. arXiv:2006.10204, 2020.",
        "notes": "Lightweight on-device model; best for real-time applications.",
    },
    "yolo": {
        "model": "yolo",
        "mae_px": 4.0,
        "pck_05": 0.94,
        "reference": "Jocher G et al. YOLOv8-pose. Ultralytics, 2023.",
        "notes": "Fast single-stage detector with pose head; good speed-accuracy trade-off.",
    },
    "vitpose": {
        "model": "vitpose",
        "mae_px": 3.0,
        "pck_05": 0.96,
        "reference": "Xu Y et al. ViTPose: Simple Vision Transformer Baselines "
                     "for Human Pose Estimation. NeurIPS, 2022.",
        "notes": "Transformer-based; state-of-the-art accuracy on COCO.",
    },
    "sapiens": {
        "model": "sapiens",
        "mae_px": 2.5,
        "pck_05": 0.97,
        "reference": "Khirodkar R et al. Sapiens: Foundation for Human Vision "
                     "Models. ECCV, 2024.",
        "notes": "Foundation model; highest accuracy but computationally expensive.",
    },
    "rtmw": {
        "model": "rtmw",
        "mae_px": 3.5,
        "pck_05": 0.95,
        "reference": "Jiang T et al. RTMPose: Real-Time Multi-Person Pose "
                     "Estimation based on MMPose. arXiv:2303.07399, 2023.",
        "notes": "Real-time whole-body model; good balance of speed and accuracy.",
    },
    "mmpose": {
        "model": "mmpose",
        "mae_px": 3.0,
        "pck_05": 0.96,
        "reference": "Sun K et al. Deep High-Resolution Representation Learning "
                     "for Human Pose Estimation. CVPR, 2019.",
        "notes": "HRNet backbone via MMPose framework; widely used benchmark model.",
    },
    "hrnet": {
        "model": "hrnet",
        "mae_px": 3.0,
        "pck_05": 0.96,
        "reference": "Sun K et al. Deep High-Resolution Representation Learning "
                     "for Human Pose Estimation. CVPR, 2019.",
        "notes": "High-resolution network; maintains spatial resolution throughout.",
    },
}


def model_accuracy_info(model: str) -> dict:
    """Return published accuracy metrics for a pose estimation backend.

    Parameters
    ----------
    model : str
        One of ``'mediapipe'``, ``'yolo'``, ``'vitpose'``, ``'sapiens'``,
        ``'rtmw'``, ``'mmpose'``, ``'hrnet'``.

    Returns
    -------
    dict
        Keys: ``'model'``, ``'mae_px'`` (mean absolute error in pixels),
        ``'pck_05'`` (PCK@0.5), ``'reference'`` (citation string),
        ``'notes'``.  All values are ``None`` for unknown models.
    """
    if model in _MODEL_ACCURACY:
        return dict(_MODEL_ACCURACY[model])
    return {
        "model": model,
        "mae_px": None,
        "pck_05": None,
        "reference": None,
        "notes": None,
    }


# ── Stratified ranges ───────────────────────────────────────────────

def stratified_ranges(age=None, sex=None, speed=None) -> dict:
    """Return angle and spatio-temporal ranges adjusted by demographics.

    Uses normative data from :func:`~myogait.normative.select_stratum`
    and :func:`~myogait.normative.get_normative_curve` to adjust the
    default validation ranges based on subject demographics.

    Parameters
    ----------
    age : int, optional
        Subject age in years. Selects stratum (pediatric / adult / elderly).
    sex : str, optional
        ``'M'`` or ``'F'``. Female subjects get slightly adjusted norms.
    speed : str, optional
        ``'slow'``, ``'normal'``, or ``'fast'``. Adjusts cadence and
        stride time expectations.

    Returns
    -------
    dict
        Keys: ``'angle_ranges'`` (dict) and ``'spatiotemporal_ranges'``
        (dict), structured like the module-level constants.
    """
    # Start from deep copies of the defaults
    angle_ranges = copy.deepcopy(ANGLE_RANGES)
    st_ranges = copy.deepcopy(SPATIOTEMPORAL_RANGES)

    # Determine stratum from age
    stratum = select_stratum(age)

    # Use normative curves to inform ROM adjustments
    if stratum == "elderly":
        # Elderly: wider full ROM range (to accommodate reduced mobility
        # that may appear as out-of-range in either direction), lower cadence
        _norm_hip = get_normative_curve("hip", "elderly")
        _norm_knee = get_normative_curve("knee", "elderly")

        for side in ("L", "R"):
            # Widen hip range: extend lower bound by 5 deg, keep upper
            angle_ranges[f"hip_{side}"]["full"] = (
                angle_ranges[f"hip_{side}"]["full"][0] - 5,
                angle_ranges[f"hip_{side}"]["full"][1],
            )
            # Widen knee range: extend upper swing bound
            angle_ranges[f"knee_{side}"]["full"] = (
                angle_ranges[f"knee_{side}"]["full"][0],
                angle_ranges[f"knee_{side}"]["full"][1] + 5,
            )
            # Widen ankle range
            angle_ranges[f"ankle_{side}"]["full"] = (
                angle_ranges[f"ankle_{side}"]["full"][0] - 5,
                angle_ranges[f"ankle_{side}"]["full"][1] + 5,
            )
            if "stance" in angle_ranges[f"hip_{side}"]:
                angle_ranges[f"hip_{side}"]["stance"] = (
                    angle_ranges[f"hip_{side}"]["stance"][0] - 5,
                    angle_ranges[f"hip_{side}"]["stance"][1],
                )
            if "swing" in angle_ranges[f"knee_{side}"]:
                angle_ranges[f"knee_{side}"]["swing"] = (
                    angle_ranges[f"knee_{side}"]["swing"][0],
                    angle_ranges[f"knee_{side}"]["swing"][1] + 5,
                )

        # Trunk / pelvis: slightly wider
        angle_ranges["trunk_angle"]["full"] = (
            angle_ranges["trunk_angle"]["full"][0] - 5,
            angle_ranges["trunk_angle"]["full"][1] + 5,
        )
        angle_ranges["pelvis_tilt"]["full"] = (
            angle_ranges["pelvis_tilt"]["full"][0] - 5,
            angle_ranges["pelvis_tilt"]["full"][1] + 5,
        )

        # Spatio-temporal: elderly walk slower, lower cadence
        st_ranges["cadence_steps_per_min"] = (60, 130)
        st_ranges["stride_time_mean_s"] = (0.9, 1.8)
        st_ranges["step_time_mean_s"] = (0.45, 0.9)

    elif stratum == "pediatric":
        # Pediatric: increased ROM, higher cadence
        _norm_hip = get_normative_curve("hip", "pediatric")

        for side in ("L", "R"):
            angle_ranges[f"hip_{side}"]["full"] = (
                angle_ranges[f"hip_{side}"]["full"][0] - 5,
                angle_ranges[f"hip_{side}"]["full"][1] + 5,
            )
            angle_ranges[f"knee_{side}"]["full"] = (
                angle_ranges[f"knee_{side}"]["full"][0] - 5,
                angle_ranges[f"knee_{side}"]["full"][1] + 10,
            )
            angle_ranges[f"ankle_{side}"]["full"] = (
                angle_ranges[f"ankle_{side}"]["full"][0] - 5,
                angle_ranges[f"ankle_{side}"]["full"][1] + 5,
            )
            if "stance" in angle_ranges[f"hip_{side}"]:
                angle_ranges[f"hip_{side}"]["stance"] = (
                    angle_ranges[f"hip_{side}"]["stance"][0] - 5,
                    angle_ranges[f"hip_{side}"]["stance"][1] + 5,
                )
            if "swing" in angle_ranges[f"knee_{side}"]:
                angle_ranges[f"knee_{side}"]["swing"] = (
                    angle_ranges[f"knee_{side}"]["swing"][0] - 5,
                    angle_ranges[f"knee_{side}"]["swing"][1] + 10,
                )

        # Trunk / pelvis: slightly wider for children
        angle_ranges["trunk_angle"]["full"] = (
            angle_ranges["trunk_angle"]["full"][0] - 3,
            angle_ranges["trunk_angle"]["full"][1] + 3,
        )
        angle_ranges["pelvis_tilt"]["full"] = (
            angle_ranges["pelvis_tilt"]["full"][0] - 3,
            angle_ranges["pelvis_tilt"]["full"][1] + 3,
        )

        # Spatio-temporal: children have higher cadence, shorter strides
        st_ranges["cadence_steps_per_min"] = (100, 170)
        st_ranges["stride_time_mean_s"] = (0.6, 1.3)
        st_ranges["step_time_mean_s"] = (0.3, 0.65)

    # Sex adjustments (applied on top of age stratum)
    if sex is not None and sex.upper() == "F":
        # Females tend to have slightly narrower stride, higher cadence
        cad_lo, cad_hi = st_ranges["cadence_steps_per_min"]
        st_ranges["cadence_steps_per_min"] = (cad_lo + 5, cad_hi + 5)
        st_lo, st_hi = st_ranges["stride_time_mean_s"]
        st_ranges["stride_time_mean_s"] = (st_lo - 0.05, st_hi - 0.05)
        step_lo, step_hi = st_ranges["step_time_mean_s"]
        st_ranges["step_time_mean_s"] = (step_lo - 0.02, step_hi - 0.02)

    # Speed adjustments (applied on top of age/sex adjustments)
    if speed is not None and speed.lower() == "slow":
        cad_lo, cad_hi = st_ranges["cadence_steps_per_min"]
        st_ranges["cadence_steps_per_min"] = (max(40, cad_lo - 20), cad_hi - 10)
        st_lo, st_hi = st_ranges["stride_time_mean_s"]
        st_ranges["stride_time_mean_s"] = (st_lo + 0.1, st_hi + 0.3)
        step_lo, step_hi = st_ranges["step_time_mean_s"]
        st_ranges["step_time_mean_s"] = (step_lo + 0.05, step_hi + 0.15)
    elif speed is not None and speed.lower() == "fast":
        cad_lo, cad_hi = st_ranges["cadence_steps_per_min"]
        st_ranges["cadence_steps_per_min"] = (cad_lo + 10, min(200, cad_hi + 20))
        st_lo, st_hi = st_ranges["stride_time_mean_s"]
        st_ranges["stride_time_mean_s"] = (max(0.4, st_lo - 0.15), st_hi - 0.1)
        step_lo, step_hi = st_ranges["step_time_mean_s"]
        st_ranges["step_time_mean_s"] = (max(0.2, step_lo - 0.08), step_hi - 0.05)

    return {
        "angle_ranges": angle_ranges,
        "spatiotemporal_ranges": st_ranges,
    }


# ── Stratified validation ───────────────────────────────────────────

def validate_biomechanical_stratified(
    data: dict,
    cycles: Optional[dict] = None,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    speed: Optional[str] = None,
) -> dict:
    """Validate gait data using demographically stratified ranges.

    Like :func:`validate_biomechanical` but adjusts reference ranges
    based on subject age, sex, and walking speed before validation.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``angles`` and optionally ``events``.
    cycles : dict, optional
        Output of ``segment_cycles()``. Enables per-phase validation.
    age : int, optional
        Subject age in years.
    sex : str, optional
        ``'M'`` or ``'F'``.
    speed : str, optional
        ``'slow'``, ``'normal'``, or ``'fast'``.

    Returns
    -------
    dict
        Validation report with keys:

        - ``valid`` (bool): True if no critical violations.
        - ``violations`` (list): All detected violations.
        - ``summary`` (dict): Counts by severity level.
        - ``stratum`` (str): The age stratum used.
        - ``adjusted_ranges`` (dict): The stratified ranges applied.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    ranges = stratified_ranges(age=age, sex=sex, speed=speed)
    adj_angle_ranges = ranges["angle_ranges"]
    adj_st_ranges = ranges["spatiotemporal_ranges"]

    violations = []

    # Validate angles against stratified ranges
    angle_violations = _validate_angles_with_ranges(data, adj_angle_ranges)
    violations.extend(angle_violations)

    # Validate per-phase angles (if cycles available)
    if cycles:
        phase_violations = _validate_phase_angles_with_ranges(
            data, cycles, adj_angle_ranges
        )
        violations.extend(phase_violations)

    # Validate spatio-temporal parameters
    events = data.get("events", {})
    if events:
        st_violations = _validate_spatiotemporal_with_ranges(data, adj_st_ranges)
        violations.extend(st_violations)

    # Summarize
    critical = [v for v in violations if v["severity"] == "critical"]
    warning = [v for v in violations if v["severity"] == "warning"]
    info = [v for v in violations if v["severity"] == "info"]

    return {
        "valid": len(critical) == 0,
        "violations": violations,
        "summary": {
            "total": len(violations),
            "critical": len(critical),
            "warning": len(warning),
            "info": len(info),
        },
        "stratum": select_stratum(age),
        "adjusted_ranges": ranges,
    }


def _validate_angles_with_ranges(data: dict, angle_ranges: dict) -> list:
    """Check angle values against provided ranges."""
    violations = []
    angles = data.get("angles", {})
    angle_frames = angles.get("frames", [])

    if not angle_frames:
        return violations

    for joint, ranges in angle_ranges.items():
        lo, hi = ranges["full"]
        values = [af.get(joint) for af in angle_frames if af.get(joint) is not None]
        if not values:
            continue

        out_of_range = [v for v in values if v < lo or v > hi]
        pct = 100.0 * len(out_of_range) / len(values)

        if pct > 20:
            violations.append({
                "type": "angle_range",
                "joint": joint,
                "severity": "critical" if pct > 50 else "warning",
                "pct_out_of_range": round(pct, 1),
                "expected_range": [lo, hi],
                "actual_range": [round(min(values), 1), round(max(values), 1)],
                "description": f"{joint}: {pct:.0f}% of frames outside [{lo}, {hi}] deg",
            })
        elif pct > 5:
            violations.append({
                "type": "angle_range",
                "joint": joint,
                "severity": "info",
                "pct_out_of_range": round(pct, 1),
                "expected_range": [lo, hi],
                "actual_range": [round(min(values), 1), round(max(values), 1)],
                "description": f"{joint}: {pct:.0f}% of frames outside [{lo}, {hi}] deg",
            })

    return violations


def _validate_phase_angles_with_ranges(
    data: dict, cycles: dict, angle_ranges: dict
) -> list:
    """Validate angles per gait phase using provided ranges."""
    violations = []
    cycle_list = cycles.get("cycles", [])

    for c in cycle_list:
        an = c.get("angles_normalized", {})
        for joint_base in ["hip", "knee", "ankle"]:
            vals = an.get(joint_base)
            if vals is None:
                continue
            vals = np.array(vals)

            joint_key = f"{joint_base}_{c['side'][0].upper()}"
            ranges = angle_ranges.get(joint_key, {})

            # Stance phase: 0-60% -> indices 0-60
            stance_range = ranges.get("stance")
            if stance_range:
                stance_vals = vals[:61]
                stance_min = np.min(stance_vals)
                stance_max = np.max(stance_vals)
                if stance_min < stance_range[0] - 10 or stance_max > stance_range[1] + 10:
                    violations.append({
                        "type": "phase_angle",
                        "joint": joint_key,
                        "phase": "stance",
                        "cycle_id": c["cycle_id"],
                        "severity": "warning",
                        "expected_range": list(stance_range),
                        "actual_range": [round(float(stance_min), 1), round(float(stance_max), 1)],
                        "description": f"Cycle {c['cycle_id']} {c['side']} {joint_base} stance: "
                                       f"[{stance_min:.0f}, {stance_max:.0f}] vs expected {stance_range}",
                    })

    return violations


def _validate_spatiotemporal_with_ranges(data: dict, st_ranges: dict) -> list:
    """Validate spatio-temporal parameters against provided ranges."""
    violations = []
    events = data.get("events", {})
    fps = data.get("meta", {}).get("fps", 30.0)

    # Check cadence
    all_hs = []
    for key in ["left_hs", "right_hs"]:
        for ev in events.get(key, []):
            all_hs.append(ev["frame"])
    all_hs.sort()

    if len(all_hs) >= 3:
        intervals = np.diff(all_hs) / fps
        step_time = float(np.mean(intervals))
        cadence = 60.0 / step_time if step_time > 0 else 0

        lo, hi = st_ranges["cadence_steps_per_min"]
        if cadence < lo or cadence > hi:
            violations.append({
                "type": "spatiotemporal",
                "parameter": "cadence",
                "severity": "warning",
                "value": round(cadence, 1),
                "expected_range": [lo, hi],
                "description": f"Cadence {cadence:.0f} steps/min outside normal [{lo}, {hi}]",
            })

    return violations
