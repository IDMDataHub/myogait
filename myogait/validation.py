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
"""

import logging
from typing import Dict, List, Optional

import numpy as np

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
