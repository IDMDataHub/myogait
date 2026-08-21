"""High-level entry point: the validated myogait pipeline in one call.

``run_pipeline`` encodes the recipe that was benchmarked against
optical motion capture (see the Validation section of the README):
Butterworth filtering, sagittal angles without neutral calibration,
direction-independent sign convention, Zeni gait events, cycle
segmentation, and direction-consistent cycle filtering.

    import myogait as mg

    result = mg.run_pipeline("walk.mp4", model="sapiens2-quick")
    result = mg.run_pipeline("walk.myogait.json")   # pre-extracted
    result = mg.run_pipeline("trial.c3d")           # optical reference

    result["data"]      # pivot dict (landmarks, angles, events)
    result["cycles"]    # segmented, filtered gait cycles
    result["stats"]     # analyze_gait output (None if no cycles)

Every step remains available individually for advanced use — this
wrapper only removes the burden of knowing the validated defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _filter_cycles_by_direction(data: dict, cycles: dict) -> dict:
    """Keep the dominant walking-direction cycle group.

    Walkway recordings often contain an outbound and a return pass;
    cycles walked against the globally-detected direction come out
    with mirrored angles and would corrupt any average.  Groups the
    cycles by mid-hip displacement sign, keeps the larger group, and
    enforces the flexion-positive convention on the kept cycles.
    """
    frames = data.get("frames", [])
    if not frames or not cycles.get("cycles"):
        return cycles
    first_idx = frames[0].get("frame_idx", 0)

    def _hip_x(fi):
        pos = int(fi) - first_idx
        if pos < 0 or pos >= len(frames):
            return None
        lm = frames[pos].get("landmarks", {})
        lh, rh = lm.get("LEFT_HIP"), lm.get("RIGHT_HIP")
        if not lh or not rh:
            return None
        try:
            return (float(lh["x"]) + float(rh["x"])) / 2.0
        except (TypeError, KeyError, ValueError):
            return None

    groups = {1.0: [], -1.0: []}
    for c in cycles["cycles"]:
        x0, x1 = _hip_x(c.get("start_frame", 0)), _hip_x(c.get("end_frame", 0))
        if x0 is None or x1 is None or abs(x1 - x0) < 1e-5:
            continue
        groups[1.0 if x1 > x0 else -1.0].append(c)

    def _knee_mean(cs):
        vals = [np.mean(c["angles_normalized"]["knee"]) for c in cs
                if c.get("angles_normalized", {}).get("knee") is not None]
        return float(np.mean(vals)) if vals else -np.inf

    keep = max(groups.values(), key=lambda cs: (len(cs), _knee_mean(cs)))
    if not keep:
        return cycles

    for side in ("left", "right"):
        side_cycles = [c for c in keep if c.get("side") == side]
        if _knee_mean(side_cycles) < 0:
            for c in side_cycles:
                an = c.get("angles_normalized", {})
                for j in ("knee", "hip"):
                    if an.get(j) is not None:
                        an[j] = [-v for v in an[j]]
        hip_curves = [c["angles_normalized"]["hip"] for c in side_cycles
                      if c.get("angles_normalized", {}).get("hip") is not None
                      and len(c["angles_normalized"]["hip"]) == 101]
        if hip_curves:
            hm = np.mean(hip_curves, axis=0)
            if np.mean(hm[:15]) < np.mean(hm[40:60]):
                for c in side_cycles:
                    an = c.get("angles_normalized", {})
                    if an.get("hip") is not None:
                        an["hip"] = [-v for v in an["hip"]]
    return {**cycles, "cycles": keep}


def _diagnose(data: dict, cycles_all: dict, cycles_kept: dict) -> dict:
    """Post-hoc quality diagnostics with actionable warnings.

    Inspects the processed recording for the conditions that are known
    to degrade accuracy, and reports them instead of silently letting
    them bias the results:

    - tracking coverage and confidence
    - out-of-sagittal-plane distortion (foreshortening) — suggests
      ``apply_perspective_correction`` when meaningful
    - a usable standing prelude — suggests ``calibrate=True`` only
      when one actually exists
    - there-and-back recordings (how many mirror cycles were dropped)
    - too few cycles for a reliable average
    - implausible ankle range (screening-grade flag)
    """
    warnings_: list = []
    frames = data.get("frames", [])
    n_frames = len(frames)
    n_tracked = sum(1 for f in frames if f.get("landmarks"))
    confs = [f.get("confidence") for f in frames if f.get("confidence") is not None]
    conf_mean = float(np.mean(confs)) if confs else None

    if n_frames and n_tracked / n_frames < 0.8:
        warnings_.append(
            f"tracking: landmarks present on only {100*n_tracked/n_frames:.0f}% "
            "of frames — results may be fragmentary")
    if conf_mean is not None and conf_mean < 0.6:
        warnings_.append(
            f"tracking: mean landmark confidence {conf_mean:.2f} < 0.6 — "
            "consider a stronger pose model or better lighting")

    # Foreshortening / out-of-plane distortion: if the apparent thigh
    # length varies a lot across the recording, the subject is not
    # moving parallel to the image plane.
    thigh = []
    for f in frames:
        lm = f.get("landmarks", {})
        h, k = lm.get("LEFT_HIP"), lm.get("LEFT_KNEE")
        if h and k:
            try:
                thigh.append(float(np.hypot(h["x"] - k["x"], h["y"] - k["y"])))
            except (TypeError, KeyError):
                pass
    distortion = None
    if len(thigh) > 30:
        t = np.asarray(thigh)
        distortion = float((np.percentile(t, 95) - np.percentile(t, 5))
                            / (np.percentile(t, 95) + 1e-9))
        if distortion > 0.35:
            warnings_.append(
                f"perspective: apparent thigh length varies {100*distortion:.0f}% "
                "across the clip — subject not parallel to the image plane; "
                "consider apply_perspective_correction() or a better camera "
                "placement")

    # Standing prelude: if the first second is near-static, neutral
    # calibration would actually be applicable.
    if n_frames > 60:
        fps = float(data.get("meta", {}).get("fps", 30.0))
        head = thigh[: int(fps)] if thigh else []
        if len(head) > 10 and np.std(head) / (np.mean(head) + 1e-9) < 0.02:
            warnings_.append(
                "calibration: the clip starts with a near-static pose — "
                "compute_angles(calibrate=True) is applicable here and would "
                "give absolute joint angles a subject-specific neutral zero")

    n_all = len(cycles_all.get("cycles", []))
    n_kept = len(cycles_kept.get("cycles", []))
    if n_all - n_kept > 0:
        warnings_.append(
            f"direction: {n_all - n_kept} cycle(s) walked against the dominant "
            "direction were dropped (there-and-back recording)")
    per_side = {s: sum(1 for c in cycles_kept.get("cycles", [])
                        if c.get("side") == s) for s in ("left", "right")}
    for s, n in per_side.items():
        if n < 3:
            warnings_.append(
                f"cycles: only {n} {s} cycle(s) — averages are unreliable, "
                "record more passes (>=3 per side recommended)")

    ankle_roms = [float(np.ptp(c["angles_normalized"]["ankle"]))
                  for c in cycles_kept.get("cycles", [])
                  if c.get("angles_normalized", {}).get("ankle") is not None]
    if ankle_roms and (np.mean(ankle_roms) < 8 or np.mean(ankle_roms) > 60):
        warnings_.append(
            f"ankle: mean ROM {np.mean(ankle_roms):.0f} deg is outside the "
            "usual range — treat absolute ankle values as screening-grade")

    return {
        "n_frames": n_frames,
        "tracking_coverage": round(n_tracked / n_frames, 3) if n_frames else None,
        "confidence_mean": round(conf_mean, 3) if conf_mean is not None else None,
        "plane_distortion": round(distortion, 3) if distortion is not None else None,
        "n_cycles_left": per_side["left"],
        "n_cycles_right": per_side["right"],
        "n_cycles_dropped_direction": n_all - n_kept,
        "warnings": warnings_,
    }


def run_pipeline(
    source,
    model: str = "sapiens2-quick",
    butterworth_cutoff: float = 4.0,
    event_method: str = "zeni",
    min_cycle_duration_s: float = 0.8,
    max_cycle_duration_s: float = 1.6,
    n_points: int = 101,
    analyze: bool = True,
    direction_filter: bool = True,
    show_progress: bool = True,
) -> dict:
    """Run the validated end-to-end gait pipeline on a video, a
    pre-extracted myogait JSON, or a C3D optical-capture file.

    Steps (identical to the benchmarked configuration):

    1. Load / extract landmarks (``extract`` for videos with the given
       ``model``; ``load_json``; ``load_c3d`` with, for C3D, the
       3-D ankle reference via :func:`compute_c3d_reference_angles`).
    2. ``normalize(filters=["butterworth"], butterworth_cutoff=4.0)``.
    3. ``compute_angles(calibrate=False)`` — neutral calibration is
       only meaningful when the clip starts with a standing pose;
       gait clips usually do not, and the built-in guard would skip
       it anyway.
    4. ``canonicalize_angle_signs()`` — flexion-positive convention
       independent of walking direction (essential for comparison and
       longitudinal follow-up).
    5. ``detect_events(method="zeni", trim_standstill=False)``.
    6. ``segment_cycles`` into ``n_points``-sample cycles within the
       physiological duration window.
    7. Direction-consistent cycle filter (there-and-back recordings).
    8. Quality diagnostics (tracking coverage, plane distortion,
       standing-prelude detection, cycle counts) with actionable
       warnings — see :func:`_diagnose`.
    9. Optional ``analyze_gait``.

    Returns ``{"data", "cycles", "stats", "quality", "source_type"}``.
    ``result["quality"]["warnings"]`` lists anything the pipeline
    detected that should temper interpretation (also logged).
    """
    from .schema import load_json
    from .normalize import normalize
    from .angles import compute_angles, canonicalize_angle_signs
    from .events import detect_events
    from .cycles import segment_cycles
    from .analysis import analyze_gait

    src = str(source)
    suffix = Path(src).suffix.lower()
    if suffix == ".c3d":
        from .experimental_vicon import load_c3d
        data = load_c3d(src)
        source_type = "c3d"
    elif suffix == ".json" or src.endswith(".myogait.json"):
        data = load_json(src)
        source_type = "json"
    elif suffix in _VIDEO_SUFFIXES:
        from .extract import extract
        data = extract(src, model=model, show_progress=show_progress)
        source_type = "video"
    else:
        raise ValueError(
            f"Unrecognised source type '{suffix}'. Expected a video "
            f"({sorted(_VIDEO_SUFFIXES)}), a .myogait.json, or a .c3d file."
        )

    data = normalize(data, filters=["butterworth"],
                     butterworth_cutoff=butterworth_cutoff)
    data = compute_angles(data, calibrate=False)
    if source_type == "c3d":
        from .experimental_vicon import compute_c3d_reference_angles
        data = compute_c3d_reference_angles(data)
    data = canonicalize_angle_signs(data)
    data = detect_events(data, method=event_method, trim_standstill=False,
                         min_cycle_duration=min(0.6, min_cycle_duration_s))
    cycles_all = segment_cycles(data, n_points=n_points,
                                 min_duration=min_cycle_duration_s,
                                 max_duration=max_cycle_duration_s)
    cycles = (_filter_cycles_by_direction(data, cycles_all)
              if direction_filter else cycles_all)

    quality = _diagnose(data, cycles_all, cycles)
    for w in quality["warnings"]:
        logger.warning("run_pipeline: %s", w)

    stats: Optional[dict] = None
    if analyze and cycles.get("cycles"):
        try:
            stats = analyze_gait(data, cycles)
        except Exception as exc:  # analysis must never sink the pipeline
            logger.warning("analyze_gait failed: %s", exc)

    return {"data": data, "cycles": cycles, "stats": stats,
            "quality": quality, "source_type": source_type}
