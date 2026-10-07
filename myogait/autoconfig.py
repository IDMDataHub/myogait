"""Automatic choice of the processing recipe from the recording itself.

The default recipe (neutral calibration on, standstill trimmed) fits a
clean clip that starts with the subject standing still and walks in one
direction.  Many real recordings do not: a marker (C3D) trial, a walkway
that starts mid-stride, or a there-and-back pass need calibration off,
the standstill kept and direction-consistent cycles -- the recipe of
:func:`myogait.run_pipeline`, benchmarked against optical motion capture.
``detect_recipe`` inspects the pivot and picks one, with a short human
rationale; ``run_auto`` runs it and, if no gait cycle is found, falls
back once to the overground recipe.

    import myogait as mg

    result = mg.run_auto("walk.mp4", model="sapiens2-quick")
    result["recipe"], result["reasons"]

The same detection is used by the myogait-app graphical interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Recipe for a clean clip with a standing start, one walking direction.
DEFAULT_RECIPE = {
    "name": "default",
    "calibrate": True,
    "trim_standstill": True,
    "direction_filter": False,
    "c3d_reference_ankle": False,
}

#: The validated overground / marker recipe (that of ``run_pipeline``).
OVERGROUND_RECIPE = {
    "name": "overground",
    "calibrate": False,
    "trim_standstill": False,
    "direction_filter": False,
    "c3d_reference_ankle": True,
}


def _mid_hip_x(frames: list) -> np.ndarray:
    """Antero-posterior progression proxy: the finite mid-hip x values.

    Detection is advisory, so an incomplete landmark is simply skipped
    rather than turning a usable recording into an error.
    """
    xs: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, dict):
            continue
        values: list[float] = []
        for key in ("LEFT_HIP", "RIGHT_HIP"):
            hip = landmarks.get(key)
            if not isinstance(hip, dict):
                continue
            try:
                value = float(hip.get("x"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            xs.append(float(np.mean(values)))
    return np.asarray(xs, dtype=float)


def has_static_start(frames: list, n: int = 20, thresh: float = 0.01) -> bool:
    """True when the first frames barely move -- a standing neutral pose.

    A standing start gives neutral calibration a real reference; a
    mid-stride start does not, and first-frame calibration would then
    shift the whole cycle.  Measured on the mid-hip x spread
    (normalised image units).
    """
    xs = _mid_hip_x(frames[: min(n, len(frames))])
    return xs.size >= 3 and float(xs.std()) < thresh


def has_direction_reversal(frames: list, thresh: float = 0.15) -> bool:
    """True for a there-and-back walkway: the progression reverses.

    The mid-hip x goes one way then comes back by more than ``thresh``
    of the frame width, in either camera direction.
    """
    xs = _mid_hip_x(frames)
    if xs.size < 10:
        return False
    start, end = float(xs[0]), float(xs[-1])
    high, low = float(np.max(xs)), float(np.min(xs))
    returned_from_high = high - start > thresh and high - end > thresh
    returned_from_low = start - low > thresh and end - low > thresh
    return returned_from_high or returned_from_low


def is_c3d_source(data: dict) -> bool:
    """True when the pivot comes from a marker-based (C3D) recording."""
    return bool(data.get("c3d_markers_3d")) or str(
        (data.get("meta") or {}).get("source") or "").lower() == "c3d"


def detect_recipe(data: dict) -> tuple[dict, list[str]]:
    """Choose the processing recipe for one pivot.

    Returns ``(recipe, reasons)`` where ``recipe`` is a copy of
    :data:`DEFAULT_RECIPE` or :data:`OVERGROUND_RECIPE` (with
    ``direction_filter`` set for there-and-back recordings) and
    ``reasons`` a list of short human-readable explanations.
    """
    frames = data.get("frames") or []
    reasons: list[str] = []

    is_c3d = is_c3d_source(data)
    reversal = has_direction_reversal(frames)
    static_start = has_static_start(frames)

    if is_c3d:
        reasons.append("marker (C3D) source: 3-D ankle reference on")
    if reversal:
        reasons.append("there-and-back walkway: direction-dependent, calibration off")
    if not static_start and not reversal:
        reasons.append("no standing neutral at the start: calibration off")

    if is_c3d or reversal or not static_start:
        reasons.append("overground recipe: standstill kept")
        recipe = dict(OVERGROUND_RECIPE)
        if reversal:
            recipe["direction_filter"] = True
            reasons.append("return-pass cycles filtered out (dominant direction kept)")
        return recipe, reasons

    reasons.append("clean standing-start clip: default recipe")
    return dict(DEFAULT_RECIPE), reasons


def run_auto(
    source,
    model: str = "sapiens2-quick",
    butterworth_cutoff: float = 4.0,
    event_method: str = "zeni",
    min_cycle_duration_s: float = 0.8,
    max_cycle_duration_s: float = 1.6,
    n_points: int = 101,
    analyze: bool = True,
    show_progress: bool = True,
) -> dict:
    """Run the pipeline with an automatically chosen recipe.

    ``source`` is a video, a pivot JSON, a C3D file, or an already
    loaded pivot ``dict``.  The recipe is chosen by
    :func:`detect_recipe`; if it segments no gait cycle, the overground
    recipe is tried once before returning.

    Returns the :func:`myogait.run_pipeline` result dictionary plus
    ``"recipe"`` (the recipe actually used) and ``"reasons"``.
    """
    import copy

    from .pipeline import _load_source, _run_steps

    data, source_type = _load_source(source, model=model,
                                     show_progress=show_progress)
    recipe, reasons = detect_recipe(data)

    def _run(rec: dict, pivot: dict) -> dict:
        return _run_steps(
            pivot, source_type,
            butterworth_cutoff=butterworth_cutoff,
            calibrate=rec["calibrate"],
            event_method=event_method,
            trim_standstill=rec["trim_standstill"],
            min_cycle_duration_s=min_cycle_duration_s,
            max_cycle_duration_s=max_cycle_duration_s,
            n_points=n_points,
            analyze=analyze,
            direction_filter=rec["direction_filter"],
        )

    pristine: Optional[dict] = (copy.deepcopy(data)
                                if recipe["name"] != "overground" else None)
    result = _run(recipe, data)
    if pristine is not None and not (result["cycles"] or {}).get("cycles"):
        alt = _run(dict(OVERGROUND_RECIPE), pristine)
        if (alt["cycles"] or {}).get("cycles"):
            reasons = reasons + [
                "no cycle with the first recipe -> fell back to overground"]
            recipe, result = dict(OVERGROUND_RECIPE), alt
    result["recipe"] = recipe
    result["reasons"] = reasons
    return result
