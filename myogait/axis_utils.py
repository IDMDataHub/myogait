"""Axis and direction utilities for gait analysis.

Provides functions to detect walking direction from foot landmark
orientation, which is more robust than displacement-based methods
and works on treadmills.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_walking_direction_from_feet(
    data: dict,
    toe_names: tuple = ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
    heel_names: tuple = ("LEFT_HEEL", "RIGHT_HEEL"),
) -> str:
    """Detect walking direction from foot orientation (toe vs heel).

    Compares the horizontal position of the toe to the heel for both
    feet across all frames.  If toes are to the right of heels on
    average the subject is walking to the right, and vice versa.

    This method is:

    - **Unit-independent**: works on normalised [0, 1] coords or mm.
    - **Treadmill-compatible**: does not rely on net displacement.
    - **Robust**: uses median across all frames and both feet.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.  Each frame must
        have ``landmarks`` with toe and heel entries.
    toe_names : tuple of str, optional
        Landmark names for left and right toes
        (default ``("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")``).
    heel_names : tuple of str, optional
        Landmark names for left and right heels
        (default ``("LEFT_HEEL", "RIGHT_HEEL")``).

    Returns
    -------
    str
        ``"right"`` if walking to the right, ``"left"`` if walking
        to the left.
    """
    frames = data.get("frames", [])
    if not frames:
        return "right"  # default: assume left-to-right

    diffs = []
    for frame in frames:
        lm = frame.get("landmarks", {})
        for toe_name, heel_name in zip(toe_names, heel_names):
            toe = lm.get(toe_name)
            heel = lm.get(heel_name)
            if (toe is not None and heel is not None
                    and toe.get("x") is not None
                    and heel.get("x") is not None):
                tx = float(toe["x"])
                hx = float(heel["x"])
                if not (np.isnan(tx) or np.isnan(hx)):
                    diffs.append(tx - hx)

    if not diffs:
        return "right"  # default

    median_diff = float(np.median(diffs))
    direction = "right" if median_diff > 0 else "left"
    logger.debug(
        "Foot direction: median(toe_x - heel_x) = %.4f -> %s",
        median_diff, direction,
    )
    return direction


def detect_walking_direction_from_feet_arrays(
    frames_landmarks: list,
    toe_idx: int = 31,
    heel_idx: int = 29,
    toe_idx_r: int = 32,
    heel_idx_r: int = 30,
) -> str:
    """Detect walking direction from raw landmark arrays.

    Same logic as :func:`detect_walking_direction_from_feet` but
    operates on raw ``(N, 3)`` landmark arrays (used internally during
    extraction before the data dict is built).

    Parameters
    ----------
    frames_landmarks : list of np.ndarray or None
        List of landmark arrays, shape ``(n_landmarks, 3+)``.
    toe_idx, heel_idx : int
        Left toe / heel indices (default MediaPipe 31/29).
    toe_idx_r, heel_idx_r : int
        Right toe / heel indices (default MediaPipe 32/30).

    Returns
    -------
    str
        ``"right"`` or ``"left"``.
    """
    diffs = []
    for lm in frames_landmarks:
        if lm is None:
            continue
        n = lm.shape[0]
        for ti, hi in [(toe_idx, heel_idx), (toe_idx_r, heel_idx_r)]:
            if ti >= n or hi >= n:
                continue
            tx = lm[ti, 0]
            hx = lm[hi, 0]
            if not (np.isnan(tx) or np.isnan(hx)):
                diffs.append(tx - hx)

    if not diffs:
        return "right"

    return "right" if float(np.median(diffs)) > 0 else "left"
