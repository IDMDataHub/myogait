"""Normalization: filtering and spatial normalization of pose data.

Each normalization step is a standalone function that can be used
independently or composed via the main normalize() orchestrator.

Steps available:

    - filter_butterworth: Low-pass Butterworth (zero-phase via filtfilt).
      Ref: Butterworth S. On the theory of filter amplifiers.
      Exp Wireless Wireless Eng. 1930;7:536-541.
      Zero-phase implementation: Gustafsson F. Determining the initial
      states in forward-backward filtering. IEEE Trans Signal Process.
      1996;44(4):988-992. doi:10.1109/78.492552
      Recommended for gait kinematics at 4-6 Hz cutoff:
      Ref: Winter DA. Biomechanics and Motor Control of Human Movement.
      4th ed. Wiley; 2009. Chapter 2.

    - filter_savgol: Savitzky-Golay polynomial smoothing.
      Ref: Savitzky A, Golay MJE. Smoothing and differentiation of
      data by simplified least squares procedures. Anal Chem.
      1964;36(8):1627-1639. doi:10.1021/ac60214a047

    - filter_moving_mean: Simple centered moving average.

    - filter_spline: Smoothing spline interpolation.
      Ref: Reinsch CH. Smoothing by spline functions.
      Numer Math. 1967;10:177-183. doi:10.1007/BF02162161
      Woltring HJ. A Fortran package for generalized, cross-validatory
      spline smoothing and differentiation. Adv Eng Softw.
      1986;8(2):104-113. doi:10.1016/0141-1195(86)90098-7

    - filter_kalman: Kalman filter for trajectory smoothing.
      Ref: Kalman RE. A new approach to linear filtering and prediction
      problems. J Basic Eng. 1960;82(1):35-45.
      doi:10.1115/1.3662552

    - center_on_torso: Center coords on torso centroid, scale to [-100,100].
    - align_skeleton: Normalize skeleton scale + center.
    - correct_bilateral: Correct right segments to match left reference.
    - correct_pixel_ratio: Fix non-square pixels (e.g. after MediaPipe resize).

General filtering reference for human motion:
    Winter DA, Sidwall HG, Hobson DA. Measurement and reduction of
    noise in kinematics of locomotion. J Biomech. 1974;7(2):157-159.
    doi:10.1016/0021-9290(74)90056-6

Usage:
    # Simple: just pass filter names
    data = normalize(data, filters=["butterworth"])

    # Advanced: full config with per-step options
    data = normalize(data, steps=[
        {"type": "butterworth", "cutoff": 6.0, "order": 4},
        {"type": "center_on_torso"},
        {"type": "correct_pixel_ratio", "input_width": 1920, "input_height": 1080,
         "processed_width": 256, "processed_height": 256},
    ])

    # Or call steps directly
    from myogait.normalize import filter_butterworth
    df = filter_butterworth(df, cutoff=4.0, order=2, fs=30.0)
"""

import copy
import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import MP_LANDMARK_NAMES

logger = logging.getLogger(__name__)


# ── DataFrame conversion ────────────────────────────────────────────


def frames_to_dataframe(frames: list) -> pd.DataFrame:
    """Convert pivot JSON frames to a DataFrame with LANDMARK_x, LANDMARK_y columns."""
    rows = []
    for f in frames:
        row = {"frame_idx": f["frame_idx"], "time_s": f["time_s"]}
        for name, coords in f.get("landmarks", {}).items():
            row[f"{name}_x"] = coords["x"]
            row[f"{name}_y"] = coords["y"]
            row[f"{name}_visibility"] = coords.get("visibility", 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_frames(df: pd.DataFrame, original_frames: list) -> list:
    """Write DataFrame values back into pivot JSON frame structure."""
    frames = copy.deepcopy(original_frames)

    for i, frame in enumerate(frames):
        if i >= len(df):
            break
        for name in list(frame.get("landmarks", {}).keys()):
            xcol = f"{name}_x"
            ycol = f"{name}_y"
            if xcol in df.columns and ycol in df.columns:
                frame["landmarks"][name]["x"] = float(df[xcol].iloc[i])
                frame["landmarks"][name]["y"] = float(df[ycol].iloc[i])

    return frames


def _apply_on_xy(df: pd.DataFrame, func) -> pd.DataFrame:
    """Apply a function to all _x / _y coordinate columns."""
    df = df.copy()
    cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            continue
        v = s.interpolate(limit_direction="both").bfill().ffill().to_numpy(float)
        try:
            df[c] = func(v)
        except Exception as e:
            logger.warning(f"Filter failed for column {c}: {e}")
            df[c] = v
    return df


# ── Signal filter steps ─────────────────────────────────────────────


def filter_butterworth(
    df: pd.DataFrame,
    cutoff: float = 4.0,
    order: int = 2,
    fs: float = 30.0,
    **kwargs,
) -> pd.DataFrame:
    """Butterworth low-pass filter (zero-phase via filtfilt).

    Args:
        df: DataFrame with pose coordinate columns.
        cutoff: Cutoff frequency in Hz.
        order: Filter order.
        fs: Sampling frequency (fps).
    """
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * float(fs)
    normal = float(cutoff) / nyq if nyq > 0 else 0.1
    normal = max(min(normal, 0.99), 1e-6)
    b, a = butter(int(order), normal, btype="low", analog=False)
    return _apply_on_xy(df, lambda v: filtfilt(b, a, v))


def filter_savgol(
    df: pd.DataFrame,
    window_length: int = 21,
    polyorder: int = 2,
    **kwargs,
) -> pd.DataFrame:
    """Savitzky-Golay polynomial smoothing filter.

    Args:
        df: DataFrame with pose coordinate columns.
        window_length: Window length (odd number).
        polyorder: Polynomial order.
    """
    from scipy.signal import savgol_filter
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    def _smooth(v):
        if len(v) >= window_length:
            return savgol_filter(v, window_length, polyorder)
        return v

    return _apply_on_xy(df, _smooth)


def filter_moving_mean(
    df: pd.DataFrame,
    window: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Simple centered moving average.

    Args:
        df: DataFrame with pose coordinate columns.
        window: Window size.
    """
    df = df.copy()
    cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        df[c] = s.rolling(int(window), min_periods=1, center=True).mean().bfill().ffill()
    return df


def filter_spline(
    df: pd.DataFrame,
    s: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Smoothing spline filter.

    Args:
        df: DataFrame with pose coordinate columns.
        s: Smoothing factor.
    """
    from scipy.interpolate import UnivariateSpline

    def _smooth(v):
        x = np.arange(len(v), dtype=float)
        try:
            return UnivariateSpline(x, v, s=float(s))(x)
        except Exception:
            return v

    return _apply_on_xy(df, _smooth)


def filter_kalman(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Kalman filter for trajectory smoothing (falls back to moving mean).

    Requires pykalman package.
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        return filter_moving_mean(df, window=5)

    df = df.copy()
    points = sorted(
        {c[:-2] for c in df.columns if c.endswith("_x") and f"{c[:-2]}_y" in df.columns}
    )

    for pt in points:
        cx, cy = f"{pt}_x", f"{pt}_y"
        x = pd.to_numeric(df[cx], errors="coerce").interpolate().bfill().ffill().values
        y = pd.to_numeric(df[cy], errors="coerce").interpolate().bfill().ffill().values
        observations = np.column_stack([x, y])

        kf = KalmanFilter(
            initial_state_mean=observations[0], n_dim_obs=2, n_dim_state=2
        )
        try:
            smoothed, _ = kf.smooth(observations)
            df[cx] = smoothed[:, 0]
            df[cy] = smoothed[:, 1]
        except Exception:
            pass

    return df


# ── Spatial normalization steps ──────────────────────────────────────


def center_on_torso(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Center coordinates on torso centroid and normalize to [-100, 100].

    Uses the mean of shoulders and hips as the torso center.
    """
    df = df.copy()
    required = [
        "LEFT_SHOULDER_x", "RIGHT_SHOULDER_x", "LEFT_HIP_x", "RIGHT_HIP_x",
        "LEFT_SHOULDER_y", "RIGHT_SHOULDER_y", "LEFT_HIP_y", "RIGHT_HIP_y",
    ]
    if not all(c in df.columns for c in required):
        return df

    center_x = (
        df["LEFT_SHOULDER_x"] + df["RIGHT_SHOULDER_x"]
        + df["LEFT_HIP_x"] + df["RIGHT_HIP_x"]
    ) / 4
    center_y = (
        df["LEFT_SHOULDER_y"] + df["RIGHT_SHOULDER_y"]
        + df["LEFT_HIP_y"] + df["RIGHT_HIP_y"]
    ) / 4

    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]

    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce") - center_x
    for c in ycols:
        df[c] = pd.to_numeric(df[c], errors="coerce") - center_y

    all_vals = pd.concat([df[xcols].stack(), df[ycols].stack()])
    scale = max(abs(all_vals.min()), abs(all_vals.max()))
    if scale > 0:
        for c in xcols + ycols:
            df[c] = df[c] / scale * 100

    return df


def align_skeleton(
    df: pd.DataFrame,
    ref_size: float = None,
    **kwargs,
) -> pd.DataFrame:
    """Align skeleton via torso centering and scale normalization.

    Args:
        df: DataFrame with pose coordinate columns.
        ref_size: Reference body size for scaling. Auto-estimated if None.
    """
    df = df.copy()
    points = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
    xcols = [f"{p}_x" for p in points]
    ycols = [f"{p}_y" for p in points]

    if not all(c in df.columns for c in xcols + ycols):
        return df

    if ref_size is None:
        if "LEFT_ANKLE_x" in df.columns:
            dx = df["LEFT_SHOULDER_x"] - df["LEFT_ANKLE_x"]
            dy = df["LEFT_SHOULDER_y"] - df["LEFT_ANKLE_y"]
            ref_size = np.median(np.sqrt(dx ** 2 + dy ** 2))
        else:
            ref_size = 1.0

    all_xcols = [c for c in df.columns if c.endswith("_x")]
    all_ycols = [c for c in df.columns if c.endswith("_y")]

    for i in range(len(df)):
        cx = np.nanmean([df[c].iloc[i] for c in xcols])
        cy = np.nanmean([df[c].iloc[i] for c in ycols])
        for c in all_xcols:
            df.loc[df.index[i], c] = (df[c].iloc[i] - cx) / ref_size * 100
        for c in all_ycols:
            df.loc[df.index[i], c] = (df[c].iloc[i] - cy) / ref_size * 100

    return df


def correct_bilateral(
    df: pd.DataFrame,
    num_ref_frames: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """Correct right-side segment lengths to match left-side reference.

    Args:
        df: DataFrame with pose coordinate columns.
        num_ref_frames: Number of frames to compute reference lengths.
    """
    df = df.copy()
    segments = [
        ("LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST", "RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_KNEE", "RIGHT_HIP", "RIGHT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE", "RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_ANKLE", "LEFT_HEEL", "RIGHT_ANKLE", "RIGHT_HEEL"),
    ]

    for p1_l, p2_l, p1_r, p2_r in segments:
        cols_l = [f"{p1_l}_x", f"{p1_l}_y", f"{p2_l}_x", f"{p2_l}_y"]
        cols_r = [f"{p1_r}_x", f"{p1_r}_y", f"{p2_r}_x", f"{p2_r}_y"]
        if not all(c in df.columns for c in cols_l + cols_r):
            continue

        dx_l = df[f"{p2_l}_x"] - df[f"{p1_l}_x"]
        dy_l = df[f"{p2_l}_y"] - df[f"{p1_l}_y"]
        lengths_l = np.sqrt(dx_l ** 2 + dy_l ** 2)
        ref_length = np.median(lengths_l.iloc[:num_ref_frames])

        if ref_length <= 0 or not np.isfinite(ref_length):
            continue

        dx_r = df[f"{p2_r}_x"] - df[f"{p1_r}_x"]
        dy_r = df[f"{p2_r}_y"] - df[f"{p1_r}_y"]
        lengths_r = np.sqrt(dx_r ** 2 + dy_r ** 2)
        scale = ref_length / lengths_r.replace(0, np.nan)

        df[f"{p2_r}_x"] = df[f"{p1_r}_x"] + dx_r * scale
        df[f"{p2_r}_y"] = df[f"{p1_r}_y"] + dy_r * scale

    return df


def correct_pixel_ratio(
    df: pd.DataFrame,
    input_width: int = 1920,
    input_height: int = 1080,
    processed_width: int = 256,
    processed_height: int = 256,
    **kwargs,
) -> pd.DataFrame:
    """Correct for non-square pixels after model processing.

    When a model (e.g. MediaPipe) internally resizes the image to a square,
    the output coordinates need to be rescaled to match the original
    aspect ratio.

    Args:
        df: DataFrame with pose coordinate columns (in [0,1] range).
        input_width: Original video width in pixels.
        input_height: Original video height in pixels.
        processed_width: Width the model processes internally.
        processed_height: Height the model processes internally.
    """
    if input_width == input_height and processed_width == processed_height:
        return df  # No correction needed

    df = df.copy()
    aspect_original = input_width / input_height
    aspect_processed = processed_width / processed_height
    ratio = aspect_original / aspect_processed

    if abs(ratio - 1.0) < 0.01:
        return df  # Close enough to square

    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]

    if ratio > 1.0:
        # Original wider than processed → x coordinates need stretching
        for c in xcols:
            df[c] = pd.to_numeric(df[c], errors="coerce") * ratio
    else:
        # Original taller than processed → y coordinates need stretching
        for c in ycols:
            df[c] = pd.to_numeric(df[c], errors="coerce") / ratio

    return df


# ── Data quality steps ───────────────────────────────────────────────


def confidence_filter(
    df: pd.DataFrame,
    threshold: float = 0.3,
    _data_frames: list = None,
    **kwargs,
) -> pd.DataFrame:
    """Filter landmarks by visibility confidence, setting low-confidence coords to NaN.

    For each frame, checks each landmark's visibility from the original frame
    data. Coordinates for landmarks with visibility below the threshold are
    replaced with NaN so that downstream interpolation or filtering can handle
    them as gaps.

    This is standard practice in pose estimation pipelines to suppress noisy
    detections that the model itself reports as uncertain.

    Args:
        df: DataFrame with LANDMARK_x, LANDMARK_y, LANDMARK_visibility columns.
        threshold: Minimum visibility score in [0, 1]. Landmarks with
            visibility < threshold have their x/y set to NaN. Default 0.3.
        _data_frames: Original frame dicts (list) from data["frames"].
            Used to read per-landmark visibility. If None, falls back to
            the _visibility columns already present in df.

    Returns:
        Modified DataFrame with low-confidence coordinates set to NaN.
    """
    df = df.copy()

    # Determine unique landmark names from column suffixes
    landmarks = sorted({
        c.rsplit("_", 1)[0]
        for c in df.columns
        if c.endswith("_x") and f"{c.rsplit('_', 1)[0]}_y" in df.columns
    })

    for lm_name in landmarks:
        xcol = f"{lm_name}_x"
        ycol = f"{lm_name}_y"
        viscol = f"{lm_name}_visibility"

        if _data_frames is not None:
            # Read visibility from original frame data
            for i in range(min(len(df), len(_data_frames))):
                frame = _data_frames[i]
                lm = frame.get("landmarks", {}).get(lm_name, {})
                vis = lm.get("visibility", 1.0)
                if vis < threshold:
                    df.at[df.index[i], xcol] = np.nan
                    df.at[df.index[i], ycol] = np.nan
        elif viscol in df.columns:
            # Fall back to visibility columns in the DataFrame
            mask = pd.to_numeric(df[viscol], errors="coerce") < threshold
            df.loc[mask, xcol] = np.nan
            df.loc[mask, ycol] = np.nan

    return df


def detect_outliers(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    **kwargs,
) -> pd.DataFrame:
    """Detect spike outliers via z-score and replace with linear interpolation.

    For each coordinate column (_x / _y), computes the z-score of each value.
    Values with |z| > z_thresh are replaced with NaN, then linearly
    interpolated. This removes sudden single-frame spikes caused by
    misdetections without affecting the overall trajectory shape.

    Args:
        df: DataFrame with pose coordinate columns.
        z_thresh: Z-score threshold for outlier detection. Values with
            |z| > z_thresh are treated as outliers. Default 3.0.

    Returns:
        Modified DataFrame with outlier spikes interpolated away.
    """
    df = df.copy()
    cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all() or s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        outliers = z.abs() > z_thresh
        if outliers.any():
            s[outliers] = np.nan
            s = s.interpolate(method="linear", limit_direction="both")
            s = s.bfill().ffill()
            df[c] = s

    return df


def data_quality_score(data: dict) -> dict:
    """Compute a composite data quality score from 0 to 100.

    Evaluates four quality dimensions of the extracted pose data:

    - **Detection rate**: Fraction of frames that have at least one landmark
      with valid (non-NaN) coordinates.
    - **Mean confidence**: Average frame-level confidence score across all
      frames (from the ``confidence`` field).
    - **Gap percentage**: Fraction of frames where all landmarks are NaN
      (complete detection failures).
    - **Jitter score**: Mean frame-to-frame displacement (Euclidean) of the
      hip center, normalized. Lower jitter = higher quality.

    The overall score is a weighted combination:
        overall = 0.3 * detection_rate + 0.3 * mean_confidence
                + 0.2 * (1 - gap_pct) + 0.2 * jitter_component

    Args:
        data: Pivot JSON dict with ``frames`` populated.

    Returns:
        Dict with keys: overall_score (0-100), detection_rate (0-1),
        mean_confidence (0-1), gap_pct (0-1), jitter_score (float).
        Also stored in data["quality"].
    """
    frames = data.get("frames", [])
    n = len(frames)
    if n == 0:
        result = {
            "overall_score": 0.0,
            "detection_rate": 0.0,
            "mean_confidence": 0.0,
            "gap_pct": 1.0,
            "jitter_score": 0.0,
        }
        data["quality"] = result
        return result

    # Detection rate: frames with at least one valid landmark
    detected = 0
    gap_count = 0
    confidences = []
    hip_centers = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        conf = frame.get("confidence", 0.0)
        confidences.append(conf if conf is not None else 0.0)

        has_valid = False
        all_nan = True
        for name, coords in lm.items():
            x = coords.get("x")
            y = coords.get("y")
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                has_valid = True
                all_nan = False
                break

        if has_valid:
            detected += 1
        if all_nan and len(lm) > 0:
            gap_count += 1

        # Compute hip center for jitter
        l_hip = lm.get("LEFT_HIP", {})
        r_hip = lm.get("RIGHT_HIP", {})
        lx = l_hip.get("x")
        ly = l_hip.get("y")
        rx = r_hip.get("x")
        ry = r_hip.get("y")
        if (lx is not None and ly is not None and rx is not None and ry is not None
                and not np.isnan(lx) and not np.isnan(ly)
                and not np.isnan(rx) and not np.isnan(ry)):
            hip_centers.append(((lx + rx) / 2, (ly + ry) / 2))
        else:
            hip_centers.append(None)

    detection_rate = detected / n
    mean_confidence = float(np.mean(confidences))
    gap_pct = gap_count / n

    # Jitter: mean frame-to-frame displacement of hip center
    displacements = []
    for i in range(1, len(hip_centers)):
        if hip_centers[i] is not None and hip_centers[i - 1] is not None:
            dx = hip_centers[i][0] - hip_centers[i - 1][0]
            dy = hip_centers[i][1] - hip_centers[i - 1][1]
            displacements.append(np.sqrt(dx ** 2 + dy ** 2))

    jitter_score = float(np.mean(displacements)) if displacements else 0.0

    # Jitter component: lower jitter = higher quality
    # Normalize: typical jitter < 0.01 is good; > 0.05 is poor
    jitter_component = max(0.0, 1.0 - jitter_score / 0.05)

    overall = (
        0.3 * detection_rate
        + 0.3 * min(mean_confidence, 1.0)
        + 0.2 * (1.0 - gap_pct)
        + 0.2 * jitter_component
    ) * 100.0

    result = {
        "overall_score": round(overall, 2),
        "detection_rate": round(detection_rate, 4),
        "mean_confidence": round(mean_confidence, 4),
        "gap_pct": round(gap_pct, 4),
        "jitter_score": round(jitter_score, 6),
    }
    data["quality"] = result
    return result


# ── Step registry ────────────────────────────────────────────────────


NORMALIZE_STEPS: Dict[str, Callable] = {
    "butterworth": filter_butterworth,
    "savgol": filter_savgol,
    "moving_mean": filter_moving_mean,
    "spline": filter_spline,
    "kalman": filter_kalman,
    "center_on_torso": center_on_torso,
    "align_skeleton": align_skeleton,
    "correct_bilateral": correct_bilateral,
    "correct_pixel_ratio": correct_pixel_ratio,
    "confidence_filter": confidence_filter,
    "detect_outliers": detect_outliers,
}


def register_normalize_step(name: str, func: Callable):
    """Register a custom normalization step.

    The function must accept (df: pd.DataFrame, **kwargs) -> pd.DataFrame.
    """
    NORMALIZE_STEPS[name] = func


def list_normalize_steps() -> list:
    """Return available normalization step names."""
    return list(NORMALIZE_STEPS.keys())


# ── Public API ───────────────────────────────────────────────────────


def normalize(
    data: dict,
    filters: Optional[List[str]] = None,
    steps: Optional[List[dict]] = None,
    butterworth_cutoff: float = 4.0,
    butterworth_order: int = 2,
    center: bool = False,
    align: bool = False,
    correct_limbs: bool = False,
    pixel_ratio: Optional[dict] = None,
    gap_max_frames: int = 10,
) -> dict:
    """Normalize and filter pose data in the pivot JSON.

    Supports two modes of operation:

    1. **Simple mode** -- pass filter names and flags::

        normalize(data, filters=["butterworth"], center=True)

    2. **Advanced mode** -- pass a list of step dicts::

        normalize(data, steps=[
            {"type": "butterworth", "cutoff": 6.0, "order": 4},
            {"type": "center_on_torso"},
        ])

    The *steps* list takes precedence over *filters*/*flags*.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    filters : list of str, optional
        Filter names to apply in order (simple mode).
    steps : list of dict, optional
        Step configurations with ``type`` and per-step params.
    butterworth_cutoff : float, optional
        Cutoff frequency for Butterworth in simple mode (default 4.0).
    butterworth_order : int, optional
        Filter order for Butterworth in simple mode (default 2).
    center : bool, optional
        Center on torso centroid (simple mode, default False).
    align : bool, optional
        Align skeleton (simple mode, default False).
    correct_limbs : bool, optional
        Correct bilateral segments (simple mode, default False).
    pixel_ratio : dict, optional
        Dict with ``input_width``, ``input_height``,
        ``processed_width``, ``processed_height``.
    gap_max_frames : int, optional
        Maximum gap length (in frames) that will be interpolated.
        Gaps longer than this are left as NaN rather than being
        interpolated across. Gap metadata is recorded in
        ``data["normalization"]["gaps"]``. Default 10.

    Returns
    -------
    dict
        Modified *data* dict (also modifies in place).

    Raises
    ------
    ValueError
        If *data* has no frames.
    """
    if not data.get("frames"):
        raise ValueError("No frames in data. Run extract() first.")

    fps = data.get("meta", {}).get("fps", 30.0)

    # Save raw frames on first call
    if "frames_raw" not in data:
        data["frames_raw"] = copy.deepcopy(data["frames"])

    # Convert to DataFrame for processing
    df = frames_to_dataframe(data["frames_raw"])

    # ── Gap handling: identify and protect long gaps ──────────────
    gap_info = []
    xy_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    # Detect long gaps BEFORE applying steps so they are preserved as NaN
    if gap_max_frames is not None and gap_max_frames > 0:
        for c in xy_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            is_nan = s.isna()
            if not is_nan.any():
                continue
            # Find contiguous NaN runs
            groups = (is_nan != is_nan.shift()).cumsum()
            for grp_id, grp in s.groupby(groups):
                if grp.isna().all() and len(grp) > gap_max_frames:
                    gap_info.append({
                        "column": c,
                        "start_frame": int(grp.index[0]),
                        "end_frame": int(grp.index[-1]),
                        "length": len(grp),
                    })

    # Build step list
    if steps is not None:
        # Advanced mode: steps provided directly
        step_list = steps
    else:
        # Simple mode: build from flags
        step_list = []
        if filters:
            for f in filters:
                params = {"type": f}
                if f == "butterworth":
                    params["cutoff"] = butterworth_cutoff
                    params["order"] = butterworth_order
                step_list.append(params)
        if correct_limbs:
            step_list.append({"type": "correct_bilateral"})
        if center:
            step_list.append({"type": "center_on_torso"})
        elif align:
            step_list.append({"type": "align_skeleton"})
        if pixel_ratio:
            if isinstance(pixel_ratio, dict):
                step_list.append({"type": "correct_pixel_ratio", **pixel_ratio})
            else:
                # pixel_ratio=True: auto-detect from video metadata
                meta = data.get("meta", {})
                step_list.append({
                    "type": "correct_pixel_ratio",
                    "input_width": meta.get("width", 1920),
                    "input_height": meta.get("height", 1080),
                })

    # Execute steps
    applied = []
    for step_config in step_list:
        step_type = step_config.get("type", "")
        if not step_type:
            continue

        func = NORMALIZE_STEPS.get(step_type)
        if func is None:
            logger.warning(f"Unknown normalize step: {step_type}, skipping")
            continue

        # Extract params (everything except 'type')
        params = {k: v for k, v in step_config.items() if k != "type"}

        # Inject fs for frequency-based filters
        if step_type in ("butterworth",):
            params.setdefault("fs", fps)

        # Inject _data_frames for confidence_filter
        if step_type == "confidence_filter":
            params.setdefault("_data_frames", data.get("frames_raw") or data.get("frames"))

        df = func(df, **params)
        applied.append(step_type)

    # ── Re-apply long-gap NaN protection after filtering ─────────
    if gap_max_frames is not None and gap_max_frames > 0:
        for gap in gap_info:
            c = gap["column"]
            if c in df.columns:
                start = gap["start_frame"]
                end = gap["end_frame"]
                idx_mask = (df.index >= start) & (df.index <= end)
                df.loc[idx_mask, c] = np.nan

    # Write back
    data["frames"] = dataframe_to_frames(df, data["frames_raw"])

    # Record normalization parameters
    data["normalization"] = {
        "steps": [dict(s) for s in step_list] if step_list else [],
        "steps_applied": applied,
        "fps_used": fps,
        "gap_max_frames": gap_max_frames,
        "gaps": gap_info,
    }

    return data
