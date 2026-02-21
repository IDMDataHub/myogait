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

    - filter_median: Median filter for spike/outlier removal.
      Standard preprocessing step in DeepLabCut and Pose2Sim pipelines.
      Ref: Pagnon D, Domalain M, Reveret L. Pose2Sim: An end-to-end
      workflow for 3D markerless kinematics. J Open Source Softw.
      2022;7(77):4362. doi:10.21105/joss.04362
      Mathis A, et al. DeepLabCut: markerless pose estimation of
      user-defined body parts with deep learning. Nat Neurosci.
      2018;21:1281-1289. doi:10.1038/s41593-018-0209-y

    - filter_kalman: Kalman filter for trajectory smoothing.
      Ref: Kalman RE. A new approach to linear filtering and prediction
      problems. J Basic Eng. 1960;82(1):35-45.
      doi:10.1115/1.3662552

    - filter_loess: LOESS/LOWESS locally weighted scatterplot smoothing.
      Ref: Cleveland WS. Robust locally weighted regression and smoothing
      scatterplots. J Am Stat Assoc. 1979;74(368):829-836.
      doi:10.1080/01621459.1979.10481038
      Adopted as the default smoothing filter in the Pose2Sim pipeline:
      Ref: Pagnon D, Domalain M, Reveret L. Pose2Sim: An end-to-end
      workflow for 3D markerless kinematics. J Open Source Softw.
      2022;7(77):4362. doi:10.21105/joss.04362

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


def filter_median(
    df: pd.DataFrame,
    kernel_size: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Median filter for spike/outlier removal.

    Applies a 1-D median filter independently to each coordinate column.
    This is the standard preprocessing step recommended by DeepLabCut and
    Pose2Sim pipelines to remove single-frame detection spikes before
    further smoothing.

    Ref: Pagnon D, Domalain M, Reveret L. Pose2Sim: An end-to-end
    workflow for 3D markerless kinematics. J Open Source Softw.
    2022;7(77):4362. doi:10.21105/joss.04362

    Args:
        df: DataFrame with pose coordinate columns.
        kernel_size: Size of the median filter window. Must be a positive
            odd integer. Default 3.

    Raises:
        ValueError: If *kernel_size* is not a positive odd integer.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )

    from scipy.signal import medfilt

    return _apply_on_xy(df, lambda v: medfilt(v, kernel_size=kernel_size))


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


def filter_loess(
    df: pd.DataFrame,
    frac: float = 0.1,
    it: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """LOESS/LOWESS locally weighted scatterplot smoothing.

    Applies the LOWESS (LOcally WEighted Scatterplot Smoothing) algorithm
    independently to each coordinate column. This is the default smoothing
    filter adopted by the Pose2Sim markerless kinematics pipeline.

    Ref: Cleveland WS. Robust locally weighted regression and smoothing
    scatterplots. J Am Stat Assoc. 1979;74(368):829-836.
    doi:10.1080/01621459.1979.10481038

    Pose2Sim convention:
        Pagnon D, Domalain M, Reveret L. Pose2Sim: An end-to-end
        workflow for 3D markerless kinematics. J Open Source Softw.
        2022;7(77):4362. doi:10.21105/joss.04362

    Requires the ``statsmodels`` package (``pip install statsmodels``).

    Args:
        df: DataFrame with pose coordinate columns.
        frac: Fraction of data used for each local regression, in (0, 1].
            Smaller values follow the data more closely; larger values
            produce smoother curves. Default 0.1.
        it: Number of robustness iterations. Higher values give more
            resistance to outliers. Default 3.

    Raises:
        ImportError: If ``statsmodels`` is not installed.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        raise ImportError(
            "statsmodels is required for the LOESS/LOWESS filter. "
            "Install it with: pip install statsmodels"
        )

    df = df.copy()
    cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            continue
        # Build valid (non-NaN) mask for LOWESS fitting
        valid = s.notna()
        if valid.sum() < 3:
            continue
        x_all = np.arange(len(s), dtype=float)
        x_valid = x_all[valid]
        y_valid = s.values[valid]
        try:
            smoothed = lowess(
                y_valid, x_valid, frac=float(frac), it=int(it),
                return_sorted=True,
            )
            # Interpolate back to all indices (including former NaN positions)
            df[c] = np.interp(x_all, smoothed[:, 0], smoothed[:, 1])
        except Exception as e:
            logger.warning(f"LOESS filter failed for column {c}: {e}")
            # Fill NaN values but leave the signal unsmoothed
            df[c] = s.interpolate(limit_direction="both").bfill().ffill()
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
            df[c] = df[c].clip(0.0, 1.0)
    else:
        # Original taller than processed → y coordinates need stretching
        for c in ycols:
            df[c] = pd.to_numeric(df[c], errors="coerce") / ratio
            df[c] = df[c].clip(0.0, 1.0)

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


def residual_analysis(
    df: pd.DataFrame,
    fs: float = 30.0,
    freq_range: tuple = (1.0, 15.0),
    freq_step: float = 0.5,
    order: int = 2,
) -> dict:
    """Determine optimal low-pass cutoff frequency via residual analysis.

    Implements Winter's residual analysis method for automatic selection
    of the cutoff frequency for low-pass Butterworth filtering of
    kinematic data.

    For each candidate cutoff frequency in *freq_range*, the signal is
    filtered and the RMS residual between the raw and filtered signal is
    computed.  At high cutoff frequencies the residual-vs-frequency curve
    is approximately linear (the filter removes only noise).  The optimal
    cutoff is the frequency at which the residual first rises above this
    noise-only regression line by more than one standard error, i.e.
    the point where the curve departs from linearity.

    Reference
    ---------
    Winter DA. *Biomechanics and Motor Control of Human Movement*.
    4th ed. Hoboken, NJ: Wiley; 2009. Chapter 2, Section 2.4.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pose coordinate columns (``*_x``, ``*_y``).
    fs : float, optional
        Sampling frequency in Hz (default 30.0).
    freq_range : tuple of (float, float), optional
        (min_freq, max_freq) in Hz to evaluate (default (1.0, 15.0)).
    freq_step : float, optional
        Step size in Hz between candidate frequencies (default 0.5).
    order : int, optional
        Butterworth filter order (default 2).

    Returns
    -------
    dict
        ``"optimal_cutoff"`` : float
            Recommended cutoff frequency (median across columns).
        ``"residuals"`` : dict[float, float]
            Mean RMS residual (across columns) at each candidate freq.
        ``"per_column"`` : dict[str, float]
            Optimal cutoff for each individual coordinate column.
    """
    from scipy.signal import butter, filtfilt

    cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    freqs = np.arange(freq_range[0], freq_range[1] + freq_step * 0.5, freq_step)
    nyq = 0.5 * float(fs)

    # Pre-process columns: interpolate NaN, convert to float arrays
    raw_signals = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            continue
        raw_signals[c] = (
            s.interpolate(limit_direction="both").bfill().ffill().to_numpy(float)
        )

    if not raw_signals:
        return {"optimal_cutoff": float(freqs[len(freqs) // 2]),
                "residuals": {float(f): 0.0 for f in freqs},
                "per_column": {}}

    # Compute RMS residuals for each frequency and column
    col_residuals = {c: {} for c in raw_signals}  # col -> {freq: rms}
    for fc in freqs:
        normal = float(fc) / nyq
        normal = max(min(normal, 0.99), 1e-6)
        b, a = butter(int(order), normal, btype="low", analog=False)
        for c, raw in raw_signals.items():
            try:
                filtered = filtfilt(b, a, raw)
            except Exception:
                col_residuals[c][float(fc)] = 0.0
                continue
            rms = float(np.sqrt(np.mean((raw - filtered) ** 2)))
            col_residuals[c][float(fc)] = rms

    # Find optimal cutoff per column
    per_column = {}
    for c, res_dict in col_residuals.items():
        per_column[c] = _find_knee_frequency(freqs, res_dict)

    # Mean residual across columns per frequency
    mean_residuals = {}
    for fc in freqs:
        vals = [col_residuals[c].get(float(fc), 0.0) for c in raw_signals]
        mean_residuals[float(fc)] = float(np.mean(vals)) if vals else 0.0

    # Overall optimal cutoff: median across per-column optima
    if per_column:
        optimal = float(np.median(list(per_column.values())))
    else:
        optimal = float(freqs[len(freqs) // 2])

    return {
        "optimal_cutoff": optimal,
        "residuals": mean_residuals,
        "per_column": per_column,
    }


def _find_knee_frequency(freqs: np.ndarray, res_dict: dict) -> float:
    """Find the frequency where the residual curve departs from linearity.

    Fits a regression line to the upper half of the residual curve (high-
    frequency region, which is approximately linear because only noise is
    being removed).  Walking down from high to low frequency, the first
    point whose residual exceeds the regression prediction by more than
    one standard error of the fit is identified as the knee / optimal
    cutoff.

    Parameters
    ----------
    freqs : np.ndarray
        Array of candidate frequencies (ascending).
    res_dict : dict[float, float]
        Mapping of frequency -> RMS residual.

    Returns
    -------
    float
        Optimal cutoff frequency.
    """
    residuals = np.array([res_dict.get(float(f), 0.0) for f in freqs])
    n = len(freqs)

    if n < 4:
        return float(freqs[n // 2])

    # Fit line to upper half of curve (high-freq region)
    upper_start = n // 2
    x_upper = freqs[upper_start:]
    y_upper = residuals[upper_start:]

    if len(x_upper) < 2 or np.std(y_upper) < 1e-15:
        return float(freqs[n // 2])

    # Linear regression on the upper portion
    coeffs = np.polyfit(x_upper, y_upper, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # Predicted values and standard error of the fit in the upper region
    y_pred_upper = slope * x_upper + intercept
    se = float(np.sqrt(np.mean((y_upper - y_pred_upper) ** 2)))

    if se < 1e-15:
        # Residual curve is essentially flat; pick middle frequency
        return float(freqs[n // 2])

    # Walk from high frequency toward low: find where residual departs
    # from the regression line by more than one SE
    predicted_all = slope * freqs + intercept
    deviations = residuals - predicted_all

    # Scan from the highest freq downward; find first freq where
    # the deviation exceeds the threshold
    for i in range(n - 1, -1, -1):
        if deviations[i] > se:
            return float(freqs[i])

    # Fallback: return lowest candidate
    return float(freqs[0])


def auto_cutoff_frequency(
    df: pd.DataFrame,
    fs: float = 30.0,
    method: str = "residual",
    **kwargs,
) -> float:
    """Automatically select a low-pass cutoff frequency.

    Convenience wrapper that dispatches to a specific method for
    determining the optimal Butterworth cutoff frequency.

    Currently supported methods:

    - ``"residual"``: Winter's residual analysis
      (see :func:`residual_analysis`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pose coordinate columns.
    fs : float, optional
        Sampling frequency in Hz (default 30.0).
    method : str, optional
        Selection method (default ``"residual"``).
    **kwargs
        Additional keyword arguments forwarded to the underlying method.

    Returns
    -------
    float
        Recommended cutoff frequency in Hz.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    supported = ("residual",)
    if method not in supported:
        raise ValueError(
            f"Unsupported method {method!r}. Choose from {supported}."
        )

    if method == "residual":
        result = residual_analysis(df, fs=fs, **kwargs)
        return result["optimal_cutoff"]

    # Future: elif method == "psd": ...


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
    "median": filter_median,
    "kalman": filter_kalman,
    "loess": filter_loess,
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


# ── Gap filling ──────────────────────────────────────────────────────


def fill_gaps(
    data: dict,
    method: str = "spline",
    max_gap_frames: int = 10,
    report: bool = False,
) -> dict:
    """Detect and interpolate gaps in landmark data.

    A gap is defined as a frame where a landmark has visibility=0 or
    NaN in its x/y coordinates. Gaps shorter than or equal to
    *max_gap_frames* are interpolated; longer gaps are left as NaN.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    method : str, optional
        Interpolation method: ``"linear"`` (np.interp), ``"spline"``
        (scipy CubicSpline), or ``"zero"`` (fill with 0). Default
        ``"spline"``.
    max_gap_frames : int, optional
        Maximum gap length (in frames) to interpolate. Gaps longer
        than this are left as NaN. Default 10.
    report : bool, optional
        If True, adds ``data["gap_fill_report"]`` with gap statistics.

    Returns
    -------
    dict
        Modified *data* dict (also modifies in place).

    Raises
    ------
    ValueError
        If *method* is not one of ``"linear"``, ``"spline"``, ``"zero"``.
    """
    valid_methods = ("linear", "spline", "zero")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    frames = data.get("frames", [])
    if not frames:
        return data

    # Collect all landmark names from the first frame
    landmark_names = list(frames[0].get("landmarks", {}).keys())
    n_frames = len(frames)

    report_data = {
        "method": method,
        "max_gap_frames": max_gap_frames,
        "landmarks_with_gaps": [],
        "total_gaps_filled": 0,
        "total_gaps_skipped": 0,
        "gap_sizes_filled": [],
    }

    for lm_name in landmark_names:
        # Extract x, y, visibility series
        x_series = np.full(n_frames, np.nan)
        y_series = np.full(n_frames, np.nan)

        for i, f in enumerate(frames):
            lm = f.get("landmarks", {}).get(lm_name, {})
            vis = lm.get("visibility", 1.0)
            x_val = lm.get("x")
            y_val = lm.get("y")

            # Mark as gap if visibility is 0 or coords are NaN
            is_gap = (vis == 0.0
                      or x_val is None or y_val is None
                      or (isinstance(x_val, float) and np.isnan(x_val))
                      or (isinstance(y_val, float) and np.isnan(y_val)))

            if not is_gap:
                x_series[i] = float(x_val)
                y_series[i] = float(y_val)

        # Detect contiguous gap runs
        is_nan = np.isnan(x_series)
        if not np.any(is_nan):
            continue

        landmark_had_gap = False

        # Find contiguous gap runs
        gap_starts = []
        gap_ends = []
        in_gap = False
        for i in range(n_frames):
            if is_nan[i] and not in_gap:
                gap_starts.append(i)
                in_gap = True
            elif not is_nan[i] and in_gap:
                gap_ends.append(i - 1)
                in_gap = False
        if in_gap:
            gap_ends.append(n_frames - 1)

        for gs, ge in zip(gap_starts, gap_ends):
            gap_len = ge - gs + 1

            if gap_len > max_gap_frames:
                report_data["total_gaps_skipped"] += 1
                continue

            # Need valid points before and after the gap for interpolation
            valid_indices = np.where(~is_nan)[0]
            if len(valid_indices) < 2:
                report_data["total_gaps_skipped"] += 1
                continue

            landmark_had_gap = True
            report_data["total_gaps_filled"] += 1
            report_data["gap_sizes_filled"].append(gap_len)

            gap_indices = np.arange(gs, ge + 1)

            if method == "zero":
                x_series[gap_indices] = 0.0
                y_series[gap_indices] = 0.0
            elif method == "linear":
                x_series[gap_indices] = np.interp(
                    gap_indices, valid_indices, x_series[valid_indices])
                y_series[gap_indices] = np.interp(
                    gap_indices, valid_indices, y_series[valid_indices])
            elif method == "spline":
                from scipy.interpolate import CubicSpline
                # Use valid indices for spline fitting
                try:
                    cs_x = CubicSpline(valid_indices, x_series[valid_indices])
                    cs_y = CubicSpline(valid_indices, y_series[valid_indices])
                    x_series[gap_indices] = cs_x(gap_indices)
                    y_series[gap_indices] = cs_y(gap_indices)
                except Exception:
                    # Fall back to linear if spline fails
                    x_series[gap_indices] = np.interp(
                        gap_indices, valid_indices, x_series[valid_indices])
                    y_series[gap_indices] = np.interp(
                        gap_indices, valid_indices, y_series[valid_indices])

            # Update valid indices after filling
            is_nan = np.isnan(x_series)

        if landmark_had_gap:
            report_data["landmarks_with_gaps"].append(lm_name)

        # Write filled values back to frames
        for i in range(n_frames):
            if not np.isnan(x_series[i]):
                frames[i]["landmarks"][lm_name]["x"] = float(x_series[i])
                frames[i]["landmarks"][lm_name]["y"] = float(y_series[i])

    # Compute summary stats for report
    if report_data["gap_sizes_filled"]:
        report_data["mean_gap_size"] = round(
            float(np.mean(report_data["gap_sizes_filled"])), 2)
    else:
        report_data["mean_gap_size"] = 0.0

    if report:
        data["gap_fill_report"] = report_data

    return data
