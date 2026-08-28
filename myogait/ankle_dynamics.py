"""Restore the ankle sagittal dynamics a markerless pose estimator attenuates.

A 2-D pose estimator behaves, on a joint-angle waveform, like a subject-
independent low-pass filter: it keeps ~80 % of the signal at 1 Hz but only
~20 % at 6-7 Hz, so the *fast* push-off plantar-flexion is flattened and the
ankle range of motion is systematically under-read (bias ≈ -11 deg on BioCV).

We model the estimator as a linear time-invariant system with transfer
function ``H(f)`` and invert it (Wiener deconvolution). ``H(f)`` was estimated
once against synchronous optical mocap (Vicon) on the BioCV / Bath BATH-01258
dataset (9 subjects, 85 walking trials, cam01 lateral view) and is embedded
below -- so no mocap is needed at run time.

Crucially the correction is applied as **mean restoration**: because the
deconvolution is linear, ``mean(deconv(cycles)) == deconv(mean(cycles))``. The
systematic deficit lives in the *mean* cycle; the cycle-to-cycle spread is the
clinical variability signal we must not touch. So we deconvolve only the mean
waveform, obtain the per-phase correction ``delta = deconv(mean) - mean`` and
add it to *every* cycle. This restores the systematic amplitude (ankle ROM
bias roughly halved) while leaving inter-cycle variability exactly unchanged
and restoring inter-patient variability toward the optical reference.

Validated leave-one-subject-out: ankle |ROM error| 10.8 -> 6.7 deg, bias
-10.6 -> -5.3 deg, inter-cycle SD 1.73 -> 1.73 deg (unchanged). It improves
every subject and makes no gait-shape assumption (the filter is in Hz,
adaptive to cadence), so it is safe on pathological gait.

The residual bias (~5 deg) is information the pose estimator never captured;
a linear inverse filter cannot recreate it.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

# ── Calibrated ankle transfer function H(f) ──────────────────────────
# Estimated on BioCV BATH-01258 (Sapiens-2 cam01 lateral) vs Visual3D Vicon,
# 9 subjects / 85 walking trials. Complex per-harmonic Wiener transfer of the
# pose estimator (video = H * truth), sampled at the stride harmonics.
ANKLE_TF_FREQ_HZ = np.array([
    0.933136, 1.866273, 2.799409, 3.732546, 4.665682, 5.598819, 6.531955, 7.465092,
])
_ANKLE_TF_H = np.array([
    0.743620 - 0.091119j, 0.705380 - 0.112409j, 0.701172 - 0.136082j, 0.390251 - 0.072741j,
    0.354046 - 0.010892j, 0.170991 - 0.048969j, 0.141292 + 0.025642j, 0.066250 - 0.001067j,
])
ANKLE_TF_H = _ANKLE_TF_H
ANKLE_TF_REG = 0.08          # Wiener regularisation (noise vs restoration trade-off)
ANKLE_TF_N_HARMONICS = 8
ANKLE_TF_METADATA = {
    "dataset": "BioCV BATH-01258 (Bath), Sapiens-2 cam01 lateral vs Visual3D Vicon",
    "n_subjects": 9, "n_trials": 85,
    "validation": "leave-one-subject-out; ankle |ROM err| 10.8->6.7 deg, "
                  "bias -10.6->-5.3 deg, inter-cycle SD unchanged (1.73)",
}


def _deconvolve_mean(mean_wave: np.ndarray, stride_s: float,
                     freq=ANKLE_TF_FREQ_HZ, H=ANKLE_TF_H, reg=ANKLE_TF_REG,
                     n_harmonics=ANKLE_TF_N_HARMONICS) -> np.ndarray:
    """Wiener-deconvolve one 101-point cycle-mean waveform.

    The cycle is one period; harmonic ``k`` sits at ``k / stride_s`` Hz, so the
    embedded ``H(f)`` is interpolated at the subject's own cadence (cadence-
    adaptive => no healthy-gait assumption). Only harmonics 1..N are touched;
    the mean (DC) is preserved.
    """
    w = np.asarray(mean_wave, dtype=float)
    if w.size < 3 or not np.isfinite(w).all() or stride_s <= 0:
        return w
    V = np.fft.rfft(w[:100])
    Vd = V.copy()
    f0 = 1.0 / stride_s
    mag = np.abs(H)
    pha = np.unwrap(np.angle(H))
    for k in range(1, min(n_harmonics, len(V) - 1) + 1):
        fk = k * f0
        Hk = np.interp(fk, freq, mag) * np.exp(1j * np.interp(fk, freq, pha))
        Vd[k] = V[k] * np.conj(Hk) / (abs(Hk) ** 2 + reg)
    wd = np.fft.irfft(Vd, n=100)
    return np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, 100), wd)


def ankle_restoration_delta(cycles: dict, side: str) -> Optional[np.ndarray]:
    """Return the per-phase (101-point) ankle correction for one side, or None.

    ``delta = deconv(mean_ankle) - mean_ankle``; add it to each cycle's ankle
    waveform to restore the systematic amplitude without altering the spread.
    """
    side_cycles = [c for c in cycles.get("cycles", [])
                   if c.get("side") == side
                   and len(c.get("angles_normalized", {}).get("ankle", [])) == 101]
    if len(side_cycles) < 2:
        return None
    durations = [c["duration"] for c in side_cycles if c.get("duration", 0) > 0]
    if not durations:
        return None
    stride_s = float(np.mean(durations))
    waves = np.array([c["angles_normalized"]["ankle"] for c in side_cycles], dtype=float)
    mean_w = waves.mean(axis=0)
    return _deconvolve_mean(mean_w, stride_s) - mean_w


def restore_ankle_dynamics(cycles: dict, inplace: bool = False) -> dict:
    """Correct the ankle waveform of every cycle by mean restoration.

    Parameters
    ----------
    cycles : dict
        Output of :func:`segment_cycles`.
    inplace : bool
        If False (default) operate on a deep copy and leave the input untouched.

    Returns
    -------
    dict
        Cycles whose ``angles_normalized["ankle"]`` have the calibrated
        push-off restoration added, per side. Cycle-to-cycle variability is
        preserved exactly. A ``dynamics_restored`` marker is set on the summary.
    """
    out = cycles if inplace else copy.deepcopy(cycles)
    applied = []
    for side in ("left", "right"):
        delta = ankle_restoration_delta(out, side)
        if delta is None:
            continue
        for c in out.get("cycles", []):
            if c.get("side") == side and len(c.get("angles_normalized", {}).get("ankle", [])) == 101:
                c["angles_normalized"]["ankle"] = (np.asarray(c["angles_normalized"]["ankle"], float) + delta).tolist()
        applied.append(side)
    if applied:
        out.setdefault("summary", {})["ankle_dynamics_restored"] = applied
    return out
