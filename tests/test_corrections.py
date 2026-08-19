"""Tests for myogait.corrections — apply_linear_detrend + typing imports.

Regression context: commit 69f21f6 introduced apply_linear_detrend()
with ``Optional[List[str]]`` annotations but without importing the
typing names.  ``from __future__ import annotations`` masked the bug
at import time (annotations become strings), but the CI lint gate
(ruff --select E,W,F) failed with 2x F821 and
``typing.get_type_hints()`` raised NameError.  These tests pin both
the typing contract and the detrend behaviour documented in the
function docstring: the anatomical mean and the per-cycle ROM are
preserved while the slow DC drift is removed.
"""

import typing

import numpy as np
import pytest

from myogait.corrections import apply_linear_detrend

# ── Typing contract (the actual CI-breaking bug) ─────────────────────


def test_type_hints_resolve():
    """get_type_hints() must resolve — it evaluates string annotations.

    Before the fix this raised NameError: name 'Optional' is not defined.
    """
    hints = typing.get_type_hints(apply_linear_detrend)
    assert "data" in hints
    assert "joints" in hints


# ── Helpers ──────────────────────────────────────────────────────────

# Synthetic baseline joint angle (deg): any constant works — detrending
# must be invariant to the DC level, which test_removes_linear_drift
# asserts via the preserved mean.
BASE_ANGLE_DEG = 10.0

# Injected drift slope (deg/frame) for the pure-ramp tests.  Magnitude
# chosen from the apply_linear_detrend docstring ("hip angle sliding by
# 10-30 deg from the first to the last cycle"): 0.2 deg/frame over a
# 100-frame recording is a 20 deg drift, mid-range of that document.
DEFAULT_DRIFT = 0.2


def _make_angle_data(n_frames=100, drift_deg_per_frame=DEFAULT_DRIFT,
                     amp=0.0, period=25, joints=None):
    """Build a pivot dict with a linear drift and optional sinusoid."""
    if joints is None:
        joints = ["hip_L", "hip_R", "knee_L", "knee_R",
                  "ankle_L", "ankle_R", "trunk_angle"]
    frames = []
    for i in range(n_frames):
        osc = amp * np.sin(2 * np.pi * i / period) if amp else 0.0
        drift = drift_deg_per_frame * i
        frame = {"frame_idx": i}
        for key in joints:
            frame[key] = float(BASE_ANGLE_DEG + drift + osc)
        frames.append(frame)
    return {"angles": {"frames": frames}}


def _slope_of(values):
    """Least-squares slope of a series (NaN-safe)."""
    vals = np.asarray(values, dtype=float)
    mask = ~np.isnan(vals)
    idx = np.arange(len(vals))
    slope, _ = np.polyfit(idx[mask], vals[mask], 1)
    return float(slope)


def _series(data, key):
    return [f.get(key) for f in data["angles"]["frames"]]


def _per_cycle_ptp(values, period=25):
    """Median peak-to-peak amplitude computed within each cycle."""
    vals = np.asarray(values, dtype=float)
    n_cycles = len(vals) // period
    ptps = []
    for c in range(n_cycles):
        seg = vals[c * period:(c + 1) * period]
        if not np.isnan(seg).any():
            ptps.append(float(np.ptp(seg)))
    return float(np.median(ptps)) if ptps else float("nan")


# ── Detrend behaviour ────────────────────────────────────────────────


def test_removes_linear_drift_preserves_mean():
    data = _make_angle_data()  # pure ramp
    before = _series(data, "hip_L")
    assert abs(_slope_of(before) - DEFAULT_DRIFT) < 1e-9  # drift present
    before_mean = float(np.mean(before))

    apply_linear_detrend(data)

    after = _series(data, "hip_L")
    assert abs(_slope_of(after)) < 1e-9  # drift gone
    assert abs(float(np.mean(after)) - before_mean) < 1e-9  # mean kept


def test_preserves_per_cycle_rom():
    """The documented contract is per-cycle ROM, not global peak-to-peak.

    A slow drift contaminates windowed ROM measurements (the ramp tilts
    each gait-cycle window); detrending must restore the oscillation
    amplitude within every cycle.

    Math note: OLS detrend subtracts its own fitted line, so the
    residual slope is ~0 by construction (the OLS residual is
    orthogonal to the regressors).  On a ramp+sinusoid composite the
    fitted slope still carries the ramp/sinusoid cross-term bias, which
    modulates the oscillation slightly; the remaining ROM deficit is
    exactly that projection (here ~0.19 deg on a 10 deg ROM), and the
    element-wise pin below reproduces the documented algorithm
    (OLS fit, subtract trend, re-add mean) from first principles
    instead of hardcoding magic numbers.
    """
    period = 20
    drift, amp = 0.1, 5.0
    data = _make_angle_data(drift_deg_per_frame=drift, amp=amp, period=period)
    before = np.asarray(_series(data, "hip_L"))
    before_ptp = _per_cycle_ptp(before, period)
    true_ptp = 2.0 * amp
    assert before_ptp < true_ptp  # drift contaminates windowed ROM

    apply_linear_detrend(data)

    after = np.asarray(_series(data, "hip_L"))

    # Anatomical mean preserved (documented contract)
    assert float(np.mean(after)) == pytest.approx(
        float(np.mean(before)), abs=1e-9)
    # Residual slope ~0: detrend subtracts its own OLS fit
    assert abs(_slope_of(after)) < 1e-9

    # Element-wise pin against the algorithm reproduced independently
    idx = np.arange(len(before))
    slope_hat, intercept_hat = np.polyfit(idx, before, 1)
    expected = (before - (slope_hat * idx + intercept_hat)
                + float(np.mean(before)))
    assert after == pytest.approx(expected, abs=1e-9)

    # Clinical meaning: ROM restored up to the cross-term distortion
    after_ptp = _per_cycle_ptp(after, period)
    assert after_ptp == pytest.approx(
        _per_cycle_ptp(expected, period), abs=1e-9)
    assert after_ptp > before_ptp
    assert after_ptp == pytest.approx(true_ptp, abs=0.2)


def test_sets_marker_and_is_idempotent():
    data = _make_angle_data()
    apply_linear_detrend(data)
    assert data["angles"]["linear_detrended"] is True

    snapshot = [dict(f) for f in data["angles"]["frames"]]
    apply_linear_detrend(data)  # second call must be a no-op
    assert data["angles"]["frames"] == snapshot


def test_short_series_untouched():
    """Joints below the internal minimum-sample guard are left unchanged.

    The guard requires >= 20 valid samples (corrections.py). Series
    clearly under any such bound must pass through bit-identical.
    """
    for n_frames in (0, 1, 5, 10, 19):
        data = _make_angle_data(n_frames=n_frames)
        before = _series(data, "knee_R")
        apply_linear_detrend(data)
        assert _series(data, "knee_R") == before, f"n={n_frames} modified"
    # Boundary (exactly 20 valid samples) and above ARE detrended
    for n_frames in (20, 25):
        data = _make_angle_data(n_frames=n_frames)
        apply_linear_detrend(data)
        assert abs(_slope_of(_series(data, "knee_R"))) < 1e-9, (
            f"n={n_frames} not detrended")


def test_nan_values_skipped():
    """NaN angle samples are preserved, valid neighbours are detrended."""
    data = _make_angle_data(n_frames=50)
    data["angles"]["frames"][7]["ankle_L"] = float("nan")
    apply_linear_detrend(data)
    assert np.isnan(data["angles"]["frames"][7]["ankle_L"])
    valid = [f["ankle_L"] for f in data["angles"]["frames"]
             if not np.isnan(f["ankle_L"])]
    assert abs(_slope_of(valid)) < 1e-9


def test_none_values_skipped():
    data = _make_angle_data(n_frames=50)
    data["angles"]["frames"][5]["hip_L"] = None
    data["angles"]["frames"][6]["hip_L"] = None
    apply_linear_detrend(data)
    assert data["angles"]["frames"][5]["hip_L"] is None
    assert data["angles"]["frames"][6]["hip_L"] is None
    # Valid frames are detrended anyway
    valid = [f["hip_L"] for f in data["angles"]["frames"]
             if f["hip_L"] is not None]
    assert abs(_slope_of(valid)) < 1e-9


def test_other_frame_keys_preserved():
    """Detrending touches only the target joints, never other keys."""
    data = _make_angle_data(n_frames=40)
    for f in data["angles"]["frames"]:
        f["landmark_positions"] = {"LEFT_HIP": [0.5, 0.5]}
    apply_linear_detrend(data)
    for f in data["angles"]["frames"]:
        assert f["landmark_positions"] == {"LEFT_HIP": [0.5, 0.5]}
        assert "frame_idx" in f


def test_custom_joints_only():
    data = _make_angle_data()  # pure ramp on all joints
    apply_linear_detrend(data, joints=["knee_L"])
    assert abs(_slope_of(_series(data, "knee_L"))) < 1e-9
    # Unlisted joints keep their drift
    assert abs(_slope_of(_series(data, "hip_L")) - DEFAULT_DRIFT) < 1e-9


def test_empty_angles_is_noop():
    data = {"angles": {"frames": []}}
    apply_linear_detrend(data)
    assert data["angles"].get("linear_detrended") is not True

    data_no_angles = {}
    result = apply_linear_detrend(data_no_angles)
    assert result is data_no_angles
