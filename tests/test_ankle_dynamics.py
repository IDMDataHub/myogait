"""Tests for the calibrated ankle-dynamics restoration (mean-restoration
deconvolution). Key invariants: the correction restores the systematic mean
amplitude while leaving inter-cycle variability EXACTLY unchanged, and never
mutates the caller's cycles."""
import copy

import numpy as np

from myogait.ankle_dynamics import (
    restore_ankle_dynamics, ankle_restoration_delta, _deconvolve_mean,
    ANKLE_TF_FREQ_HZ, ANKLE_TF_H,
)


def _make_cycles(n=6, seed=0, side="left"):
    """Synthetic cycles: an attenuated ankle waveform (push-off dip) + per-cycle
    noise, so restoration has harmonics to act on."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 101)
    base = 8 * np.sin(2 * np.pi * t) - 10 * np.exp(-((t - 0.6) / 0.05) ** 2)
    cycles = {"cycles": []}
    for i in range(n):
        w = base + rng.normal(0, 1.2, 101)
        cycles["cycles"].append({
            "cycle_id": i, "side": side, "duration": 1.05,
            "start_frame": i * 40, "end_frame": i * 40 + 40, "toe_off_frame": i * 40 + 24,
            "angles_normalized": {"ankle": w.tolist(),
                                  "knee": (30 + 25 * np.sin(2 * np.pi * t)).tolist(),
                                  "hip": (10 * np.sin(2 * np.pi * t)).tolist()},
        })
    return cycles


def _per_phase_sd(cycles, side="left"):
    a = np.array([c["angles_normalized"]["ankle"] for c in cycles["cycles"] if c["side"] == side])
    return a.std(axis=0)


def test_inter_cycle_variability_is_preserved_exactly():
    """The whole point of mean-restoration: the per-phase spread is untouched."""
    cyc = _make_cycles()
    before = _per_phase_sd(cyc)
    out = restore_ankle_dynamics(cyc)
    after = _per_phase_sd(out)
    assert np.allclose(before, after, atol=1e-9)


def test_restoration_changes_the_mean_waveform():
    cyc = _make_cycles()
    a0 = np.array([c["angles_normalized"]["ankle"] for c in cyc["cycles"]]).mean(0)
    out = restore_ankle_dynamics(cyc)
    a1 = np.array([c["angles_normalized"]["ankle"] for c in out["cycles"]]).mean(0)
    assert not np.allclose(a0, a1)                     # a correction was applied
    assert np.ptp(a1) >= np.ptp(a0) - 1e-6             # amplitude not reduced (push-off deepened)


def test_input_cycles_are_not_mutated():
    cyc = _make_cycles()
    ref = copy.deepcopy(cyc)
    restore_ankle_dynamics(cyc)                        # inplace defaults to False
    assert cyc["cycles"][0]["angles_normalized"]["ankle"] == ref["cycles"][0]["angles_normalized"]["ankle"]


def test_marker_and_delta_are_constant_across_cycles():
    cyc = _make_cycles()
    out = restore_ankle_dynamics(cyc)
    assert "left" in out["summary"]["ankle_dynamics_restored"]
    # delta is the same for every cycle (that is what preserves the spread)
    deltas = [np.array(o["angles_normalized"]["ankle"]) - np.array(i["angles_normalized"]["ankle"])
              for i, o in zip(cyc["cycles"], out["cycles"])]
    for d in deltas[1:]:
        assert np.allclose(d, deltas[0])


def test_too_few_cycles_is_a_noop():
    cyc = _make_cycles(n=1)
    out = restore_ankle_dynamics(cyc)
    assert ankle_restoration_delta(cyc, "left") is None
    assert "ankle_dynamics_restored" not in out.get("summary", {})


def test_deconvolve_preserves_length_and_finiteness():
    w = np.array((8 * np.sin(2 * np.pi * np.linspace(0, 1, 101))).tolist())
    out = _deconvolve_mean(w, stride_s=1.1)
    assert out.shape == (101,) and np.isfinite(out).all()


def test_filter_is_a_lowpass():
    """Embedded transfer function must attenuate high frequencies (|H| decreasing)."""
    mag = np.abs(ANKLE_TF_H)
    assert mag[0] > mag[5]                              # keeps more at ~1 Hz than ~5-6 Hz
    assert (ANKLE_TF_FREQ_HZ[1:] > ANKLE_TF_FREQ_HZ[:-1]).all()


def test_analyze_gait_flag_runs_and_is_optin():
    from conftest import run_full_pipeline
    data, cycles, _ = run_full_pipeline(n_frames=300, fps=30.0)
    import myogait as mg
    s_off = mg.analyze_gait(data, cycles)
    s_on = mg.analyze_gait(data, cycles, restore_ankle_dynamics=True)
    assert isinstance(s_off, dict) and isinstance(s_on, dict)
