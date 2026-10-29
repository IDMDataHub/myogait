"""Tests for summarize_cycles / reject_outlier_cycles."""
import numpy as np

from myogait import reject_outlier_cycles, summarize_cycles


def _cycle(side, ankle_amp, shift=0, i=0):
    x = np.linspace(0, 2 * np.pi, 101)
    base = {"hip": 20 * np.cos(x), "knee": 30 * (1 - np.cos(x)),
            "ankle": ankle_amp * np.sin(np.roll(x, shift))}
    return {"cycle_id": i, "side": side,
            "angles_normalized": {k: v.tolist() for k, v in base.items()}}


def test_rejects_shape_outlier_and_recomputes_summary():
    rng = np.random.default_rng(0)
    cycles = [_cycle("left", 15 + rng.normal(0, 0.5), i=i) for i in range(8)]
    cycles.append(_cycle("left", 15, shift=25, i=99))  # mistimed push-off
    out = reject_outlier_cycles({"cycles": cycles, "summary": {}})
    ids = [c["cycle_id"] for c in out["cycles"]]
    assert 99 not in ids and len(ids) == 8
    assert out["summary"]["n_rejected_outlier"] == 1
    assert out["summary"]["left"]["n_cycles"] == 8
    expected = np.mean([c["angles_normalized"]["ankle"] for c in out["cycles"]], 0)
    np.testing.assert_allclose(out["summary"]["left"]["ankle_mean"], expected)


def test_small_side_untouched():
    cycles = [_cycle("right", 15, i=i) for i in range(3)] + [_cycle("right", 15, shift=30, i=9)]
    out = reject_outlier_cycles({"cycles": cycles, "summary": {}})
    assert len(out["cycles"]) == 4
    assert out["summary"]["n_rejected_outlier"] == 0


def test_summarize_cycles_per_side():
    cycles = [_cycle("left", 10), _cycle("right", 20)]
    s = summarize_cycles(cycles)
    assert s["left"]["n_cycles"] == 1 and s["right"]["n_cycles"] == 1
    assert np.isclose(max(s["right"]["ankle_mean"]), 20, atol=0.1)
