"""Tests for frontal-plane angle support in cycle segmentation.

Verifies that segment_cycles() correctly includes frontal angles
(pelvis_list, hip_adduction, knee_valgus) in cycle summaries when
they are present in the angle frame data, and that backward
compatibility is preserved when they are absent.
"""

import numpy as np
import pytest

from conftest import run_full_pipeline, make_walking_data

from myogait.cycles import segment_cycles


# ── Helpers ──────────────────────────────────────────────────────────


def _add_frontal_angles_to_data(data):
    """Add frontal angle keys to angle frames for testing."""
    if data.get("angles") and data["angles"].get("frames"):
        for i, af in enumerate(data["angles"]["frames"]):
            t = i / max(len(data["angles"]["frames"]) - 1, 1)
            af["pelvis_list"] = 2.0 + 3.0 * np.sin(2 * np.pi * t)
            af["hip_adduction_L"] = 5.0 + 4.0 * np.sin(2 * np.pi * t)
            af["hip_adduction_R"] = 4.8 + 4.0 * np.sin(2 * np.pi * t + 0.1)
            af["knee_valgus_L"] = 1.0 + 2.0 * np.sin(2 * np.pi * t)
            af["knee_valgus_R"] = 1.2 + 2.0 * np.sin(2 * np.pi * t + 0.1)


def _run_pipeline_with_frontal(n_frames=300, fps=30.0):
    """Run full pipeline and inject frontal angles before cycle segmentation."""
    from myogait import normalize, compute_angles, detect_events
    data = make_walking_data(n_frames, fps)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    _add_frontal_angles_to_data(data)
    detect_events(data)
    cycles = segment_cycles(data)
    return data, cycles


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_no_frontal():
    """Pipeline without frontal angles."""
    data, cycles, stats = run_full_pipeline()
    return data, cycles, stats


@pytest.fixture(scope="module")
def pipeline_with_frontal():
    """Pipeline with frontal angles injected."""
    data, cycles = _run_pipeline_with_frontal()
    return data, cycles


# ── Tests: frontal keys present ──────────────────────────────────────


class TestFrontalAnglesPresent:
    """Tests when frontal angles are present in the data."""

    def test_cycle_has_pelvis_list(self, pipeline_with_frontal):
        """Cycle summary should include pelvis_list when present in data."""
        _data, cycles = pipeline_with_frontal
        cycle_list = cycles.get("cycles", [])
        assert len(cycle_list) > 0
        # At least one cycle should have pelvis_list in angles_normalized
        has_pelvis = any(
            "pelvis_list" in c.get("angles_normalized", {})
            for c in cycle_list
        )
        assert has_pelvis, "No cycle has pelvis_list in angles_normalized"

    def test_cycle_has_hip_adduction(self, pipeline_with_frontal):
        """Cycle should include hip_adduction when present."""
        _data, cycles = pipeline_with_frontal
        cycle_list = cycles.get("cycles", [])
        has_hip_add = any(
            "hip_adduction" in c.get("angles_normalized", {})
            for c in cycle_list
        )
        assert has_hip_add, "No cycle has hip_adduction in angles_normalized"

    def test_cycle_has_knee_valgus(self, pipeline_with_frontal):
        """Cycle should include knee_valgus when present."""
        _data, cycles = pipeline_with_frontal
        cycle_list = cycles.get("cycles", [])
        has_kv = any(
            "knee_valgus" in c.get("angles_normalized", {})
            for c in cycle_list
        )
        assert has_kv, "No cycle has knee_valgus in angles_normalized"

    def test_frontal_values_normalized_to_101(self, pipeline_with_frontal):
        """Each frontal angle array should be normalized to 101 points."""
        _data, cycles = pipeline_with_frontal
        for c in cycles.get("cycles", []):
            an = c.get("angles_normalized", {})
            for key in ["pelvis_list", "hip_adduction", "knee_valgus"]:
                if key in an:
                    assert len(an[key]) == 101, (
                        f"Expected 101 points for {key}, got {len(an[key])}"
                    )

    def test_summary_has_frontal_mean_std(self, pipeline_with_frontal):
        """Summary should contain mean/std for frontal joints."""
        _data, cycles = pipeline_with_frontal
        summary = cycles.get("summary", {})
        # At least one side should have frontal summary
        found_frontal = False
        for side in ("left", "right"):
            ss = summary.get(side, {})
            if "pelvis_list_mean" in ss:
                assert "pelvis_list_std" in ss
                assert len(ss["pelvis_list_mean"]) == 101
                assert len(ss["pelvis_list_std"]) == 101
                found_frontal = True
        assert found_frontal, "No frontal summary found in either side"

    def test_summary_hip_adduction_mean(self, pipeline_with_frontal):
        """Summary should contain hip_adduction_mean when present."""
        _data, cycles = pipeline_with_frontal
        summary = cycles.get("summary", {})
        found = False
        for side in ("left", "right"):
            ss = summary.get(side, {})
            if "hip_adduction_mean" in ss:
                assert len(ss["hip_adduction_mean"]) == 101
                found = True
        assert found, "No hip_adduction_mean in any side summary"

    def test_sagittal_still_present(self, pipeline_with_frontal):
        """Sagittal angles should still be present alongside frontal."""
        _data, cycles = pipeline_with_frontal
        cycle_list = cycles.get("cycles", [])
        assert len(cycle_list) > 0
        # At least one cycle should have hip (sagittal) too
        has_hip = any("hip" in c.get("angles_normalized", {}) for c in cycle_list)
        assert has_hip, "Sagittal hip should still be present"


# ── Tests: frontal keys absent (backward compatibility) ──────────────


class TestFrontalAnglesAbsent:
    """Tests when frontal angles are NOT present (backward compat)."""

    def test_no_frontal_keys_in_cycle(self, pipeline_no_frontal):
        """Without frontal data, cycles should not contain frontal keys."""
        _data, cycles, _stats = pipeline_no_frontal
        for c in cycles.get("cycles", []):
            an = c.get("angles_normalized", {})
            assert "pelvis_list" not in an
            assert "hip_adduction" not in an
            assert "knee_valgus" not in an

    def test_no_frontal_keys_in_summary(self, pipeline_no_frontal):
        """Without frontal data, summary should not contain frontal keys."""
        _data, cycles, _stats = pipeline_no_frontal
        summary = cycles.get("summary", {})
        for side in ("left", "right"):
            ss = summary.get(side, {})
            assert "pelvis_list_mean" not in ss
            assert "hip_adduction_mean" not in ss
            assert "knee_valgus_mean" not in ss

    def test_sagittal_unaffected(self, pipeline_no_frontal):
        """Sagittal angles should work exactly as before."""
        _data, cycles, _stats = pipeline_no_frontal
        cycle_list = cycles.get("cycles", [])
        assert len(cycle_list) > 0
        has_hip = any("hip" in c.get("angles_normalized", {}) for c in cycle_list)
        assert has_hip
        summary = cycles.get("summary", {})
        has_summary = False
        for side in ("left", "right"):
            ss = summary.get(side, {})
            if "hip_mean" in ss:
                has_summary = True
        assert has_summary
