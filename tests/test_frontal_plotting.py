"""Tests for frontal-plane plotting support in myogait.plotting.

Verifies that plot_normative_comparison handles the ``plane`` parameter
correctly and that the plot_frontal_comparison convenience function works.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

import numpy as np
import pytest

from conftest import run_full_pipeline, make_walking_data

from myogait.plotting import plot_normative_comparison, plot_frontal_comparison
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


def _pipeline_with_frontal():
    """Run full pipeline and inject frontal angles."""
    from myogait import normalize, compute_angles, detect_events
    data = make_walking_data(300, 30.0)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    _add_frontal_angles_to_data(data)
    detect_events(data)
    cycles = segment_cycles(data)
    return data, cycles


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_data():
    """Standard pipeline (sagittal only)."""
    data, cycles, stats = run_full_pipeline()
    return data, cycles, stats


@pytest.fixture(scope="module")
def frontal_pipeline():
    """Pipeline with frontal angles."""
    data, cycles = _pipeline_with_frontal()
    return data, cycles


# ── Tests: plane="sagittal" (backward compat) ───────────────────────


class TestSagittalPlane:
    """plane='sagittal' should be the default and backward-compatible."""

    def test_default_returns_figure(self, pipeline_data):
        data, cycles, _ = pipeline_data
        fig = plot_normative_comparison(data, cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_explicit_sagittal(self, pipeline_data):
        data, cycles, _ = pipeline_data
        fig = plot_normative_comparison(data, cycles, plane="sagittal")
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        # Default sagittal joints: hip, knee, ankle, trunk = 4 subplots
        assert len(axes) == 4
        plt.close(fig)


# ── Tests: plane="frontal" ──────────────────────────────────────────


class TestFrontalPlane:
    """Tests for plane='frontal'."""

    def test_frontal_returns_figure(self, frontal_pipeline):
        data, cycles = frontal_pipeline
        fig = plot_normative_comparison(data, cycles, plane="frontal")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_frontal_has_3_subplots(self, frontal_pipeline):
        data, cycles = frontal_pipeline
        fig = plot_normative_comparison(data, cycles, plane="frontal")
        axes = fig.get_axes()
        # Default frontal joints: pelvis_list, hip_adduction, knee_valgus = 3
        assert len(axes) == 3
        plt.close(fig)

    def test_frontal_without_frontal_data(self, pipeline_data):
        """Frontal plot should not crash even without frontal cycle data."""
        data, cycles, _ = pipeline_data
        fig = plot_normative_comparison(data, cycles, plane="frontal")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── Tests: plane="both" ─────────────────────────────────────────────


class TestBothPlanes:
    """Tests for plane='both'."""

    def test_both_returns_figure(self, frontal_pipeline):
        data, cycles = frontal_pipeline
        fig = plot_normative_comparison(data, cycles, plane="both")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_both_has_7_subplots(self, frontal_pipeline):
        data, cycles = frontal_pipeline
        fig = plot_normative_comparison(data, cycles, plane="both")
        axes = fig.get_axes()
        # 4 sagittal + 3 frontal = 7
        assert len(axes) == 7
        plt.close(fig)


# ── Tests: plot_frontal_comparison convenience function ──────────────


class TestPlotFrontalComparison:
    """Tests for the plot_frontal_comparison convenience wrapper."""

    def test_returns_figure(self, frontal_pipeline):
        _data, cycles = frontal_pipeline
        fig = plot_frontal_comparison(cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_stratum(self, frontal_pipeline):
        _data, cycles = frontal_pipeline
        fig = plot_frontal_comparison(cycles, stratum="pediatric")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
