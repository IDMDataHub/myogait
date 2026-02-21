"""Tests for extended plotting functions in myogait.plotting.

Verifies that each new plotting function returns a matplotlib Figure
and does not crash on synthetic data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

import numpy as np
import pytest

from conftest import run_full_pipeline, make_walking_data

from myogait.plotting import (
    plot_normative_comparison,
    plot_gvs_profile,
    plot_quality_dashboard,
    plot_longitudinal,
    plot_arm_swing,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def pipeline_data():
    """Run full pipeline and return (data, cycles, stats)."""
    data, cycles, stats = run_full_pipeline()
    return data, cycles, stats


@pytest.fixture
def walking_data():
    """Return raw walking data (no pipeline run)."""
    return make_walking_data(n_frames=300, fps=30.0)


# ── plot_normative_comparison ────────────────────────────────────────

class TestPlotNormativeComparison:
    """Tests for plot_normative_comparison."""

    def test_returns_figure(self, pipeline_data):
        data, cycles, _stats = pipeline_data
        fig = plot_normative_comparison(data, cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_joints(self, pipeline_data):
        data, cycles, _stats = pipeline_data
        fig = plot_normative_comparison(data, cycles, joints=["hip", "knee"])
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_stratum_pediatric(self, pipeline_data):
        data, cycles, _stats = pipeline_data
        fig = plot_normative_comparison(data, cycles, stratum="pediatric")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_gvs_profile ────────────────────────────────────────────────

class TestPlotGVSProfile:
    """Tests for plot_gvs_profile."""

    def test_returns_figure(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_gvs_profile(cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_stratum_elderly(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_gvs_profile(cycles, stratum="elderly")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_quality_dashboard ──────────────────────────────────────────

class TestPlotQualityDashboard:
    """Tests for plot_quality_dashboard."""

    def test_returns_figure(self, walking_data):
        fig = plot_quality_dashboard(walking_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_four_panels(self, walking_data):
        fig = plot_quality_dashboard(walking_data)
        axes = fig.get_axes()
        # 4 subplots + 1 colorbar axis = 5 axes total
        assert len(axes) >= 4
        plt.close(fig)

    def test_empty_data(self):
        """Should not crash even with no frames."""
        data = {"frames": [], "meta": {"fps": 30.0}}
        fig = plot_quality_dashboard(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_longitudinal ───────────────────────────────────────────────

class TestPlotLongitudinal:
    """Tests for plot_longitudinal."""

    def test_returns_figure_cadence(self):
        sessions = [
            {
                "date": "2024-01-01",
                "stats": {"spatiotemporal": {"cadence_steps_per_min": 110}},
            },
            {
                "date": "2024-02-01",
                "stats": {"spatiotemporal": {"cadence_steps_per_min": 108}},
            },
            {
                "date": "2024-03-01",
                "stats": {"spatiotemporal": {"cadence_steps_per_min": 112}},
            },
        ]
        fig = plot_longitudinal(sessions, metric="cadence")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_returns_figure_symmetry(self):
        sessions = [
            {
                "date": "2024-01-15",
                "stats": {"symmetry": {"overall_si": 5.2}},
            },
            {
                "date": "2024-04-15",
                "stats": {"symmetry": {"overall_si": 3.1}},
            },
        ]
        fig = plot_longitudinal(sessions, metric="symmetry")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_error_bars(self):
        sessions = [
            {
                "date": "2024-01-01",
                "stats": {"gps_2d_overall": 8.5},
                "error": 1.2,
            },
            {
                "date": "2024-06-01",
                "stats": {"gps_2d_overall": 7.1},
                "error": 0.9,
            },
        ]
        fig = plot_longitudinal(sessions, metric="gps_2d_overall")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_empty_sessions(self):
        fig = plot_longitudinal([], metric="cadence")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_arm_swing ──────────────────────────────────────────────────

class TestPlotArmSwing:
    """Tests for plot_arm_swing."""

    def test_returns_figure(self, pipeline_data):
        data, cycles, _stats = pipeline_data
        fig = plot_arm_swing(data, cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_has_two_subplots(self, pipeline_data):
        data, cycles, _stats = pipeline_data
        fig = plot_arm_swing(data, cycles)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)
