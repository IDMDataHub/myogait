"""Tests for the five new plotting functions added to myogait.plotting.

Covers:
    - plot_session_comparison
    - plot_cadence_profile
    - plot_rom_summary
    - plot_butterfly
    - animate_normative_comparison
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

import pytest

from conftest import run_full_pipeline, make_walking_data

from myogait.plotting import (
    plot_session_comparison,
    plot_cadence_profile,
    plot_rom_summary,
    plot_butterfly,
    animate_normative_comparison,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def pipeline_data():
    """Run full pipeline and return (data, cycles, stats)."""
    data, cycles, stats = run_full_pipeline()
    return data, cycles, stats


@pytest.fixture
def two_sessions():
    """Create two sessions using the full pipeline."""
    data_a, cycles_a, stats_a = run_full_pipeline(n_frames=300, fps=30.0)
    data_b, cycles_b, stats_b = run_full_pipeline(n_frames=300, fps=30.0)
    session_a = {"data": data_a, "cycles": cycles_a, "stats": stats_a, "label": "Pre-op"}
    session_b = {"data": data_b, "cycles": cycles_b, "stats": stats_b, "label": "Post-op"}
    return session_a, session_b


@pytest.fixture
def walking_data_with_events():
    """Return walking data with events detected."""
    from myogait import normalize, compute_angles, detect_events
    data = make_walking_data(n_frames=300, fps=30.0)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data)
    return data


@pytest.fixture
def rom_data():
    """Return a synthetic ROM data dict."""
    return {
        "hip_L": {"rom_mean": 42.0, "rom_std": 3.5},
        "hip_R": {"rom_mean": 40.0, "rom_std": 4.0},
        "knee_L": {"rom_mean": 58.0, "rom_std": 5.0},
        "knee_R": {"rom_mean": 55.0, "rom_std": 4.5},
        "ankle_L": {"rom_mean": 25.0, "rom_std": 2.5},
        "ankle_R": {"rom_mean": 24.0, "rom_std": 3.0},
    }


# ── plot_session_comparison ──────────────────────────────────────────

class TestPlotSessionComparison:
    """Tests for plot_session_comparison."""

    def test_returns_figure(self, two_sessions):
        session_a, session_b = two_sessions
        fig = plot_session_comparison(session_a, session_b)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_default_joints_grid(self, two_sessions):
        session_a, session_b = two_sessions
        fig = plot_session_comparison(session_a, session_b)
        axes = fig.get_axes()
        # 2 rows x 3 joints = 6 subplots
        assert len(axes) == 6
        plt.close("all")

    def test_custom_joints(self, two_sessions):
        session_a, session_b = two_sessions
        fig = plot_session_comparison(session_a, session_b, joints=["hip", "knee"])
        axes = fig.get_axes()
        # 2 rows x 2 joints = 4 subplots
        assert len(axes) == 4
        plt.close("all")

    def test_single_joint(self, two_sessions):
        session_a, session_b = two_sessions
        fig = plot_session_comparison(session_a, session_b, joints=["knee"])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_figsize(self, two_sessions):
        session_a, session_b = two_sessions
        fig = plot_session_comparison(session_a, session_b, figsize=(20, 10))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ── plot_cadence_profile ─────────────────────────────────────────────

class TestPlotCadenceProfile:
    """Tests for plot_cadence_profile."""

    def test_returns_figure(self, walking_data_with_events):
        fig = plot_cadence_profile(walking_data_with_events)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_has_labels(self, walking_data_with_events):
        fig = plot_cadence_profile(walking_data_with_events)
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Cadence (steps/min)"
        plt.close("all")

    def test_raises_without_events(self):
        data = {"meta": {"fps": 30.0}}
        with pytest.raises(ValueError, match="No events"):
            plot_cadence_profile(data)
        plt.close("all")

    def test_few_hs(self):
        """Handles case with very few heel strikes."""
        data = {
            "events": {"left_hs": [{"time": 1.0}], "right_hs": []},
            "meta": {"fps": 30.0},
        }
        fig = plot_cadence_profile(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ── plot_rom_summary ─────────────────────────────────────────────────

class TestPlotRomSummary:
    """Tests for plot_rom_summary."""

    def test_returns_figure(self, rom_data):
        fig = plot_rom_summary(rom_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_stratum_elderly(self, rom_data):
        fig = plot_rom_summary(rom_data, stratum="elderly")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_figsize(self, rom_data):
        fig = plot_rom_summary(rom_data, figsize=(14, 8))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_empty_rom_data(self):
        """Should not crash with empty data."""
        fig = plot_rom_summary({})
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_partial_rom_data(self):
        """Works when only some joints are present."""
        fig = plot_rom_summary({"hip_L": {"rom_mean": 40.0, "rom_std": 2.0}})
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ── plot_butterfly ───────────────────────────────────────────────────

class TestPlotButterfly:
    """Tests for plot_butterfly."""

    def test_returns_figure(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_butterfly(cycles)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_two_subplots(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_butterfly(cycles)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close("all")

    def test_hip_joint(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_butterfly(cycles, joint="hip")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_ankle_joint(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_butterfly(cycles, joint="ankle")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_title(self, pipeline_data):
        _data, cycles, _stats = pipeline_data
        fig = plot_butterfly(cycles, joint="knee")
        ax_top = fig.get_axes()[0]
        assert "Butterfly Plot - Knee" in ax_top.get_title()
        plt.close("all")

    def test_empty_cycles(self):
        """No crash with empty cycles dict."""
        fig = plot_butterfly({"cycles": [], "summary": {}})
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ── animate_normative_comparison ─────────────────────────────────────

class TestAnimateNormativeComparison:
    """Tests for animate_normative_comparison."""

    def test_returns_string(self, pipeline_data, tmp_path):
        _data, cycles, _stats = pipeline_data
        out = str(tmp_path / "test_anim.gif")
        result = animate_normative_comparison(cycles, output_path=out)
        assert isinstance(result, str)
        plt.close("all")

    def test_file_created(self, pipeline_data, tmp_path):
        _data, cycles, _stats = pipeline_data
        out = str(tmp_path / "test_anim2.gif")
        result = animate_normative_comparison(cycles, output_path=out)
        import os
        assert os.path.exists(result)
        plt.close("all")

    def test_stratum_pediatric(self, pipeline_data, tmp_path):
        _data, cycles, _stats = pipeline_data
        out = str(tmp_path / "test_anim_ped.gif")
        result = animate_normative_comparison(cycles, stratum="pediatric", output_path=out)
        assert isinstance(result, str)
        plt.close("all")

    def test_empty_cycles(self, tmp_path):
        """Should not crash with empty cycles."""
        out = str(tmp_path / "test_anim_empty.gif")
        result = animate_normative_comparison(
            {"cycles": [], "summary": {}}, output_path=out,
        )
        assert isinstance(result, str)
        plt.close("all")
