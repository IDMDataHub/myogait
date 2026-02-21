"""Tests for pca_waveform_analysis."""

import numpy as np
import pytest

from myogait.analysis import pca_waveform_analysis


def _make_cycles(n_cycles=10, n_frames=30):
    """Generate synthetic cycle data for PCA testing."""
    rng = np.random.RandomState(42)
    cycles_list = []
    for i in range(n_cycles):
        t = np.linspace(0, 2 * np.pi, n_frames)
        # Add variation across cycles
        hip = 20 * np.sin(t) + rng.randn() * 2
        knee = 60 * np.sin(t + 0.5) + rng.randn() * 3
        ankle = 15 * np.sin(t + 1.0) + rng.randn() * 1
        cycles_list.append({
            "cycle_id": i + 1,
            "side": "left" if i % 2 == 0 else "right",
            "start_frame": i * n_frames,
            "end_frame": (i + 1) * n_frames - 1,
            "duration": n_frames / 30.0,
            "stance_pct": 60.0,
            "angles": {
                "hip_L": hip.tolist(),
                "knee_L": knee.tolist(),
                "ankle_L": ankle.tolist(),
            },
        })
    return {"cycles": cycles_list}


class TestPCAWaveformAnalysis:
    """Tests for pca_waveform_analysis."""

    def test_returns_per_joint_results(self):
        """Result has one entry per requested joint."""
        cycles = _make_cycles()
        result = pca_waveform_analysis(cycles)
        # Default joints
        assert "hip_L" in result
        assert "knee_L" in result
        assert "ankle_L" in result
        assert len(result) == 3

    def test_components_shape(self):
        """Components have shape (n_components, n_points)."""
        cycles = _make_cycles()
        n_components = 3
        n_points = 101
        result = pca_waveform_analysis(
            cycles, n_components=n_components, n_points=n_points
        )
        for joint in ("hip_L", "knee_L", "ankle_L"):
            assert result[joint]["components"].shape == (n_components, n_points)

    def test_scores_shape(self):
        """Scores have shape (n_cycles, n_components)."""
        n_cycles = 10
        n_components = 3
        cycles = _make_cycles(n_cycles=n_cycles)
        result = pca_waveform_analysis(cycles, n_components=n_components)
        for joint in ("hip_L", "knee_L", "ankle_L"):
            assert result[joint]["scores"].shape == (n_cycles, n_components)

    def test_explained_variance_sums_to_one(self):
        """Sum of all eigenvalues (explained variance ratios) approximately equals 1.0.

        When n_components equals the number of cycles (capturing all
        variance), the ratios must sum to 1.
        """
        n_cycles = 10
        cycles = _make_cycles(n_cycles=n_cycles)
        # Request all components to capture all variance
        result = pca_waveform_analysis(cycles, n_components=n_cycles)
        for joint in ("hip_L", "knee_L", "ankle_L"):
            total = np.sum(result[joint]["explained_variance_ratio"])
            assert abs(total - 1.0) < 1e-10, f"Sum was {total} for {joint}"

    def test_first_pc_captures_most_variance(self):
        """First PC captures more than 50% of variance."""
        cycles = _make_cycles()
        result = pca_waveform_analysis(cycles)
        for joint in ("hip_L", "knee_L", "ankle_L"):
            first_pct = result[joint]["explained_variance_ratio"][0]
            assert first_pct > 0.5, (
                f"First PC for {joint} only explains {first_pct:.2%}"
            )

    def test_mean_waveform_correct(self):
        """Mean matches np.mean of time-normalized input waveforms."""
        cycles = _make_cycles(n_cycles=8, n_frames=25)
        n_points = 101
        result = pca_waveform_analysis(cycles, n_points=n_points)

        # Manually compute expected mean for hip_L
        waveforms = []
        for c in cycles["cycles"]:
            wf = c["angles"]["hip_L"]
            x_in = np.linspace(0, 1, len(wf))
            x_out = np.linspace(0, 1, n_points)
            waveforms.append(np.interp(x_out, x_in, wf))
        expected_mean = np.mean(waveforms, axis=0)

        np.testing.assert_allclose(
            result["hip_L"]["mean"], expected_mean, atol=1e-12
        )

    def test_too_few_cycles_raises(self):
        """Less than 3 cycles raises ValueError."""
        cycles = _make_cycles(n_cycles=2)
        with pytest.raises(ValueError, match="at least 3"):
            pca_waveform_analysis(cycles)

    def test_custom_joints(self):
        """Only specified joints are analyzed."""
        cycles = _make_cycles()
        result = pca_waveform_analysis(cycles, joints=["knee_L"])
        assert list(result.keys()) == ["knee_L"]

    def test_reconstruction(self):
        """Mean + scores @ components approximates original data."""
        n_cycles = 10
        n_points = 101
        cycles = _make_cycles(n_cycles=n_cycles)
        # Use all components so reconstruction is exact
        result = pca_waveform_analysis(
            cycles, n_components=n_cycles, n_points=n_points
        )

        for joint in ("hip_L", "knee_L", "ankle_L"):
            r = result[joint]
            mean = r["mean"]
            scores = r["scores"]
            components = r["components"]

            # Reconstruct
            reconstructed = mean[np.newaxis, :] + scores @ components

            # Build original normalized matrix
            waveforms = []
            for c in cycles["cycles"]:
                wf = c["angles"][joint]
                x_in = np.linspace(0, 1, len(wf))
                x_out = np.linspace(0, 1, n_points)
                waveforms.append(np.interp(x_out, x_in, wf))
            original = np.array(waveforms)

            np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_identical_waveforms_zero_variance(self):
        """All identical waveforms produce zero explained variance."""
        n_cycles = 5
        n_frames = 30
        t = np.linspace(0, 2 * np.pi, n_frames)
        waveform = (20 * np.sin(t)).tolist()

        cycles_list = []
        for i in range(n_cycles):
            cycles_list.append({
                "cycle_id": i + 1,
                "side": "left",
                "start_frame": i * n_frames,
                "end_frame": (i + 1) * n_frames - 1,
                "duration": n_frames / 30.0,
                "stance_pct": 60.0,
                "angles": {
                    "hip_L": list(waveform),
                    "knee_L": list(waveform),
                    "ankle_L": list(waveform),
                },
            })
        cycles = {"cycles": cycles_list}
        result = pca_waveform_analysis(cycles)

        for joint in ("hip_L", "knee_L", "ankle_L"):
            evr = result[joint]["explained_variance_ratio"]
            assert np.allclose(evr, 0.0), (
                f"Expected zero variance for identical waveforms on {joint}, "
                f"got {evr}"
            )
