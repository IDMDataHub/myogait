"""Tests for compute_derivatives (angular velocity & acceleration).

Validates central-difference derivative computation on synthetic
angle waveforms with known analytical derivatives.

Reference: Winter DA. Biomechanics and Motor Control of Human
Movement. 4th ed. Wiley; 2009. Chapter 2.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myogait.analysis import compute_derivatives


# ── Helper ───────────────────────────────────────────────────────────


def _make_data(angles_dict, fps=30.0, n_frames=100):
    """Build a minimal myogait data dict for derivative tests."""
    frames = []
    for i in range(n_frames):
        f = {"landmarks": {}, "frame_idx": i}
        frames.append(f)
    angle_frames = []
    for i in range(n_frames):
        af = {}
        for joint, values in angles_dict.items():
            af[joint] = values[i] if values[i] is not None else None
        angle_frames.append(af)
    return {
        "frames": frames,
        "meta": {"fps": fps},
        "angles": {"frames": angle_frames, "method": "test"},
    }


# ══════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════


class TestComputeDerivatives:
    """Tests for compute_derivatives."""

    # 1. Constant angles → velocity = 0, acceleration = 0
    def test_constant_angle_zero_derivatives(self):
        n = 100
        fps = 30.0
        angles_dict = {
            "knee_L": [45.0] * n,
        }
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["knee_L"], max_order=2)

        np.testing.assert_allclose(
            result["knee_L"]["velocity"], 0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            result["knee_L"]["acceleration"], 0.0, atol=1e-10,
        )

    # 2. Linear angle increase → constant velocity, zero acceleration
    def test_linear_angle_constant_velocity(self):
        n = 200
        fps = 100.0
        dt = 1.0 / fps
        slope = 30.0  # 30 deg/s
        angles_dict = {
            "hip_L": [slope * i * dt for i in range(n)],
        }
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["hip_L"], max_order=2)

        vel = result["hip_L"]["velocity"]
        acc = result["hip_L"]["acceleration"]

        # Interior points should all be exactly the slope
        np.testing.assert_allclose(vel[1:-1], slope, atol=1e-8)

        # Acceleration should be zero (linear function)
        np.testing.assert_allclose(acc[2:-2], 0.0, atol=1e-6)

    # 3. Quadratic angle → linear velocity, constant acceleration
    def test_quadratic_angle_constant_acceleration(self):
        n = 200
        fps = 100.0
        dt = 1.0 / fps
        # angle(t) = 0.5 * a * t^2 → vel = a*t, acc = a
        a_const = 500.0  # deg/s²
        t_arr = np.arange(n) * dt
        angles_dict = {
            "ankle_R": list(0.5 * a_const * t_arr ** 2),
        }
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["ankle_R"], max_order=2)

        vel = result["ankle_R"]["velocity"]
        acc = result["ankle_R"]["acceleration"]

        # Velocity should be linear: a * t
        expected_vel = a_const * t_arr
        np.testing.assert_allclose(vel[1:-1], expected_vel[1:-1], rtol=1e-6)

        # Acceleration should be constant a_const (interior points)
        np.testing.assert_allclose(acc[2:-2], a_const, rtol=1e-4)

    # 4. Sine wave → cos velocity, -sin acceleration
    def test_sine_wave_derivatives(self):
        n = 1000
        fps = 500.0
        dt = 1.0 / fps
        freq = 2.0  # Hz
        omega = 2.0 * np.pi * freq
        t_arr = np.arange(n) * dt
        amplitude = 20.0  # degrees

        # angle(t) = A * sin(omega * t)
        # velocity = A * omega * cos(omega * t)
        # accel    = -A * omega^2 * sin(omega * t)
        angle_vals = amplitude * np.sin(omega * t_arr)
        angles_dict = {"hip_R": list(angle_vals)}
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["hip_R"], max_order=2)

        vel = result["hip_R"]["velocity"]
        acc = result["hip_R"]["acceleration"]

        expected_vel = amplitude * omega * np.cos(omega * t_arr)
        expected_acc = -amplitude * omega ** 2 * np.sin(omega * t_arr)

        # Use interior points (avoid boundary effects of np.gradient)
        # Tolerance is generous because finite differences approximate
        np.testing.assert_allclose(
            vel[10:-10], expected_vel[10:-10], rtol=5e-4, atol=0.5,
        )
        np.testing.assert_allclose(
            acc[10:-10], expected_acc[10:-10], rtol=5e-3, atol=5.0,
        )

    # 5. NaN handling: NaN angles produce NaN derivatives
    def test_nan_handling(self):
        n = 50
        fps = 30.0
        # Use a non-constant waveform so the NaN is clearly visible
        vals = [float(i) for i in range(n)]
        vals[20] = None  # will become NaN
        vals[25] = float("nan")
        angles_dict = {"knee_R": vals}
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["knee_R"], max_order=2)

        vel = result["knee_R"]["velocity"]
        acc = result["knee_R"]["acceleration"]

        # Central differences propagate NaN to neighbors:
        # gradient at i uses f[i-1] and f[i+1], so indices
        # adjacent to a NaN input will also be NaN in velocity.
        assert np.isnan(vel[19]), "NaN should propagate to neighbor velocity"
        assert np.isnan(vel[21]), "NaN should propagate to neighbor velocity"
        assert np.isnan(vel[24]), "NaN should propagate to neighbor velocity"
        assert np.isnan(vel[26]), "NaN should propagate to neighbor velocity"

        # Acceleration (2nd gradient of velocity) spreads NaN further.
        # The NaN indices in vel [19,21,24,26] cause NaN at their
        # neighbors in acc: [18,20,22,23,25,27].
        assert np.isnan(acc[18]), "NaN should propagate to acceleration"
        assert np.isnan(acc[20]), "NaN should propagate to acceleration"
        assert np.isnan(acc[22]), "NaN should propagate to acceleration"
        assert np.isnan(acc[25]), "NaN should propagate to acceleration"

    # 6. max_order=1 → only velocity computed, no acceleration key
    def test_max_order_1(self):
        n = 50
        fps = 30.0
        angles_dict = {"hip_L": [float(i) for i in range(n)]}
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["hip_L"], max_order=1)

        assert "velocity" in result["hip_L"]
        assert "acceleration" not in result["hip_L"]

    # 7. Custom joints: only specified joints are computed
    def test_custom_joints(self):
        n = 50
        fps = 30.0
        angles_dict = {
            "hip_L": [10.0] * n,
            "hip_R": [20.0] * n,
            "knee_L": [30.0] * n,
            "knee_R": [40.0] * n,
            "ankle_L": [5.0] * n,
            "ankle_R": [15.0] * n,
        }
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["hip_L", "ankle_R"])

        assert "hip_L" in result
        assert "ankle_R" in result
        assert "knee_L" not in result
        assert "knee_R" not in result
        assert "hip_R" not in result
        assert "ankle_L" not in result

    # 8. Units check: velocity matches expected deg/s
    def test_units_deg_per_sec(self):
        n = 100
        fps = 50.0
        dt = 1.0 / fps
        slope = 90.0  # 90 deg/s
        angles_dict = {
            "ankle_L": [slope * i * dt for i in range(n)],
        }
        data = _make_data(angles_dict, fps=fps, n_frames=n)
        result = compute_derivatives(data, joints=["ankle_L"], max_order=1)

        vel = result["ankle_L"]["velocity"]
        # Interior velocity should be 90 deg/s
        np.testing.assert_allclose(vel[5:-5], slope, atol=1e-8)

    # 9. Missing angles raises ValueError
    def test_empty_angles_raises(self):
        data = {
            "frames": [{"landmarks": {}, "frame_idx": 0}],
            "meta": {"fps": 30.0},
        }
        with pytest.raises(ValueError, match="angles"):
            compute_derivatives(data)

        # Also test with empty frames list
        data2 = {
            "frames": [],
            "meta": {"fps": 30.0},
            "angles": {"frames": []},
        }
        with pytest.raises(ValueError):
            compute_derivatives(data2)

    # 10. Results stored in data["derivatives"]
    def test_output_stored_in_data(self):
        n = 50
        fps = 30.0
        angles_dict = {"hip_L": [float(i) for i in range(n)]}
        data = _make_data(angles_dict, fps=fps, n_frames=n)

        assert "derivatives" not in data
        result = compute_derivatives(data, joints=["hip_L"])

        # Result is stored in data dict
        assert "derivatives" in data
        assert data["derivatives"] is result
        assert "hip_L" in data["derivatives"]
        assert isinstance(data["derivatives"]["hip_L"]["velocity"], np.ndarray)
        assert isinstance(data["derivatives"]["hip_L"]["acceleration"], np.ndarray)
        assert len(data["derivatives"]["hip_L"]["velocity"]) == n
        assert len(data["derivatives"]["hip_L"]["acceleration"]) == n
