"""Tests for time_frequency_analysis."""

import numpy as np

from myogait.analysis import time_frequency_analysis


def _make_data(angles_dict, fps=30.0, n_frames=100):
    """Build a minimal myogait data dict for testing."""
    frames = [{"landmarks": {}, "frame_idx": i} for i in range(n_frames)]
    angle_frames = []
    for i in range(n_frames):
        af = {}
        for joint, values in angles_dict.items():
            af[joint] = float(values[i]) if values[i] is not None else None
        angle_frames.append(af)
    return {
        "frames": frames,
        "meta": {"fps": fps},
        "angles": {"frames": angle_frames, "method": "test"},
    }


# ── 1. CWT returns expected keys ─────────────────────────────────────


def test_cwt_returns_expected_keys():
    """Result dict has power, frequencies, times, dominant_frequency, method."""
    n = 100
    t = np.arange(n) / 30.0
    signal = np.sin(2 * np.pi * 2 * t)
    data = _make_data({"knee_L": signal}, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data, joints=["knee_L"])

    assert "knee_L" in result
    entry = result["knee_L"]
    for key in ("power", "frequencies", "times", "dominant_frequency", "method"):
        assert key in entry, f"Missing key: {key}"
    assert entry["method"] == "cwt"


# ── 2. Power matrix shape ────────────────────────────────────────────


def test_power_matrix_shape():
    """power shape is (n_freqs, n_frames) for CWT."""
    n = 120
    n_freqs = 40
    t = np.arange(n) / 30.0
    signal = np.sin(2 * np.pi * 3 * t)
    data = _make_data({"hip_L": signal}, fps=30.0, n_frames=n)

    result = time_frequency_analysis(
        data, joints=["hip_L"], method="cwt", n_freqs=n_freqs,
    )

    power = result["hip_L"]["power"]
    assert power.shape == (n_freqs, n)


# ── 3. Dominant frequency for pure sine ──────────────────────────────


def test_dominant_frequency_for_sine():
    """Pure 2 Hz sine -> dominant frequency close to 2 Hz."""
    fps = 60.0
    n = 300
    t = np.arange(n) / fps
    signal = np.sin(2 * np.pi * 2.0 * t)
    data = _make_data({"ankle_L": signal}, fps=fps, n_frames=n)

    result = time_frequency_analysis(
        data,
        joints=["ankle_L"],
        method="cwt",
        freq_range=(0.5, 10.0),
        n_freqs=100,
    )

    dominant = result["ankle_L"]["dominant_frequency"]
    assert abs(dominant - 2.0) < 0.5, f"Expected ~2 Hz, got {dominant}"


# ── 4. STFT method works ────────────────────────────────────────────


def test_stft_method_works():
    """method='stft' produces valid output with expected keys."""
    n = 128
    t = np.arange(n) / 30.0
    signal = np.sin(2 * np.pi * 3 * t)
    data = _make_data({"knee_L": signal}, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data, joints=["knee_L"], method="stft")

    entry = result["knee_L"]
    assert entry["method"] == "stft"
    assert entry["power"].ndim == 2
    assert len(entry["frequencies"]) == entry["power"].shape[0]
    assert len(entry["times"]) == entry["power"].shape[1]
    assert isinstance(entry["dominant_frequency"], float)


# ── 5. Frequency range respected ────────────────────────────────────


def test_frequency_range_respected():
    """frequencies array values stay within freq_range for CWT."""
    n = 100
    t = np.arange(n) / 30.0
    signal = np.sin(2 * np.pi * 5 * t)
    data = _make_data({"hip_L": signal}, fps=30.0, n_frames=n)

    freq_range = (1.0, 10.0)
    result = time_frequency_analysis(
        data, joints=["hip_L"], method="cwt", freq_range=freq_range,
    )

    freqs = result["hip_L"]["frequencies"]
    assert freqs.min() >= freq_range[0] - 1e-9
    assert freqs.max() <= freq_range[1] + 1e-9


# ── 6. Multiple joints ──────────────────────────────────────────────


def test_multiple_joints():
    """Default joints produces results for hip_L, knee_L, ankle_L."""
    n = 100
    t = np.arange(n) / 30.0
    angles = {
        "hip_L": np.sin(2 * np.pi * 1 * t),
        "knee_L": np.sin(2 * np.pi * 2 * t),
        "ankle_L": np.sin(2 * np.pi * 3 * t),
    }
    data = _make_data(angles, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data)

    assert "hip_L" in result
    assert "knee_L" in result
    assert "ankle_L" in result
    assert len(result) == 3


# ── 7. Custom joints ────────────────────────────────────────────────


def test_custom_joints():
    """Only specified joints are analyzed."""
    n = 100
    t = np.arange(n) / 30.0
    angles = {
        "hip_L": np.sin(2 * np.pi * 1 * t),
        "knee_L": np.sin(2 * np.pi * 2 * t),
        "ankle_L": np.sin(2 * np.pi * 3 * t),
    }
    data = _make_data(angles, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data, joints=["knee_L"])

    assert "knee_L" in result
    assert "hip_L" not in result
    assert "ankle_L" not in result


# ── 8. NaN handling ──────────────────────────────────────────────────


def test_nan_handling():
    """NaN values in the signal don't crash the analysis."""
    n = 100
    t = np.arange(n) / 30.0
    signal = np.sin(2 * np.pi * 2 * t).tolist()
    # Inject some None/NaN values
    signal[10] = None
    signal[20] = None
    signal[50] = None
    data = _make_data({"knee_L": signal}, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data, joints=["knee_L"])

    entry = result["knee_L"]
    assert not np.any(np.isnan(entry["power"]))
    assert isinstance(entry["dominant_frequency"], float)
    assert not np.isnan(entry["dominant_frequency"])


# ── 9. Constant signal has very low power ────────────────────────────


def test_constant_signal_low_power():
    """A constant (DC) signal should have very low power after mean removal."""
    n = 100
    constant = [5.0] * n
    data = _make_data({"hip_L": constant}, fps=30.0, n_frames=n)

    result = time_frequency_analysis(data, joints=["hip_L"])

    total_power = np.sum(result["hip_L"]["power"])
    assert total_power < 1e-10, f"Expected near-zero power, got {total_power}"
