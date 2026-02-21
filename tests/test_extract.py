"""Tests for myogait package."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ── Import / API ──────────────────────────────────────────────────────

def test_import():
    import myogait
    assert hasattr(myogait, "__version__")
    assert myogait.__version__ == "0.3.1"


def test_public_api():
    from myogait import extract, normalize, compute_angles, load_json, save_json
    assert callable(extract)
    assert callable(normalize)
    assert callable(compute_angles)
    assert callable(load_json)
    assert callable(save_json)


def test_list_models():
    from myogait.models import list_models
    models = list_models()
    assert "mediapipe" in models
    assert "yolo" in models
    assert "sapiens-quick" in models
    assert "sapiens-mid" in models
    assert "sapiens-top" in models
    assert "hrnet" in models
    assert "mmpose" in models


# ── Schema ────────────────────────────────────────────────────────────

def test_schema_create_empty():
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=30.0, width=1920, height=1080, n_frames=100)
    from myogait import __version__
    assert data["myogait_version"] == __version__
    assert data["meta"]["fps"] == 30.0
    assert data["meta"]["n_frames"] == 100
    assert data["meta"]["duration_s"] == pytest.approx(100 / 30.0, abs=0.01)
    assert data["frames"] == []
    assert data["extraction"] is None
    assert data["angles"] is None
    assert data["events"] is None


def test_schema_save_load(tmp_path):
    from myogait.schema import create_empty, save_json, load_json
    data = create_empty("test.mp4", fps=30.0, width=1920, height=1080, n_frames=10)
    path = tmp_path / "test.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["meta"]["fps"] == 30.0


def test_schema_save_numpy_types(tmp_path):
    """Ensure numpy types are serialized correctly."""
    from myogait.schema import save_json, load_json
    data = {
        "myogait_version": "0.1.0",
        "meta": {"fps": np.float64(30.0), "n_frames": np.int64(10)},
        "frames": [],
    }
    path = tmp_path / "np.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["meta"]["fps"] == 30.0
    assert loaded["meta"]["n_frames"] == 10


# ── Constants ─────────────────────────────────────────────────────────

def test_constants():
    from myogait.constants import MP_LANDMARK_NAMES, COCO_LANDMARK_NAMES, COCO_TO_MP
    assert len(MP_LANDMARK_NAMES) == 33
    assert len(COCO_LANDMARK_NAMES) == 17
    assert all(v in MP_LANDMARK_NAMES for v in COCO_TO_MP.values())


# ── Extract helpers ───────────────────────────────────────────────────

def test_coco_to_mediapipe():
    from myogait.extract import _coco_to_mediapipe
    coco = np.random.rand(17, 3)
    mp33 = _coco_to_mediapipe(coco)
    assert mp33.shape == (33, 3)
    np.testing.assert_array_equal(mp33[0], coco[0])  # NOSE


def test_coco_to_mediapipe_missing_are_nan():
    from myogait.extract import _coco_to_mediapipe
    from myogait.constants import MP_NAME_TO_INDEX
    coco = np.ones((17, 3))
    mp33 = _coco_to_mediapipe(coco)
    # LEFT_HEEL is not in COCO, should be NaN
    heel_idx = MP_NAME_TO_INDEX["LEFT_HEEL"]
    assert np.isnan(mp33[heel_idx, 0])


# ── Filters ───────────────────────────────────────────────────────────

def test_filters_import():
    from myogait.normalize import (
        NORMALIZE_STEPS,
    )
    assert "butterworth" in NORMALIZE_STEPS
    assert "savgol" in NORMALIZE_STEPS
    assert "center_on_torso" in NORMALIZE_STEPS


def test_butterworth_smooths():
    from myogait.normalize import filter_butterworth
    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    clean = np.sin(2 * np.pi * 2 * t)
    noisy = clean + np.random.normal(0, 0.3, 100)
    df = pd.DataFrame({"NOSE_x": noisy, "NOSE_y": clean})
    filtered = filter_butterworth(df, order=2, cutoff=5.0, fs=100.0)
    # Filtered should be closer to clean than noisy
    error_noisy = np.mean((noisy - clean) ** 2)
    error_filtered = np.mean((filtered["NOSE_x"].values - clean) ** 2)
    assert error_filtered < error_noisy


def test_filter_pipeline():
    from myogait.normalize import filter_butterworth, filter_moving_mean
    np.random.seed(42)
    df = pd.DataFrame({
        "NOSE_x": np.random.rand(50),
        "NOSE_y": np.random.rand(50),
    })
    # Apply filters sequentially (same effect as pipeline)
    result = filter_butterworth(df, order=2, cutoff=5.0, fs=30.0)
    result = filter_moving_mean(result, window=3)
    assert len(result) == 50
    assert "NOSE_x" in result.columns


# ── Normalize ─────────────────────────────────────────────────────────

def test_normalize_empty_raises():
    from myogait import normalize
    from myogait.schema import create_empty
    data = create_empty("test.mp4")
    with pytest.raises(ValueError, match="No frames"):
        normalize(data)


def test_normalize_preserves_raw():
    """normalize() should save original frames in frames_raw."""
    from myogait import normalize
    data = _make_fake_data(20)
    normalize(data, filters=["butterworth"])
    assert "frames_raw" in data
    assert len(data["frames_raw"]) == 20


# ── Angles ────────────────────────────────────────────────────────────

def test_angles_empty_raises():
    from myogait import compute_angles
    from myogait.schema import create_empty
    data = create_empty("test.mp4")
    with pytest.raises(ValueError, match="No frames"):
        compute_angles(data)


def test_angles_vertical_axis_method():
    """Standing pose should give ~0 hip angle."""
    from myogait import compute_angles
    data = _make_standing_data(30)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    angles = data["angles"]
    assert angles["method"] == "sagittal_vertical_axis"
    # In standing pose, hip should be near 0
    mid = angles["frames"][15]
    assert mid["hip_L"] is not None
    assert abs(mid["hip_L"]) < 15  # should be near 0


def test_angles_has_trunk_and_pelvis():
    from myogait import compute_angles
    data = _make_standing_data(10)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    af = data["angles"]["frames"][5]
    assert "trunk_angle" in af
    assert "pelvis_tilt" in af


def test_angles_has_landmark_positions():
    from myogait import compute_angles
    data = _make_standing_data(10)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    af = data["angles"]["frames"][5]
    assert "landmark_positions" in af
    lp = af["landmark_positions"]
    assert "left_hip" in lp
    assert "right_knee" in lp
    assert len(lp["left_hip"]) == 3  # x, y, visibility


def test_angles_correction_factor():
    from myogait import compute_angles
    import copy
    data1 = _make_standing_data(10)
    data2 = copy.deepcopy(data1)
    compute_angles(data1, correction_factor=1.0, calibrate=False)
    compute_angles(data2, correction_factor=0.5, calibrate=False)
    k1 = data1["angles"]["frames"][5]["knee_L"]
    k2 = data2["angles"]["frames"][5]["knee_L"]
    if k1 is not None and k2 is not None and k1 != 0:
        assert abs(k2 / k1 - 0.5) < 0.1


def test_pelvis_tilt_nan_in_lateral_view():
    """In lateral view, hips overlap → pelvis_tilt should be NaN/None."""
    from myogait import compute_angles
    data = _make_lateral_view_data(10)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    af = data["angles"]["frames"][5]
    # Pelvis tilt should be None (NaN converted) when hips are close
    assert af["pelvis_tilt"] is None


# ── Helpers ───────────────────────────────────────────────────────────

def _make_fake_data(n_frames):
    """Create minimal fake data with landmarks for testing."""
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=30.0, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    for i in range(n_frames):
        lm = {}
        for name in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                      "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                      "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                      "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]:
            lm[name] = {"x": 0.5, "y": 0.5, "visibility": 1.0}
        frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.9})
    data["frames"] = frames
    return data


def _make_standing_data(n_frames):
    """Create data simulating a standing person in profile view."""
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=30.0, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    for i in range(n_frames):
        # Standing person: landmarks roughly aligned vertically
        lm = {
            "NOSE":             {"x": 0.50, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "LEFT_HIP":         {"x": 0.50, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP":        {"x": 0.50, "y": 0.50, "visibility": 1.0},
            "LEFT_KNEE":        {"x": 0.50, "y": 0.65, "visibility": 1.0},
            "RIGHT_KNEE":       {"x": 0.50, "y": 0.65, "visibility": 1.0},
            "LEFT_ANKLE":       {"x": 0.50, "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE":      {"x": 0.50, "y": 0.80, "visibility": 1.0},
            "LEFT_HEEL":        {"x": 0.51, "y": 0.82, "visibility": 1.0},
            "RIGHT_HEEL":       {"x": 0.51, "y": 0.82, "visibility": 1.0},
            "LEFT_FOOT_INDEX":  {"x": 0.47, "y": 0.82, "visibility": 1.0},
            "RIGHT_FOOT_INDEX": {"x": 0.47, "y": 0.82, "visibility": 1.0},
        }
        frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.95})
    data["frames"] = frames
    return data


def _make_walking_data(n_frames=300, fps=30.0):
    """Create data simulating a person walking (sinusoidal ankle motion).

    Ankles oscillate in x relative to hips, simulating gait cycles.
    Period ~1s (30 frames at 30fps), 10 cycles in 300 frames.
    """
    from myogait.schema import create_empty
    data = create_empty("test_walk.mp4", fps=fps, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    cycle_period = 1.0  # seconds per gait cycle
    for i in range(n_frames):
        t = i / fps
        # Ankles oscillate in antero-posterior (x) direction, 180° out of phase
        phase_l = 2 * np.pi * t / cycle_period
        phase_r = phase_l + np.pi
        ankle_amp = 0.08  # amplitude of oscillation
        hip_x = 0.50
        lm = {
            "NOSE":             {"x": hip_x, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": hip_x, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": hip_x + 0.01, "y": 0.25, "visibility": 1.0},
            "LEFT_HIP":         {"x": hip_x, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP":        {"x": hip_x + 0.01, "y": 0.50, "visibility": 1.0},
            "LEFT_KNEE":        {"x": hip_x + 0.04 * np.sin(phase_l), "y": 0.65, "visibility": 1.0},
            "RIGHT_KNEE":       {"x": hip_x + 0.01 + 0.04 * np.sin(phase_r), "y": 0.65, "visibility": 1.0},
            "LEFT_ANKLE":       {"x": hip_x + ankle_amp * np.sin(phase_l), "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE":      {"x": hip_x + 0.01 + ankle_amp * np.sin(phase_r), "y": 0.80, "visibility": 1.0},
            "LEFT_HEEL":        {"x": hip_x + ankle_amp * np.sin(phase_l) + 0.01, "y": 0.82, "visibility": 1.0},
            "RIGHT_HEEL":       {"x": hip_x + 0.01 + ankle_amp * np.sin(phase_r) + 0.01, "y": 0.82, "visibility": 1.0},
            "LEFT_FOOT_INDEX":  {"x": hip_x + ankle_amp * np.sin(phase_l) - 0.03, "y": 0.82, "visibility": 1.0},
            "RIGHT_FOOT_INDEX": {"x": hip_x + 0.01 + ankle_amp * np.sin(phase_r) - 0.03, "y": 0.82, "visibility": 1.0},
        }
        frames.append({"frame_idx": i, "time_s": round(t, 4), "landmarks": lm, "confidence": 0.95})
    data["frames"] = frames
    return data


def _make_lateral_view_data(n_frames):
    """Create data where left/right hips overlap (lateral/profile view)."""
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=30.0, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    for i in range(n_frames):
        lm = {
            "NOSE":             {"x": 0.50, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": 0.505, "y": 0.25, "visibility": 1.0},
            "LEFT_HIP":         {"x": 0.50, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP":        {"x": 0.505, "y": 0.50, "visibility": 1.0},  # <2% apart
            "LEFT_KNEE":        {"x": 0.50, "y": 0.65, "visibility": 1.0},
            "RIGHT_KNEE":       {"x": 0.52, "y": 0.65, "visibility": 1.0},
            "LEFT_ANKLE":       {"x": 0.50, "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE":      {"x": 0.52, "y": 0.80, "visibility": 1.0},
            "LEFT_HEEL":        {"x": 0.51, "y": 0.82, "visibility": 1.0},
            "RIGHT_HEEL":       {"x": 0.53, "y": 0.82, "visibility": 1.0},
            "LEFT_FOOT_INDEX":  {"x": 0.47, "y": 0.82, "visibility": 1.0},
            "RIGHT_FOOT_INDEX": {"x": 0.49, "y": 0.82, "visibility": 1.0},
        }
        frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.95})
    data["frames"] = frames
    return data


# ── Events ───────────────────────────────────────────────────────────

def _walking_data_with_angles():
    """Helper: walking data + angles computed."""
    from myogait import compute_angles
    data = _make_walking_data(300, fps=30.0)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    return data


def test_detect_events_zeni():
    from myogait import detect_events
    data = _walking_data_with_angles()
    detect_events(data)
    events = data["events"]
    assert events["method"] == "zeni"
    # 300 frames / 30fps = 10s, cycle = 1s → ~10 HS per side
    assert len(events["left_hs"]) >= 5
    assert len(events["right_hs"]) >= 5
    assert len(events["left_to"]) >= 5
    assert len(events["right_to"]) >= 5


def test_detect_events_format():
    from myogait import detect_events
    data = _walking_data_with_angles()
    detect_events(data)
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        for ev in data["events"][key]:
            assert "frame" in ev
            assert "time" in ev
            assert "confidence" in ev
            assert isinstance(ev["frame"], int)
            assert 0 <= ev["confidence"] <= 1


def test_detect_events_empty_raises():
    from myogait import detect_events
    from myogait.schema import create_empty
    data = create_empty("test.mp4")
    with pytest.raises(ValueError, match="No frames"):
        detect_events(data)


# ── Cycles ───────────────────────────────────────────────────────────

def test_segment_cycles():
    from myogait import detect_events, segment_cycles
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    assert "cycles" in cycles
    assert "summary" in cycles
    assert len(cycles["cycles"]) >= 4  # at least some cycles per side


def test_normalize_cycles_101_points():
    from myogait import detect_events, segment_cycles
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data, n_points=101)
    for c in cycles["cycles"]:
        for joint in ["hip", "knee", "ankle"]:
            vals = c["angles_normalized"].get(joint)
            if vals is not None:
                assert len(vals) == 101


def test_cycle_mean_std():
    from myogait import detect_events, segment_cycles
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    for side in ("left", "right"):
        summary = cycles["summary"].get(side)
        if summary is None:
            continue
        for joint in ["hip", "knee", "ankle"]:
            mean = summary.get(f"{joint}_mean")
            std = summary.get(f"{joint}_std")
            if mean is not None:
                assert len(mean) == 101
                assert len(std) == 101


# ── Analysis ─────────────────────────────────────────────────────────

def test_analyze_gait_stats():
    from myogait import detect_events, segment_cycles, analyze_gait
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    assert "spatiotemporal" in stats
    assert "symmetry" in stats
    assert "variability" in stats
    assert "pathology_flags" in stats
    st = stats["spatiotemporal"]
    assert st["cadence_steps_per_min"] > 0
    assert st["n_cycles_total"] >= 4


# ── Plotting ─────────────────────────────────────────────────────────

def test_plot_angles_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from myogait import plot_angles
    data = _walking_data_with_angles()
    fig = plot_angles(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_events_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from myogait import detect_events, plot_events
    data = _walking_data_with_angles()
    detect_events(data)
    fig = plot_events(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cycles_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from myogait import detect_events, segment_cycles, plot_cycles
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    fig = plot_cycles(cycles, side="left")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── Report (PDF) ────────────────────────────────────────────────────

def test_generate_report_creates_pdf(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from myogait import detect_events, segment_cycles, analyze_gait, generate_report
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    pdf_path = tmp_path / "report.pdf"
    generate_report(data, cycles, stats, str(pdf_path))
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000  # non-trivial PDF


# ── Angle methods ───────────────────────────────────────────────────

def test_angle_method_classic():
    from myogait import compute_angles
    data = _make_standing_data(30)
    compute_angles(data, method="sagittal_classic", correction_factor=1.0, calibrate=False)
    angles = data["angles"]
    assert angles["method"] == "sagittal_classic"
    mid = angles["frames"][15]
    assert mid["hip_L"] is not None
    assert mid["knee_L"] is not None


def test_angle_method_vertical_axis():
    from myogait import compute_angles
    data = _make_standing_data(30)
    compute_angles(data, method="sagittal_vertical_axis", correction_factor=1.0, calibrate=False)
    assert data["angles"]["method"] == "sagittal_vertical_axis"


def test_angle_method_unknown_raises():
    from myogait import compute_angles
    data = _make_standing_data(10)
    with pytest.raises(ValueError, match="Unknown angle method"):
        compute_angles(data, method="nonexistent")


def test_list_angle_methods():
    from myogait.angles import list_angle_methods
    methods = list_angle_methods()
    assert "sagittal_vertical_axis" in methods
    assert "sagittal_classic" in methods


def test_register_custom_angle_method():
    from myogait.angles import register_angle_method, ANGLE_METHODS
    def dummy_method(frame, model):
        return {"frame_idx": frame["frame_idx"],
                "hip_L": 10.0, "hip_R": 10.0,
                "knee_L": 5.0, "knee_R": 5.0,
                "ankle_L": 0.0, "ankle_R": 0.0,
                "trunk_angle": 0.0, "pelvis_tilt": 0.0,
                "landmark_positions": {}}
    register_angle_method("dummy_test", dummy_method)
    assert "dummy_test" in ANGLE_METHODS
    # Clean up
    del ANGLE_METHODS["dummy_test"]


def test_calibration_joints_custom():
    from myogait import compute_angles
    data = _make_standing_data(30)
    compute_angles(data, correction_factor=1.0, calibrate=True,
                   calibration_joints=["knee_L", "knee_R"])
    assert data["angles"]["calibration_joints"] == ["knee_L", "knee_R"]


# ── Normalize steps ─────────────────────────────────────────────────

def test_normalize_with_steps():
    from myogait import normalize
    data = _make_fake_data(50)
    normalize(data, steps=[
        {"type": "butterworth", "cutoff": 5.0, "order": 2},
        {"type": "moving_mean", "window": 3},
    ])
    assert data["normalization"]["steps_applied"] == ["butterworth", "moving_mean"]


def test_normalize_pixel_ratio():
    from myogait import normalize
    data = _make_fake_data(20)
    normalize(data, pixel_ratio={
        "input_width": 1920, "input_height": 1080,
        "processed_width": 256, "processed_height": 256,
    })
    assert "correct_pixel_ratio" in data["normalization"]["steps_applied"]


def test_normalize_center_on_torso():
    from myogait import normalize
    data = _make_fake_data(20)
    normalize(data, center=True)
    assert "center_on_torso" in data["normalization"]["steps_applied"]


def test_list_normalize_steps():
    from myogait.normalize import list_normalize_steps
    steps = list_normalize_steps()
    assert "butterworth" in steps
    assert "correct_pixel_ratio" in steps
    assert "center_on_torso" in steps


def test_normalize_backward_compat_filters():
    """Old-style filters= still works."""
    from myogait import normalize
    data = _make_fake_data(50)
    normalize(data, filters=["butterworth"], butterworth_cutoff=5.0)
    assert "butterworth" in data["normalization"]["steps_applied"]


def test_correct_pixel_ratio_standalone():
    from myogait.normalize import correct_pixel_ratio
    import pandas as pd
    df = pd.DataFrame({
        "NOSE_x": [0.5, 0.5], "NOSE_y": [0.5, 0.5],
    })
    result = correct_pixel_ratio(df, input_width=1920, input_height=1080,
                                  processed_width=256, processed_height=256)
    # 1920/1080 = 1.78, 256/256 = 1.0, ratio = 1.78 → x stretched
    assert result["NOSE_x"].iloc[0] > 0.5


# ── Event methods ──────────────────────────────────────────────────


def test_list_event_methods():
    from myogait.events import list_event_methods
    methods = list_event_methods()
    assert "zeni" in methods
    assert "crossing" in methods
    assert "velocity" in methods
    assert "oconnor" in methods


def test_detect_events_crossing():
    from myogait import detect_events
    data = _walking_data_with_angles()
    detect_events(data, method="crossing")
    assert data["events"]["method"] == "crossing"
    assert len(data["events"]["left_hs"]) >= 1
    assert len(data["events"]["right_hs"]) >= 1


def test_detect_events_velocity():
    from myogait import detect_events
    data = _walking_data_with_angles()
    detect_events(data, method="velocity")
    assert data["events"]["method"] == "velocity"


def test_detect_events_oconnor():
    from myogait import detect_events
    data = _walking_data_with_angles()
    detect_events(data, method="oconnor")
    assert data["events"]["method"] == "oconnor"


def test_detect_events_unknown_raises():
    from myogait import detect_events
    data = _walking_data_with_angles()
    with pytest.raises(ValueError, match="Unknown method"):
        detect_events(data, method="nonexistent")


def test_register_event_method():
    from myogait.events import register_event_method, EVENT_METHODS
    def dummy_detect(frames, fps, min_dur, cutoff):
        return {"left_hs": [], "right_hs": [], "left_to": [], "right_to": []}
    register_event_method("dummy_ev", dummy_detect)
    assert "dummy_ev" in EVENT_METHODS
    del EVENT_METHODS["dummy_ev"]


# ── Export ─────────────────────────────────────────────────────────


def test_export_csv(tmp_path):
    from myogait import detect_events, segment_cycles, analyze_gait
    from myogait.export import export_csv
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    files = export_csv(data, str(tmp_path), cycles, stats)
    assert len(files) >= 3  # angles.csv, events.csv, cycles.csv, stats.csv
    for f in files:
        assert Path(f).exists()


def test_export_mot(tmp_path):
    from myogait.export import export_mot
    data = _walking_data_with_angles()
    mot_path = str(tmp_path / "test.mot")
    export_mot(data, mot_path)
    assert Path(mot_path).exists()
    content = Path(mot_path).read_text()
    assert "endheader" in content
    assert "hip_flexion_l" in content


def test_export_trc(tmp_path):
    from myogait.export import export_trc
    data = _make_walking_data(30)
    trc_path = str(tmp_path / "test.trc")
    export_trc(data, trc_path)
    assert Path(trc_path).exists()
    content = Path(trc_path).read_text()
    assert "LEFT_HIP" in content


def test_export_excel(tmp_path):
    pytest.importorskip("openpyxl")
    from myogait import detect_events, segment_cycles, analyze_gait
    from myogait.export import export_excel
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    xlsx_path = str(tmp_path / "test.xlsx")
    export_excel(data, xlsx_path, cycles, stats)
    assert Path(xlsx_path).exists()
    assert Path(xlsx_path).stat().st_size > 1000


# ── Analysis: new metrics ────────────────────────────────────────


def test_regularity_index():
    from myogait import regularity_index
    data = _make_walking_data(300)
    result = regularity_index(data)
    assert "step_regularity" in result
    assert "stride_regularity" in result
    # Walking data should have decent regularity
    if result["step_regularity"] is not None:
        assert 0 < result["step_regularity"] <= 1.0


def test_harmonic_ratio():
    from myogait import harmonic_ratio
    data = _make_walking_data(300)
    result = harmonic_ratio(data)
    assert "hr_ap" in result
    assert "hr_vertical" in result


def test_step_length():
    from myogait import detect_events, segment_cycles, step_length
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    result = step_length(data, cycles)
    assert "step_length_left" in result
    assert "stride_length_left" in result
    assert result["unit"] == "normalized"


def test_step_length_calibrated():
    from myogait import detect_events, segment_cycles, step_length
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    result = step_length(data, cycles, height_m=1.75)
    assert result["calibrated"] is True
    assert result["unit"] == "m"


def test_walking_speed():
    from myogait import detect_events, segment_cycles, walking_speed
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    result = walking_speed(data, cycles)
    assert "speed_mean" in result


def test_step_length_treadmill_returns_not_reliable():
    from myogait import detect_events, segment_cycles, step_length
    data = _walking_data_with_angles()
    data["extraction"] = {"treadmill": True}
    detect_events(data)
    cycles = segment_cycles(data)
    result = step_length(data, cycles)
    assert result["step_length_left"] is None
    assert result["valid_for_progression"] is False


def test_walking_speed_treadmill_returns_not_reliable():
    from myogait import detect_events, segment_cycles, walking_speed
    data = _walking_data_with_angles()
    data["extraction"] = {"treadmill": True}
    detect_events(data)
    cycles = segment_cycles(data)
    result = walking_speed(data, cycles)
    assert result["speed_mean"] is None
    assert result["valid_for_progression"] is False


def test_detect_pathologies():
    from myogait import detect_events, segment_cycles, detect_pathologies
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    result = detect_pathologies(data, cycles)
    assert isinstance(result, list)
    for p in result:
        assert "pattern" in p
        assert "side" in p
        assert "severity" in p
        assert "confidence" in p


def test_analyze_gait_includes_new_metrics():
    from myogait import detect_events, segment_cycles, analyze_gait
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    assert "regularity" in stats
    assert "harmonic_ratio" in stats
    assert "step_length" in stats
    assert "walking_speed" in stats
    assert "pathologies" in stats


# ── Validation ───────────────────────────────────────────────────


def test_validate_biomechanical():
    from myogait import validate_biomechanical
    data = _walking_data_with_angles()
    report = validate_biomechanical(data)
    assert "valid" in report
    assert "violations" in report
    assert "summary" in report
    assert isinstance(report["violations"], list)


def test_validate_with_cycles():
    from myogait import detect_events, segment_cycles, validate_biomechanical
    data = _walking_data_with_angles()
    detect_events(data)
    cycles = segment_cycles(data)
    report = validate_biomechanical(data, cycles)
    assert "summary" in report


def test_get_angle_ranges():
    from myogait.validation import get_angle_ranges
    ranges = get_angle_ranges()
    assert "hip_L" in ranges
    assert "knee_L" in ranges
    assert "full" in ranges["hip_L"]


# ── Subject metadata ────────────────────────────────────────────


def test_set_subject():
    from myogait import set_subject
    from myogait.schema import create_empty
    data = create_empty("test.mp4")
    set_subject(data, age=45, sex="M", height_m=1.75, weight_kg=80,
                pathology="DMD", notes="Baseline visit")
    assert data["subject"]["age"] == 45
    assert data["subject"]["sex"] == "M"
    assert data["subject"]["height_m"] == 1.75
    assert data["subject"]["pathology"] == "DMD"


def test_schema_has_subject():
    from myogait.schema import create_empty
    data = create_empty("test.mp4")
    assert "subject" in data
    assert data["subject"] is None


# ── Config ───────────────────────────────────────────────────────


def test_config_save_load(tmp_path):
    from myogait import save_config, load_config, DEFAULT_CONFIG
    path = tmp_path / "test_config.json"
    save_config(DEFAULT_CONFIG, str(path))
    loaded = load_config(str(path))
    assert loaded["extract"]["model"] == "mediapipe"
    assert loaded["events"]["method"] == "zeni"


def test_config_merge():
    from myogait.config import _deep_merge
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 10}, "e": 5}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 10
    assert result["b"]["d"] == 3
    assert result["e"] == 5


def test_default_config():
    from myogait import DEFAULT_CONFIG
    assert "extract" in DEFAULT_CONFIG
    assert "normalize" in DEFAULT_CONFIG
    assert "angles" in DEFAULT_CONFIG
    assert "events" in DEFAULT_CONFIG
    assert "cycles" in DEFAULT_CONFIG
    assert "subject" in DEFAULT_CONFIG
    assert "experimental" in DEFAULT_CONFIG["extract"]
    assert DEFAULT_CONFIG["extract"]["experimental"]["enabled"] is False


# ── Phase plane plot ─────────────────────────────────────────────


def test_plot_phase_plane():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from myogait import plot_phase_plane
    data = _walking_data_with_angles()
    fig = plot_phase_plane(data, joint="knee_L")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── Version ──────────────────────────────────────────────────────


def test_version_updated():
    import myogait
    assert myogait.__version__ == "0.3.1"


# ── Goliath 308 / Sapiens ───────────────────────────────────────────

def test_goliath_landmark_names_count():
    from myogait.constants import GOLIATH_LANDMARK_NAMES, GOLIATH_NAME_TO_INDEX
    assert len(GOLIATH_LANDMARK_NAMES) == 308
    assert len(GOLIATH_NAME_TO_INDEX) == 308
    assert GOLIATH_LANDMARK_NAMES[0] == "nose"
    assert GOLIATH_LANDMARK_NAMES[307] == "r_border_of_pupil_midpoint_2"


def test_goliath_to_coco_mapping():
    from myogait.constants import GOLIATH_TO_COCO, GOLIATH_LANDMARK_NAMES
    # Check key body landmarks
    assert GOLIATH_TO_COCO[0] == 0   # nose → nose
    assert GOLIATH_TO_COCO[5] == 5   # left_shoulder → left_shoulder
    assert GOLIATH_TO_COCO[9] == 11  # left_hip → left_hip (NOT 9)
    assert GOLIATH_TO_COCO[41] == 10  # right_wrist → right_wrist
    assert GOLIATH_TO_COCO[62] == 9   # left_wrist → left_wrist
    # Verify names match
    assert GOLIATH_LANDMARK_NAMES[9] == "left_hip"
    assert GOLIATH_LANDMARK_NAMES[41] == "right_wrist"
    assert GOLIATH_LANDMARK_NAMES[62] == "left_wrist"


def test_heatmaps_to_all():
    from myogait.models.sapiens import _heatmaps_to_all
    # Synthetic heatmaps: 308 channels, 8x6 resolution
    rng = np.random.RandomState(42)
    heatmaps = rng.rand(308, 8, 6).astype(np.float32)
    result = _heatmaps_to_all(heatmaps)
    assert result.shape == (308, 3)
    # x,y in [0,1], conf in [0,1]
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 1)
    assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1)
    assert np.all(result[:, 2] >= 0) and np.all(result[:, 2] <= 1)


def test_heatmaps_to_coco():
    from myogait.models.sapiens import _heatmaps_to_coco
    rng = np.random.RandomState(42)
    heatmaps = rng.rand(308, 8, 6).astype(np.float32)
    result = _heatmaps_to_coco(heatmaps)
    assert result.shape == (17, 3)
    # All 17 COCO keypoints should be populated
    assert not np.any(np.isnan(result))


def test_extract_handles_dict_result():
    """extract.py correctly handles dict return from process_frame."""
    from myogait.extract import _coco_to_mediapipe
    # Simulate what extract does when it gets a dict
    coco_17 = np.random.rand(17, 3).astype(np.float32)
    coco_17[:, 2] = np.clip(coco_17[:, 2], 0, 1)
    aux_308 = np.random.rand(308, 3).astype(np.float32)

    result = {"landmarks": coco_17, "auxiliary_goliath308": aux_308}

    # Extract logic
    auxiliary = None
    if isinstance(result, dict):
        auxiliary = result.get("auxiliary_goliath308")
        lm = result.get("landmarks")
    else:
        lm = result

    assert lm.shape == (17, 3)
    assert auxiliary.shape == (308, 3)
    mp33 = _coco_to_mediapipe(lm)
    assert mp33.shape == (33, 3)


def test_sapiens_model_registry():
    from myogait.models.sapiens import _MODELS
    assert "0.3b" in _MODELS
    assert "0.6b" in _MODELS
    assert "1b" in _MODELS
    # Each entry: (filename, repo_id)
    for size, (filename, repo_id) in _MODELS.items():
        assert filename.endswith(".pt2")
        assert repo_id.startswith("facebook/sapiens-pose-")


def test_get_device_returns_valid():
    """_get_device returns cpu, cuda, or xpu."""
    from myogait.models.sapiens import _get_device
    pytest.importorskip("torch")
    device = _get_device()
    assert device.type in ("cpu", "cuda", "xpu")


# ── New model registries ─────────────────────────────────────────────

def test_list_models_includes_new():
    from myogait.models import list_models
    models = list_models()
    assert "vitpose" in models
    assert "vitpose-large" in models
    assert "vitpose-huge" in models
    assert "rtmw" in models


def test_depth_model_registry():
    from myogait.models.sapiens_depth import _DEPTH_MODELS
    assert "0.3b" in _DEPTH_MODELS
    assert "0.6b" in _DEPTH_MODELS
    assert "1b" in _DEPTH_MODELS
    assert "2b" in _DEPTH_MODELS
    for size, (filename, repo_id) in _DEPTH_MODELS.items():
        assert filename.endswith(".pt2")
        assert repo_id.startswith("facebook/sapiens-depth-")


def test_seg_model_registry():
    from myogait.models.sapiens_seg import _SEG_MODELS
    assert "0.3b" in _SEG_MODELS
    assert "0.6b" in _SEG_MODELS
    assert "1b" in _SEG_MODELS
    for size, (filename, repo_id) in _SEG_MODELS.items():
        assert filename.endswith(".pt2")
        assert repo_id.startswith("facebook/sapiens-seg-")


def test_goliath_seg_classes():
    from myogait.constants import GOLIATH_SEG_CLASSES
    assert len(GOLIATH_SEG_CLASSES) == 28
    assert GOLIATH_SEG_CLASSES[0] == "Background"
    assert "Torso" in GOLIATH_SEG_CLASSES
    assert "Face_Neck" in GOLIATH_SEG_CLASSES


def test_wholebody_landmark_names():
    from myogait.constants import WHOLEBODY_LANDMARK_NAMES
    assert len(WHOLEBODY_LANDMARK_NAMES) == 133
    # First 17 are COCO body keypoints
    assert WHOLEBODY_LANDMARK_NAMES[0] == "nose"
    assert WHOLEBODY_LANDMARK_NAMES[16] == "right_ankle"
    # Feet
    assert WHOLEBODY_LANDMARK_NAMES[17] == "left_big_toe"
    # Face
    assert WHOLEBODY_LANDMARK_NAMES[23] == "face_0"
    # Hands
    assert WHOLEBODY_LANDMARK_NAMES[91] == "left_hand_root"
    assert WHOLEBODY_LANDMARK_NAMES[112] == "right_hand_root"


def test_wholebody_to_coco_mapping():
    from myogait.constants import WHOLEBODY_TO_COCO
    assert len(WHOLEBODY_TO_COCO) == 17
    for i in range(17):
        assert WHOLEBODY_TO_COCO[i] == i


def test_vitpose_model_variants():
    from myogait.models.vitpose import _VITPOSE_MODELS
    assert "base" in _VITPOSE_MODELS
    assert "large" in _VITPOSE_MODELS
    assert "huge" in _VITPOSE_MODELS


def test_depth_estimator_sample_at_landmarks():
    """Test depth sampling at landmark positions."""
    from myogait.models.sapiens_depth import SapiensDepthEstimator

    estimator = SapiensDepthEstimator()

    # Synthetic depth map: gradient from 0 (top) to 1 (bottom)
    depth_map = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)

    # Landmark at center: should get ~0.5
    landmarks = np.full((3, 3), np.nan)
    landmarks[0] = [0.5, 0.5, 1.0]  # center
    landmarks[1] = [0.0, 0.0, 1.0]  # top-left

    depths = estimator.sample_at_landmarks(depth_map, landmarks, flipped=False)
    assert depths.shape == (3,)
    assert not np.isnan(depths[0])
    assert not np.isnan(depths[1])
    assert np.isnan(depths[2])  # was NaN landmark
    assert depths[0] > depths[1]  # center deeper than top


def test_seg_estimator_classes():
    from myogait.models.sapiens_seg import SapiensSegEstimator
    assert SapiensSegEstimator.n_classes == 28
    assert len(SapiensSegEstimator.classes) == 28


def test_extract_with_depth_flag():
    """Test that extract() accepts optional extraction flags."""
    import inspect
    from myogait.extract import extract
    sig = inspect.signature(extract)
    assert "with_depth" in sig.parameters
    assert "with_seg" in sig.parameters
    assert "experimental" in sig.parameters
    assert sig.parameters["with_depth"].default is False
    assert sig.parameters["with_seg"].default is False
    assert sig.parameters["experimental"].default is None


def test_sapiens_size_from_model():
    from myogait.extract import _sapiens_size_from_model
    assert _sapiens_size_from_model("sapiens-quick") == "0.3b"
    assert _sapiens_size_from_model("sapiens-mid") == "0.6b"
    assert _sapiens_size_from_model("sapiens-top") == "1b"
    assert _sapiens_size_from_model("mediapipe") == "0.3b"  # default


def test_extract_handles_wholebody_auxiliary():
    """Test that extract handles RTMW dict with auxiliary_wholebody133."""
    from myogait.extract import _coco_to_mediapipe

    # Simulate RTMW dict result
    result = {
        "landmarks": np.random.rand(17, 3).astype(np.float32),
        "auxiliary_wholebody133": np.random.rand(133, 3).astype(np.float32),
    }
    assert result["landmarks"].shape == (17, 3)
    assert result["auxiliary_wholebody133"].shape == (133, 3)

    # COCO to MP conversion should work on landmarks
    mp33 = _coco_to_mediapipe(result["landmarks"])
    assert mp33.shape == (33, 3)
