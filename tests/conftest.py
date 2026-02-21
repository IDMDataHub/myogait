"""Shared test fixtures for myogait test suite.

Provides reusable fake data generators and pipeline helpers
used across all test modules.
"""

import numpy as np
import pytest


def make_fake_data(n_frames, fps=30.0):
    """Create minimal fake data with landmarks for testing."""
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=fps, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    for i in range(n_frames):
        lm = {}
        for name in ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
                      "LEFT_SHOULDER", "RIGHT_SHOULDER",
                      "LEFT_ELBOW", "RIGHT_ELBOW",
                      "LEFT_WRIST", "RIGHT_WRIST",
                      "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                      "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                      "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]:
            lm[name] = {"x": 0.5, "y": 0.5, "visibility": 1.0}
        frames.append({"frame_idx": i, "time_s": i / fps, "landmarks": lm, "confidence": 0.9})
    data["frames"] = frames
    return data


def make_standing_data(n_frames, fps=30.0):
    """Create data simulating a standing person in profile view."""
    from myogait.schema import create_empty
    data = create_empty("test.mp4", fps=fps, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    for i in range(n_frames):
        lm = {
            "NOSE":             {"x": 0.50, "y": 0.10, "visibility": 1.0},
            "LEFT_EYE":         {"x": 0.49, "y": 0.08, "visibility": 1.0},
            "RIGHT_EYE":        {"x": 0.51, "y": 0.08, "visibility": 1.0},
            "LEFT_EAR":         {"x": 0.48, "y": 0.10, "visibility": 1.0},
            "RIGHT_EAR":        {"x": 0.52, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "LEFT_ELBOW":       {"x": 0.50, "y": 0.37, "visibility": 1.0},
            "RIGHT_ELBOW":      {"x": 0.50, "y": 0.37, "visibility": 1.0},
            "LEFT_WRIST":       {"x": 0.50, "y": 0.48, "visibility": 1.0},
            "RIGHT_WRIST":      {"x": 0.50, "y": 0.48, "visibility": 1.0},
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
        frames.append({"frame_idx": i, "time_s": i / fps, "landmarks": lm, "confidence": 0.95})
    data["frames"] = frames
    return data


def make_walking_data(n_frames=300, fps=30.0):
    """Create data simulating a person walking (sinusoidal ankle motion).

    Ankles oscillate in x relative to hips, simulating gait cycles.
    Period ~1s (30 frames at 30fps), 10 cycles in 300 frames.
    Arms swing in anti-phase with legs.
    """
    from myogait.schema import create_empty
    data = create_empty("test_walk.mp4", fps=fps, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    cycle_period = 1.0
    for i in range(n_frames):
        t = i / fps
        phase_l = 2 * np.pi * t / cycle_period
        phase_r = phase_l + np.pi
        ankle_amp = 0.08
        arm_amp = 0.04
        hip_x = 0.50
        lm = {
            "NOSE":             {"x": hip_x, "y": 0.10, "visibility": 1.0},
            "LEFT_EYE":         {"x": hip_x - 0.01, "y": 0.08, "visibility": 1.0},
            "RIGHT_EYE":        {"x": hip_x + 0.01, "y": 0.08, "visibility": 1.0},
            "LEFT_EAR":         {"x": hip_x - 0.02, "y": 0.10, "visibility": 1.0},
            "RIGHT_EAR":        {"x": hip_x + 0.02, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": hip_x, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": hip_x + 0.01, "y": 0.25, "visibility": 1.0},
            # Arms swing in anti-phase with ipsilateral leg
            "LEFT_ELBOW":       {"x": hip_x + arm_amp * np.sin(phase_r) * 0.5, "y": 0.37, "visibility": 1.0},
            "RIGHT_ELBOW":      {"x": hip_x + 0.01 + arm_amp * np.sin(phase_l) * 0.5, "y": 0.37, "visibility": 1.0},
            "LEFT_WRIST":       {"x": hip_x + arm_amp * np.sin(phase_r), "y": 0.48, "visibility": 1.0},
            "RIGHT_WRIST":      {"x": hip_x + 0.01 + arm_amp * np.sin(phase_l), "y": 0.48, "visibility": 1.0},
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


def make_walking_data_with_low_confidence(n_frames=300, fps=30.0, low_vis_rate=0.3):
    """Walking data where some landmarks have low visibility scores."""
    data = make_walking_data(n_frames, fps)
    rng = np.random.RandomState(42)
    for frame in data["frames"]:
        for name, lm in frame["landmarks"].items():
            if rng.random() < low_vis_rate:
                lm["visibility"] = rng.uniform(0.0, 0.2)
    return data


def make_walking_data_with_gaps(n_frames=300, fps=30.0, gap_frames=None):
    """Walking data with NaN gaps at specified frame ranges."""
    data = make_walking_data(n_frames, fps)
    if gap_frames is None:
        gap_frames = list(range(50, 65)) + list(range(150, 155))
    for i in gap_frames:
        if i < len(data["frames"]):
            for name in list(data["frames"][i]["landmarks"].keys()):
                data["frames"][i]["landmarks"][name] = {
                    "x": float("nan"), "y": float("nan"), "visibility": 0.0
                }
            data["frames"][i]["confidence"] = 0.0
    return data


def make_walking_data_with_depth(n_frames=100, fps=30.0):
    """Walking data with simulated depth values per landmark."""
    data = make_walking_data(n_frames, fps)
    for frame in data["frames"]:
        depths = {}
        for name, lm in frame["landmarks"].items():
            # Simulate depth: closer landmarks (lower y) have lower depth
            depths[name] = round(lm["y"] * 2.0 + 0.5, 4)
        frame["landmark_depths"] = depths
    return data


def walking_data_with_angles(n_frames=300, fps=30.0):
    """Walking data with angles computed."""
    from myogait import compute_angles
    data = make_walking_data(n_frames, fps)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    return data


def run_full_pipeline(n_frames=300, fps=30.0):
    """Run the complete pipeline and return (data, cycles, stats)."""
    from myogait import normalize, compute_angles, detect_events, segment_cycles, analyze_gait
    data = make_walking_data(n_frames, fps)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    return data, cycles, stats


def make_normative_like_data(n_frames=300, fps=30.0):
    """Walking data that mimics normal adult gait kinematics.

    Produces angles close to normative values:
    - Hip: -10 to 30 deg
    - Knee: 0 to 60 deg
    - Ankle: -15 to 15 deg
    """
    from myogait.schema import create_empty
    data = create_empty("test_normal.mp4", fps=fps, width=1920, height=1080, n_frames=n_frames)
    data["extraction"] = {"model": "mediapipe"}
    frames = []
    cycle_period = 1.1  # ~109 steps/min
    for i in range(n_frames):
        t = i / fps
        phase = 2 * np.pi * t / cycle_period
        # Use realistic oscillation amplitudes
        hip_x = 0.50
        # Knee oscillation in y to produce realistic knee angle
        knee_y_offset = 0.02 * np.sin(phase)
        lm = {
            "NOSE":             {"x": hip_x, "y": 0.10, "visibility": 1.0},
            "LEFT_EYE":         {"x": hip_x - 0.01, "y": 0.08, "visibility": 1.0},
            "RIGHT_EYE":        {"x": hip_x + 0.01, "y": 0.08, "visibility": 1.0},
            "LEFT_EAR":         {"x": hip_x - 0.02, "y": 0.10, "visibility": 1.0},
            "RIGHT_EAR":        {"x": hip_x + 0.02, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER":    {"x": hip_x, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER":   {"x": hip_x + 0.01, "y": 0.25, "visibility": 1.0},
            "LEFT_ELBOW":       {"x": hip_x, "y": 0.37, "visibility": 1.0},
            "RIGHT_ELBOW":      {"x": hip_x + 0.01, "y": 0.37, "visibility": 1.0},
            "LEFT_WRIST":       {"x": hip_x, "y": 0.48, "visibility": 1.0},
            "RIGHT_WRIST":      {"x": hip_x + 0.01, "y": 0.48, "visibility": 1.0},
            "LEFT_HIP":         {"x": hip_x, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP":        {"x": hip_x + 0.01, "y": 0.50, "visibility": 1.0},
            "LEFT_KNEE":        {"x": hip_x + 0.06 * np.sin(phase), "y": 0.65 + knee_y_offset, "visibility": 1.0},
            "RIGHT_KNEE":       {"x": hip_x + 0.01 + 0.06 * np.sin(phase + np.pi), "y": 0.65 - knee_y_offset, "visibility": 1.0},
            "LEFT_ANKLE":       {"x": hip_x + 0.10 * np.sin(phase), "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE":      {"x": hip_x + 0.01 + 0.10 * np.sin(phase + np.pi), "y": 0.80, "visibility": 1.0},
            "LEFT_HEEL":        {"x": hip_x + 0.10 * np.sin(phase) + 0.01, "y": 0.82, "visibility": 1.0},
            "RIGHT_HEEL":       {"x": hip_x + 0.01 + 0.10 * np.sin(phase + np.pi) + 0.01, "y": 0.82, "visibility": 1.0},
            "LEFT_FOOT_INDEX":  {"x": hip_x + 0.10 * np.sin(phase) - 0.03, "y": 0.82, "visibility": 1.0},
            "RIGHT_FOOT_INDEX": {"x": hip_x + 0.01 + 0.10 * np.sin(phase + np.pi) - 0.03, "y": 0.82, "visibility": 1.0},
        }
        frames.append({"frame_idx": i, "time_s": round(t, 4), "landmarks": lm, "confidence": 0.95})
    data["frames"] = frames
    return data
