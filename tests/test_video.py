"""Tests for myogait.video module."""

import os

import cv2
import numpy as np
import pytest

from conftest import make_walking_data
from myogait.video import (
    SKELETON_CONNECTIONS,
    render_skeleton_frame,
    render_skeleton_video,
    render_stickfigure_animation,
)
from myogait.constants import GOLIATH_SKELETON_CONNECTIONS, GOLIATH_FACE_START


# ── Helpers ──────────────────────────────────────────────────────


def _make_synthetic_video(path, n_frames=10, width=320, height=240, fps=30.0):
    """Create a short synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a gradient to make frames distinguishable
        frame[:, :, 0] = int(255 * i / max(n_frames - 1, 1))
        writer.write(frame)
    writer.release()


def _sample_landmarks():
    """Return a minimal landmark dict with normalised coordinates."""
    return {
        "NOSE":              {"x": 0.50, "y": 0.10, "visibility": 1.0},
        "LEFT_EYE":          {"x": 0.49, "y": 0.08, "visibility": 1.0},
        "RIGHT_EYE":         {"x": 0.51, "y": 0.08, "visibility": 1.0},
        "LEFT_EAR":          {"x": 0.48, "y": 0.10, "visibility": 1.0},
        "RIGHT_EAR":         {"x": 0.52, "y": 0.10, "visibility": 1.0},
        "LEFT_SHOULDER":     {"x": 0.45, "y": 0.25, "visibility": 1.0},
        "RIGHT_SHOULDER":    {"x": 0.55, "y": 0.25, "visibility": 1.0},
        "LEFT_ELBOW":        {"x": 0.42, "y": 0.37, "visibility": 1.0},
        "RIGHT_ELBOW":       {"x": 0.58, "y": 0.37, "visibility": 1.0},
        "LEFT_WRIST":        {"x": 0.40, "y": 0.48, "visibility": 1.0},
        "RIGHT_WRIST":       {"x": 0.60, "y": 0.48, "visibility": 1.0},
        "LEFT_HIP":          {"x": 0.47, "y": 0.50, "visibility": 1.0},
        "RIGHT_HIP":         {"x": 0.53, "y": 0.50, "visibility": 1.0},
        "LEFT_KNEE":         {"x": 0.46, "y": 0.65, "visibility": 1.0},
        "RIGHT_KNEE":        {"x": 0.54, "y": 0.65, "visibility": 1.0},
        "LEFT_ANKLE":        {"x": 0.45, "y": 0.80, "visibility": 1.0},
        "RIGHT_ANKLE":       {"x": 0.55, "y": 0.80, "visibility": 1.0},
        "LEFT_HEEL":         {"x": 0.46, "y": 0.82, "visibility": 1.0},
        "RIGHT_HEEL":        {"x": 0.56, "y": 0.82, "visibility": 1.0},
        "LEFT_FOOT_INDEX":   {"x": 0.43, "y": 0.82, "visibility": 1.0},
        "RIGHT_FOOT_INDEX":  {"x": 0.57, "y": 0.82, "visibility": 1.0},
    }


# ── render_skeleton_frame ────────────────────────────────────────


class TestRenderSkeletonFrame:

    def test_returns_ndarray_with_correct_shape(self):
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        lm = _sample_landmarks()
        result = render_skeleton_frame(frame, lm)
        assert isinstance(result, np.ndarray)
        assert result.shape == (h, w, 3)

    def test_does_not_modify_original_frame(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        original = frame.copy()
        render_skeleton_frame(frame, _sample_landmarks())
        np.testing.assert_array_equal(frame, original)

    def test_with_angles(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        lm = _sample_landmarks()
        angles = {"hip_L": 25.3, "knee_R": 10.0}
        result = render_skeleton_frame(frame, lm, angles=angles)
        assert isinstance(result, np.ndarray)

    def test_with_events(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        lm = _sample_landmarks()
        events = {"type": "HS", "side": "left"}
        result = render_skeleton_frame(frame, lm, events=events)
        assert isinstance(result, np.ndarray)

    def test_with_empty_landmarks(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        result = render_skeleton_frame(frame, {})
        assert result.shape == frame.shape

    def test_handles_nan_landmarks(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        lm = {"NOSE": {"x": float("nan"), "y": float("nan"), "visibility": 0.0}}
        result = render_skeleton_frame(frame, lm)
        assert isinstance(result, np.ndarray)

    def test_skeleton_color_non_auto(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        result = render_skeleton_frame(
            frame, _sample_landmarks(), skeleton_color="green"
        )
        assert isinstance(result, np.ndarray)


# ── render_skeleton_video ────────────────────────────────────────


class TestRenderSkeletonVideo:

    def test_creates_output_file(self, tmp_path):
        n_frames = 5
        w, h = 320, 240
        video_in = str(tmp_path / "input.mp4")
        video_out = str(tmp_path / "output.mp4")
        _make_synthetic_video(video_in, n_frames=n_frames, width=w, height=h)

        data = make_walking_data(n_frames=n_frames, fps=30.0)
        result = render_skeleton_video(video_in, data, video_out)

        assert result == video_out
        assert os.path.exists(video_out)
        # Verify the output is a valid video
        cap = cv2.VideoCapture(video_out)
        assert cap.isOpened()
        cap.release()

    def test_with_show_angles_and_events(self, tmp_path):
        n_frames = 5
        video_in = str(tmp_path / "input.mp4")
        video_out = str(tmp_path / "output.mp4")
        _make_synthetic_video(video_in, n_frames=n_frames)

        data = make_walking_data(n_frames=n_frames, fps=30.0)
        # Add minimal angles data
        data["angles"] = {
            "frames": [
                {"hip_L": 10.0, "knee_L": 5.0} for _ in range(n_frames)
            ]
        }
        # Add minimal events
        data["events"] = {
            "left_hs": [{"frame": 0}],
            "right_to": [{"frame": 2}],
        }

        result = render_skeleton_video(
            video_in, data, video_out,
            show_angles=True, show_events=True,
        )
        assert os.path.exists(result)

    def test_raises_on_bad_video(self, tmp_path):
        data = make_walking_data(n_frames=5)
        with pytest.raises(ValueError, match="Cannot open video"):
            render_skeleton_video(
                str(tmp_path / "nonexistent.mp4"), data,
                str(tmp_path / "out.mp4"),
            )


# ── render_stickfigure_animation ─────────────────────────────────


class TestRenderStickfigureAnimation:

    def test_creates_gif(self, tmp_path):
        data = make_walking_data(n_frames=10, fps=10.0)
        out = str(tmp_path / "anim.gif")
        result = render_stickfigure_animation(data, out, format="gif", fps=10)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_with_show_trail(self, tmp_path):
        data = make_walking_data(n_frames=10, fps=10.0)
        out = str(tmp_path / "trail.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10, show_trail=True,
        )
        assert os.path.exists(result)

    def test_with_cycles(self, tmp_path):
        data = make_walking_data(n_frames=10, fps=10.0)
        cycles = {
            "cycles": [
                {"hs_frame": 0, "to_frame": 3, "end_frame": 9, "side": "left"}
            ]
        }
        out = str(tmp_path / "cycles.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10, cycles=cycles,
        )
        assert os.path.exists(result)

    def test_raises_on_empty_data(self, tmp_path):
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0, width=320, height=240, n_frames=0)
        data["frames"] = []
        with pytest.raises(ValueError, match="No frames"):
            render_stickfigure_animation(
                data, str(tmp_path / "empty.gif"), format="gif",
            )

    def test_raises_on_unsupported_format(self, tmp_path):
        data = make_walking_data(n_frames=5, fps=10.0)
        with pytest.raises(ValueError, match="Unsupported format"):
            render_stickfigure_animation(
                data, str(tmp_path / "bad.avi"), format="avi",
            )


# ── SKELETON_CONNECTIONS ─────────────────────────────────────────


class TestSkeletonConnections:

    def test_is_list_of_tuples(self):
        assert isinstance(SKELETON_CONNECTIONS, list)
        for conn in SKELETON_CONNECTIONS:
            assert isinstance(conn, tuple)
            assert len(conn) == 2

    def test_expected_length(self):
        assert len(SKELETON_CONNECTIONS) == 20


# ── Helpers for Goliath tests ────────────────────────────────────


def _sample_goliath308(conf=0.9):
    """Return a plausible 308-point Goliath landmark list.

    Body keypoints get realistic positions; face/hand points get
    approximate values so rendering logic is exercised.
    """
    # Start with NaN (invisible)
    g = [[float("nan"), float("nan"), 0.0]] * 308

    # Body (0-14) — same layout as _sample_landmarks
    body = {
        0: (0.50, 0.10),   # nose
        1: (0.49, 0.08),   # left_eye
        2: (0.51, 0.08),   # right_eye
        3: (0.48, 0.10),   # left_ear
        4: (0.52, 0.10),   # right_ear
        5: (0.45, 0.25),   # left_shoulder
        6: (0.55, 0.25),   # right_shoulder
        7: (0.42, 0.37),   # left_elbow
        8: (0.58, 0.37),   # right_elbow
        9: (0.47, 0.50),   # left_hip
        10: (0.53, 0.50),  # right_hip
        11: (0.46, 0.65),  # left_knee
        12: (0.54, 0.65),  # right_knee
        13: (0.45, 0.80),  # left_ankle
        14: (0.55, 0.80),  # right_ankle
    }
    for idx, (x, y) in body.items():
        g[idx] = [x, y, conf]

    # Feet (15-20)
    feet = {
        15: (0.43, 0.83), 16: (0.44, 0.83), 17: (0.46, 0.82),
        18: (0.57, 0.83), 19: (0.56, 0.83), 20: (0.56, 0.82),
    }
    for idx, (x, y) in feet.items():
        g[idx] = [x, y, conf]

    # Wrists (41=right, 62=left)
    g[41] = [0.60, 0.48, conf]
    g[62] = [0.40, 0.48, conf]

    # Additional body (63-69)
    g[63] = [0.41, 0.37, conf]  # left_olecranon
    g[64] = [0.59, 0.37, conf]  # right_olecranon
    g[65] = [0.43, 0.37, conf]  # left_cubital_fossa
    g[66] = [0.57, 0.37, conf]  # right_cubital_fossa
    g[67] = [0.44, 0.24, conf]  # left_acromion
    g[68] = [0.56, 0.24, conf]  # right_acromion
    g[69] = [0.50, 0.15, conf]  # neck

    return g


def _make_goliath_data(n_frames=5, fps=30.0, conf=0.9):
    """Build a data dict with goliath308 in each frame."""
    data = make_walking_data(n_frames=n_frames, fps=fps)
    for fd in data["frames"]:
        fd["goliath308"] = _sample_goliath308(conf=conf)
        fd["confidence"] = conf
    return data


# ── GOLIATH_SKELETON_CONNECTIONS ────────────────────────────────


class TestGoliathSkeletonConnections:

    def test_is_list_of_index_pairs(self):
        assert isinstance(GOLIATH_SKELETON_CONNECTIONS, list)
        for conn in GOLIATH_SKELETON_CONNECTIONS:
            assert isinstance(conn, tuple)
            assert len(conn) == 2
            assert isinstance(conn[0], int) and isinstance(conn[1], int)

    def test_all_indices_in_range(self):
        for a, b in GOLIATH_SKELETON_CONNECTIONS:
            assert 0 <= a < GOLIATH_FACE_START, f"idx {a} is face or out of range"
            assert 0 <= b < GOLIATH_FACE_START, f"idx {b} is face or out of range"

    def test_no_self_loops(self):
        for a, b in GOLIATH_SKELETON_CONNECTIONS:
            assert a != b


# ── render_skeleton_frame with goliath308 ────────────────────────


class TestRenderGoliathFrame:

    def test_renders_with_goliath308(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        g308 = _sample_goliath308()
        result = render_skeleton_frame(frame, {}, goliath308=g308)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        # Should have drawn something (not all zeros)
        assert result.sum() > 0

    def test_does_not_modify_original(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        original = frame.copy()
        render_skeleton_frame(frame, {}, goliath308=_sample_goliath308())
        np.testing.assert_array_equal(frame, original)

    def test_with_angles_and_events(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        g308 = _sample_goliath308()
        angles = {"hip_L": 25.3, "knee_R": 10.0}
        events = {"type": "HS", "side": "left"}
        result = render_skeleton_frame(
            frame, {}, angles=angles, events=events, goliath308=g308,
        )
        assert isinstance(result, np.ndarray)

    def test_low_conf_points_skipped(self):
        """Points with confidence < 0.1 should not be drawn."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        g308 = _sample_goliath308(conf=0.05)  # all below threshold
        result = render_skeleton_frame(frame, {}, goliath308=g308)
        # Nothing should be drawn
        assert result.sum() == 0

    def test_goliath_takes_precedence_over_landmarks(self):
        """When goliath308 is provided, it should be used instead of landmarks."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        lm = _sample_landmarks()
        g308 = _sample_goliath308()
        result = render_skeleton_frame(frame, lm, goliath308=g308)
        assert isinstance(result, np.ndarray)


# ── render_skeleton_video with use_goliath ──────────────────────


class TestRenderSkeletonVideoGoliath:

    def test_use_goliath(self, tmp_path):
        n_frames = 5
        video_in = str(tmp_path / "in.mp4")
        video_out = str(tmp_path / "out.mp4")
        _make_synthetic_video(video_in, n_frames=n_frames)

        data = _make_goliath_data(n_frames=n_frames)
        result = render_skeleton_video(
            video_in, data, video_out, use_goliath=True,
        )
        assert os.path.exists(result)
        cap = cv2.VideoCapture(result)
        assert cap.isOpened()
        cap.release()

    def test_min_confidence_skips_frames(self, tmp_path):
        n_frames = 5
        video_in = str(tmp_path / "in.mp4")
        video_out = str(tmp_path / "out.mp4")
        _make_synthetic_video(video_in, n_frames=n_frames)

        data = _make_goliath_data(n_frames=n_frames, conf=0.9)
        # Set frame 2 confidence below threshold
        data["frames"][2]["confidence"] = 0.1
        result = render_skeleton_video(
            video_in, data, video_out, min_confidence=0.5,
        )
        assert os.path.exists(result)


# ── render_stickfigure_animation with use_goliath ───────────────


class TestStickfigureGoliath:

    def test_goliath_stickfigure_gif(self, tmp_path):
        data = _make_goliath_data(n_frames=5, fps=10.0)
        out = str(tmp_path / "goliath.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10, use_goliath=True,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_min_confidence_filters_frames(self, tmp_path):
        data = _make_goliath_data(n_frames=10, fps=10.0, conf=0.9)
        # Set half the frames to low confidence
        for i in range(0, 10, 2):
            data["frames"][i]["confidence"] = 0.05
        out = str(tmp_path / "filtered.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10, min_confidence=0.3,
        )
        assert os.path.exists(result)

    def test_all_below_threshold_raises(self, tmp_path):
        data = _make_goliath_data(n_frames=5, fps=10.0, conf=0.01)
        for fd in data["frames"]:
            fd["confidence"] = 0.01
        out = str(tmp_path / "empty.gif")
        with pytest.raises(ValueError, match="No frames"):
            render_stickfigure_animation(
                data, out, format="gif", fps=10, min_confidence=0.5,
            )

    def test_goliath_with_angles(self, tmp_path):
        data = _make_goliath_data(n_frames=5, fps=10.0)
        data["angles"] = {
            "frames": [{"hip_L": 10.0, "knee_R": 5.0} for _ in range(5)]
        }
        out = str(tmp_path / "angles.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10, use_goliath=True,
            show_angles=True,
        )
        assert os.path.exists(result)

    def test_mediapipe_still_works(self, tmp_path):
        """Ensure the default (MediaPipe) path is not broken."""
        data = make_walking_data(n_frames=5, fps=10.0)
        out = str(tmp_path / "mp.gif")
        result = render_stickfigure_animation(
            data, out, format="gif", fps=10,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
