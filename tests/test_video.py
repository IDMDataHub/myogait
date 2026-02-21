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
