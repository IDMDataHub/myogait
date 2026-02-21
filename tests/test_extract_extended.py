"""Extended tests for myogait.extract module.

Tests for detect_sagittal_alignment, auto_crop_roi, and select_person.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from conftest import make_walking_data, make_standing_data
from myogait.extract import (
    detect_sagittal_alignment,
    auto_crop_roi,
    select_person,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_synthetic_video(path, n_frames=10, width=320, height=240, fps=30.0):
    """Create a short synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ── detect_sagittal_alignment ────────────────────────────────────


class TestDetectSagittalAlignment:

    def test_returns_expected_keys(self):
        data = make_walking_data(n_frames=30)
        result = detect_sagittal_alignment(data)
        assert isinstance(result, dict)
        expected_keys = {
            "deviation_angle_deg",
            "is_sagittal",
            "hip_width_ratio",
            "confidence",
            "warning",
        }
        assert set(result.keys()) == expected_keys

    def test_sagittal_data_detected_as_sagittal(self):
        """Standing data with overlapping hips (profile view) should be sagittal."""
        data = make_standing_data(n_frames=30)
        result = detect_sagittal_alignment(data)
        assert result["is_sagittal"] is True
        assert result["deviation_angle_deg"] < 15.0
        assert result["warning"] is None

    def test_oblique_data_detected(self):
        """Data with wide hip separation should be flagged as oblique."""
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0, width=640, height=480, n_frames=30)
        data["extraction"] = {"model": "mediapipe"}
        frames = []
        for i in range(30):
            lm = {
                "LEFT_HIP":   {"x": 0.20, "y": 0.50, "visibility": 1.0},
                "RIGHT_HIP":  {"x": 0.80, "y": 0.50, "visibility": 1.0},
                "LEFT_KNEE":  {"x": 0.20, "y": 0.65, "visibility": 1.0},
                "RIGHT_KNEE": {"x": 0.80, "y": 0.65, "visibility": 1.0},
            }
            frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.9})
        data["frames"] = frames

        result = detect_sagittal_alignment(data)
        assert result["is_sagittal"] is False
        assert result["deviation_angle_deg"] > 15.0
        assert result["warning"] is not None

    def test_empty_frames(self):
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0)
        data["frames"] = []
        result = detect_sagittal_alignment(data)
        assert result["confidence"] == 0.0
        assert result["is_sagittal"] is True

    def test_custom_threshold(self):
        data = make_walking_data(n_frames=30)
        result_strict = detect_sagittal_alignment(data, threshold_deg=0.1)
        result_loose = detect_sagittal_alignment(data, threshold_deg=90.0)
        # With a very loose threshold, everything should be sagittal
        assert result_loose["is_sagittal"] is True


# ── auto_crop_roi ────────────────────────────────────────────────


class TestAutoCropRoi:

    def test_returns_bbox(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=5)
        data = make_walking_data(n_frames=5)

        result = auto_crop_roi(video_path, data=data)
        assert "bbox" in result
        assert len(result["bbox"]) == 4
        x1, y1, x2, y2 = result["bbox"]
        assert x1 < x2
        assert y1 < y2
        assert result["output_path"] is None

    def test_without_data(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=5, width=320, height=240)

        result = auto_crop_roi(video_path, data=None)
        assert "bbox" in result
        x1, y1, x2, y2 = result["bbox"]
        # Without data, bbox should cover the full frame
        assert x1 == 0
        assert y1 == 0

    def test_with_output_path(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        out_path = str(tmp_path / "cropped.mp4")
        _make_synthetic_video(video_path, n_frames=5)
        data = make_walking_data(n_frames=5)

        result = auto_crop_roi(video_path, data=data, output_path=out_path)
        assert result["output_path"] == out_path
        assert os.path.exists(out_path)

    def test_raises_on_bad_video(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot open video"):
            auto_crop_roi(str(tmp_path / "nonexistent.mp4"))

    def test_custom_padding(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=5)
        data = make_walking_data(n_frames=5)

        r_small = auto_crop_roi(video_path, data=data, padding=0.05)
        r_large = auto_crop_roi(video_path, data=data, padding=0.50)
        # Larger padding should produce a wider bbox
        x1s, _, x2s, _ = r_small["bbox"]
        x1l, _, x2l, _ = r_large["bbox"]
        assert (x2l - x1l) >= (x2s - x1s)


# ── select_person ────────────────────────────────────────────────


class TestSelectPerson:

    def test_returns_expected_keys(self):
        data = make_walking_data(n_frames=30)
        result = select_person(data)
        assert isinstance(result, dict)
        expected_keys = {
            "selected",
            "strategy",
            "n_frames_with_landmarks",
            "bbox",
            "multi_person_warning",
        }
        assert set(result.keys()) == expected_keys

    def test_finds_person_in_walking_data(self):
        data = make_walking_data(n_frames=30)
        result = select_person(data, strategy="largest")
        assert result["selected"] is True
        assert result["strategy"] == "largest"
        assert result["n_frames_with_landmarks"] == 30
        assert result["bbox"] is not None

    def test_center_strategy(self):
        data = make_walking_data(n_frames=30)
        result = select_person(data, strategy="center")
        assert result["strategy"] == "center"
        assert result["selected"] is True

    def test_with_bbox_filter(self):
        data = make_walking_data(n_frames=30)
        # Use a bbox that covers everything
        result = select_person(data, bbox=(0.0, 0.0, 1.0, 1.0))
        assert result["selected"] is True
        assert result["n_frames_with_landmarks"] == 30

    def test_with_restrictive_bbox(self):
        data = make_walking_data(n_frames=30)
        # Use a very small bbox that won't contain any landmarks
        result = select_person(data, bbox=(0.99, 0.99, 1.0, 1.0))
        assert result["n_frames_with_landmarks"] == 0
        assert result["selected"] is False

    def test_empty_frames(self):
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0)
        data["frames"] = []
        result = select_person(data)
        assert result["selected"] is False
        assert result["n_frames_with_landmarks"] == 0
        assert result["bbox"] is None

    def test_multi_person_warning_from_extraction(self):
        data = make_walking_data(n_frames=10)
        data["extraction"]["multi_person_warning"] = True
        result = select_person(data)
        assert result["multi_person_warning"] is True
