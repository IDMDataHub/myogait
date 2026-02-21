"""Tests for OpenPose JSON export and inverse mapping tables."""

import json
import os

import pytest

from conftest import make_walking_data
from myogait.export import export_openpose_json
from myogait.constants import (
    MP_TO_COCO_17,
    MP_TO_BODY25,
    MP_TO_HALPE26,
    BODY_25_LANDMARK_NAMES,
    HALPE_26_LANDMARK_NAMES,
    OPENSIM_MARKER_MAP,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _load_first_json(paths):
    """Load and return the first OpenPose JSON from a list of paths."""
    with open(paths[0]) as f:
        return json.load(f)


# ── Mapping table tests ──────────────────────────────────────────────


class TestMappingTables:
    """Verify inverse mapping tables exist and have correct sizes."""

    def test_mp_to_coco_17_size(self):
        assert len(MP_TO_COCO_17) == 17

    def test_mp_to_coco_17_values_in_range(self):
        for name, idx in MP_TO_COCO_17.items():
            assert 0 <= idx < 17, f"{name} maps to out-of-range index {idx}"

    def test_mp_to_body25_size(self):
        # At least 20 direct mappings (excludes composites Neck, MidHip)
        assert len(MP_TO_BODY25) >= 20

    def test_mp_to_body25_values_in_range(self):
        for name, idx in MP_TO_BODY25.items():
            assert 0 <= idx < 25, f"{name} maps to out-of-range index {idx}"

    def test_mp_to_halpe26_size(self):
        # At least 18 direct mappings (excludes composites Head, Neck, Hip)
        assert len(MP_TO_HALPE26) >= 18

    def test_mp_to_halpe26_values_in_range(self):
        for name, idx in MP_TO_HALPE26.items():
            assert 0 <= idx < 26, f"{name} maps to out-of-range index {idx}"

    def test_body25_landmark_names_length(self):
        assert len(BODY_25_LANDMARK_NAMES) == 25

    def test_halpe26_landmark_names_length(self):
        assert len(HALPE_26_LANDMARK_NAMES) == 26

    def test_opensim_marker_map_keys(self):
        assert "gait2392" in OPENSIM_MARKER_MAP
        assert "rajagopal2015" in OPENSIM_MARKER_MAP

    def test_opensim_gait2392_has_markers(self):
        markers = OPENSIM_MARKER_MAP["gait2392"]
        assert "RIGHT_SHOULDER" in markers
        assert "LEFT_ANKLE" in markers
        assert "LEFT_FOOT_INDEX" in markers

    def test_opensim_rajagopal2015_has_markers(self):
        markers = OPENSIM_MARKER_MAP["rajagopal2015"]
        assert "RIGHT_SHOULDER" in markers
        assert "LEFT_ANKLE" in markers
        assert "LEFT_FOOT_INDEX" in markers


# ── File creation tests ──────────────────────────────────────────────


class TestOpenPoseFileCreation:
    """Verify that export_openpose_json creates the correct number of files."""

    def test_creates_one_file_per_frame_coco(self, tmp_path):
        data = make_walking_data(n_frames=5)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        assert len(paths) == 5
        for p in paths:
            assert os.path.isfile(p)

    def test_creates_one_file_per_frame_body25(self, tmp_path):
        data = make_walking_data(n_frames=3)
        paths = export_openpose_json(data, str(tmp_path), model="BODY_25")
        assert len(paths) == 3

    def test_creates_one_file_per_frame_halpe26(self, tmp_path):
        data = make_walking_data(n_frames=4)
        paths = export_openpose_json(data, str(tmp_path), model="HALPE_26")
        assert len(paths) == 4

    def test_file_naming_convention(self, tmp_path):
        data = make_walking_data(n_frames=2)
        paths = export_openpose_json(data, str(tmp_path), model="COCO", prefix="test_")
        assert paths[0].endswith("test_000000000000_keypoints.json")
        assert paths[1].endswith("test_000000000001_keypoints.json")

    def test_file_naming_no_prefix(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        assert os.path.basename(paths[0]) == "000000000000_keypoints.json"


# ── Format validation tests ──────────────────────────────────────────


class TestOpenPoseFormat:
    """Verify the JSON structure and keypoint array sizes."""

    def test_coco_keypoints_length(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        assert len(kps) == 17 * 3  # 51

    def test_body25_keypoints_length(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="BODY_25")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        assert len(kps) == 25 * 3  # 75

    def test_halpe26_keypoints_length(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="HALPE_26")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        assert len(kps) == 26 * 3  # 78

    def test_openpose_structure_keys(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        doc = _load_first_json(paths)
        assert "version" in doc
        assert doc["version"] == 1.1
        assert "people" in doc
        person = doc["people"][0]
        assert "person_id" in person
        assert "pose_keypoints_2d" in person
        assert "face_keypoints_2d" in person
        assert "hand_left_keypoints_2d" in person
        assert "hand_right_keypoints_2d" in person

    def test_person_id_is_minus_one(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        doc = _load_first_json(paths)
        assert doc["people"][0]["person_id"] == [-1]


# ── Denormalization tests ────────────────────────────────────────────


class TestDenormalization:
    """Verify that keypoints are denormalized to pixel coordinates."""

    def test_keypoints_are_in_pixel_range(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]   # 1920
        height = data["meta"]["height"]  # 1080
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        # Check that non-zero x values are in pixel range (not 0-1)
        for i in range(17):
            x = kps[i * 3]
            y = kps[i * 3 + 1]
            if x > 0:
                assert x > 1.0, f"Keypoint {i} x={x} appears normalized (should be pixels)"
                assert x <= width, f"Keypoint {i} x={x} exceeds width {width}"
            if y > 0:
                assert y > 1.0, f"Keypoint {i} y={y} appears normalized (should be pixels)"
                assert y <= height, f"Keypoint {i} y={y} exceeds height {height}"

    def test_nose_denormalized_correctly(self, tmp_path):
        """NOSE is at x=0.5, y=0.10 -> should be 960.0, 108.0 in pixels."""
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="COCO")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        # COCO index 0 = NOSE
        nose_x = kps[0]
        nose_y = kps[1]
        assert abs(nose_x - 960.0) < 1.0  # 0.5 * 1920
        assert abs(nose_y - 108.0) < 1.0  # 0.10 * 1080


# ── Composite keypoint tests ────────────────────────────────────────


class TestCompositeKeypoints:
    """Verify composite keypoints (Neck, MidHip, Head)."""

    def test_body25_neck_is_midpoint_shoulders(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]
        height = data["meta"]["height"]
        paths = export_openpose_json(data, str(tmp_path), model="BODY_25")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]

        lm = data["frames"][0]["landmarks"]
        ls = lm["LEFT_SHOULDER"]
        rs = lm["RIGHT_SHOULDER"]
        expected_x = (ls["x"] + rs["x"]) / 2.0 * width
        expected_y = (ls["y"] + rs["y"]) / 2.0 * height

        # Neck is idx 1 in BODY_25
        neck_x = kps[1 * 3]
        neck_y = kps[1 * 3 + 1]
        assert abs(neck_x - expected_x) < 0.01
        assert abs(neck_y - expected_y) < 0.01

    def test_body25_midhip_is_midpoint_hips(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]
        height = data["meta"]["height"]
        paths = export_openpose_json(data, str(tmp_path), model="BODY_25")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]

        lm = data["frames"][0]["landmarks"]
        lh = lm["LEFT_HIP"]
        rh = lm["RIGHT_HIP"]
        expected_x = (lh["x"] + rh["x"]) / 2.0 * width
        expected_y = (lh["y"] + rh["y"]) / 2.0 * height

        # MidHip is idx 8 in BODY_25
        midhip_x = kps[8 * 3]
        midhip_y = kps[8 * 3 + 1]
        assert abs(midhip_x - expected_x) < 0.01
        assert abs(midhip_y - expected_y) < 0.01

    def test_body25_neck_confidence(self, tmp_path):
        data = make_walking_data(n_frames=1)
        paths = export_openpose_json(data, str(tmp_path), model="BODY_25")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]
        # Neck confidence should be min of two shoulders
        neck_conf = kps[1 * 3 + 2]
        assert neck_conf > 0.0

    def test_halpe26_head_equals_nose(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]
        height = data["meta"]["height"]
        paths = export_openpose_json(data, str(tmp_path), model="HALPE_26")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]

        lm = data["frames"][0]["landmarks"]
        nose = lm["NOSE"]
        # Head is idx 17 in HALPE_26
        head_x = kps[17 * 3]
        head_y = kps[17 * 3 + 1]
        assert abs(head_x - nose["x"] * width) < 0.01
        assert abs(head_y - nose["y"] * height) < 0.01

    def test_halpe26_neck_is_midpoint_shoulders(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]
        height = data["meta"]["height"]
        paths = export_openpose_json(data, str(tmp_path), model="HALPE_26")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]

        lm = data["frames"][0]["landmarks"]
        ls = lm["LEFT_SHOULDER"]
        rs = lm["RIGHT_SHOULDER"]
        expected_x = (ls["x"] + rs["x"]) / 2.0 * width
        expected_y = (ls["y"] + rs["y"]) / 2.0 * height

        # Neck is idx 18 in HALPE_26
        neck_x = kps[18 * 3]
        neck_y = kps[18 * 3 + 1]
        assert abs(neck_x - expected_x) < 0.01
        assert abs(neck_y - expected_y) < 0.01

    def test_halpe26_hip_is_midpoint_hips(self, tmp_path):
        data = make_walking_data(n_frames=1)
        width = data["meta"]["width"]
        height = data["meta"]["height"]
        paths = export_openpose_json(data, str(tmp_path), model="HALPE_26")
        doc = _load_first_json(paths)
        kps = doc["people"][0]["pose_keypoints_2d"]

        lm = data["frames"][0]["landmarks"]
        lh = lm["LEFT_HIP"]
        rh = lm["RIGHT_HIP"]
        expected_x = (lh["x"] + rh["x"]) / 2.0 * width
        expected_y = (lh["y"] + rh["y"]) / 2.0 * height

        # Hip is idx 19 in HALPE_26
        hip_x = kps[19 * 3]
        hip_y = kps[19 * 3 + 1]
        assert abs(hip_x - expected_x) < 0.01
        assert abs(hip_y - expected_y) < 0.01


# ── Error handling tests ─────────────────────────────────────────────


class TestOpenPoseErrors:
    """Verify error handling."""

    def test_raises_on_invalid_model(self, tmp_path):
        data = make_walking_data(n_frames=1)
        with pytest.raises(ValueError, match="Unknown model"):
            export_openpose_json(data, str(tmp_path), model="INVALID")

    def test_raises_on_non_dict(self, tmp_path):
        with pytest.raises(TypeError):
            export_openpose_json("not a dict", str(tmp_path))

    def test_raises_on_empty_frames(self, tmp_path):
        with pytest.raises(ValueError, match="No frames"):
            export_openpose_json({"frames": []}, str(tmp_path))
