"""Useful runtime-behavior tests for extract utility functions."""

from pathlib import Path

import cv2
import numpy as np

from conftest import make_walking_data


def _write_tiny_video(path: Path, n_frames: int = 8, w: int = 64, h: int = 48, fps: float = 20.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_detect_treadmill_empty_defaults_false():
    from myogait.extract import detect_treadmill

    data = {"frames": [], "extraction": None}
    out = detect_treadmill(data)
    assert out["extraction"]["treadmill"] is False
    assert out["extraction"]["treadmill_confidence"] == 0.0


def test_detect_treadmill_detects_stationary_hip():
    from myogait.extract import detect_treadmill

    data = {"frames": [], "extraction": {}}
    for i in range(60):
        x = 0.5 + 0.002 * np.sin(i / 10.0)
        data["frames"].append(
            {
                "frame_idx": i,
                "confidence": 0.9,
                "landmarks": {
                    "LEFT_HIP": {"x": x, "y": 0.5},
                    "RIGHT_HIP": {"x": x + 0.01, "y": 0.5},
                },
            }
        )
    out = detect_treadmill(data)
    assert out["extraction"]["treadmill"] is True
    assert out["extraction"]["treadmill_confidence"] >= 0.5


def test_detect_multi_person_flags_large_jumps():
    from myogait.extract import detect_multi_person

    data = {"frames": [], "extraction": {}}
    for i in range(5):
        base = 0.2 + i * 0.01
        data["frames"].append(
            {
                "frame_idx": i,
                "confidence": 0.9,
                "landmarks": {
                    "LEFT_HIP": {"x": base, "y": 0.5},
                    "RIGHT_HIP": {"x": base + 0.05, "y": 0.5},
                    "LEFT_SHOULDER": {"x": base, "y": 0.3},
                    "RIGHT_SHOULDER": {"x": base + 0.05, "y": 0.3},
                    "NOSE": {"x": base + 0.02, "y": 0.2},
                },
            }
        )
    # Inject a suspicious switch
    data["frames"][3]["landmarks"]["LEFT_HIP"]["x"] = 0.9
    data["frames"][3]["landmarks"]["RIGHT_HIP"]["x"] = 0.95

    out = detect_multi_person(data)
    assert out["extraction"]["multi_person_warning"] is True
    assert len(out["extraction"]["suspicious_frames"]) >= 1


def test_detect_sagittal_alignment_no_valid_frames():
    from myogait.extract import detect_sagittal_alignment

    data = {"frames": [{"landmarks": {}}]}
    result = detect_sagittal_alignment(data)
    assert result["confidence"] == 0.0
    assert result["warning"] is not None


def test_detect_sagittal_alignment_oblique_case():
    from myogait.extract import detect_sagittal_alignment

    data = make_walking_data(20, fps=30.0)
    # Enforce wide hip distance relative to femur -> oblique
    for f in data["frames"]:
        f["landmarks"]["LEFT_HIP"]["x"] = 0.2
        f["landmarks"]["RIGHT_HIP"]["x"] = 0.8
        f["landmarks"]["LEFT_KNEE"]["x"] = 0.25
        f["landmarks"]["RIGHT_KNEE"]["x"] = 0.75
    result = detect_sagittal_alignment(data, threshold_deg=10.0)
    assert result["is_sagittal"] is False
    assert result["warning"] is not None


def test_auto_crop_roi_full_frame_without_data(tmp_path):
    from myogait.extract import auto_crop_roi

    vpath = tmp_path / "tiny.mp4"
    _write_tiny_video(vpath, w=80, h=60)
    out = auto_crop_roi(str(vpath), data=None, output_path=None)
    assert out["bbox"] == (0, 0, 80, 60)
    assert out["output_path"] is None


def test_auto_crop_roi_writes_cropped_video(tmp_path):
    from myogait.extract import auto_crop_roi

    vpath = tmp_path / "tiny.mp4"
    outpath = tmp_path / "crop.mp4"
    _write_tiny_video(vpath, w=100, h=80)

    data = {
        "frames": [
            {"landmarks": {"NOSE": {"x": 0.4, "y": 0.4}, "LEFT_ANKLE": {"x": 0.45, "y": 0.8}}},
            {"landmarks": {"NOSE": {"x": 0.6, "y": 0.45}, "RIGHT_ANKLE": {"x": 0.55, "y": 0.82}}},
        ]
    }
    out = auto_crop_roi(str(vpath), data=data, padding=0.05, output_path=str(outpath))
    assert out["output_path"] == str(outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_select_person_bbox_filter_and_metadata():
    from myogait.extract import select_person

    data = {
        "extraction": {"multi_person_warning": True},
        "frames": [
            {"landmarks": {"NOSE": {"x": 0.2, "y": 0.2}, "LEFT_HIP": {"x": 0.25, "y": 0.6}}},
            {"landmarks": {"NOSE": {"x": 0.8, "y": 0.2}, "LEFT_HIP": {"x": 0.75, "y": 0.6}}},
        ],
    }
    res = select_person(data, strategy="center", bbox=(0.0, 0.0, 0.5, 1.0))
    assert res["selected"] is True
    assert res["strategy"] == "center"
    assert res["multi_person_warning"] is True
    assert res["bbox"] is not None
