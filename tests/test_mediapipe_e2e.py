"""End-to-end extraction test with a REAL pose backend (MediaPipe).

The main CI matrix installs only the [dev] extra, so every backend test
skips itself — the green badge proves nothing about extraction. This
file runs in a dedicated CI job that installs myogait[mediapipe]
(CPU-only) and exercises the real extraction path: video decode →
MediaPipe inference → pivot dict.

Locally it skips cleanly when mediapipe is not installed.
"""

import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")

import cv2  # noqa: E402

import myogait as mg  # noqa: E402
from myogait.models import available_models, get_extractor  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """A short synthetic video: a high-contrast humanoid figure moving
    horizontally. MediaPipe may or may not detect it as a person — the
    e2e contract is that the full pipeline runs without crashing and
    returns a well-formed pivot dict either way."""
    path = tmp_path_factory.mktemp("e2e") / "walk.mp4"
    w, h, fps, n = 320, 240, 30, 45
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (w, h))
    for i in range(n):
        img = np.full((h, w, 3), 255, np.uint8)
        x = 60 + int(i * 3)
        # head, trunk, legs, arms — a stick figure
        cv2.circle(img, (x, 50), 12, (0, 0, 0), -1)
        cv2.line(img, (x, 62), (x, 140), (0, 0, 0), 8)
        cv2.line(img, (x, 140), (x - 18, 200), (0, 0, 0), 6)
        cv2.line(img, (x, 140), (x + 18, 200), (0, 0, 0), 6)
        cv2.line(img, (x, 80), (x - 22, 120), (0, 0, 0), 5)
        cv2.line(img, (x, 80), (x + 22, 120), (0, 0, 0), 5)
        out.write(img)
    out.release()
    return str(path)


def test_available_models_reports_mediapipe():
    assert available_models()["mediapipe"] is True


def test_extractor_lifecycle():
    ext = get_extractor("mediapipe")
    ext.setup()
    try:
        frame = np.full((240, 320, 3), 255, np.uint8)
        result = ext.process_frame(frame)   # blank frame: no person
        assert result is None or hasattr(result, "landmarks") or isinstance(result, (dict, np.ndarray))
    finally:
        ext.teardown()


def test_extract_end_to_end(synthetic_video):
    data = mg.extract(synthetic_video, model="mediapipe",
                      show_progress=False)
    assert data["meta"]["fps"] > 0
    assert isinstance(data["frames"], list)
    assert data["extraction"]["model"] == "mediapipe"
    # Every frame entry is well-formed whether or not a person was found
    for f in data["frames"]:
        assert "landmarks" in f


def test_extract_unreadable_video_raises(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(mg.UnreadableVideoError):
        mg.extract(str(bad), model="mediapipe", show_progress=False)
