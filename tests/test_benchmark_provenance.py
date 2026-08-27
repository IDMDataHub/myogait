"""Reproducibility metadata for experimental VICON benchmarks."""

import hashlib

from myogait.experimental_benchmark import benchmark_input_fingerprints


def test_benchmark_input_fingerprints_identify_existing_inputs(tmp_path):
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"video")
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "res_angles_t.mat").write_bytes(b"angles")

    fingerprints = benchmark_input_fingerprints(video, trial)

    assert fingerprints["video_sha256"] == hashlib.sha256(b"video").hexdigest()
    assert fingerprints["vicon_res_angles_t.mat_sha256"] == hashlib.sha256(b"angles").hexdigest()
    assert fingerprints["vicon_points3D_t.mat_sha256"] is None
    assert fingerprints["vicon_cycle.mat_sha256"] is None
