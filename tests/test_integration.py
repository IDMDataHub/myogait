"""Comprehensive integration tests for the myogait package.

Tests cover end-to-end pipelines, JSON round-trips, edge cases,
I/O error handling, config management, export formats, validation,
reporting, determinism, and cross-method consistency.
"""

import copy
import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from pathlib import Path


# ── Helper functions (duplicated from test_extract.py for independence) ──


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
    cycle_period = 1.0
    for i in range(n_frames):
        t = i / fps
        phase_l = 2 * np.pi * t / cycle_period
        phase_r = phase_l + np.pi
        ankle_amp = 0.08
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


def _walking_data_with_angles():
    """Helper: walking data + angles computed."""
    from myogait import compute_angles
    data = _make_walking_data(300, fps=30.0)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    return data


def _run_full_pipeline(n_frames=300, fps=30.0):
    """Run the complete pipeline and return (data, cycles, stats)."""
    from myogait import normalize, compute_angles, detect_events, segment_cycles, analyze_gait
    data = _make_walking_data(n_frames, fps)
    normalize(data, filters=["butterworth"])
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data)
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    return data, cycles, stats


# ── 1. Full pipeline test ─────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end pipeline tests using synthetic walking data."""

    def test_full_pipeline_end_to_end(self):
        """Run extract-like data through normalize, angles, events, cycles, analyze."""
        data, cycles, stats = _run_full_pipeline()

        # Normalization was applied
        assert data.get("normalization") is not None
        assert "butterworth" in data["normalization"]["steps_applied"]

        # Angles were computed
        assert data.get("angles") is not None
        assert len(data["angles"]["frames"]) == 300

        # Events were detected
        assert data.get("events") is not None
        assert len(data["events"]["left_hs"]) >= 5
        assert len(data["events"]["right_hs"]) >= 5

        # Cycles were segmented
        assert len(cycles["cycles"]) >= 4

        # Stats were computed
        assert stats["spatiotemporal"]["cadence_steps_per_min"] > 0
        assert stats["spatiotemporal"]["n_cycles_total"] >= 4
        assert "symmetry" in stats
        assert "variability" in stats

    def test_analyze_gait_keeps_legacy_summary_keys(self):
        """Backward compatibility for tutorial keys at top-level stats."""
        _, _, stats = _run_full_pipeline()
        assert "cadence" in stats
        assert "speed" in stats
        assert "stance_pct" in stats
        assert stats["cadence"] == stats["spatiotemporal"]["cadence_steps_per_min"]
        assert stats["speed"] == stats["walking_speed"]["speed_mean"]

    def test_pipeline_preserves_original_frames(self):
        """Normalization should preserve raw frames in frames_raw."""
        data, _, _ = _run_full_pipeline()
        assert "frames_raw" in data
        assert len(data["frames_raw"]) == 300
        # Raw frames should differ from filtered frames
        raw_x = data["frames_raw"][50]["landmarks"]["LEFT_ANKLE"]["x"]
        filt_x = data["frames"][50]["landmarks"]["LEFT_ANKLE"]["x"]
        # They may be close but the key is frames_raw is present
        assert raw_x is not None
        assert filt_x is not None

    def test_pipeline_meta_consistency(self):
        """Verify that meta information is consistent throughout the pipeline."""
        data, _, _ = _run_full_pipeline()
        assert data["meta"]["fps"] == 30.0
        assert data["meta"]["n_frames"] == 300
        assert data["meta"]["width"] == 1920
        assert data["meta"]["height"] == 1080


# ── 2. JSON round-trip ────────────────────────────────────────────────


class TestJsonRoundTrip:
    """Test saving and loading JSON data preserves all fields."""

    def test_basic_json_round_trip(self, tmp_path):
        """Create data, save_json, load_json, verify all fields preserved."""
        from myogait.schema import save_json, load_json
        data = _make_walking_data(30)
        path = tmp_path / "basic.json"
        save_json(data, path)
        loaded = load_json(path)

        assert loaded["meta"]["fps"] == data["meta"]["fps"]
        assert loaded["meta"]["n_frames"] == data["meta"]["n_frames"]
        assert len(loaded["frames"]) == len(data["frames"])
        # Verify a specific landmark value
        original_x = data["frames"][10]["landmarks"]["LEFT_ANKLE"]["x"]
        loaded_x = loaded["frames"][10]["landmarks"]["LEFT_ANKLE"]["x"]
        assert loaded_x == pytest.approx(original_x, abs=1e-6)

    def test_json_round_trip_after_full_pipeline(self, tmp_path):
        """Run full pipeline, save, load, verify events/angles/cycles survive."""
        from myogait.schema import save_json, load_json
        data, cycles, stats = _run_full_pipeline()

        path = tmp_path / "pipeline.json"
        save_json(data, path)
        loaded = load_json(path)

        # Angles survived
        assert loaded["angles"] is not None
        assert loaded["angles"]["method"] == data["angles"]["method"]
        assert len(loaded["angles"]["frames"]) == len(data["angles"]["frames"])

        # Events survived
        assert loaded["events"] is not None
        assert loaded["events"]["method"] == data["events"]["method"]
        assert len(loaded["events"]["left_hs"]) == len(data["events"]["left_hs"])
        assert len(loaded["events"]["right_hs"]) == len(data["events"]["right_hs"])

        # Normalization info survived
        assert loaded["normalization"] is not None
        assert loaded["normalization"]["steps_applied"] == data["normalization"]["steps_applied"]

    def test_json_round_trip_with_numpy_types(self, tmp_path):
        """Verify numpy scalar types serialize and deserialize correctly."""
        from myogait.schema import save_json, load_json
        data = _make_fake_data(5)
        data["meta"]["custom_float"] = np.float64(3.14)
        data["meta"]["custom_int"] = np.int64(42)
        data["meta"]["custom_bool"] = np.bool_(True)

        path = tmp_path / "numpy.json"
        save_json(data, path)
        loaded = load_json(path)

        assert loaded["meta"]["custom_float"] == pytest.approx(3.14)
        assert loaded["meta"]["custom_int"] == 42
        assert loaded["meta"]["custom_bool"] is True

    def test_json_round_trip_with_subject(self, tmp_path):
        """Verify subject metadata survives JSON round-trip."""
        from myogait import set_subject
        from myogait.schema import save_json, load_json
        data = _make_fake_data(5)
        set_subject(data, age=45, sex="M", height_m=1.75, weight_kg=80, pathology="DMD")

        path = tmp_path / "subject.json"
        save_json(data, path)
        loaded = load_json(path)

        assert loaded["subject"]["age"] == 45
        assert loaded["subject"]["sex"] == "M"
        assert loaded["subject"]["height_m"] == 1.75
        assert loaded["subject"]["pathology"] == "DMD"


# ── 3. Empty / edge cases ────────────────────────────────────────────


class TestEdgeCases:
    """Tests for empty, minimal, and degenerate inputs."""

    def test_normalize_empty_frames_raises(self):
        """Normalization on empty frames should raise ValueError."""
        from myogait import normalize
        from myogait.schema import create_empty
        data = create_empty("test.mp4")
        with pytest.raises(ValueError, match="No frames"):
            normalize(data)

    def test_compute_angles_empty_frames_raises(self):
        """Angle computation on empty frames should raise ValueError."""
        from myogait import compute_angles
        from myogait.schema import create_empty
        data = create_empty("test.mp4")
        with pytest.raises(ValueError, match="No frames"):
            compute_angles(data)

    def test_detect_events_empty_frames_raises(self):
        """Event detection on empty frames should raise ValueError."""
        from myogait import detect_events
        from myogait.schema import create_empty
        data = create_empty("test.mp4")
        with pytest.raises(ValueError, match="No frames"):
            detect_events(data)

    def test_single_frame_normalize(self):
        """A single frame should pass through normalization without error."""
        from myogait import normalize
        data = _make_fake_data(1)
        # Should not crash, though filtering a single point is degenerate
        normalize(data, filters=["moving_mean"])
        assert len(data["frames"]) == 1

    def test_single_frame_angles(self):
        """A single frame should produce one angle frame."""
        from myogait import compute_angles
        data = _make_standing_data(1)
        compute_angles(data, correction_factor=1.0, calibrate=False)
        assert len(data["angles"]["frames"]) == 1

    def test_very_short_video_10_frames(self):
        """10 frames of walking data should not crash but may find few events."""
        from myogait import normalize, compute_angles, detect_events
        data = _make_walking_data(10, fps=30.0)
        normalize(data, filters=["moving_mean"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        # With only 10 frames there may be 0 events, but no crash
        assert data["events"] is not None

    def test_all_nan_landmarks(self):
        """Frames with all NaN landmarks should produce NaN angles gracefully."""
        from myogait import compute_angles
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0, width=100, height=100, n_frames=5)
        data["extraction"] = {"model": "mediapipe"}
        frames = []
        for i in range(5):
            lm = {}
            for name in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                          "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                          "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                          "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]:
                lm[name] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
            frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.0})
        data["frames"] = frames
        compute_angles(data, correction_factor=1.0, calibrate=False)
        # All angles should be None (NaN converted)
        for af in data["angles"]["frames"]:
            assert af["hip_L"] is None
            assert af["knee_L"] is None

    def test_min_confidence_skips_low_frames(self):
        """min_confidence skips angle computation on low-confidence frames."""
        from myogait import compute_angles
        data = _make_standing_data(10)
        # Set frames 3, 7 to low confidence
        data["frames"][3]["confidence"] = 0.0
        data["frames"][7]["confidence"] = 0.05
        compute_angles(data, correction_factor=1.0, calibrate=False, min_confidence=0.1)
        afs = data["angles"]["frames"]
        assert len(afs) == 10  # all frames present
        # Skipped frames have None angles
        assert afs[3]["hip_L"] is None
        assert afs[7]["hip_L"] is None
        # Normal frames have computed angles
        assert afs[0]["hip_L"] is not None
        assert afs[5]["hip_L"] is not None

    def test_missing_landmark_keys(self):
        """Frames missing some landmark keys should still compute available angles."""
        from myogait import compute_angles
        from myogait.schema import create_empty
        data = create_empty("test.mp4", fps=30.0, width=100, height=100, n_frames=5)
        data["extraction"] = {"model": "mediapipe"}
        frames = []
        for i in range(5):
            # Only shoulders and hips, no knees/ankles
            lm = {
                "LEFT_SHOULDER":  {"x": 0.5, "y": 0.25, "visibility": 1.0},
                "RIGHT_SHOULDER": {"x": 0.5, "y": 0.25, "visibility": 1.0},
                "LEFT_HIP":       {"x": 0.5, "y": 0.50, "visibility": 1.0},
                "RIGHT_HIP":      {"x": 0.5, "y": 0.50, "visibility": 1.0},
            }
            frames.append({"frame_idx": i, "time_s": i / 30.0, "landmarks": lm, "confidence": 0.5})
        data["frames"] = frames
        compute_angles(data, correction_factor=1.0, calibrate=False)
        # Trunk angle should be computed; hip/knee/ankle should be None
        af = data["angles"]["frames"][2]
        assert af["trunk_angle"] is not None
        assert af["hip_L"] is None
        assert af["knee_L"] is None
        assert af["ankle_L"] is None


# ── 4. I/O error tests ───────────────────────────────────────────────


class TestIOErrors:
    """Tests for expected I/O error conditions."""

    def test_load_json_nonexistent_file(self):
        """load_json with nonexistent file should raise FileNotFoundError."""
        from myogait.schema import load_json
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path/does_not_exist.json")

    def test_load_json_invalid_json(self, tmp_path):
        """load_json with invalid JSON should raise json.JSONDecodeError."""
        from myogait.schema import load_json
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("this is { not valid json !!!")
        with pytest.raises(json.JSONDecodeError):
            load_json(bad_path)

    def test_load_json_missing_meta_key(self, tmp_path):
        """load_json with valid JSON but missing 'meta' key should raise ValueError."""
        from myogait.schema import load_json
        path = tmp_path / "no_meta.json"
        path.write_text(json.dumps({"frames": []}))
        with pytest.raises(ValueError, match="meta"):
            load_json(path)

    def test_load_json_missing_frames_key(self, tmp_path):
        """load_json with valid JSON but missing 'frames' key should raise ValueError."""
        from myogait.schema import load_json
        path = tmp_path / "no_frames.json"
        path.write_text(json.dumps({"meta": {"fps": 30}}))
        with pytest.raises(ValueError, match="frames"):
            load_json(path)

    def test_load_config_nonexistent_file(self):
        """load_config with nonexistent file should raise FileNotFoundError."""
        from myogait import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")

    def test_export_mot_no_angles_raises(self, tmp_path):
        """export_mot with no angles should raise ValueError."""
        from myogait.export import export_mot
        data = _make_fake_data(10)
        # data has no angles computed
        mot_path = str(tmp_path / "test.mot")
        with pytest.raises(ValueError, match="No angles"):
            export_mot(data, mot_path)

    def test_export_trc_no_frames_raises(self, tmp_path):
        """export_trc with no frames should raise ValueError."""
        from myogait.export import export_trc
        from myogait.schema import create_empty
        data = create_empty("test.mp4")
        trc_path = str(tmp_path / "test.trc")
        with pytest.raises(ValueError, match="No frames"):
            export_trc(data, trc_path)


# ── 5. Config round-trip ─────────────────────────────────────────────


class TestConfigRoundTrip:
    """Tests for configuration save/load."""

    def test_config_round_trip(self, tmp_path):
        """save_config then load_config should preserve all settings."""
        from myogait import save_config, load_config, DEFAULT_CONFIG
        path = tmp_path / "config.json"
        save_config(DEFAULT_CONFIG, str(path))
        loaded = load_config(str(path))

        assert loaded["extract"]["model"] == DEFAULT_CONFIG["extract"]["model"]
        assert loaded["events"]["method"] == DEFAULT_CONFIG["events"]["method"]
        assert loaded["angles"]["method"] == DEFAULT_CONFIG["angles"]["method"]
        assert loaded["normalize"]["butterworth_cutoff"] == DEFAULT_CONFIG["normalize"]["butterworth_cutoff"]
        assert loaded["cycles"]["n_points"] == DEFAULT_CONFIG["cycles"]["n_points"]

    def test_config_round_trip_with_overrides(self, tmp_path):
        """Custom config values should survive save/load."""
        from myogait import save_config, load_config
        custom = {
            "extract": {"model": "yolo"},
            "events": {"method": "velocity", "min_cycle_duration": 0.5},
            "angles": {"correction_factor": 0.9},
        }
        path = tmp_path / "custom_config.json"
        save_config(custom, str(path))
        loaded = load_config(str(path))

        assert loaded["extract"]["model"] == "yolo"
        assert loaded["events"]["method"] == "velocity"
        assert loaded["events"]["min_cycle_duration"] == 0.5
        assert loaded["angles"]["correction_factor"] == 0.9
        # Defaults should be merged in for keys not in custom
        assert "butterworth_cutoff" in loaded["normalize"]


# ── 6. Export integration ─────────────────────────────────────────────


class TestExportIntegration:
    """Full pipeline then export to CSV, MOT, TRC."""

    def test_export_csv_after_pipeline(self, tmp_path):
        """Full pipeline then export_csv produces valid files."""
        from myogait.export import export_csv
        data, cycles, stats = _run_full_pipeline()
        files = export_csv(data, str(tmp_path), cycles, stats)
        assert len(files) >= 3
        for f in files:
            p = Path(f)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_export_mot_after_pipeline(self, tmp_path):
        """Full pipeline then export_mot produces a valid .mot file."""
        from myogait.export import export_mot
        data, _, _ = _run_full_pipeline()
        mot_path = str(tmp_path / "gait.mot")
        export_mot(data, mot_path)
        p = Path(mot_path)
        assert p.exists()
        content = p.read_text()
        assert "endheader" in content
        assert "hip_flexion_l" in content
        lines = content.strip().split("\n")
        # Header (6 lines) + column header (1) + data rows
        assert len(lines) > 10

    def test_export_trc_after_pipeline(self, tmp_path):
        """Full pipeline then export_trc produces a valid .trc file."""
        from myogait.export import export_trc
        data, _, _ = _run_full_pipeline()
        trc_path = str(tmp_path / "markers.trc")
        export_trc(data, trc_path)
        p = Path(trc_path)
        assert p.exists()
        content = p.read_text()
        assert "LEFT_HIP" in content
        assert "RIGHT_KNEE" in content
        assert p.stat().st_size > 100

    def test_export_excel_after_pipeline(self, tmp_path):
        """Full pipeline then export_excel produces a valid .xlsx file."""
        pytest.importorskip("openpyxl")
        from myogait.export import export_excel
        data, cycles, stats = _run_full_pipeline()
        xlsx_path = str(tmp_path / "report.xlsx")
        export_excel(data, xlsx_path, cycles, stats)
        p = Path(xlsx_path)
        assert p.exists()
        assert p.stat().st_size > 1000


# ── 7. Validation integration ─────────────────────────────────────────


class TestValidationIntegration:
    """Full pipeline then biomechanical validation."""

    def test_validate_after_pipeline(self):
        """Validation report should have correct structure after full pipeline."""
        from myogait import validate_biomechanical
        data, cycles, _ = _run_full_pipeline()
        report = validate_biomechanical(data, cycles)

        assert "valid" in report
        assert isinstance(report["valid"], bool)
        assert "violations" in report
        assert isinstance(report["violations"], list)
        assert "summary" in report
        assert "total" in report["summary"]
        assert "critical" in report["summary"]
        assert "warning" in report["summary"]
        assert "info" in report["summary"]

    def test_validate_without_cycles(self):
        """Validation should work without cycles (angles only)."""
        from myogait import validate_biomechanical
        data = _walking_data_with_angles()
        report = validate_biomechanical(data)
        assert "valid" in report
        assert "violations" in report

    def test_validate_violation_structure(self):
        """Each violation should have required fields."""
        from myogait import validate_biomechanical
        data, cycles, _ = _run_full_pipeline()
        report = validate_biomechanical(data, cycles)
        for v in report["violations"]:
            assert "type" in v
            assert "severity" in v
            assert v["severity"] in ("critical", "warning", "info")


# ── 8. Report integration ─────────────────────────────────────────────


class TestReportIntegration:
    """Full pipeline then PDF report generation."""

    def test_generate_report_after_pipeline(self, tmp_path):
        """Full pipeline then generate_report produces a valid PDF."""
        from myogait import generate_report
        import matplotlib.pyplot as plt
        data, cycles, stats = _run_full_pipeline()
        pdf_path = tmp_path / "report.pdf"
        generate_report(data, cycles, stats, str(pdf_path))
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000
        plt.close("all")

    def test_report_path_returned(self, tmp_path):
        """generate_report should return the output path."""
        from myogait import generate_report
        import matplotlib.pyplot as plt
        data, cycles, stats = _run_full_pipeline()
        pdf_path = tmp_path / "report2.pdf"
        result = generate_report(data, cycles, stats, str(pdf_path))
        assert result == str(pdf_path)
        plt.close("all")


# ── 9. Batch-style processing / determinism ──────────────────────────


class TestDeterminism:
    """Verify that processing the same data twice yields identical results."""

    def test_deterministic_angles(self):
        """Angle computation should produce identical output on same input."""
        from myogait import compute_angles
        data1 = _make_walking_data(60)
        data2 = copy.deepcopy(data1)
        compute_angles(data1, correction_factor=1.0, calibrate=False)
        compute_angles(data2, correction_factor=1.0, calibrate=False)

        for i in range(len(data1["angles"]["frames"])):
            af1 = data1["angles"]["frames"][i]
            af2 = data2["angles"]["frames"][i]
            for key in ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]:
                if af1[key] is None:
                    assert af2[key] is None
                else:
                    assert af1[key] == pytest.approx(af2[key], abs=1e-10)

    def test_deterministic_events(self):
        """Event detection should produce identical output on same input."""
        from myogait import detect_events
        data1 = _walking_data_with_angles()
        data2 = copy.deepcopy(data1)
        detect_events(data1)
        detect_events(data2)

        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            assert len(data1["events"][key]) == len(data2["events"][key])
            for ev1, ev2 in zip(data1["events"][key], data2["events"][key]):
                assert ev1["frame"] == ev2["frame"]
                assert ev1["time"] == ev2["time"]

    def test_deterministic_full_pipeline(self):
        """Full pipeline run twice should produce identical statistics."""
        _, _, stats1 = _run_full_pipeline()
        _, _, stats2 = _run_full_pipeline()

        assert stats1["spatiotemporal"]["cadence_steps_per_min"] == stats2["spatiotemporal"]["cadence_steps_per_min"]
        assert stats1["spatiotemporal"]["n_cycles_total"] == stats2["spatiotemporal"]["n_cycles_total"]
        assert stats1["symmetry"] == stats2["symmetry"]


# ── 10. Cross-method consistency (events) ────────────────────────────


class TestCrossMethodEvents:
    """Verify all 4 event detection methods produce valid output format."""

    @pytest.mark.parametrize("method", ["zeni", "crossing", "velocity", "oconnor"])
    def test_event_method_valid_format(self, method):
        """Each event detection method should return events with correct structure."""
        from myogait import detect_events
        data = _walking_data_with_angles()
        detect_events(data, method=method)
        events = data["events"]

        assert events["method"] == method
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            assert key in events
            assert isinstance(events[key], list)
            for ev in events[key]:
                assert "frame" in ev
                assert "time" in ev
                assert "confidence" in ev
                assert isinstance(ev["frame"], int)
                assert ev["frame"] >= 0
                assert isinstance(ev["time"], float)
                assert 0 <= ev["confidence"] <= 1

    @pytest.mark.parametrize("method", ["zeni", "crossing", "oconnor"])
    def test_event_method_detects_some_events(self, method):
        """AP-based event methods should detect at least some events on walking data."""
        from myogait import detect_events
        data = _walking_data_with_angles()
        detect_events(data, method=method)
        events = data["events"]
        total = sum(len(events[k]) for k in ["left_hs", "right_hs", "left_to", "right_to"])
        assert total >= 1, f"Method {method} detected 0 events on walking data"

    def test_velocity_method_runs_without_error(self):
        """Velocity method should run without error even if synthetic data lacks y-oscillation."""
        from myogait import detect_events
        data = _walking_data_with_angles()
        detect_events(data, method="velocity")
        events = data["events"]
        assert events["method"] == "velocity"
        # Velocity method uses y-coordinate changes; synthetic data has constant y,
        # so 0 events is acceptable here.
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            assert isinstance(events[key], list)

    def test_unknown_event_method_raises(self):
        """Using an unknown event method should raise ValueError."""
        from myogait import detect_events
        data = _walking_data_with_angles()
        with pytest.raises(ValueError, match="Unknown method"):
            detect_events(data, method="nonexistent_method")


# ── 11. Angle method consistency ──────────────────────────────────────


class TestCrossMethodAngles:
    """Verify both angle computation methods produce valid output."""

    @pytest.mark.parametrize("method", ["sagittal_vertical_axis", "sagittal_classic"])
    def test_angle_method_valid_output(self, method):
        """Each angle method should produce angles with correct keys."""
        from myogait import compute_angles
        data = _make_standing_data(30)
        compute_angles(data, method=method, correction_factor=1.0, calibrate=False)
        angles = data["angles"]

        assert angles["method"] == method
        assert len(angles["frames"]) == 30
        for af in angles["frames"]:
            assert "frame_idx" in af
            assert "hip_L" in af
            assert "hip_R" in af
            assert "knee_L" in af
            assert "knee_R" in af
            assert "ankle_L" in af
            assert "ankle_R" in af
            assert "trunk_angle" in af
            assert "pelvis_tilt" in af
            assert "landmark_positions" in af

    @pytest.mark.parametrize("method", ["sagittal_vertical_axis", "sagittal_classic"])
    def test_angle_method_standing_near_zero(self, method):
        """In standing pose, joint angles should be small (near zero)."""
        from myogait import compute_angles
        data = _make_standing_data(30)
        compute_angles(data, method=method, correction_factor=1.0, calibrate=False)
        mid = data["angles"]["frames"][15]
        # Standing: hip and knee should be relatively small
        if mid["hip_L"] is not None:
            assert abs(mid["hip_L"]) < 30
        if mid["knee_L"] is not None:
            assert abs(mid["knee_L"]) < 30

    def test_unknown_angle_method_raises(self):
        """Using an unknown angle method should raise ValueError."""
        from myogait import compute_angles
        data = _make_standing_data(10)
        with pytest.raises(ValueError, match="Unknown angle method"):
            compute_angles(data, method="nonexistent_method")


# ── 12. Additional integration tests ─────────────────────────────────


class TestAdditionalIntegration:
    """Additional integration tests for completeness."""

    def test_pipeline_with_subject_metadata(self, tmp_path):
        """Full pipeline with subject metadata attached."""
        from myogait import set_subject
        from myogait.schema import save_json, load_json
        data, cycles, stats = _run_full_pipeline()
        set_subject(data, age=25, sex="F", height_m=1.65, weight_kg=60, pathology=None)

        path = tmp_path / "with_subject.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["subject"]["age"] == 25
        assert loaded["subject"]["sex"] == "F"
        assert loaded["subject"]["height_m"] == 1.65

    def test_segment_cycles_requires_events(self):
        """segment_cycles without events should raise ValueError."""
        from myogait import segment_cycles
        data = _walking_data_with_angles()
        # No events detected yet
        with pytest.raises(ValueError, match="No events"):
            segment_cycles(data)

    def test_segment_cycles_requires_angles(self):
        """segment_cycles without angles should raise ValueError."""
        from myogait import detect_events, segment_cycles
        data = _make_walking_data(300)
        # We need frames with landmark data to detect events but no angles
        # First detect events (they use raw frames, not angles)
        detect_events(data)
        # Remove angles key explicitly
        data["angles"] = None
        with pytest.raises(ValueError, match="No angles"):
            segment_cycles(data)

    def test_pipeline_different_fps(self):
        """Pipeline should work with non-standard FPS values."""
        from myogait import normalize, compute_angles, detect_events
        data = _make_walking_data(150, fps=15.0)
        normalize(data, filters=["butterworth"])
        compute_angles(data, correction_factor=1.0, calibrate=False)
        detect_events(data)
        assert data["events"] is not None
        assert data["meta"]["fps"] == 15.0

    def test_normalize_then_angles_consistency(self):
        """Normalized data should still produce valid angle values."""
        from myogait import normalize, compute_angles
        data = _make_walking_data(60)
        normalize(data, filters=["butterworth"], center=True)
        compute_angles(data, correction_factor=1.0, calibrate=False)
        # At least some angles should be non-None
        non_none_count = sum(
            1 for af in data["angles"]["frames"]
            if af["knee_L"] is not None
        )
        assert non_none_count > 0

    def test_export_csv_creates_angles_file(self, tmp_path):
        """export_csv should create an angles.csv with correct columns."""
        import pandas as pd
        from myogait.export import export_csv
        data, cycles, stats = _run_full_pipeline()
        files = export_csv(data, str(tmp_path), cycles, stats)

        angles_files = [f for f in files if "angles.csv" in f]
        assert len(angles_files) == 1

        df = pd.read_csv(angles_files[0])
        assert "frame_idx" in df.columns
        assert "hip_L" in df.columns
        assert "knee_L" in df.columns
        assert "ankle_L" in df.columns
        assert len(df) == 300
