"""Tests for extended foot landmark extraction and usage.

Covers:
- _enrich_foot_landmarks with mock goliath308 data
- _enrich_foot_landmarks with mock wholebody133 data
- Foot landmarks used instead of geometric estimates when available
- Backward compatibility: no auxiliary data -> geometric estimation works
- FOOT_INDEX computed as midpoint of toes
- TRC export includes extended landmarks when available
- Angles use real toe data when available
- foot_progression_angle uses real toe data
"""


import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_walking_data

from myogait.extract import _enrich_foot_landmarks, _estimate_missing_foot_landmarks, _is_landmark_nan
from myogait.angles import (
    _estimate_foot_landmarks,
    _get_foot_index_from_toes,
    compute_angles,
    foot_progression_angle,
)
from myogait.constants import (
    EXTENDED_FOOT_LANDMARKS,
    GOLIATH_FOOT_INDICES,
    RTMW_FOOT_INDICES,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_base_frame():
    """Create a single frame with standard MediaPipe landmarks."""
    return {
        "frame_idx": 0,
        "time_s": 0.0,
        "landmarks": {
            "NOSE": {"x": 0.50, "y": 0.10, "visibility": 1.0},
            "LEFT_EYE": {"x": 0.49, "y": 0.08, "visibility": 1.0},
            "RIGHT_EYE": {"x": 0.51, "y": 0.08, "visibility": 1.0},
            "LEFT_EAR": {"x": 0.48, "y": 0.10, "visibility": 1.0},
            "RIGHT_EAR": {"x": 0.52, "y": 0.10, "visibility": 1.0},
            "LEFT_SHOULDER": {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "RIGHT_SHOULDER": {"x": 0.50, "y": 0.25, "visibility": 1.0},
            "LEFT_ELBOW": {"x": 0.50, "y": 0.37, "visibility": 1.0},
            "RIGHT_ELBOW": {"x": 0.50, "y": 0.37, "visibility": 1.0},
            "LEFT_WRIST": {"x": 0.50, "y": 0.48, "visibility": 1.0},
            "RIGHT_WRIST": {"x": 0.50, "y": 0.48, "visibility": 1.0},
            "LEFT_HIP": {"x": 0.50, "y": 0.50, "visibility": 1.0},
            "RIGHT_HIP": {"x": 0.50, "y": 0.50, "visibility": 1.0},
            "LEFT_KNEE": {"x": 0.50, "y": 0.65, "visibility": 1.0},
            "RIGHT_KNEE": {"x": 0.50, "y": 0.65, "visibility": 1.0},
            "LEFT_ANKLE": {"x": 0.50, "y": 0.80, "visibility": 1.0},
            "RIGHT_ANKLE": {"x": 0.50, "y": 0.80, "visibility": 1.0},
        },
        "confidence": 0.95,
    }


def _add_mock_goliath308(data):
    """Add mock Goliath308 auxiliary data with foot landmarks."""
    for frame in data["frames"]:
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        # Foot landmarks (indices 15-20)
        goliath[15] = [0.45, 0.92, 0.9]  # left_big_toe
        goliath[16] = [0.43, 0.92, 0.9]  # left_small_toe
        goliath[17] = [0.47, 0.90, 0.9]  # left_heel
        goliath[18] = [0.55, 0.92, 0.9]  # right_big_toe
        goliath[19] = [0.57, 0.92, 0.9]  # right_small_toe
        goliath[20] = [0.53, 0.90, 0.9]  # right_heel
        frame["goliath308"] = goliath
    data["extraction"] = data.get("extraction", {})
    data["extraction"]["auxiliary_format"] = "goliath308"


def _add_mock_wholebody133(data):
    """Add mock WholeBody133 auxiliary data with foot landmarks."""
    for frame in data["frames"]:
        wb = [[float("nan"), float("nan"), 0.0]] * 133
        # Foot landmarks (indices 17-22)
        wb[17] = [0.45, 0.92, 0.85]  # left_big_toe
        wb[18] = [0.43, 0.92, 0.85]  # left_small_toe
        wb[19] = [0.47, 0.90, 0.85]  # left_heel
        wb[20] = [0.55, 0.92, 0.85]  # right_big_toe
        wb[21] = [0.57, 0.92, 0.85]  # right_small_toe
        wb[22] = [0.53, 0.90, 0.85]  # right_heel
        frame["wholebody133"] = wb
    data["extraction"] = data.get("extraction", {})
    data["extraction"]["auxiliary_format"] = "wholebody133"


# ── Constants tests ────────────────────────────────────────────────────


class TestFootLandmarkConstants:

    def test_extended_foot_landmarks_list(self):
        """EXTENDED_FOOT_LANDMARKS should contain 4 toe landmarks."""
        assert len(EXTENDED_FOOT_LANDMARKS) == 4
        assert "LEFT_BIG_TOE" in EXTENDED_FOOT_LANDMARKS
        assert "RIGHT_SMALL_TOE" in EXTENDED_FOOT_LANDMARKS

    def test_goliath_foot_indices_mapping(self):
        """GOLIATH_FOOT_INDICES should map indices 15-20."""
        assert GOLIATH_FOOT_INDICES[15] == "LEFT_BIG_TOE"
        assert GOLIATH_FOOT_INDICES[17] == "LEFT_HEEL"
        assert GOLIATH_FOOT_INDICES[20] == "RIGHT_HEEL"

    def test_rtmw_foot_indices_mapping(self):
        """RTMW_FOOT_INDICES should map indices 17-22."""
        assert RTMW_FOOT_INDICES[17] == "LEFT_BIG_TOE"
        assert RTMW_FOOT_INDICES[19] == "LEFT_HEEL"
        assert RTMW_FOOT_INDICES[22] == "RIGHT_HEEL"


# ── _enrich_foot_landmarks tests ──────────────────────────────────────


class TestEnrichFootLandmarksGoliath:

    def test_goliath308_injects_foot_landmarks(self):
        """With goliath308 data, foot landmarks should be injected."""
        frame = _make_base_frame()
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        goliath[15] = [0.45, 0.92, 0.9]  # left_big_toe
        goliath[16] = [0.43, 0.92, 0.9]  # left_small_toe
        goliath[17] = [0.47, 0.90, 0.9]  # left_heel
        goliath[18] = [0.55, 0.92, 0.9]  # right_big_toe
        goliath[19] = [0.57, 0.92, 0.9]  # right_small_toe
        goliath[20] = [0.53, 0.90, 0.9]  # right_heel
        frame["goliath308"] = goliath

        _enrich_foot_landmarks(frame)

        lm = frame["landmarks"]
        assert "LEFT_BIG_TOE" in lm
        assert "LEFT_SMALL_TOE" in lm
        assert "LEFT_HEEL" in lm
        assert "RIGHT_BIG_TOE" in lm
        assert "RIGHT_SMALL_TOE" in lm
        assert "RIGHT_HEEL" in lm

    def test_goliath308_correct_coordinates(self):
        """Injected foot landmarks should have correct x/y values."""
        frame = _make_base_frame()
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        goliath[15] = [0.45, 0.92, 0.9]
        goliath[16] = [0.43, 0.92, 0.9]
        goliath[17] = [0.47, 0.90, 0.9]
        goliath[18] = [0.55, 0.92, 0.9]
        goliath[19] = [0.57, 0.92, 0.9]
        goliath[20] = [0.53, 0.90, 0.9]
        frame["goliath308"] = goliath

        _enrich_foot_landmarks(frame)

        assert frame["landmarks"]["LEFT_BIG_TOE"]["x"] == pytest.approx(0.45)
        assert frame["landmarks"]["LEFT_BIG_TOE"]["y"] == pytest.approx(0.92)
        assert frame["landmarks"]["LEFT_BIG_TOE"]["visibility"] == pytest.approx(0.9)
        assert frame["landmarks"]["RIGHT_HEEL"]["x"] == pytest.approx(0.53)

    def test_goliath308_sets_detected_flag(self):
        """foot_landmarks_source should be set to 'detected'."""
        frame = _make_base_frame()
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        goliath[15] = [0.45, 0.92, 0.9]
        goliath[16] = [0.43, 0.92, 0.9]
        goliath[17] = [0.47, 0.90, 0.9]
        goliath[18] = [0.55, 0.92, 0.9]
        goliath[19] = [0.57, 0.92, 0.9]
        goliath[20] = [0.53, 0.90, 0.9]
        frame["goliath308"] = goliath

        _enrich_foot_landmarks(frame)

        assert frame.get("foot_landmarks_source") == "detected"


class TestEnrichFootLandmarksWholebody:

    def test_wholebody133_injects_foot_landmarks(self):
        """With wholebody133 data, foot landmarks should be injected."""
        frame = _make_base_frame()
        wb = [[float("nan"), float("nan"), 0.0]] * 133
        wb[17] = [0.45, 0.92, 0.85]
        wb[18] = [0.43, 0.92, 0.85]
        wb[19] = [0.47, 0.90, 0.85]
        wb[20] = [0.55, 0.92, 0.85]
        wb[21] = [0.57, 0.92, 0.85]
        wb[22] = [0.53, 0.90, 0.85]
        frame["wholebody133"] = wb

        _enrich_foot_landmarks(frame)

        lm = frame["landmarks"]
        assert "LEFT_BIG_TOE" in lm
        assert "RIGHT_SMALL_TOE" in lm
        assert frame.get("foot_landmarks_source") == "detected"

    def test_wholebody133_correct_values(self):
        """WholeBody133 foot landmarks should have correct values."""
        frame = _make_base_frame()
        wb = [[float("nan"), float("nan"), 0.0]] * 133
        wb[17] = [0.45, 0.92, 0.85]
        wb[18] = [0.43, 0.92, 0.85]
        wb[19] = [0.47, 0.90, 0.85]
        wb[20] = [0.55, 0.92, 0.85]
        wb[21] = [0.57, 0.92, 0.85]
        wb[22] = [0.53, 0.90, 0.85]
        frame["wholebody133"] = wb

        _enrich_foot_landmarks(frame)

        assert frame["landmarks"]["RIGHT_BIG_TOE"]["x"] == pytest.approx(0.55)
        assert frame["landmarks"]["RIGHT_BIG_TOE"]["visibility"] == pytest.approx(0.85)


class TestEnrichFootLandmarksFootIndex:

    def test_foot_index_computed_as_midpoint(self):
        """FOOT_INDEX should be the midpoint of big_toe and small_toe."""
        frame = _make_base_frame()
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        goliath[15] = [0.45, 0.92, 0.9]  # left_big_toe
        goliath[16] = [0.43, 0.92, 0.9]  # left_small_toe
        goliath[17] = [0.47, 0.90, 0.9]  # left_heel
        goliath[18] = [0.55, 0.92, 0.9]  # right_big_toe
        goliath[19] = [0.57, 0.92, 0.9]  # right_small_toe
        goliath[20] = [0.53, 0.90, 0.9]  # right_heel
        frame["goliath308"] = goliath

        _enrich_foot_landmarks(frame)

        lm = frame["landmarks"]
        # LEFT_FOOT_INDEX = midpoint(0.45, 0.43) = 0.44
        assert lm["LEFT_FOOT_INDEX"]["x"] == pytest.approx(0.44, abs=1e-6)
        assert lm["LEFT_FOOT_INDEX"]["y"] == pytest.approx(0.92, abs=1e-6)
        # RIGHT_FOOT_INDEX = midpoint(0.55, 0.57) = 0.56
        assert lm["RIGHT_FOOT_INDEX"]["x"] == pytest.approx(0.56, abs=1e-6)
        assert lm["RIGHT_FOOT_INDEX"]["y"] == pytest.approx(0.92, abs=1e-6)


class TestEnrichFootLandmarksNoAuxiliary:

    def test_no_auxiliary_data_no_change(self):
        """Without auxiliary data, frame should not be modified."""
        frame = _make_base_frame()
        original_keys = set(frame["landmarks"].keys())

        _enrich_foot_landmarks(frame)

        # No new landmarks should be added
        assert set(frame["landmarks"].keys()) == original_keys
        assert "foot_landmarks_source" not in frame

    def test_low_confidence_aux_skipped(self):
        """Auxiliary landmarks with very low confidence should be skipped."""
        frame = _make_base_frame()
        goliath = [[float("nan"), float("nan"), 0.0]] * 308
        # All foot landmarks have confidence below threshold (0.1)
        goliath[15] = [0.45, 0.92, 0.05]
        goliath[16] = [0.43, 0.92, 0.05]
        goliath[17] = [0.47, 0.90, 0.05]
        goliath[18] = [0.55, 0.92, 0.05]
        goliath[19] = [0.57, 0.92, 0.05]
        goliath[20] = [0.53, 0.90, 0.05]
        frame["goliath308"] = goliath

        _enrich_foot_landmarks(frame)

        # No foot landmarks should be injected
        assert "LEFT_BIG_TOE" not in frame["landmarks"]
        assert "foot_landmarks_source" not in frame


# ── _estimate_foot_landmarks (backward compat) ──────────────────────


class TestEstimateFootLandmarksBackwardCompat:

    def test_estimation_still_works_without_aux(self):
        """Without auxiliary data, geometric estimation should still work."""
        frame = _make_base_frame()
        result = _estimate_foot_landmarks(frame)
        lm = result["landmarks"]
        # Should have estimated HEEL and FOOT_INDEX
        assert "LEFT_HEEL" in lm
        assert "RIGHT_HEEL" in lm
        assert "LEFT_FOOT_INDEX" in lm
        assert "RIGHT_FOOT_INDEX" in lm
        # Should be at visibility 0.5 (estimated)
        assert lm["LEFT_HEEL"]["visibility"] == pytest.approx(0.5)

    def test_estimation_skipped_when_detected(self):
        """When real foot landmarks are present, estimation should be skipped."""
        frame = _make_base_frame()
        # Inject real foot landmarks with high visibility
        frame["landmarks"]["LEFT_HEEL"] = {"x": 0.47, "y": 0.90, "visibility": 0.9}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": 0.44, "y": 0.92, "visibility": 0.9}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": 0.53, "y": 0.90, "visibility": 0.9}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": 0.56, "y": 0.92, "visibility": 0.9}

        result = _estimate_foot_landmarks(frame)
        lm = result["landmarks"]

        # Should keep the detected values, not overwrite with estimates
        assert lm["LEFT_HEEL"]["x"] == pytest.approx(0.47)
        assert lm["LEFT_HEEL"]["visibility"] == pytest.approx(0.9)
        assert lm["LEFT_FOOT_INDEX"]["x"] == pytest.approx(0.44)


# ── _get_foot_index_from_toes ──────────────────────────────────────


class TestGetFootIndexFromToes:

    def test_returns_midpoint_when_toes_available(self):
        """Should return midpoint of big_toe and small_toe."""
        frame = {
            "landmarks": {
                "LEFT_BIG_TOE": {"x": 0.45, "y": 0.92, "visibility": 0.9},
                "LEFT_SMALL_TOE": {"x": 0.43, "y": 0.92, "visibility": 0.9},
                "LEFT_FOOT_INDEX": {"x": 0.40, "y": 0.80, "visibility": 0.5},
            }
        }
        result = _get_foot_index_from_toes(frame, "LEFT")
        assert result is not None
        assert result[0] == pytest.approx(0.44, abs=1e-6)
        assert result[1] == pytest.approx(0.92, abs=1e-6)

    def test_falls_back_to_foot_index(self):
        """Without toe data, should fall back to FOOT_INDEX."""
        frame = {
            "landmarks": {
                "LEFT_FOOT_INDEX": {"x": 0.40, "y": 0.80, "visibility": 0.5},
            }
        }
        result = _get_foot_index_from_toes(frame, "LEFT")
        assert result is not None
        assert result[0] == pytest.approx(0.40)

    def test_returns_none_when_nothing(self):
        """Without any toe or foot index, should return None."""
        frame = {"landmarks": {}}
        result = _get_foot_index_from_toes(frame, "LEFT")
        assert result is None


# ── Angles with real toe data ─────────────────────────────────────────


class TestAnglesWithRealToeData:

    def test_ankle_angle_uses_toe_midpoint(self):
        """Ankle angle should use toe midpoint when big/small toes available."""
        data = make_walking_data(n_frames=60)
        _add_mock_goliath308(data)
        # Enrich foot landmarks
        for frame in data["frames"]:
            _enrich_foot_landmarks(frame)

        # Verify foot landmarks were injected
        assert "LEFT_BIG_TOE" in data["frames"][0]["landmarks"]

        # Compute angles -- should use real toe data
        compute_angles(data, correction_factor=1.0, calibrate=False)

        # Ankle angles should be numeric (not None/NaN)
        af = data["angles"]["frames"][10]
        assert af["ankle_L"] is not None
        assert af["ankle_R"] is not None

    def test_foot_progression_uses_toe_midpoint(self):
        """foot_progression_angle should use toe midpoint when available."""
        data = make_walking_data(n_frames=30)
        _add_mock_goliath308(data)
        for frame in data["frames"]:
            _enrich_foot_landmarks(frame)

        result = foot_progression_angle(data)
        # Should produce numeric angles (not None)
        assert result["foot_angle_L"][0] is not None
        assert result["foot_angle_R"][0] is not None


# ── TRC export ────────────────────────────────────────────────────────


class TestTrcExportExtendedFoot:

    def test_trc_includes_extended_landmarks(self, tmp_path):
        """TRC export should include big_toe/small_toe when available."""
        from myogait.export import export_trc

        data = make_walking_data(n_frames=10)
        _add_mock_goliath308(data)
        for frame in data["frames"]:
            _enrich_foot_landmarks(frame)

        trc_path = str(tmp_path / "foot_ext.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        # Marker header line should include extended foot names
        assert "LEFT_BIG_TOE" in content
        assert "RIGHT_SMALL_TOE" in content

    def test_trc_no_extended_without_aux(self, tmp_path):
        """TRC export should NOT include big_toe/small_toe without aux data."""
        from myogait.export import export_trc

        data = make_walking_data(n_frames=10)
        trc_path = str(tmp_path / "no_ext.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        assert "LEFT_BIG_TOE" not in content
        assert "RIGHT_BIG_TOE" not in content


# ── Integration: full pipeline backward compat ─────────────────────


class TestFullPipelineBackwardCompat:

    def test_pipeline_without_aux_data(self):
        """Full pipeline without auxiliary data should still work (no regression)."""
        from myogait import compute_angles

        data = make_walking_data(n_frames=100)
        compute_angles(data, correction_factor=1.0, calibrate=False)

        # All angle frames should have valid angles
        for af in data["angles"]["frames"]:
            assert "ankle_L" in af
            assert "ankle_R" in af
            # At least some should be non-None
        non_none = sum(1 for af in data["angles"]["frames"] if af["ankle_L"] is not None)
        assert non_none > 50

    def test_pipeline_with_goliath_aux(self):
        """Pipeline with Goliath auxiliary data should produce valid angles."""
        from myogait import compute_angles

        data = make_walking_data(n_frames=100)
        _add_mock_goliath308(data)
        for frame in data["frames"]:
            _enrich_foot_landmarks(frame)

        compute_angles(data, correction_factor=1.0, calibrate=False)

        non_none = sum(1 for af in data["angles"]["frames"] if af["ankle_L"] is not None)
        assert non_none > 50


# ── _is_landmark_nan ─────────────────────────────────────────────────


class TestIsLandmarkNan:

    def test_none_entry(self):
        assert _is_landmark_nan(None) is True

    def test_missing_x(self):
        assert _is_landmark_nan({"y": 0.5, "visibility": 0.9}) is True

    def test_nan_x(self):
        assert _is_landmark_nan({"x": float("nan"), "y": 0.5, "visibility": 0.9}) is True

    def test_zero_visibility(self):
        assert _is_landmark_nan({"x": 0.5, "y": 0.5, "visibility": 0.0}) is True

    def test_valid_landmark(self):
        assert _is_landmark_nan({"x": 0.5, "y": 0.5, "visibility": 0.9}) is False

    def test_low_but_valid_visibility(self):
        assert _is_landmark_nan({"x": 0.5, "y": 0.5, "visibility": 0.5}) is False


# ── _estimate_missing_foot_landmarks ──────────────────────────────────


class TestEstimateMissingFootLandmarks:

    def test_estimates_heel_and_foot_index_for_coco(self):
        """COCO-17 frame (no aux) should get estimated HEEL and FOOT_INDEX."""
        frame = _make_base_frame()
        # Simulate COCO-17: HEEL/FOOT_INDEX present but NaN
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        lm = frame["landmarks"]
        assert not _is_landmark_nan(lm.get("LEFT_HEEL"))
        assert not _is_landmark_nan(lm.get("RIGHT_HEEL"))
        assert not _is_landmark_nan(lm.get("LEFT_FOOT_INDEX"))
        assert not _is_landmark_nan(lm.get("RIGHT_FOOT_INDEX"))

    def test_estimated_visibility_is_half(self):
        """Estimated landmarks should have visibility=0.5."""
        frame = _make_base_frame()
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        lm = frame["landmarks"]
        assert lm["LEFT_HEEL"]["visibility"] == pytest.approx(0.5)
        assert lm["RIGHT_FOOT_INDEX"]["visibility"] == pytest.approx(0.5)

    def test_sets_estimated_source_flag(self):
        """foot_landmarks_source should be 'estimated'."""
        frame = _make_base_frame()
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)
        assert frame.get("foot_landmarks_source") == "estimated"

    def test_skips_when_detected_flag_set(self):
        """Should not overwrite detected foot landmarks."""
        frame = _make_base_frame()
        frame["landmarks"]["LEFT_HEEL"] = {"x": 0.47, "y": 0.90, "visibility": 0.9}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": 0.53, "y": 0.90, "visibility": 0.9}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": 0.44, "y": 0.92, "visibility": 0.9}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": 0.56, "y": 0.92, "visibility": 0.9}
        frame["foot_landmarks_source"] = "detected"

        _estimate_missing_foot_landmarks(frame)

        # Should keep original values
        assert frame["landmarks"]["LEFT_HEEL"]["x"] == pytest.approx(0.47)
        assert frame["landmarks"]["LEFT_HEEL"]["visibility"] == pytest.approx(0.9)

    def test_no_estimation_when_heel_already_valid(self):
        """If HEEL already has valid coords, skip estimation for it."""
        frame = _make_base_frame()
        frame["landmarks"]["LEFT_HEEL"] = {"x": 0.47, "y": 0.90, "visibility": 0.8}
        frame["landmarks"]["RIGHT_HEEL"] = {"x": 0.53, "y": 0.90, "visibility": 0.8}
        # Only FOOT_INDEX is NaN
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["RIGHT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        lm = frame["landmarks"]
        # HEEL should keep its original value
        assert lm["LEFT_HEEL"]["x"] == pytest.approx(0.47)
        assert lm["LEFT_HEEL"]["visibility"] == pytest.approx(0.8)
        # FOOT_INDEX should be estimated
        assert not _is_landmark_nan(lm.get("LEFT_FOOT_INDEX"))
        assert lm["LEFT_FOOT_INDEX"]["visibility"] == pytest.approx(0.5)

    def test_estimated_coords_in_bounds(self):
        """Estimated coordinates should be clipped to [0, 1]."""
        frame = _make_base_frame()
        # Ankle near image edge
        frame["landmarks"]["LEFT_ANKLE"] = {"x": 0.02, "y": 0.98, "visibility": 1.0}
        frame["landmarks"]["LEFT_KNEE"] = {"x": 0.02, "y": 0.80, "visibility": 1.0}
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        lm = frame["landmarks"]
        assert 0.0 <= lm["LEFT_HEEL"]["x"] <= 1.0
        assert 0.0 <= lm["LEFT_HEEL"]["y"] <= 1.0
        assert 0.0 <= lm["LEFT_FOOT_INDEX"]["x"] <= 1.0
        assert 0.0 <= lm["LEFT_FOOT_INDEX"]["y"] <= 1.0

    def test_no_estimation_without_ankle(self):
        """Without ankle, estimation should not happen."""
        frame = _make_base_frame()
        del frame["landmarks"]["LEFT_ANKLE"]
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}
        frame["landmarks"]["LEFT_FOOT_INDEX"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        # LEFT side should still be NaN (no ankle to estimate from)
        assert _is_landmark_nan(frame["landmarks"]["LEFT_HEEL"])

    def test_no_estimation_without_knee(self):
        """Without knee, estimation should not happen."""
        frame = _make_base_frame()
        del frame["landmarks"]["LEFT_KNEE"]
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        assert _is_landmark_nan(frame["landmarks"]["LEFT_HEEL"])

    def test_empty_landmarks_noop(self):
        """Frame with empty/missing landmarks should not crash."""
        frame = {"frame_idx": 0, "landmarks": {}}
        _estimate_missing_foot_landmarks(frame)
        assert "foot_landmarks_source" not in frame

    def test_heel_below_ankle(self):
        """Estimated HEEL y should be >= ankle y (foot is below ankle)."""
        frame = _make_base_frame()
        frame["landmarks"]["LEFT_HEEL"] = {"x": float("nan"), "y": float("nan"), "visibility": 0.0}

        _estimate_missing_foot_landmarks(frame)

        lm = frame["landmarks"]
        ankle_y = frame["landmarks"]["LEFT_ANKLE"]["y"]
        # Heel should be below ankle (y increases downward)
        assert lm["LEFT_HEEL"]["y"] >= ankle_y - 0.01
