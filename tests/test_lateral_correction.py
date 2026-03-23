"""Tests for correct_lateral_labels() — toggle-based L/R inversion correction."""

import numpy as np
import copy

from myogait.normalize import correct_lateral_labels


# ── Helpers ──────────────────────────────────────────────────────────


def _make_landmark(x, y, vis=1.0):
    return {"x": float(x), "y": float(y), "visibility": vis}


def _make_walking_frames(n_frames=30, direction="right"):
    """Build frames of a person walking with stable L/R labelling.

    LEFT landmarks are at x ~0.45, RIGHT at x ~0.55 (walking right).
    Knee oscillates vertically to simulate gait. Foot/heel at bottom.
    """
    frames = []
    sign = 1.0 if direction == "right" else -1.0
    for i in range(n_frames):
        phase = 2 * np.pi * i / 20  # ~20 frames per cycle
        # Small horizontal displacement (walking)
        dx = 0.001 * i * sign

        lm = {
            "LEFT_HIP": _make_landmark(0.45 + dx, 0.45),
            "RIGHT_HIP": _make_landmark(0.55 + dx, 0.45),
            "LEFT_SHOULDER": _make_landmark(0.44 + dx, 0.25),
            "RIGHT_SHOULDER": _make_landmark(0.56 + dx, 0.25),
            "LEFT_KNEE": _make_landmark(
                0.45 + dx + 0.02 * np.sin(phase), 0.60),
            "RIGHT_KNEE": _make_landmark(
                0.55 + dx - 0.02 * np.sin(phase), 0.60),
            "LEFT_ANKLE": _make_landmark(
                0.45 + dx + 0.03 * np.sin(phase), 0.80),
            "RIGHT_ANKLE": _make_landmark(
                0.55 + dx - 0.03 * np.sin(phase), 0.80),
            "LEFT_HEEL": _make_landmark(
                0.43 + dx + 0.03 * np.sin(phase), 0.82),
            "RIGHT_HEEL": _make_landmark(
                0.53 + dx - 0.03 * np.sin(phase), 0.82),
            "LEFT_FOOT_INDEX": _make_landmark(
                0.49 + dx + 0.03 * np.sin(phase), 0.82),
            "RIGHT_FOOT_INDEX": _make_landmark(
                0.59 + dx - 0.03 * np.sin(phase), 0.82),
        }
        frames.append({
            "frame_idx": i,
            "time_s": i / 30.0,
            "confidence": 0.9,
            "landmarks": lm,
        })
    return frames


def _make_data(n_frames=30, direction="right"):
    """Build a minimal myogait data dict with walking frames."""
    return {
        "meta": {"fps": 30.0},
        "extraction": {"model": "mediapipe"},
        "frames": _make_walking_frames(n_frames, direction),
        "normalization": None,
    }


def _swap_all_pairs(frame):
    """Swap ALL L/R pairs in a frame (simulates a full MediaPipe inversion)."""
    lm = frame["landmarks"]
    pairs = [
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_KNEE", "RIGHT_KNEE"),
        ("LEFT_ANKLE", "RIGHT_ANKLE"),
        ("LEFT_HEEL", "RIGHT_HEEL"),
        ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ]
    for l_name, r_name in pairs:
        if l_name in lm and r_name in lm:
            lm[l_name], lm[r_name] = lm[r_name], lm[l_name]


def _swap_pair(frame, l_name, r_name):
    """Swap a single bilateral pair."""
    lm = frame["landmarks"]
    lm[l_name], lm[r_name] = lm[r_name], lm[l_name]


# ── Tests: no inversions ────────────────────────────────────────────


class TestNoInversions:
    """When landmarks are correct, no corrections should be applied."""

    def test_no_corrections(self):
        data = _make_data(n_frames=30)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_frames_corrected"] == 0

    def test_no_inversions_detected(self):
        data = _make_data(n_frames=30)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_inversions"] == 0

    def test_longer_sequence(self):
        data = _make_data(n_frames=60)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_frames_corrected"] == 0


# ── Tests: full inversion detected and corrected ────────────────────


class TestFullInversion:
    """All landmarks swapped for a few frames should be detected."""

    def test_inversion_detected(self):
        data = _make_data(n_frames=30)
        # Swap ALL landmarks at frames 10, 11, 12
        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_inversions"] >= 1
        assert meta["n_frames_corrected"] >= 2

    def test_inversion_values_restored(self):
        """After correction, positions should match original."""
        data_orig = _make_data(n_frames=30)
        data = copy.deepcopy(data_orig)

        # Swap ALL landmarks at frames 10, 11, 12
        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)

        # Compare corrected vs original
        for i in [10, 11, 12]:
            orig_lh = data_orig["frames"][i]["landmarks"]["LEFT_HIP"]
            corr_lh = result["frames"][i]["landmarks"]["LEFT_HIP"]
            assert abs(orig_lh["x"] - corr_lh["x"]) < 0.001
            assert abs(orig_lh["y"] - corr_lh["y"]) < 0.001

    def test_non_inverted_frames_untouched(self):
        """Frames outside the inversion should not be modified."""
        data_orig = _make_data(n_frames=30)
        data = copy.deepcopy(data_orig)

        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)

        # Frame 5 should be unchanged
        orig_la = data_orig["frames"][5]["landmarks"]["LEFT_ANKLE"]
        corr_la = result["frames"][5]["landmarks"]["LEFT_ANKLE"]
        assert abs(orig_la["x"] - corr_la["x"]) < 0.001

    def test_two_inversion_bursts(self):
        """Two separate inversion periods should both be corrected."""
        data = _make_data(n_frames=50)

        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])
        for i in [30, 31, 32]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        # Should detect transitions for both bursts
        assert meta["n_inversions"] >= 2
        assert meta["n_frames_corrected"] >= 4


# ── Tests: edge cases ───────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_frames(self):
        data = {"frames": [], "normalization": None}
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_frames_corrected"] == 0
        assert meta["n_inversions"] == 0

    def test_single_frame(self):
        data = _make_data(n_frames=1)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_frames_corrected"] == 0

    def test_missing_landmarks(self):
        """Frames with missing landmarks should not crash."""
        data = _make_data(n_frames=10)
        del data["frames"][5]["landmarks"]["LEFT_HIP"]
        del data["frames"][5]["landmarks"]["RIGHT_HIP"]

        result = correct_lateral_labels(data)
        assert "lateral_correction" in result["normalization"]

    def test_normalization_none_init(self):
        data = _make_data()
        data["normalization"] = None
        result = correct_lateral_labels(data)
        assert result["normalization"] is not None
        assert "lateral_correction" in result["normalization"]

    def test_kwargs_ignored(self):
        """Old keyword arguments should be silently ignored."""
        data = _make_data(n_frames=10)
        result = correct_lateral_labels(
            data, method="transition", ratio=0.25, window=2)
        assert "lateral_correction" in result["normalization"]


# ── Tests: idempotence ──────────────────────────────────────────────


class TestIdempotence:

    def test_second_pass_zero_corrections(self):
        data = _make_data(n_frames=30)
        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])

        # First pass fixes them
        result = correct_lateral_labels(data)
        n_first = result["normalization"]["lateral_correction"][
            "n_frames_corrected"]
        assert n_first > 0

        # Second pass should find nothing
        result2 = correct_lateral_labels(result)
        n_second = result2["normalization"]["lateral_correction"][
            "n_frames_corrected"]
        assert n_second == 0


# ── Tests: metadata ─────────────────────────────────────────────────


class TestMetadata:

    def test_metadata_keys(self):
        data = _make_data()
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert "n_inversions" in meta
        assert "n_frames_corrected" in meta
        assert "inversion_frames" in meta

    def test_inversion_frames_list(self):
        """inversion_frames should list the corrected frame indices."""
        data = _make_data(n_frames=30)
        for i in [10, 11, 12]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert isinstance(meta["inversion_frames"], list)
        assert len(meta["inversion_frames"]) > 0


# ── Tests: temporal continuity ──────────────────────────────────────


class TestTemporalContinuity:
    """Verify that correction preserves temporal continuity of signals."""

    def test_hip_x_continuity(self):
        """LEFT_HIP x should be smooth after correction (no jumps)."""
        data = _make_data(n_frames=40)
        for i in [15, 16, 17]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)

        # Extract LEFT_HIP x after correction
        xs = []
        for f in result["frames"]:
            xs.append(f["landmarks"]["LEFT_HIP"]["x"])

        # Max frame-to-frame change should be small
        diffs = [abs(xs[i+1] - xs[i]) for i in range(len(xs) - 1)]
        max_diff = max(diffs)
        # Normal walking: dx ~ 0.001/frame + noise
        # A label swap would cause ~0.1 jump
        assert max_diff < 0.05, (
            f"Max frame-to-frame hip_x diff = {max_diff:.4f}, "
            "suggesting broken temporal continuity"
        )

    def test_knee_x_continuity(self):
        """LEFT_KNEE x should be smooth after correction."""
        data = _make_data(n_frames=40)
        for i in [15, 16, 17]:
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)

        xs = []
        for f in result["frames"]:
            xs.append(f["landmarks"]["LEFT_KNEE"]["x"])

        diffs = [abs(xs[i+1] - xs[i]) for i in range(len(xs) - 1)]
        max_diff = max(diffs)
        assert max_diff < 0.05


# ── Tests: bootstrap polarity ──────────────────────────────────────


class TestBootstrapPolarity:
    """When majority of frames are inverted, polarity should flip."""

    def test_majority_inverted(self):
        """If most frames are swapped, only the minority gets corrected."""
        data = _make_data(n_frames=30)
        # Swap 20 out of 30 frames (majority)
        for i in range(5, 25):
            _swap_all_pairs(data["frames"][i])

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        # Should correct the minority (frames 0-4 and 25-29 = ~10)
        # not the majority (20 frames)
        assert meta["n_frames_corrected"] <= 15


class TestPartialMode:
    """Opt-in partial mode should fix isolated pair swaps."""

    def test_partial_mode_corrects_ankle_only_swap(self):
        data = _make_data(n_frames=30)
        original = copy.deepcopy(data)

        for i in [10, 11, 12]:
            _swap_pair(data["frames"][i], "LEFT_ANKLE", "RIGHT_ANKLE")

        result = correct_lateral_labels(data, mode="partial")
        meta = result["normalization"]["lateral_correction"]

        assert meta["mode"] == "partial"
        assert meta["pair_results"]["ankle"]["n_corrections"] >= 2
        assert meta["pair_results"]["hip"]["n_corrections"] == 0

        for i in [10, 11, 12]:
            expected = original["frames"][i]["landmarks"]["LEFT_ANKLE"]["x"]
            got = result["frames"][i]["landmarks"]["LEFT_ANKLE"]["x"]
            assert abs(expected - got) < 0.001

    def test_default_mode_remains_global(self):
        data = _make_data(n_frames=30)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["mode"] == "global"
