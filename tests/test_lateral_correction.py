"""Tests for correct_lateral_labels() — per-pair L/R swap correction."""

import numpy as np
import pytest
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


def _swap_pair(frame, l_name, r_name):
    """Swap a single L/R pair in a frame's landmarks."""
    lm = frame["landmarks"]
    lm[l_name], lm[r_name] = lm[r_name], lm[l_name]


# ── Tests: no swaps needed ───────────────────────────────────────────


class TestNoSwaps:
    """When landmarks are correct, no corrections should be applied."""

    def test_no_corrections(self):
        data = _make_data(n_frames=30)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_total_frame_corrections"] == 0

    def test_all_pairs_zero(self):
        data = _make_data(n_frames=30)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        for pair_name, info in meta["pairs"].items():
            assert info["n_corrections"] == 0, (
                f"Pair {pair_name} has {info['n_corrections']} corrections"
            )

    def test_walking_direction_detected(self):
        data = _make_data(direction="right")
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["walking_direction"] == "right"


# ── Tests: ankle swap only ───────────────────────────────────────────


class TestAnkleSwap:
    """ANKLE swapped for a few frames while rest stays correct."""

    def test_ankle_swap_detected(self):
        data = _make_data(n_frames=30)
        # Swap ANKLE at frames 10, 11, 12
        for i in [10, 11, 12]:
            _swap_pair(data["frames"][i], "LEFT_ANKLE", "RIGHT_ANKLE")

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]

        assert meta["pairs"]["ankle"]["n_corrections"] >= 2
        # Other pairs should be untouched
        assert meta["pairs"]["hip"]["n_corrections"] == 0
        assert meta["pairs"]["knee"]["n_corrections"] == 0

    def test_ankle_swap_values_restored(self):
        """After correction, ankle positions should match original."""
        data_orig = _make_data(n_frames=30)
        data = copy.deepcopy(data_orig)

        # Swap ANKLE at frame 10
        _swap_pair(data["frames"][10], "LEFT_ANKLE", "RIGHT_ANKLE")

        result = correct_lateral_labels(data)

        # Compare corrected ankle to original
        orig_la = data_orig["frames"][10]["landmarks"]["LEFT_ANKLE"]
        corr_la = result["frames"][10]["landmarks"]["LEFT_ANKLE"]
        assert abs(orig_la["x"] - corr_la["x"]) < 0.001
        assert abs(orig_la["y"] - corr_la["y"]) < 0.001


# ── Tests: hip swap only ─────────────────────────────────────────────


class TestHipSwap:
    """HIP swapped for a few frames while rest stays correct."""

    def test_hip_swap_detected(self):
        data = _make_data(n_frames=30)
        for i in [5, 6]:
            _swap_pair(data["frames"][i], "LEFT_HIP", "RIGHT_HIP")

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]

        assert meta["pairs"]["hip"]["n_corrections"] >= 1
        assert meta["pairs"]["ankle"]["n_corrections"] == 0


# ── Tests: knee + lower leg swap ─────────────────────────────────────


class TestKneePlusLowerLeg:
    """KNEE + ANKLE + HEEL + FOOT_INDEX swapped together."""

    def test_knee_plus_lower_leg(self):
        data = _make_data(n_frames=30)
        # Swap knee + ankle + heel + foot_index at frames 15, 16
        for i in [15, 16]:
            _swap_pair(data["frames"][i], "LEFT_KNEE", "RIGHT_KNEE")
            _swap_pair(data["frames"][i], "LEFT_ANKLE", "RIGHT_ANKLE")
            _swap_pair(data["frames"][i], "LEFT_HEEL", "RIGHT_HEEL")
            _swap_pair(
                data["frames"][i],
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")

        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]

        # All four pairs should be corrected
        assert meta["pairs"]["knee"]["n_corrections"] >= 1
        assert meta["pairs"]["ankle"]["n_corrections"] >= 1
        assert meta["pairs"]["heel"]["n_corrections"] >= 1
        assert meta["pairs"]["foot_index"]["n_corrections"] >= 1
        # Hip should be untouched
        assert meta["pairs"]["hip"]["n_corrections"] == 0


# ── Tests: empty and edge cases ──────────────────────────────────────


class TestEdgeCases:

    def test_empty_frames(self):
        data = {"frames": [], "normalization": None}
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_total_frame_corrections"] == 0

    def test_single_frame(self):
        data = _make_data(n_frames=1)
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]
        assert meta["n_total_frame_corrections"] == 0

    def test_missing_landmarks(self):
        """Frames with missing landmarks should not crash."""
        data = _make_data(n_frames=10)
        # Remove some landmarks from frame 5
        del data["frames"][5]["landmarks"]["LEFT_ANKLE"]
        del data["frames"][5]["landmarks"]["RIGHT_ANKLE"]

        result = correct_lateral_labels(data)
        assert "lateral_correction" in result["normalization"]

    def test_normalization_none_init(self):
        data = _make_data()
        data["normalization"] = None
        result = correct_lateral_labels(data)
        assert result["normalization"] is not None
        assert "lateral_correction" in result["normalization"]


# ── Tests: idempotence ───────────────────────────────────────────────


class TestIdempotence:

    def test_second_pass_zero_corrections(self):
        data = _make_data(n_frames=30)
        # Create swaps
        for i in [10, 11, 12]:
            _swap_pair(data["frames"][i], "LEFT_ANKLE", "RIGHT_ANKLE")

        # First pass fixes them
        result = correct_lateral_labels(data)
        n_first = result["normalization"]["lateral_correction"][
            "n_total_frame_corrections"]
        assert n_first > 0

        # Second pass should find nothing
        result2 = correct_lateral_labels(result)
        n_second = result2["normalization"]["lateral_correction"][
            "n_total_frame_corrections"]
        assert n_second == 0


# ── Tests: metadata ──────────────────────────────────────────────────


class TestMetadata:

    def test_metadata_keys(self):
        data = _make_data()
        result = correct_lateral_labels(data)
        meta = result["normalization"]["lateral_correction"]

        assert "walking_direction" in meta
        assert "pairs" in meta
        assert "n_total_frame_corrections" in meta

    def test_all_pairs_present(self):
        data = _make_data()
        result = correct_lateral_labels(data)
        pairs = result["normalization"]["lateral_correction"]["pairs"]

        for name in ["ankle", "knee", "hip", "heel",
                     "foot_index", "shoulder"]:
            assert name in pairs
            assert "n_corrections" in pairs[name]
            assert "pct" in pairs[name]
            assert "corrected_frames" in pairs[name]


# ── Tests: anatomical coherence ──────────────────────────────────────


class TestAnatomicalCoherence:
    """Verify post-correction coherence check works."""

    def test_ankle_heel_coherence(self):
        """After correction, ankle should be closer to same-side heel."""
        data = _make_data(n_frames=30)
        # Swap ankle only (not heel) at frame 10
        _swap_pair(data["frames"][10], "LEFT_ANKLE", "RIGHT_ANKLE")

        result = correct_lateral_labels(data)

        # After correction, check coherence:
        # LEFT_ANKLE should be closer to LEFT_HEEL than RIGHT_HEEL
        lm = result["frames"][10]["landmarks"]
        la = (lm["LEFT_ANKLE"]["x"], lm["LEFT_ANKLE"]["y"])
        lh = (lm["LEFT_HEEL"]["x"], lm["LEFT_HEEL"]["y"])
        rh = (lm["RIGHT_HEEL"]["x"], lm["RIGHT_HEEL"]["y"])

        d_same = (la[0] - lh[0]) ** 2 + (la[1] - lh[1]) ** 2
        d_cross = (la[0] - rh[0]) ** 2 + (la[1] - rh[1]) ** 2
        assert d_same < d_cross, (
            f"Ankle should be closer to same-side heel: "
            f"d_same={d_same:.4f}, d_cross={d_cross:.4f}"
        )
