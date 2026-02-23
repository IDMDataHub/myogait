"""Tests for procrustes_align (normalize module).

Validates Procrustes superimposition: translation, scaling, and rotation
alignment of pose landmarks across frames.

References
----------
Dryden IL, Mardia KV. Statistical Shape Analysis. Wiley; 1998.
Gower JC. Generalized procrustes analysis. Psychometrika.
1975;40(1):33-51. doi:10.1007/BF02291478
"""


import numpy as np

from myogait.normalize import procrustes_align


# ── Helpers ──────────────────────────────────────────────────────────


def _make_data(n_frames=20, n_landmarks=5, seed=42):
    """Create synthetic data with known landmark positions.

    Base shape is a regular polygon.  Each frame has random translation,
    small rotation and small scale jitter applied so Procrustes has
    something meaningful to align.
    """
    rng = np.random.RandomState(seed)
    lm_names = [f"LM_{i}" for i in range(n_landmarks)]
    base_x = np.cos(np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False))
    base_y = np.sin(np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False))

    frames = []
    for i in range(n_frames):
        angle = rng.randn() * 0.1
        scale = 1.0 + rng.randn() * 0.05
        tx, ty = rng.randn(2) * 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        landmarks = {}
        for j, name in enumerate(lm_names):
            x = scale * (cos_a * base_x[j] - sin_a * base_y[j]) + tx
            y = scale * (sin_a * base_x[j] + cos_a * base_y[j]) + ty
            landmarks[name] = {"x": float(x), "y": float(y), "visibility": 1.0}

        frames.append({"landmarks": landmarks, "frame_idx": i})

    return {"frames": frames, "meta": {"fps": 30.0}}, lm_names


def _shape_matrix(frame, lm_names):
    """Extract (n_landmarks, 2) array from a frame."""
    mat = np.array(
        [[frame["landmarks"][n]["x"], frame["landmarks"][n]["y"]] for n in lm_names]
    )
    return mat


# ── Tests ────────────────────────────────────────────────────────────


class TestProcrustesAlign:

    def test_aligned_shapes_centered(self):
        """After alignment all shapes should be centered near origin."""
        data, lm_names = _make_data()
        result = procrustes_align(data)

        for f in result["frames"]:
            mat = _shape_matrix(f, lm_names)
            centroid = mat.mean(axis=0)
            assert np.allclose(centroid, 0.0, atol=1e-6), (
                f"Centroid {centroid} not near origin"
            )

    def test_aligned_shapes_unit_scale(self):
        """After alignment all shapes should have similar centroid size."""
        data, lm_names = _make_data()
        result = procrustes_align(data)

        sizes = []
        for f in result["frames"]:
            mat = _shape_matrix(f, lm_names)
            centered = mat - mat.mean(axis=0)
            sizes.append(np.sqrt(np.sum(centered ** 2)))

        sizes = np.array(sizes)
        # All centroid sizes should be very close to each other
        assert np.std(sizes) < 1e-6, f"Scale std {np.std(sizes):.2e} too high"

    def test_translated_shape_aligned(self):
        """Pure translation should be perfectly removed."""
        lm_names = ["A", "B", "C"]
        base = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
        frames = []
        for tx in [0.0, 5.0, -3.0, 100.0]:
            landmarks = {}
            for name, (bx, by) in base.items():
                landmarks[name] = {"x": bx + tx, "y": by + tx, "visibility": 1.0}
            frames.append({"landmarks": landmarks, "frame_idx": len(frames)})

        data = {"frames": frames, "meta": {"fps": 30.0}}
        result = procrustes_align(data, landmarks=lm_names)

        # All frames should now be identical (translation removed, same shape)
        ref_mat = _shape_matrix(result["frames"][0], lm_names)
        for f in result["frames"][1:]:
            mat = _shape_matrix(f, lm_names)
            np.testing.assert_allclose(mat, ref_mat, atol=1e-10)

    def test_scaled_shape_aligned(self):
        """Pure scaling should be removed, shapes match after alignment."""
        lm_names = ["A", "B", "C"]
        base = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
        frames = []
        for s in [1.0, 2.0, 0.5, 10.0]:
            landmarks = {}
            for name, (bx, by) in base.items():
                landmarks[name] = {"x": bx * s, "y": by * s, "visibility": 1.0}
            frames.append({"landmarks": landmarks, "frame_idx": len(frames)})

        data = {"frames": frames, "meta": {"fps": 30.0}}
        result = procrustes_align(data, landmarks=lm_names)

        ref_mat = _shape_matrix(result["frames"][0], lm_names)
        for f in result["frames"][1:]:
            mat = _shape_matrix(f, lm_names)
            np.testing.assert_allclose(mat, ref_mat, atol=1e-10)

    def test_rotated_shape_aligned(self):
        """Pure rotation should be removed."""
        lm_names = ["A", "B", "C", "D"]
        angles_deg = [0, 30, 90, 180]
        # Asymmetric base shape to avoid rotational symmetry ambiguity
        base_x = np.array([1.0, 0.0, -1.0, -0.5])
        base_y = np.array([0.0, 1.0, 0.0, -0.8])

        frames = []
        for deg in angles_deg:
            rad = np.radians(deg)
            c, s = np.cos(rad), np.sin(rad)
            landmarks = {}
            for j, name in enumerate(lm_names):
                rx = c * base_x[j] - s * base_y[j]
                ry = s * base_x[j] + c * base_y[j]
                landmarks[name] = {"x": float(rx), "y": float(ry), "visibility": 1.0}
            frames.append({"landmarks": landmarks, "frame_idx": len(frames)})

        data = {"frames": frames, "meta": {"fps": 30.0}}
        result = procrustes_align(data, landmarks=lm_names, reference_frame=0)

        ref_mat = _shape_matrix(result["frames"][0], lm_names)
        for f in result["frames"][1:]:
            mat = _shape_matrix(f, lm_names)
            np.testing.assert_allclose(mat, ref_mat, atol=1e-10)

    def test_reference_frame_index(self):
        """Using a specific reference frame stores that index in metadata."""
        data, _ = _make_data(n_frames=10)
        result = procrustes_align(data, reference_frame=3)
        assert result["procrustes"]["reference"] == 3

    def test_mean_reference(self):
        """Default (no reference_frame) uses mean and stores 'mean'."""
        data, _ = _make_data(n_frames=10)
        result = procrustes_align(data)
        assert result["procrustes"]["reference"] == "mean"

    def test_metadata_stored(self):
        """data['procrustes'] contains scale_factors and rotation_angles."""
        data, _ = _make_data(n_frames=8)
        result = procrustes_align(data)

        meta = result["procrustes"]
        assert "scale_factors" in meta
        assert "rotation_angles" in meta
        assert len(meta["scale_factors"]) == 8
        assert len(meta["rotation_angles"]) == 8

        # All valid frames should have float values
        for sf in meta["scale_factors"]:
            assert isinstance(sf, float)
        for ra in meta["rotation_angles"]:
            assert isinstance(ra, float)

    def test_missing_landmarks_skipped(self):
        """Frames with NaN/missing landmarks are handled gracefully."""
        data, lm_names = _make_data(n_frames=10)

        # Corrupt frame 3 by setting a landmark to NaN
        data["frames"][3]["landmarks"][lm_names[0]]["x"] = float("nan")

        result = procrustes_align(data)

        # Frame 3 should have None scale/rotation
        assert result["procrustes"]["scale_factors"][3] is None
        assert result["procrustes"]["rotation_angles"][3] is None

        # Other frames should be aligned normally
        for i in [0, 1, 2, 4, 5]:
            assert result["procrustes"]["scale_factors"][i] is not None

    def test_original_data_not_modified(self):
        """Deep copy ensures original data dict is unchanged."""
        data, lm_names = _make_data(n_frames=5)
        original_x = data["frames"][0]["landmarks"][lm_names[0]]["x"]

        result = procrustes_align(data)

        # Original should be untouched
        assert data["frames"][0]["landmarks"][lm_names[0]]["x"] == original_x
        # Result should be different (aligned)
        assert "procrustes" not in data
        assert "procrustes" in result

    def test_single_frame_no_crash(self):
        """Single frame should return without error."""
        frames = [{
            "landmarks": {
                "A": {"x": 0.0, "y": 0.0, "visibility": 1.0},
                "B": {"x": 1.0, "y": 1.0, "visibility": 1.0},
            },
            "frame_idx": 0,
        }]
        data = {"frames": frames, "meta": {"fps": 30.0}}
        result = procrustes_align(data)
        assert "procrustes" in result

    def test_empty_frames(self):
        """Empty frames list returns gracefully."""
        data = {"frames": [], "meta": {"fps": 30.0}}
        result = procrustes_align(data)
        assert result["procrustes"]["scale_factors"] == []
        assert result["procrustes"]["rotation_angles"] == []
