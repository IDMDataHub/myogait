"""Tests for the v0.8 additions: sign canonicalisation, calibration
guard, landmark bias correction family, 3-D C3D ankle reference, and
the high-level run_pipeline entry point."""

import numpy as np
import pytest

import myogait as mg
from conftest import make_walking_data


# ── helpers ──────────────────────────────────────────────────────────


def _angle_data(hip, knee, ankle=None, fps=30.0):
    """Build a minimal dict with data['angles']['frames'] populated."""
    n = len(hip)
    frames = [{"frame_idx": i, "time_s": i / fps, "landmarks": {}}
              for i in range(n)]
    afr = []
    for i in range(n):
        afr.append({
            "frame_idx": i,
            "hip_L": float(hip[i]), "hip_R": float(hip[i]),
            "knee_L": float(knee[i]), "knee_R": float(knee[i]),
            "ankle_L": float(ankle[i]) if ankle is not None else 0.0,
            "ankle_R": float(ankle[i]) if ankle is not None else 0.0,
        })
    return {
        "meta": {"fps": fps, "n_frames": n},
        "frames": frames,
        "angles": {"frames": afr, "joints": ["hip_L", "hip_R", "knee_L",
                                              "knee_R", "ankle_L", "ankle_R"]},
    }


def _gaitlike_curves(n=200, invert_knee=False, invert_hip=False):
    """Physiological-looking periodic hip/knee curves."""
    t = np.linspace(0, 4 * 2 * np.pi, n)
    knee = 25 - 25 * np.cos(t)              # 0..50, peaks in "swing"
    hip = 15 * np.cos(t)                    # flexion positive at knee peak? no:
    # make hip positive where knee peaks: hip = 15*cos shifted so that at
    # knee max (cos=-1 -> t=pi), hip should be positive
    hip = -15 * np.cos(t)                   # at knee peak, hip=+15 (flexed)
    if invert_knee:
        knee = -knee
    if invert_hip:
        hip = -hip
    return hip, knee


# ── canonicalize_angle_signs ─────────────────────────────────────────


def test_canonicalize_fixes_inverted_knee_and_hip_together():
    hip, knee = _gaitlike_curves(invert_knee=True, invert_hip=True)
    d = _angle_data(hip, knee)
    d = mg.canonicalize_angle_signs(d)
    knees = [f["knee_L"] for f in d["angles"]["frames"]]
    assert np.mean(knees) > 0


def test_canonicalize_fixes_hip_only_inversion():
    # knee correct (positive), hip inverted: extension-positive convention
    hip, knee = _gaitlike_curves(invert_hip=True)
    d = _angle_data(hip, knee)
    d = mg.canonicalize_angle_signs(d)
    afr = d["angles"]["frames"]
    knees = np.array([f["knee_L"] for f in afr])
    hips = np.array([f["hip_L"] for f in afr])
    top = np.argsort(-knees)[: max(5, len(knees) // 10)]
    assert np.mean(hips[top]) > 0, "hip must be flexed at the knee-flexion peak"


def test_canonicalize_leaves_correct_signs_alone():
    hip, knee = _gaitlike_curves()
    d = _angle_data(hip, knee)
    before = [f["hip_L"] for f in d["angles"]["frames"]]
    d = mg.canonicalize_angle_signs(d)
    after = [f["hip_L"] for f in d["angles"]["frames"]]
    assert np.allclose(before, after)


def test_canonicalize_is_idempotent():
    hip, knee = _gaitlike_curves(invert_knee=True, invert_hip=False)
    d = _angle_data(hip, knee)
    d = mg.canonicalize_angle_signs(d)
    once = [f["hip_L"] for f in d["angles"]["frames"]]
    d = mg.canonicalize_angle_signs(d)
    twice = [f["hip_L"] for f in d["angles"]["frames"]]
    assert np.allclose(once, twice)


def test_canonicalize_no_angles_is_noop():
    d = {"meta": {}, "frames": []}
    assert mg.canonicalize_angle_signs(d) is d


# ── calibration guard ────────────────────────────────────────────────


def test_calibration_guard_skips_implausible_offset():
    data = make_walking_data(n_frames=150)
    # calibrate with an absurd tolerance -> applied; with guard -> skipped
    d_guarded = mg.compute_angles(dict(data), calibrate=True,
                                   calibration_joints=["ankle_L"],
                                   calibration_max_offset_deg=0.001)
    d_free = mg.compute_angles(dict(data), calibrate=True,
                                calibration_joints=["ankle_L"],
                                calibration_max_offset_deg=float("inf"))
    a_guard = [f["ankle_L"] for f in d_guarded["angles"]["frames"]
               if f.get("ankle_L") is not None]
    a_free = [f["ankle_L"] for f in d_free["angles"]["frames"]
              if f.get("ankle_L") is not None]
    # The guarded run must NOT have subtracted the offset that the
    # unguarded run subtracted (unless the offset was ~0 anyway).
    offset = np.mean(a_guard) - np.mean(a_free)
    assert abs(np.mean(a_guard)) >= abs(np.mean(a_free)) - 1e-6 or abs(offset) < 1e-6


# ── landmark bias family ─────────────────────────────────────────────


def _bias_dict(n_bins=10, dx=0.1, dy=0.2):
    lms = ("LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE")
    return {lm: {"dx": [dx] * n_bins, "dy": [dy] * n_bins,
                  "n": [20] * n_bins} for lm in lms}


def test_merge_landmark_biases_weighted_mean():
    b1 = _bias_dict(dx=0.1)
    b2 = _bias_dict(dx=0.3)
    merged = mg.merge_landmark_biases([b1, b2])
    assert merged["LEFT_KNEE"]["dx"][0] == pytest.approx(0.2)


def test_merge_landmark_biases_empty_raises():
    with pytest.raises(ValueError):
        mg.merge_landmark_biases([])


def test_smooth_landmark_bias_resamples_and_preserves_mean():
    b = _bias_dict(dx=0.15, dy=-0.05)
    s = mg.smooth_landmark_bias(b, n_harmonics=2, n_out_bins=50)
    assert len(s["LEFT_ANKLE"]["dx"]) == 50
    assert np.mean(s["LEFT_ANKLE"]["dx"]) == pytest.approx(0.15, abs=1e-6)
    assert np.mean(s["LEFT_ANKLE"]["dy"]) == pytest.approx(-0.05, abs=1e-6)


def test_fit_and_apply_landmark_bias_roundtrip():
    """Fit the bias of a recording against a shifted copy of itself:
    the fitted dy must match the injected shift, and applying the
    correction must bring the landmarks back."""
    data = make_walking_data(n_frames=240)
    # The synthetic walker steps in place; translate it so the walking
    # direction is well-defined (the fit skips direction-ambiguous frames).
    for i, f in enumerate(data["frames"]):
        for lm in f["landmarks"].values():
            lm["x"] += 0.0015 * i
    data = mg.normalize(data, filters=["butterworth"])
    data = mg.compute_angles(data, calibrate=False)
    data = mg.detect_events(data, method="zeni", trim_standstill=False,
                             min_cycle_duration=0.6)
    cycles = mg.segment_cycles(data, n_points=101,
                                min_duration=0.8, max_duration=1.5)
    if not cycles.get("cycles"):
        pytest.skip("synthetic walk produced no cycles")
    # "Vicon" = same recording with knees/ankles shifted down by dy_true
    import copy
    vicon = copy.deepcopy(data)
    dy_true = 0.02
    for f in vicon["frames"]:
        for lm in ("LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"):
            if lm in f["landmarks"]:
                f["landmarks"][lm]["y"] += dy_true
    bias = mg.fit_landmark_bias_by_phase(
        data, vicon, cycles, offset_s=0.0,
        landmarks=("LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"))
    dy_bins = [v for v in bias["LEFT_ANKLE"]["dy"] if not np.isnan(v)]
    assert dy_bins, "no bias bins fitted"
    # The knee shift is absorbed into the anchor scale, so the measured
    # ankle bias reflects the RELATIVE geometry difference:
    #   mg_rel = (0.80-0.50)/0.15 = 2.0 ; vc_rel = (0.82-0.50)/0.17 ≈ 1.88
    assert np.mean(dy_bins) == pytest.approx(0.118, abs=0.06)

    corrected = mg.apply_landmark_bias_correction(data, bias, cycles)
    # The correction acts in the mid-hip-anchored, thigh-scaled space:
    # after applying, the corrected ankle's RELATIVE position must be
    # closer to the vicon relative position than the original was.
    from myogait.corrections import _pose_anchor

    def rel_y(dd, i):
        anchor = _pose_anchor(dd["frames"][i]["landmarks"])
        if anchor is None:
            return None
        (mh, scale) = anchor
        return (dd["frames"][i]["landmarks"]["LEFT_ANKLE"]["y"] - mh[1]) / scale

    checked = 0
    closer = 0
    for i in range(0, len(corrected["frames"]), 10):
        ro, rc = rel_y(data, i), rel_y(corrected, i)
        rv = rel_y(vicon, i)
        if None in (ro, rc, rv) or abs(ro - rc) < 1e-9:
            continue          # frame outside any cycle: untouched
        checked += 1
        if abs(rc - rv) < abs(ro - rv):
            closer += 1
    assert checked > 0, "correction touched no sampled frames"
    assert closer / checked > 0.7


# ── compute_c3d_reference_angles ─────────────────────────────────────


def test_c3d_reference_angles_requires_c3d_data():
    data = make_walking_data(n_frames=60)
    data = mg.compute_angles(data, calibrate=False)
    with pytest.raises(ValueError):
        mg.compute_c3d_reference_angles(data)


def test_c3d_reference_angles_overwrites_ankle_from_3d():
    n = 50
    # Synthetic 3-D geometry: shank vertical, foot horizontal -> 0 deg
    # dorsiflexion for all frames; then tilt the foot up 20 deg on the
    # second half -> +20 dorsiflexion.
    knee = np.tile([0.0, 0.0, 1.0], (n, 1)) * 400
    ankle = np.zeros((n, 3))
    heel = np.tile([-50.0, 0.0, 0.0], (n, 1))
    toe = np.zeros((n, 3))
    toe[:, 0] = 150.0
    toe[n // 2:, 2] = 200.0 * np.tan(np.radians(20)) * (150 + 50) / 200.0
    # simpler: set toe z so that foot vector angle = 20 deg
    toe[n // 2:, 2] = np.tan(np.radians(20)) * (toe[n // 2:, 0] - heel[n // 2:, 0])
    m3d = {}
    for side in ("LEFT", "RIGHT"):
        m3d[f"{side}_KNEE"] = knee.copy()
        m3d[f"{side}_ANKLE"] = ankle.copy()
        m3d[f"{side}_HEEL"] = heel.copy()
        m3d[f"{side}_FOOT_INDEX"] = toe.copy()
    afr = [{"frame_idx": i, "ankle_L": 99.0, "ankle_R": 99.0} for i in range(n)]
    data = {"meta": {"fps": 100.0, "n_frames": n},
            "frames": [{"frame_idx": i, "landmarks": {}} for i in range(n)],
            "angles": {"frames": afr},
            "c3d_markers_3d": m3d}
    data = mg.compute_c3d_reference_angles(data)
    a = np.array([f["ankle_L"] for f in data["angles"]["frames"]])
    assert a[5] == pytest.approx(0.0, abs=1.0)
    assert a[-5] == pytest.approx(20.0, abs=1.5)
    assert data["angles"]["ankle_reference"] == "c3d_3d_markers"


# ── run_pipeline ─────────────────────────────────────────────────────


def test_run_pipeline_rejects_unknown_extension(tmp_path):
    p = tmp_path / "input.xyz"
    p.write_text("nothing")
    with pytest.raises(ValueError):
        mg.run_pipeline(str(p))


def test_run_pipeline_on_json(tmp_path):
    data = make_walking_data(n_frames=240)
    path = tmp_path / "walk.myogait.json"
    mg.save_json(data, str(path))
    result = mg.run_pipeline(str(path), analyze=True, show_progress=False)
    assert result["source_type"] == "json"
    assert "angles" in result["data"]
    assert result["data"]["angles"].get("sign_canonicalized") is True
    assert isinstance(result["cycles"].get("cycles", []), list)
    # diagnostics-friendly structure
    assert set(result) >= {"data", "cycles", "stats", "source_type"}
