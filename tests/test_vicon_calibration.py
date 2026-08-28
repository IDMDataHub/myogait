"""Tests for myogait.vicon_calibration -- tiers 2 and 3."""

import numpy as np
import pytest

from myogait.isb import _pelvis_frame
from myogait.vicon_calibration import (
    HjcCalibration,
    TechnicalCalibration,
    VSKData,
    apply_technical_calibration,
    calibrate_hjc_from_static,
    calibrate_technical_frames,
    estimate_hjc_harrington,
    hjc_from_calibration,
    parse_protocol,
    parse_vsk,
    reconstruct_isb_angles_tier2,
    reconstruct_isb_angles_tier3,
    technical_frame_from_vsk,
)

# Same synthetic pelvis geometry as test_isb.py's fixtures, duplicated
# rather than imported -- keeps each test file runnable standalone and
# avoids one file's fixture edits silently changing the other's results.
_RASIS = np.array([0.0, 1000.0, 100.0])
_LASIS = np.array([0.0, 1000.0, -100.0])
_RPSIS = np.array([-200.0, 1000.0, 100.0])
_LPSIS = np.array([-200.0, 1000.0, -100.0])


# ── Tier 2: Harrington regression ──────────────────────────────────────


def test_harrington_hjc_is_lateral_posterior_inferior_to_asis_plane():
    RHJC, LHJC = estimate_hjc_harrington(_RASIS, _LASIS, _RPSIS, _LPSIS)
    pelvis = _pelvis_frame(_RASIS, _LASIS, _RPSIS, _LPSIS)
    origin, x_axis, y_axis, z_axis = pelvis[:3, 3], pelvis[:3, 0], pelvis[:3, 1], pelvis[:3, 2]

    r_local = np.array([np.dot(RHJC - origin, x_axis), np.dot(RHJC - origin, y_axis), np.dot(RHJC - origin, z_axis)])
    # Anatomically-expected magnitude range (see module docstring): a few
    # cm lateral, posterior and inferior to the ASIS-plane pelvis origin.
    assert 30.0 < abs(r_local[2]) < 150.0  # lateral (z)
    assert -120.0 < r_local[0] < -10.0     # posterior (x, negative)
    assert -150.0 < r_local[1] < -10.0     # inferior (y, negative)


def test_harrington_hjc_left_right_mirror_for_symmetric_pelvis():
    RHJC, LHJC = estimate_hjc_harrington(_RASIS, _LASIS, _RPSIS, _LPSIS)
    pelvis = _pelvis_frame(_RASIS, _LASIS, _RPSIS, _LPSIS)
    origin, z_axis = pelvis[:3, 3], pelvis[:3, 2]
    # This pelvis is mirror-symmetric across z=0 in world space, and the
    # regression's ML term flips sign between sides -- so R and L should
    # sit at the same distance from the origin, on opposite sides of it
    # along the pelvis's own lateral axis.
    assert np.dot(RHJC - origin, z_axis) == pytest.approx(-np.dot(LHJC - origin, z_axis), rel=1e-6)


def test_harrington_hjc_leg_length_variant_differs_from_default():
    r_default, _ = estimate_hjc_harrington(_RASIS, _LASIS, _RPSIS, _LPSIS)
    r_with_leg, _ = estimate_hjc_harrington(_RASIS, _LASIS, _RPSIS, _LPSIS, leg_length_mm=850.0)
    assert not np.allclose(r_default, r_with_leg)


def test_hjc_calibration_round_trip_at_the_calibration_pose():
    calibration = calibrate_hjc_from_static({
        "RIGHT_ASIS": np.tile(_RASIS, (3, 1)), "LEFT_ASIS": np.tile(_LASIS, (3, 1)),
        "RIGHT_PSIS": np.tile(_RPSIS, (3, 1)), "LEFT_PSIS": np.tile(_LPSIS, (3, 1)),
    })
    pelvis = _pelvis_frame(_RASIS, _LASIS, _RPSIS, _LPSIS)
    RHJC, LHJC = hjc_from_calibration(calibration, pelvis)
    expected_R, expected_L = estimate_hjc_harrington(_RASIS, _LASIS, _RPSIS, _LPSIS)
    assert RHJC == pytest.approx(expected_R, abs=1e-6)
    assert LHJC == pytest.approx(expected_L, abs=1e-6)


def test_hjc_calibration_moves_rigidly_with_a_translated_pelvis():
    # The whole point of calibrating in the pelvis's *local* frame: a
    # pelvis translated 500 mm forward should carry the HJC with it,
    # unchanged relative to the pelvis itself.
    calibration = calibrate_hjc_from_static({
        "RIGHT_ASIS": np.tile(_RASIS, (2, 1)), "LEFT_ASIS": np.tile(_LASIS, (2, 1)),
        "RIGHT_PSIS": np.tile(_RPSIS, (2, 1)), "LEFT_PSIS": np.tile(_LPSIS, (2, 1)),
    })
    shift = np.array([500.0, 0.0, 0.0])
    pelvis_static = _pelvis_frame(_RASIS, _LASIS, _RPSIS, _LPSIS)
    pelvis_moved = _pelvis_frame(_RASIS + shift, _LASIS + shift, _RPSIS + shift, _LPSIS + shift)

    RHJC_static, _ = hjc_from_calibration(calibration, pelvis_static)
    RHJC_moved, _ = hjc_from_calibration(calibration, pelvis_moved)
    assert RHJC_moved == pytest.approx(RHJC_static + shift, abs=1e-6)


def test_reconstruct_isb_angles_tier2_matches_tier1_shape():
    # Reuses tier 1's own fixture-building approach inline (kept small
    # and local rather than importing test_isb.py's private helpers).
    n = 3
    markers = {
        "RIGHT_ASIS": _RASIS, "LEFT_ASIS": _LASIS, "RIGHT_PSIS": _RPSIS, "LEFT_PSIS": _LPSIS,
        "RIGHT_KNEE_LATERAL": np.array([-70.0, 600.0, 130.0]), "RIGHT_KNEE_MEDIAL": np.array([-70.0, 600.0, 70.0]),
        "LEFT_KNEE_LATERAL": np.array([-70.0, 600.0, -130.0]), "LEFT_KNEE_MEDIAL": np.array([-70.0, 600.0, -70.0]),
        "RIGHT_ANKLE_LATERAL": np.array([-70.0, 200.0, 120.0]), "RIGHT_ANKLE_MEDIAL": np.array([-70.0, 200.0, 80.0]),
        "LEFT_ANKLE_LATERAL": np.array([-70.0, 200.0, -120.0]), "LEFT_ANKLE_MEDIAL": np.array([-70.0, 200.0, -80.0]),
        "RIGHT_HEEL": np.array([-150.0, 120.0, 100.0]), "LEFT_HEEL": np.array([-150.0, 120.0, -100.0]),
        "RIGHT_FOOT_INDEX_MEDIAL": np.array([80.0, 120.0, 80.0]), "RIGHT_FOOT_INDEX_LATERAL": np.array([80.0, 120.0, 120.0]),
        "LEFT_FOOT_INDEX_MEDIAL": np.array([80.0, 120.0, -80.0]), "LEFT_FOOT_INDEX_LATERAL": np.array([80.0, 120.0, -120.0]),
    }
    static_markers_3d = {k: np.tile(v, (n, 1)) for k, v in markers.items()}
    data = {
        "meta": {"fps": 100.0, "n_frames": n},
        "frames": [{"frame_idx": i, "landmarks": {}} for i in range(n)],
        "angles": {"frames": [{"frame_idx": i} for i in range(n)]},
        "c3d_markers_3d": {k: v.copy() for k, v in static_markers_3d.items()},
    }
    data = reconstruct_isb_angles_tier2(data, static_markers_3d)
    assert data["angles"]["isb_reference"] == "isb_3d_tier2_static_hjc"
    frame = data["angles"]["frames"][0]
    for joint in ("hip", "knee", "ankle"):
        for side in ("L", "R"):
            assert isinstance(frame[f"{joint}_{side}"], float)


# ── Tier 3: VSK / protocol parsing ─────────────────────────────────────


_SYNTHETIC_VSK = """<?xml version="1.0"?>
<KinematicModel>
  <Skeleton>
    <Parameters>
      <Parameter NAME="RKNEE_X" VALUE="50.0"/>
      <Parameter NAME="RKNEE_Y" VALUE="0.0"/>
      <Parameter NAME="RKNEE_Z" VALUE="0.0"/>
      <Parameter NAME="RANKLE_X" VALUE="-30.0"/>
      <Parameter NAME="RANKLE_Y" VALUE="0.0"/>
      <Parameter NAME="RANKLE_Z" VALUE="-400.0"/>
    </Parameters>
    <MarkerSet>
      <Markers>
        <TargetLocalPointToWorldPoint SEGMENT="RightTibia" MARKER="RKNE" POSITION="'RKNEE_X' 'RKNEE_Y' 'RKNEE_Z'"/>
        <TargetLocalPointToWorldPoint SEGMENT="RightTibia" MARKER="RANK" POSITION="'RANKLE_X' 'RANKLE_Y' 'RANKLE_Z'"/>
      </Markers>
    </MarkerSet>
    <JointBallSet>
      <JointBalls>
        <JointBall NAME="pelvis_RightThigh" PRE-POSITION="70.0 -30.0 90.0"/>
      </JointBalls>
    </JointBallSet>
  </Skeleton>
</KinematicModel>
"""

_SYNTHETIC_PROT = """#SEGMENTS
pelvis=RASIS,LASIS,RPSIS,LPSIS
Rtibia=RKNE,RANK

#ARTICULATIONS
Rknee=Rtibia,Rfemur

#FICHIER_STATIQUE
reference=statref
"""


def test_parse_vsk_reads_segment_markers_and_joint_balls(tmp_path):
    vsk_path = tmp_path / "subject.vsk"
    vsk_path.write_text(_SYNTHETIC_VSK, encoding="utf-8")
    vsk = parse_vsk(vsk_path)
    assert "RightTibia" in vsk.segment_markers_local
    assert set(vsk.segment_markers_local["RightTibia"]) == {"RKNE", "RANK"}
    assert vsk.segment_markers_local["RightTibia"]["RKNE"] == pytest.approx([50.0, 0.0, 0.0])
    assert "pelvis_RightThigh" in vsk.joint_balls_local
    assert vsk.joint_balls_local["pelvis_RightThigh"] == pytest.approx([70.0, -30.0, 90.0])


def test_parse_protocol_reads_sections_and_lists(tmp_path):
    prot_path = tmp_path / "protocol.prot"
    prot_path.write_text(_SYNTHETIC_PROT, encoding="utf-8")
    protocol = parse_protocol(prot_path)
    assert protocol["SEGMENTS"]["pelvis"] == ["RASIS", "LASIS", "RPSIS", "LPSIS"]
    assert protocol["SEGMENTS"]["Rtibia"] == ["RKNE", "RANK"]
    assert protocol["ARTICULATIONS"]["Rknee"] == ["Rtibia", "Rfemur"]
    assert protocol["FICHIER_STATIQUE"]["reference"] == "statref"


def test_technical_frame_from_vsk_recovers_a_known_rigid_transform():
    vsk = VSKData(segment_markers_local={
        "seg": {"M1": np.array([0.0, 0.0, 0.0]), "M2": np.array([100.0, 0.0, 0.0]), "M3": np.array([0.0, 100.0, 0.0])}
    })
    # A known rotation (90 deg about z) + translation.
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([10.0, 20.0, 30.0])
    world = {name: R @ local + t for name, local in vsk.segment_markers_local["seg"].items()}

    T, used = technical_frame_from_vsk("seg", vsk.segment_markers_local, world)
    assert set(used) == {"M1", "M2", "M3"}
    assert T[:3, :3] == pytest.approx(R, abs=1e-6)
    assert T[:3, 3] == pytest.approx(t, abs=1e-6)


def test_technical_frame_from_vsk_requires_minimum_points():
    vsk = VSKData(segment_markers_local={"seg": {"M1": np.zeros(3), "M2": np.ones(3)}})
    with pytest.raises(ValueError):
        technical_frame_from_vsk("seg", vsk.segment_markers_local, {"M1": np.zeros(3), "M2": np.ones(3)})


def test_technical_frame_from_vsk_unknown_segment_raises_keyerror():
    vsk = VSKData()
    with pytest.raises(KeyError):
        technical_frame_from_vsk("nope", vsk.segment_markers_local, {})


# ── Tier 3: full synthetic integration ─────────────────────────────────


#: Off-axis "wand" marker per thigh, needed because two knee markers
#: alone cannot be rigidly fit (a 2-point correspondence has a free
#: rotation about the axis joining them) -- real VSKs add exactly this
#: kind of extra marker to the thigh for the same reason (Myokinesis's
#: own VSK carries four, RTH1-4). Placed anterior to the knee midpoint,
#: not collinear with the two knee markers.
_R_THIGH_WAND = np.array([30.0, 600.0, 100.0])
_L_THIGH_WAND = np.array([30.0, 600.0, -100.0])


def _synthetic_vsk_matching_isb_landmarks() -> VSKData:
    """A VSK whose technical clusters are the ISB landmarks themselves
    plus one off-axis wand marker per thigh (see ``_R_THIGH_WAND``) --
    the simplest case where tier 3's technical and anatomical frames
    should coincide almost exactly, since there is nothing for the
    calibration offset to correct for beyond floating-point noise.
    """
    return VSKData(
        segment_markers_local={
            "pelvis": {"RASIS": _RASIS, "LASIS": _LASIS, "RPSIS": _RPSIS, "LPSIS": _LPSIS},
            "RightThigh": {
                "RKNL": np.array([-70.0, 600.0, 130.0]), "RKNM": np.array([-70.0, 600.0, 70.0]),
                "RTHW": _R_THIGH_WAND,
            },
            "LeftThigh": {
                "LKNL": np.array([-70.0, 600.0, -130.0]), "LKNM": np.array([-70.0, 600.0, -70.0]),
                "LTHW": _L_THIGH_WAND,
            },
            "RightTibia": {
                "RKNL": np.array([-70.0, 600.0, 130.0]), "RKNM": np.array([-70.0, 600.0, 70.0]),
                "RANL": np.array([-70.0, 200.0, 120.0]), "RANM": np.array([-70.0, 200.0, 80.0]),
            },
            "LeftTibia": {
                "LKNL": np.array([-70.0, 600.0, -130.0]), "LKNM": np.array([-70.0, 600.0, -70.0]),
                "LANL": np.array([-70.0, 200.0, -120.0]), "LANM": np.array([-70.0, 200.0, -80.0]),
            },
            "RightFoot": {
                "RHEE": np.array([-150.0, 120.0, 100.0]),
                "RFM1": np.array([80.0, 120.0, 80.0]), "RFM5": np.array([80.0, 120.0, 120.0]),
                "RANL": np.array([-70.0, 200.0, 120.0]), "RANM": np.array([-70.0, 200.0, 80.0]),
            },
            "LeftFoot": {
                "LHEE": np.array([-150.0, 120.0, -100.0]),
                "LFM1": np.array([80.0, 120.0, -80.0]), "LFM5": np.array([80.0, 120.0, -120.0]),
                "LANL": np.array([-70.0, 200.0, -120.0]), "LANM": np.array([-70.0, 200.0, -80.0]),
            },
        },
        joint_balls_local={},  # none -- forces the Harrington fallback
    )


def _rename_for_vsk(markers: dict) -> dict:
    rename = {
        "RIGHT_ASIS": "RASIS", "LEFT_ASIS": "LASIS", "RIGHT_PSIS": "RPSIS", "LEFT_PSIS": "LPSIS",
        "RIGHT_KNEE_LATERAL": "RKNL", "RIGHT_KNEE_MEDIAL": "RKNM",
        "LEFT_KNEE_LATERAL": "LKNL", "LEFT_KNEE_MEDIAL": "LKNM",
        "RIGHT_ANKLE_LATERAL": "RANL", "RIGHT_ANKLE_MEDIAL": "RANM",
        "LEFT_ANKLE_LATERAL": "LANL", "LEFT_ANKLE_MEDIAL": "LANM",
        "RIGHT_HEEL": "RHEE", "LEFT_HEEL": "LHEE",
        "RIGHT_FOOT_INDEX_MEDIAL": "RFM1", "RIGHT_FOOT_INDEX_LATERAL": "RFM5",
        "LEFT_FOOT_INDEX_MEDIAL": "LFM1", "LEFT_FOOT_INDEX_LATERAL": "LFM5",
    }
    return {rename[k]: v for k, v in markers.items() if k in rename}


def _with_thigh_wands(raw: dict) -> dict:
    """Add the two synthetic thigh-wand markers (see ``_R_THIGH_WAND``)
    to a raw-marker dict already built by :func:`_rename_for_vsk`."""
    out = dict(raw)
    out["RTHW"] = _R_THIGH_WAND
    out["LTHW"] = _L_THIGH_WAND
    return out


def test_tier3_calibrates_every_segment_and_reconstructs_plausible_angles():
    static_landmarks = {
        "RIGHT_ASIS": _RASIS, "LEFT_ASIS": _LASIS, "RIGHT_PSIS": _RPSIS, "LEFT_PSIS": _LPSIS,
        "RIGHT_KNEE_LATERAL": np.array([-70.0, 600.0, 130.0]), "RIGHT_KNEE_MEDIAL": np.array([-70.0, 600.0, 70.0]),
        "LEFT_KNEE_LATERAL": np.array([-70.0, 600.0, -130.0]), "LEFT_KNEE_MEDIAL": np.array([-70.0, 600.0, -70.0]),
        "RIGHT_ANKLE_LATERAL": np.array([-70.0, 200.0, 120.0]), "RIGHT_ANKLE_MEDIAL": np.array([-70.0, 200.0, 80.0]),
        "LEFT_ANKLE_LATERAL": np.array([-70.0, 200.0, -120.0]), "LEFT_ANKLE_MEDIAL": np.array([-70.0, 200.0, -80.0]),
        "RIGHT_HEEL": np.array([-150.0, 120.0, 100.0]), "LEFT_HEEL": np.array([-150.0, 120.0, -100.0]),
        "RIGHT_FOOT_INDEX_MEDIAL": np.array([80.0, 120.0, 80.0]), "RIGHT_FOOT_INDEX_LATERAL": np.array([80.0, 120.0, 120.0]),
        "LEFT_FOOT_INDEX_MEDIAL": np.array([80.0, 120.0, -80.0]), "LEFT_FOOT_INDEX_LATERAL": np.array([80.0, 120.0, -120.0]),
    }
    static_raw = _with_thigh_wands(_rename_for_vsk(static_landmarks))
    vsk = _synthetic_vsk_matching_isb_landmarks()

    calibration = calibrate_technical_frames(vsk, static_raw, static_landmarks)
    assert set(calibration.segments) == {"pelvis", "RightThigh", "LeftThigh", "RightTibia", "LeftTibia", "RightFoot", "LeftFoot"}
    # No VSK joint ball supplied -- must have fallen back to Harrington.
    for offset in calibration.offsets.values():
        assert np.isfinite(offset).all()

    n = 3
    dynamic_raw = {k: np.tile(v, (n, 1)) for k, v in static_raw.items()}
    data = {
        "angles": {"frames": [{"frame_idx": i} for i in range(n)]},
    }
    data = reconstruct_isb_angles_tier3(data, dynamic_raw, calibration)
    assert data["angles"]["isb_reference"] == "isb_3d_tier3_calibrated"
    frame = data["angles"]["frames"][0]
    for joint in ("hip", "knee", "ankle"):
        for side in ("L", "R"):
            value = frame[f"{joint}_{side}"]
            assert isinstance(value, float)
            assert -30.0 < value < 30.0  # near-neutral standing posture


def test_tier3_left_right_symmetry():
    # Same regression rationale as test_isb.py's tier-1 symmetry test:
    # a mirror-symmetric posture must give matching left/right angles.
    static_landmarks = {
        "RIGHT_ASIS": _RASIS, "LEFT_ASIS": _LASIS, "RIGHT_PSIS": _RPSIS, "LEFT_PSIS": _LPSIS,
        "RIGHT_KNEE_LATERAL": np.array([-70.0, 600.0, 130.0]), "RIGHT_KNEE_MEDIAL": np.array([-70.0, 600.0, 70.0]),
        "LEFT_KNEE_LATERAL": np.array([-70.0, 600.0, -130.0]), "LEFT_KNEE_MEDIAL": np.array([-70.0, 600.0, -70.0]),
        "RIGHT_ANKLE_LATERAL": np.array([-270.0, 253.6, 120.0]), "RIGHT_ANKLE_MEDIAL": np.array([-270.0, 253.6, 80.0]),
        "LEFT_ANKLE_LATERAL": np.array([-270.0, 253.6, -120.0]), "LEFT_ANKLE_MEDIAL": np.array([-270.0, 253.6, -80.0]),
        "RIGHT_HEEL": np.array([-350.0, 173.6, 100.0]), "LEFT_HEEL": np.array([-350.0, 173.6, -100.0]),
        "RIGHT_FOOT_INDEX_MEDIAL": np.array([-120.0, 173.6, 80.0]), "RIGHT_FOOT_INDEX_LATERAL": np.array([-120.0, 173.6, 120.0]),
        "LEFT_FOOT_INDEX_MEDIAL": np.array([-120.0, 173.6, -80.0]), "LEFT_FOOT_INDEX_LATERAL": np.array([-120.0, 173.6, -120.0]),
    }
    static_raw = _with_thigh_wands(_rename_for_vsk(static_landmarks))
    vsk = _synthetic_vsk_matching_isb_landmarks()
    calibration = calibrate_technical_frames(vsk, static_raw, static_landmarks)

    data = {"angles": {"frames": [{"frame_idx": 0}]}}
    dynamic_raw = {k: v.reshape(1, 3) for k, v in static_raw.items()}
    data = reconstruct_isb_angles_tier3(data, dynamic_raw, calibration)
    frame = data["angles"]["frames"][0]
    for joint in ("hip", "knee", "ankle"):
        assert frame[f"{joint}_L"] == pytest.approx(frame[f"{joint}_R"], abs=1.0)


def test_tier3_raises_when_calibration_is_incomplete():
    calibration = TechnicalCalibration(vsk=VSKData(), offsets={}, segments=("pelvis",))
    data = {"angles": {"frames": [{"frame_idx": 0}]}}
    with pytest.raises(Exception):
        reconstruct_isb_angles_tier3(data, {}, calibration)
