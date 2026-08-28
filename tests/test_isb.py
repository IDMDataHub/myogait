"""Tests for myogait.isb -- ISB 3-D joint angle reconstruction (tier 1)."""

import numpy as np
import pytest

from myogait.isb import (
    ISB_REQUIRED_LANDMARKS,
    InsufficientLandmarksForISBError,
    reconstruct_isb_angles,
)


def _pelvis_markers(z_sign=1.0):
    """A simple rectangular pelvis, ASIS 200 mm anterior of PSIS.

    ``z_sign`` mirrors the whole marker set across the sagittal midline
    (z=0) when set to -1.0 -- used to build a mirror-image posture for the
    left-vs-right symmetry regression test below.
    """
    return {
        "RIGHT_ASIS": np.array([0.0, 1000.0, 100.0 * z_sign]),
        "LEFT_ASIS": np.array([0.0, 1000.0, -100.0 * z_sign]),
        "RIGHT_PSIS": np.array([-200.0, 1000.0, 100.0 * z_sign]),
        "LEFT_PSIS": np.array([-200.0, 1000.0, -100.0 * z_sign]),
    }


def _leg_markers(side: str, knee_center, ankle_center, z_sign=1.0):
    """Knee/ankle/heel/forefoot markers around given centres, one leg.

    ``side`` is ``"RIGHT"`` or ``"LEFT"``. Lateral is the side away from
    the midline: +z for the right leg, -z for the left (or the mirror of
    that when ``z_sign=-1.0``).
    """
    lateral_sign = z_sign if side == "RIGHT" else -z_sign
    kx, ky, kz = knee_center
    ax, ay, az = ankle_center
    return {
        f"{side}_KNEE_LATERAL": np.array([kx, ky, kz + 30.0 * lateral_sign]),
        f"{side}_KNEE_MEDIAL": np.array([kx, ky, kz - 30.0 * lateral_sign]),
        f"{side}_ANKLE_LATERAL": np.array([ax, ay, az + 20.0 * lateral_sign]),
        f"{side}_ANKLE_MEDIAL": np.array([ax, ay, az - 20.0 * lateral_sign]),
        f"{side}_HEEL": np.array([ax - 80.0, ay - 80.0, az]),
        f"{side}_FOOT_INDEX_MEDIAL": np.array([ax + 150.0, ay - 80.0, az - 40.0 * lateral_sign]),
        f"{side}_FOOT_INDEX_LATERAL": np.array([ax + 150.0, ay - 80.0, az + 40.0 * lateral_sign]),
    }


def _standing_pivot(n_frames=5, knee_flex_deg=0.0, z_sign=1.0):
    """A synthetic, roughly neutral standing posture, both legs identical
    apart from mirroring, with an optional knee flexion applied to both
    sides equally (rotating the shank posteriorly about the knee's
    mediolateral axis -- see the module docstring's axis convention:
    x=anterior, y=proximal/up, z=to the subject's right).

    Every frame is identical -- this tests the per-frame geometry, not
    gait dynamics.
    """
    hip_offset_x = -70.0  # x of the (fixed-ratio) hip joint centre this
    # geometry lands on, given the pelvis markers above and the module's
    # default 0.30 HJC ratio -- see the module docstring's tier-1 formula.
    knee_center_neutral = np.array([hip_offset_x, 600.0, 30.0 * z_sign])

    theta = np.radians(knee_flex_deg)
    rot_xy = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    dx, dy = rot_xy @ np.array([0.0, -1.0])
    # Posterior (-x) tilt as flexion increases -- the rotation above
    # tilts +x (anterior) for positive theta, so negate x to point the
    # shank backward instead, matching ISB's flexion-positive convention.
    shank_vector = np.array([-dx, dy, 0.0]) * 400.0
    ankle_center = knee_center_neutral + shank_vector

    markers = {}
    markers.update(_pelvis_markers(z_sign))
    for side in ("RIGHT", "LEFT"):
        markers.update(_leg_markers(side, knee_center_neutral, ankle_center, z_sign))

    m3d = {lm: np.tile(pos, (n_frames, 1)) for lm, pos in markers.items()}
    data = {
        "meta": {"fps": 100.0, "n_frames": n_frames},
        "frames": [{"frame_idx": i, "landmarks": {}} for i in range(n_frames)],
        "angles": {"frames": [{"frame_idx": i} for i in range(n_frames)]},
        "c3d_markers_3d": m3d,
    }
    return data


# ── Guardrails ──────────────────────────────────────────────────────


def test_reconstruct_isb_requires_c3d_markers_3d():
    data = {"angles": {"frames": [{"frame_idx": 0}]}}
    with pytest.raises(InsufficientLandmarksForISBError):
        reconstruct_isb_angles(data)


def test_reconstruct_isb_requires_every_landmark():
    data = _standing_pivot()
    # Drop one required landmark -- should name it in the error, not fail
    # silently or with a generic KeyError deep in the geometry code.
    del data["c3d_markers_3d"]["LEFT_ANKLE_MEDIAL"]
    with pytest.raises(InsufficientLandmarksForISBError) as exc_info:
        reconstruct_isb_angles(data)
    assert "LEFT_ANKLE_MEDIAL" in str(exc_info.value)


def test_reconstruct_isb_requires_compute_angles_first():
    data = _standing_pivot()
    del data["angles"]
    with pytest.raises(ValueError):
        reconstruct_isb_angles(data)


def test_isb_required_landmarks_are_all_present_in_fixture():
    # The fixture builder above must stay in sync with the module's own
    # requirement list, or the guardrail tests above pass for the wrong
    # reason (missing landmarks that were never going to be resolved).
    data = _standing_pivot()
    for lm in ISB_REQUIRED_LANDMARKS:
        assert lm in data["c3d_markers_3d"], lm


# ── Output contract ───────────────────────────────────────────────────


def test_reconstruct_isb_writes_backward_compatible_flexion_and_new_dof():
    data = _standing_pivot(knee_flex_deg=20.0)
    data = reconstruct_isb_angles(data)
    frame = data["angles"]["frames"][0]
    for joint in ("hip", "knee", "ankle"):
        for side in ("L", "R"):
            # Old key: still a plain float, exactly what compute_angles's
            # sagittal method already produced -- every existing reader
            # (charts, export, bias corrections) keeps working unmodified.
            assert isinstance(frame[f"{joint}_{side}"], float)
            # New keys: additive only.
            assert isinstance(frame[f"{joint}_{side}_abd_add_deg"], float)
            assert isinstance(frame[f"{joint}_{side}_int_ext_rot_deg"], float)
    assert data["angles"]["isb_reference"] == "isb_3d_direct"


def test_reconstruct_isb_only_recomputes_requested_joints():
    data = _standing_pivot()
    data["angles"]["frames"][0]["hip_L"] = 12345.0
    data = reconstruct_isb_angles(data, joints=("knee", "ankle"))
    assert data["angles"]["frames"][0]["hip_L"] == 12345.0
    assert "knee_L_abd_add_deg" in data["angles"]["frames"][0]


# ── Geometry: known flexion angle recovered correctly ──────────────────


def test_reconstruct_isb_knee_flexion_matches_known_angle():
    # 30 deg of pure sagittal-plane shank rotation about the knee, femur
    # left exactly vertical -- see _standing_pivot()'s derivation. Loose
    # tolerance: the hip joint centre proxy nudges the femur slightly off
    # vertical too, so this is not an exact closed-form match.
    data = _standing_pivot(knee_flex_deg=30.0)
    data = reconstruct_isb_angles(data)
    for side in ("L", "R"):
        assert data["angles"]["frames"][0][f"knee_{side}"] == pytest.approx(30.0, abs=3.0)


def test_reconstruct_isb_neutral_standing_is_near_zero_flexion():
    data = _standing_pivot(knee_flex_deg=0.0)
    data = reconstruct_isb_angles(data)
    for side in ("L", "R"):
        for joint in ("hip", "knee", "ankle"):
            assert data["angles"]["frames"][0][f"{joint}_{side}"] == pytest.approx(0.0, abs=5.0)


# ── Regression: the actual bug found while integrating this module ─────
#
# reconstruct_isb_angles's femur/tibia/foot frame builders need the
# marker that sits toward the subject's global left in a "body_left"
# slot and the one toward global right in "body_right" -- not simply
# "lateral" and "medial" -- because which anatomical marker is on which
# global side flips between the left and right leg. The first
# implementation used a fixed lateral-then-medial order for both sides;
# it matched a hand-validated reference on the right leg by coincidence
# (lateral happens to be body-right there) and mirrored every angle on
# the left leg (near-perfect *negative* correlation against the
# reference, and physically impossible values like a 175-degree ankle).
# A mirror-symmetric synthetic posture -- identical flexion on both
# legs, geometry reflected across the sagittal midline -- catches this
# directly: left and right must come out equal.


def test_reconstruct_isb_left_right_symmetry_regression():
    data = _standing_pivot(knee_flex_deg=25.0)
    data = reconstruct_isb_angles(data)
    frame = data["angles"]["frames"][0]
    for joint in ("hip", "knee", "ankle"):
        left = frame[f"{joint}_L"]
        right = frame[f"{joint}_R"]
        assert left == pytest.approx(right, abs=1.0), (
            f"{joint}: left={left:.2f} right={right:.2f} -- a mirrored "
            "frame on one side (the body-left/body-right marker-order "
            "bug this test guards against) shows up as a large or "
            "sign-flipped left/right disagreement here."
        )
