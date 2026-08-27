"""ISB-recommended 3-D joint angle reconstruction from C3D marker data.

:func:`compute_angles` (see ``angles.py``) works from a single point per
joint (hip/knee/ankle centres already averaged together by
:func:`myogait.experimental_vicon.load_c3d`) and reports flexion/extension
referenced to the trunk, in the sagittal plane. That is deliberately
lightweight -- it is the only option markerless/video sources and sparse
marker sets can support -- but it collapses two things a richer C3D file
can do better:

- **Reference segment**: the trunk (shoulder->hip vector) is not the
  pelvis. ISB defines hip flexion relative to the *pelvis* segment, not
  the trunk -- a real anatomical difference, not just a precision gap
  (validated empirically: r ~= 0.99 between the two, but a ~10-17 degree
  constant offset on hip/knee, see the module's companion audit report).
- **Degrees of freedom**: only flexion/extension is reported; abduction/
  adduction and internal/external rotation are not computed at all.

This module recomputes hip, knee and ankle angles from C3D markers using
proper ISB anatomical segment frames (pelvis, thigh, shank, foot), each
built from real anatomical landmarks rather than a single averaged joint
centre, and decomposes the relative rotation between adjacent segments
with the ISB-recommended Z-X-Y (flexion/adduction/rotation) sequence.

Ref: Wu G, Siegler S, Allard P, et al. ISB recommendation on definitions
of joint coordinate system of various joints for the reporting of human
joint motion -- part I: ankle, hip, and spine. J Biomech.
2002;35(4):543-548. doi:10.1016/S0021-9290(01)00222-6

Calibration tiers
------------------
Three tiers trade file requirements for accuracy (see the companion audit
report for the measured cost of each):

1. **Direct** (this module, no extra file): the anatomical frame is
   rebuilt from scratch every frame, straight from that frame's own
   markers. The hip joint centre -- invisible to any marker -- is
   approximated by a fixed fraction of the way from the pelvis centroid
   toward each side's ASIS (see :func:`_direct_hip_joint_centers`).
   Measured against tier 3 on a real Vicon+VSK trial: 2-4 degrees RMSE,
   r > 0.98, on hip/knee/ankle flexion-extension.
2. **Static-only** (planned, not in this module): the same direct
   per-frame reconstruction, but with the hip joint centre estimated by
   an anthropometric regression (e.g. Harrington 2007) fitted on
   measurements taken from a static trial -- no ``.vsk``/``.prot``
   needed, just one extra C3D file.
3. **Calibrated** (planned, not in this module): a technical marker
   cluster (from a ``.vsk``) is rigidly fit to the markers every frame,
   and a technical->anatomical offset computed once on a static trial is
   applied on top -- the full CGM/Plug-in-Gait-equivalent approach.

Only tier 1 is implemented here. It intentionally does not touch
:data:`myogait.angles.ANGLE_METHODS` -- unlike a per-frame angle method,
ISB reconstruction needs the full 3-D marker trajectories
(``data["c3d_markers_3d"]``), not one frame's 2-D projected landmarks, so
it follows the same "call after compute_angles(), overwrite in place"
shape as :func:`myogait.experimental_vicon.compute_c3d_reference_angles`
rather than registering as a ``method=`` value.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError as exc:  # pragma: no cover - scipy is already a myogait dep
    raise ImportError(
        "scipy is required for ISB angle reconstruction (already a myogait "
        "dependency -- check your installation)."
    ) from exc

logger = logging.getLogger(__name__)


class InsufficientLandmarksForISBError(ValueError):
    """Raised when a file's marker mapping cannot support ISB reconstruction.

    ISB reconstruction needs paired medial/lateral markers per joint to
    build a true anatomical segment frame -- a single joint-centre point
    (all that markerless/video sources or a sparse marker set provide) is
    not enough to define an anatomical frame's axes. Callers should catch
    this and fall back to ``compute_angles(method="sagittal_vertical_axis")``
    rather than silently presenting a wrong or degenerate result.
    """


#: Landmarks reconstruct_isb_angles() needs in data["c3d_markers_3d"],
#: beyond the six load_c3d resolves by default. A marker_mapping passed to
#: load_c3d must resolve these as *separate* points -- not averaged
#: together the way LEFT_KNEE/LEFT_ANKLE/etc. normally are, since
#: load_c3d averages every candidate marker listed for one landmark name
#: into a single point -- or the anatomical frame this module builds
#: collapses to the same single-point approximation compute_angles()
#: already gives for free, defeating the purpose.
ISB_REQUIRED_LANDMARKS: tuple[str, ...] = (
    "LEFT_ASIS", "RIGHT_ASIS", "LEFT_PSIS", "RIGHT_PSIS",
    "LEFT_KNEE_LATERAL", "LEFT_KNEE_MEDIAL",
    "RIGHT_KNEE_LATERAL", "RIGHT_KNEE_MEDIAL",
    "LEFT_ANKLE_LATERAL", "LEFT_ANKLE_MEDIAL",
    "RIGHT_ANKLE_LATERAL", "RIGHT_ANKLE_MEDIAL",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX_MEDIAL", "LEFT_FOOT_INDEX_LATERAL",
    "RIGHT_FOOT_INDEX_MEDIAL", "RIGHT_FOOT_INDEX_LATERAL",
)

#: Default fraction of the way from the pelvis centroid toward each
#: side's ASIS used as the tier-1 hip joint centre proxy. Not a
#: regression -- see the module docstring's tier 2 for a subject-specific
#: alternative that still needs no calibration *file*, planned but not
#: implemented here.
_DEFAULT_HJC_RATIO = 0.30


# ── Geometry helpers ─────────────────────────────────────────────────


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("Zero-length or non-finite vector: cannot normalize.")
    return v / n


def _project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    return v - np.dot(v, normal) * normal


def _make_transform(x: np.ndarray, y: np.ndarray, z: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal frame as a 4x4 homogeneous transform."""
    x = _normalize(x)
    y = _normalize(_project_onto_plane(y, x))
    z = _normalize(np.cross(x, y))
    y = _normalize(np.cross(z, x))
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = np.asarray(origin, dtype=float)
    return T


# ── ISB anatomical segment frames (Wu et al. 2002) ───────────────────
#
# Axis convention throughout: x = anterior, y = proximal (points toward
# the parent segment), z = to the subject's right. Matches the
# right-handed ZXY decomposition in _joint_angles_zxy below.


def _pelvis_frame(RASIS, LASIS, RPSIS, LPSIS) -> np.ndarray:
    origin = (RASIS + LASIS) / 2.0
    z = _normalize(RASIS - LASIS)
    psis_mid = (RPSIS + LPSIS) / 2.0
    x = _normalize(_project_onto_plane(origin - psis_mid, z))
    y = _normalize(np.cross(z, x))
    x = _normalize(np.cross(y, z))
    return _make_transform(x, y, z, origin)


def _femur_frame(cond_body_left, cond_body_right, hjc) -> np.ndarray:
    """*_body_left/_body_right* = whichever condyle sits toward the
    subject's global left/right -- **not** anatomical lateral/medial.
    For a right femur that is medial=body_left, lateral=body_right; for a
    left femur it is the other way round. Passing anatomical lateral/
    medial directly (fixed order, ignoring which side the leg is on)
    mirrors the frame's z-axis on one side and silently inverts every
    angle computed from it -- caught in Step 1 testing (left side gave a
    near-perfect *negative* correlation against the validated reference).
    Callers must pick the right marker for each slot per side; see
    ``reconstruct_isb_angles``.
    """
    knee_center = (cond_body_left + cond_body_right) / 2.0
    y = _normalize(hjc - knee_center)
    z = _normalize(_project_onto_plane(cond_body_right - cond_body_left, y))
    x = _normalize(np.cross(y, z))
    z = _normalize(np.cross(x, y))
    return _make_transform(x, y, z, hjc)


def _tibia_frame(cond_body_left, cond_body_right, mal_body_left, mal_body_right) -> np.ndarray:
    """Same body-left/body-right convention as :func:`_femur_frame`, for
    both the knee and ankle marker pairs."""
    knee_center = (cond_body_left + cond_body_right) / 2.0
    ankle_center = (mal_body_left + mal_body_right) / 2.0
    y = _normalize(knee_center - ankle_center)
    z_seed = (cond_body_right - cond_body_left) + (mal_body_right - mal_body_left)
    z = _normalize(_project_onto_plane(z_seed, y))
    x = _normalize(np.cross(y, z))
    z = _normalize(np.cross(x, y))
    return _make_transform(x, y, z, knee_center)


def _foot_frame(heel, fmh_body_left, fmh_body_right, ankle_lateral, ankle_medial) -> np.ndarray:
    """*fmh_body_left/right* follow the same body-left/right convention as
    the femur/tibia frames (needed here: the forefoot z-axis is order-
    sensitive). *ankle_lateral/medial* are not order-sensitive -- only
    their symmetric average (the ankle centre/origin) is used."""
    ankle_center = (ankle_lateral + ankle_medial) / 2.0
    forefoot_center = (fmh_body_left + fmh_body_right) / 2.0
    z = _normalize(fmh_body_right - fmh_body_left)
    y = _normalize(np.cross(fmh_body_right - heel, fmh_body_left - heel))
    z = _normalize(_project_onto_plane(z, y))
    x = _normalize(np.cross(y, z))
    if np.dot(forefoot_center - ankle_center, x) < 0:
        x, z = -x, -z
    y = _normalize(np.cross(z, x))
    return _make_transform(x, y, z, ankle_center)


def _joint_angles_zxy(T_prox: np.ndarray, T_dist: np.ndarray, invert_flexion: bool = False) -> dict:
    """ISB Z-X-Y decomposition of the proximal->distal relative rotation.

    Z = flexion/extension, X = abduction/adduction, Y = internal/external
    rotation. The knee flips its flexion sign (``invert_flexion=True``) to
    keep flexion positive, matching this package's documented convention
    (``angles.py``'s module docstring) -- the raw Z-axis rotation is
    negative for knee flexion given how the tibia/femur frames above are
    built, same as every other ISB-convention implementation.
    """
    R_rel = T_prox[:3, :3].T @ T_dist[:3, :3]
    z_deg, x_deg, y_deg = _Rotation.from_matrix(R_rel).as_euler("zxy", degrees=True)
    if invert_flexion:
        z_deg = -z_deg
    return {"flex_ext_deg": float(z_deg), "abd_add_deg": float(x_deg), "int_ext_rot_deg": float(y_deg)}


def _direct_hip_joint_centers(RASIS, LASIS, RPSIS, LPSIS, ratio: float = _DEFAULT_HJC_RATIO):
    """Tier-1 HJC proxy: recomputed fresh every frame, no calibration file.

    A fixed fraction of the way from the pelvis centroid toward each
    side's ASIS. Deliberately simple -- the hip joint centre is invisible
    to any marker, and a subject-specific regression (tier 2/3) needs
    either a static trial or a VSK; this proxy needs neither.
    """
    pelvis = 0.25 * (RASIS + LASIS + RPSIS + LPSIS)
    RHJC = pelvis + ratio * (RASIS - pelvis)
    LHJC = pelvis + ratio * (LASIS - pelvis)
    return RHJC, LHJC


# ── Public API ───────────────────────────────────────────────────────


def reconstruct_isb_angles(data: dict, joints: Sequence[str] = ("hip", "knee", "ankle")) -> dict:
    """Recompute hip/knee/ankle angles from C3D markers using ISB frames.

    Tier 1 (direct, no calibration file) -- see the module docstring for
    the full tier description. Call after :func:`myogait.angles.
    compute_angles`; overwrites ``flex_ext`` in place at the existing
    ``hip_{L,R}``/``knee_{L,R}``/``ankle_{L,R}`` keys (so every existing
    reader -- charts, export, bias corrections -- keeps working
    unchanged) and additionally writes ``{joint}_{side}_abd_add_deg`` and
    ``{joint}_{side}_int_ext_rot_deg`` (new keys, nothing reads these yet
    upstream).

    Parameters
    ----------
    data : dict
        Pivot dict from :func:`myogait.experimental_vicon.load_c3d`, with
        :func:`myogait.angles.compute_angles` already run and a
        ``marker_mapping`` that resolved every landmark in
        :data:`ISB_REQUIRED_LANDMARKS` as a *separate* point.
    joints : sequence of str, optional
        Which joints to recompute (default: all three). A joint not
        requested keeps whatever ``compute_angles`` already produced.

    Returns
    -------
    dict
        The same *data* dict, modified in place and returned for
        chaining (matches ``compute_c3d_reference_angles``'s convention).

    Raises
    ------
    InsufficientLandmarksForISBError
        If ``data["c3d_markers_3d"]`` is missing any landmark in
        :data:`ISB_REQUIRED_LANDMARKS`. Catch this and fall back to the
        existing sagittal method rather than proceeding -- this is the
        gate that keeps markerless/sparse-marker sources on their
        current, correct path.
    ValueError
        If ``compute_angles()`` has not been run yet.
    """
    m3d = data.get("c3d_markers_3d")
    if not m3d:
        raise InsufficientLandmarksForISBError(
            "reconstruct_isb_angles requires data from load_c3d "
            "(missing c3d_markers_3d)."
        )
    missing = [lm for lm in ISB_REQUIRED_LANDMARKS if lm not in m3d]
    if missing:
        raise InsufficientLandmarksForISBError(
            "This file's marker_mapping does not resolve the paired "
            f"medial/lateral landmarks ISB reconstruction needs: {missing}. "
            "Fall back to compute_angles(method='sagittal_vertical_axis')."
        )
    if "angles" not in data or "frames" not in data["angles"]:
        raise ValueError("Run compute_angles() before reconstruct_isb_angles().")

    frames = data["angles"]["frames"]
    n_frames = len(frames)

    for i in range(n_frames):
        pelvis_ready = True
        try:
            RASIS = np.asarray(m3d["RIGHT_ASIS"][i], dtype=float)
            LASIS = np.asarray(m3d["LEFT_ASIS"][i], dtype=float)
            RPSIS = np.asarray(m3d["RIGHT_PSIS"][i], dtype=float)
            LPSIS = np.asarray(m3d["LEFT_PSIS"][i], dtype=float)
            if not np.isfinite([RASIS, LASIS, RPSIS, LPSIS]).all():
                pelvis_ready = False
        except (KeyError, IndexError):
            pelvis_ready = False

        if not pelvis_ready:
            _write_none(frames[i], joints)
            continue

        try:
            pelvis = _pelvis_frame(RASIS, LASIS, RPSIS, LPSIS)
            RHJC, LHJC = _direct_hip_joint_centers(RASIS, LASIS, RPSIS, LPSIS)
        except ValueError:
            _write_none(frames[i], joints)
            continue

        for side, hjc in (("R", RHJC), ("L", LHJC)):
            try:
                kl = np.asarray(m3d[f"{_SIDE_WORD[side]}_KNEE_LATERAL"][i], dtype=float)
                km = np.asarray(m3d[f"{_SIDE_WORD[side]}_KNEE_MEDIAL"][i], dtype=float)
                al = np.asarray(m3d[f"{_SIDE_WORD[side]}_ANKLE_LATERAL"][i], dtype=float)
                am = np.asarray(m3d[f"{_SIDE_WORD[side]}_ANKLE_MEDIAL"][i], dtype=float)
                heel = np.asarray(m3d[f"{_SIDE_WORD[side]}_HEEL"][i], dtype=float)
                fm = np.asarray(m3d[f"{_SIDE_WORD[side]}_FOOT_INDEX_MEDIAL"][i], dtype=float)
                fl = np.asarray(m3d[f"{_SIDE_WORD[side]}_FOOT_INDEX_LATERAL"][i], dtype=float)
                pts = (kl, km, al, am, heel, fm, fl)
                if not all(np.isfinite(p).all() for p in pts):
                    _write_none_side(frames[i], side, joints)
                    continue

                # body_left/body_right, not lateral/medial: on the right
                # leg the medial marker sits toward the body's global
                # left and the lateral marker toward its global right; on
                # the left leg it is the reverse. _femur_frame/_tibia_
                # frame/_foot_frame need the marker that is actually on
                # the body's left in the body_left slot, or their z-axis
                # (and therefore every angle) comes out mirrored on one
                # side -- see their docstrings.
                if side == "R":
                    knee_body_left, knee_body_right = km, kl
                    ankle_body_left, ankle_body_right = am, al
                    fmh_body_left, fmh_body_right = fm, fl
                else:
                    knee_body_left, knee_body_right = kl, km
                    ankle_body_left, ankle_body_right = al, am
                    fmh_body_left, fmh_body_right = fl, fm

                thigh = _femur_frame(knee_body_left, knee_body_right, hjc)
                tibia = _tibia_frame(knee_body_left, knee_body_right, ankle_body_left, ankle_body_right)
                foot = _foot_frame(heel, fmh_body_left, fmh_body_right, al, am)

                if "hip" in joints:
                    _write_joint(frames[i], "hip", side, _joint_angles_zxy(pelvis, thigh))
                if "knee" in joints:
                    _write_joint(frames[i], "knee", side, _joint_angles_zxy(thigh, tibia, invert_flexion=True))
                if "ankle" in joints:
                    _write_joint(frames[i], "ankle", side, _joint_angles_zxy(tibia, foot))
            except (KeyError, IndexError, ValueError):
                _write_none_side(frames[i], side, joints)
                continue

    data["angles"]["isb_reference"] = "isb_3d_direct"
    logger.info(
        "reconstruct_isb_angles: tier 1 (direct, no calibration file), "
        "%d frames, joints=%s", n_frames, list(joints),
    )
    return data


_SIDE_WORD = {"R": "RIGHT", "L": "LEFT"}


def _write_joint(frame: dict, joint: str, side: str, angles: dict) -> None:
    frame[f"{joint}_{side}"] = angles["flex_ext_deg"]
    frame[f"{joint}_{side}_abd_add_deg"] = angles["abd_add_deg"]
    frame[f"{joint}_{side}_int_ext_rot_deg"] = angles["int_ext_rot_deg"]


def _write_none_side(frame: dict, side: str, joints: Sequence[str]) -> None:
    for joint in joints:
        frame[f"{joint}_{side}"] = None
        frame[f"{joint}_{side}_abd_add_deg"] = None
        frame[f"{joint}_{side}_int_ext_rot_deg"] = None


def _write_none(frame: dict, joints: Sequence[str]) -> None:
    for side in ("L", "R"):
        _write_none_side(frame, side, joints)
