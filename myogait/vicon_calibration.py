"""Tier 2 and tier 3 hip-joint-centre calibration for ISB reconstruction.

:mod:`myogait.isb`'s tier 1 (no extra file) approximates the hip joint
centre (HJC) -- invisible to any marker -- with a fixed ratio from the
pelvis centroid toward each side's ASIS, recomputed fresh every frame.
That is deliberately simple and needs nothing beyond the dynamic trial
itself, but it is not subject-specific.

This module adds two calibration tiers that trade one or more extra
files for a more accurate, subject-specific HJC:

- **Tier 2** (:func:`estimate_hjc_harrington`, :func:`calibrate_hjc_from_static`):
  one extra C3D file (a static trial) is enough. The Harrington et al.
  2007 regression predicts the HJC from pelvis width, pelvis depth and
  (optionally) leg length, all measurable on that one static frame. The
  offset is computed once, in the pelvis's own local frame, so it can be
  carried through the dynamic trial by that frame's own motion --
  exactly like tier 3 below, just without a VSK/protocol.

- **Tier 3** (:func:`parse_protocol`, :func:`parse_vsk`,
  :func:`calibrate_technical_frames`, :func:`apply_technical_calibration`):
  the full CGM/Plug-in-Gait-equivalent approach. A ``.vsk`` defines a
  *technical* marker cluster per segment (which may include markers with
  no direct anatomical role, e.g. thigh wands, tracked for robustness
  during gait even when anatomical landmarks are briefly occluded); a
  rigid-body fit of that cluster is computed on a static trial and
  related, once, to the true *anatomical* frame built from landmarks
  (:mod:`myogait.isb`'s frame builders) on that same static trial. The
  resulting technical->anatomical offset is then applied every dynamic
  frame on top of a fresh rigid refit of the technical cluster.

Ref (tier 2): Harrington ME, Zavatsky AB, Lawson SEM, Yuan Z, Theologis TN.
Prediction of the hip joint centre in adults, children, and patients
with cerebral palsy based on magnetic resonance imaging. J Biomech.
2007;40(3):595-602. doi:10.1016/j.jbiomech.2006.02.003

The coefficients below were transcribed from C-Motion/HAS-Motion's
Visual3D documentation (which reproduces Harrington's published
regression for its users), not from the primary paper directly --
https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:hip_joint_landmarks
-- and cross-checked only by confirming the predicted HJC falls in the
anatomically-expected range (~50-120 mm lateral, ~30-100 mm posterior
and inferior to the ASIS-plane pelvis origin) for realistic adult
measurements. Worth an independent check against the primary paper
before relying on this tier for a clinical decision.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .isb import (
    InsufficientLandmarksForISBError,
    _femur_frame,
    _foot_frame,
    _joint_angles_zxy,
    _pelvis_frame,
    _tibia_frame,
    reconstruct_isb_angles,
)

# ── Tier 2: Harrington regression (static trial only) ─────────────────


def estimate_hjc_harrington(
    RASIS: np.ndarray,
    LASIS: np.ndarray,
    RPSIS: np.ndarray,
    LPSIS: np.ndarray,
    leg_length_mm: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Harrington et al. 2007 HJC regression, world-space output.

    Uses the leg-length equation set (Harrington's more accurate of the
    two, ~14-17 mm reported error) when ``leg_length_mm`` is given,
    otherwise the ASIS-distance/pelvis-depth-only set. ``leg_length_mm``
    can itself be measured on the static trial (e.g. ASIS to lateral
    malleolus) -- still no ``.vsk``/``.prot`` needed.

    Parameters
    ----------
    RASIS, LASIS, RPSIS, LPSIS : ndarray, shape (3,)
        Pelvis landmark positions, millimetres, one static frame (or a
        frame-averaged position -- see :func:`calibrate_hjc_from_static`).
    leg_length_mm : float, optional
        Measured leg length. Omit to use the ASIS-distance-only
        equation set.

    Returns
    -------
    (RHJC, LHJC) : tuple of ndarray, shape (3,)
        World-space hip joint centre positions, millimetres, in the same
        frame the input markers were given in.
    """
    asis_distance_mm = float(np.linalg.norm(RASIS - LASIS))
    pelvis_depth_mm = float(np.linalg.norm(
        (RASIS + LASIS) / 2.0 - (RPSIS + LPSIS) / 2.0
    ))

    # Published in metres; convert, apply, convert back.
    asis_m = asis_distance_mm / 1000.0
    depth_m = pelvis_depth_mm / 1000.0

    if leg_length_mm is not None:
        leg_m = leg_length_mm / 1000.0
        ml_m = 0.28 * depth_m + 0.16 * asis_m + 0.0079
        ap_m = -0.24 * depth_m - 0.0099
        ax_m = -0.16 * asis_m - 0.04 * leg_m - 0.0071
    else:
        ml_m = 0.33 * asis_m + 0.0073
        ap_m = -0.24 * depth_m - 0.0099
        ax_m = -0.30 * asis_m - 0.0109

    # ML/AP/Axial -> this package's pelvis-frame axes (isb.py's
    # _pelvis_frame: x=anterior, y=proximal/up, z=to the subject's
    # right). AP and Axial are negative for both sides (the HJC sits
    # posterior and inferior to the ASIS-midpoint origin); ML flips sign
    # between sides (lateral is +z on the right, -z on the left).
    pelvis = _pelvis_frame(RASIS, LASIS, RPSIS, LPSIS)
    origin = pelvis[:3, 3]
    x_axis, y_axis, z_axis = pelvis[:3, 0], pelvis[:3, 1], pelvis[:3, 2]

    offset_mm_right = (ap_m * x_axis + ax_m * y_axis + ml_m * z_axis) * 1000.0
    offset_mm_left = (ap_m * x_axis + ax_m * y_axis - ml_m * z_axis) * 1000.0

    return origin + offset_mm_right, origin + offset_mm_left


@dataclass(frozen=True)
class HjcCalibration:
    """A static-trial-derived HJC, expressed in the pelvis's own local
    frame so it can be carried through a dynamic trial by that frame's
    own motion (``pelvis_frame(t) @ local_offset``), the same principle
    tier 3 uses for the whole technical cluster."""

    right_local: np.ndarray  # homogeneous local position, shape (4,)
    left_local: np.ndarray


def calibrate_hjc_from_static(
    static_markers_3d: dict, leg_length_mm: Optional[float] = None
) -> HjcCalibration:
    """Run :func:`estimate_hjc_harrington` on a static trial's mean
    marker positions and express the result in the pelvis's own local
    frame for later use with :func:`hjc_from_calibration`.

    Parameters
    ----------
    static_markers_3d : dict
        ``{"LEFT_ASIS": (n_frames, 3), ...}`` -- same shape
        :func:`myogait.experimental_vicon.load_c3d` produces in
        ``c3d_markers_3d``, but for a *static* trial's C3D file.
    leg_length_mm : float, optional
        See :func:`estimate_hjc_harrington`.
    """
    required_markers = ("RIGHT_ASIS", "LEFT_ASIS", "RIGHT_PSIS", "LEFT_PSIS")
    mean = {
        name: _mean_static_marker(static_markers_3d, name)
        for name in required_markers
    }
    RHJC, LHJC = estimate_hjc_harrington(
        mean["RIGHT_ASIS"], mean["LEFT_ASIS"], mean["RIGHT_PSIS"], mean["LEFT_PSIS"],
        leg_length_mm=leg_length_mm,
    )
    pelvis_static = _pelvis_frame(mean["RIGHT_ASIS"], mean["LEFT_ASIS"], mean["RIGHT_PSIS"], mean["LEFT_PSIS"])
    inv = np.linalg.inv(pelvis_static)
    right_local = inv @ np.append(RHJC, 1.0)
    left_local = inv @ np.append(LHJC, 1.0)
    return HjcCalibration(right_local=right_local, left_local=left_local)


def _mean_static_marker(static_markers_3d: dict, marker_name: str) -> np.ndarray:
    """Return a finite mean marker position from a static trial.

    C3D files commonly encode an occluded marker as rows of ``NaN``.  A
    direct ``nanmean`` would then return another ``NaN`` and defer the useful
    error until an unrelated matrix operation.  Reject that input at the
    calibration boundary instead.  A single ``(3,)`` position is accepted as
    a convenience for callers that have already averaged the static trial.
    """
    if marker_name not in static_markers_3d:
        raise ValueError(f"Static HJC calibration requires marker {marker_name!r}.")

    positions = np.asarray(static_markers_3d[marker_name], dtype=float)
    if positions.shape == (3,):
        positions = positions.reshape(1, 3)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Static marker {marker_name!r} must have shape (n_frames, 3); "
            f"got {positions.shape}."
        )

    finite_positions = positions[np.isfinite(positions).all(axis=1)]
    if not len(finite_positions):
        raise ValueError(
            f"Static HJC calibration cannot use {marker_name!r}: no finite frames are available."
        )
    return finite_positions.mean(axis=0)


def hjc_from_calibration(calibration: HjcCalibration, pelvis_dynamic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry a static-derived HJC (see :func:`calibrate_hjc_from_static`)
    through to one dynamic frame's own pelvis pose."""
    RHJC = (pelvis_dynamic @ calibration.right_local)[:3]
    LHJC = (pelvis_dynamic @ calibration.left_local)[:3]
    return RHJC, LHJC


# ── Tier 3: VSK/protocol technical-cluster calibration ─────────────────


def parse_protocol(prot_path) -> dict[str, dict]:
    """Parse a ``.prot`` protocol file into ``{section: {key: value}}``.

    Generic ``#SECTION`` / ``key=value`` (comma-separated values become a
    list) text format -- makes no assumption about which segments or
    articulations a given protocol defines.
    """
    sections: dict[str, dict] = {}
    current = None
    text = Path(prot_path).read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line[1:].strip()
            sections[current] = {}
            continue
        if current is None or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        sections[current][key] = [v.strip() for v in value.split(",")] if "," in value else value
    return sections


@dataclass
class VSKData:
    segment_markers_local: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    joint_balls_local: dict[str, np.ndarray] = field(default_factory=dict)


def parse_vsk(vsk_path) -> VSKData:
    """Parse a Vicon ``.vsk`` skeleton file's segment marker templates
    and joint-ball (virtual point) definitions.

    Only reads what :func:`calibrate_technical_frames` needs: each
    segment's local marker-template positions (``TargetLocalPointToWorldPoint``)
    and any named joint balls (``JointBall``, e.g. a Vicon-calibrated,
    subject-specific hip joint centre already baked into the file). Does
    not assume any particular segment names -- whatever the VSK defines
    is what comes back.
    """
    tree = ET.parse(vsk_path)
    root = tree.getroot()

    params: dict[str, float] = {}
    for p in root.findall(".//Parameters/Parameter"):
        try:
            params[p.attrib["NAME"]] = float(p.attrib["VALUE"])
        except (KeyError, ValueError):
            continue

    segment_markers_local: dict[str, dict[str, np.ndarray]] = {}
    for target in root.findall(".//TargetLocalPointToWorldPoint"):
        marker = target.attrib.get("MARKER")
        segment = target.attrib.get("SEGMENT")
        pos = target.attrib.get("POSITION")
        if not marker or not segment or not pos:
            continue
        refs = re.findall(r"'([^']+)'", pos)
        if len(refs) != 3 or any(r not in params for r in refs):
            continue
        segment_markers_local.setdefault(segment, {})[marker] = np.array(
            [params[r] for r in refs], dtype=float
        )

    joint_balls_local: dict[str, np.ndarray] = {}
    for jb in root.findall(".//JointBall"):
        name = jb.attrib.get("NAME")
        pos = jb.attrib.get("PRE-POSITION")
        if name and pos:
            joint_balls_local[name] = np.fromstring(pos, sep=" ", dtype=float)

    return VSKData(segment_markers_local=segment_markers_local, joint_balls_local=joint_balls_local)


def technical_frame_from_vsk(
    segment_name: str, segment_markers_local: dict[str, dict[str, np.ndarray]],
    world_markers: dict[str, np.ndarray], min_points: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Rigid-body (Kabsch) fit of a VSK segment's marker template to
    wherever those markers actually are this frame.

    Returns the fitted 4x4 transform and which markers were actually
    used (some may be occluded this frame, or simply absent from the
    file -- as long as at least ``min_points`` common ones remain).

    Raises
    ------
    KeyError
        If *segment_name* is not defined in the VSK.
    ValueError
        If fewer than *min_points* of that segment's markers are present
        in *world_markers* this frame.
    """
    if segment_name not in segment_markers_local:
        raise KeyError(f"Segment {segment_name!r} is not defined in this VSK.")
    local = segment_markers_local[segment_name]
    common = [m for m in local if m in world_markers and np.isfinite(world_markers[m]).all()]
    if len(common) < min_points:
        raise ValueError(
            f"Segment {segment_name!r}: only {len(common)} of its VSK markers "
            f"are present this frame ({min_points} required)."
        )

    A = np.array([local[m] for m in common], dtype=float)
    B = np.array([world_markers[m] for m in common], dtype=float)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - ca, B - cb
    U, _, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, common


#: Segment name -> the two anatomical-frame builder's positional marker
#: roles it needs, expressed as canonical landmark names (see
#: myogait.isb.ISB_REQUIRED_LANDMARKS). Drives calibrate_technical_frames
#: so it works for whatever segment names a given VSK/protocol actually
#: defines -- pass a different map for a VSK that names its segments
#: differently, rather than hardcoding one lab's naming.
DEFAULT_SEGMENT_LANDMARKS: dict[str, tuple[str, ...]] = {
    "pelvis": ("RIGHT_ASIS", "LEFT_ASIS", "RIGHT_PSIS", "LEFT_PSIS"),
    "RightThigh": ("RIGHT_KNEE_LATERAL", "RIGHT_KNEE_MEDIAL"),
    "LeftThigh": ("LEFT_KNEE_LATERAL", "LEFT_KNEE_MEDIAL"),
    "RightTibia": ("RIGHT_KNEE_LATERAL", "RIGHT_KNEE_MEDIAL", "RIGHT_ANKLE_LATERAL", "RIGHT_ANKLE_MEDIAL"),
    "LeftTibia": ("LEFT_KNEE_LATERAL", "LEFT_KNEE_MEDIAL", "LEFT_ANKLE_LATERAL", "LEFT_ANKLE_MEDIAL"),
    "RightFoot": ("RIGHT_HEEL", "RIGHT_FOOT_INDEX_MEDIAL", "RIGHT_FOOT_INDEX_LATERAL", "RIGHT_ANKLE_LATERAL", "RIGHT_ANKLE_MEDIAL"),
    "LeftFoot": ("LEFT_HEEL", "LEFT_FOOT_INDEX_MEDIAL", "LEFT_FOOT_INDEX_LATERAL", "LEFT_ANKLE_LATERAL", "LEFT_ANKLE_MEDIAL"),
}

#: pelvis -> {"RightThigh": vsk joint-ball name, "LeftThigh": ...}. Vicon
#: VSKs store the subject-specific HJC as a joint ball named after the
#: segment pair it connects (e.g. Myokinesis's own VSK uses
#: "pelvis_RightThigh"/"pelvis_LeftThigh"). Override if a VSK names them
#: differently.
DEFAULT_HJC_JOINT_BALLS = {"R": "pelvis_RightThigh", "L": "pelvis_LeftThigh"}


def _anatomical_frame(segment: str, landmarks: dict[str, np.ndarray], hjc: dict[str, np.ndarray]) -> np.ndarray:
    if segment == "pelvis":
        return _pelvis_frame(landmarks["RIGHT_ASIS"], landmarks["LEFT_ASIS"], landmarks["RIGHT_PSIS"], landmarks["LEFT_PSIS"])
    if segment == "RightThigh":
        return _femur_frame(landmarks["RIGHT_KNEE_MEDIAL"], landmarks["RIGHT_KNEE_LATERAL"], hjc["R"])
    if segment == "LeftThigh":
        return _femur_frame(landmarks["LEFT_KNEE_LATERAL"], landmarks["LEFT_KNEE_MEDIAL"], hjc["L"])
    if segment == "RightTibia":
        return _tibia_frame(landmarks["RIGHT_KNEE_MEDIAL"], landmarks["RIGHT_KNEE_LATERAL"], landmarks["RIGHT_ANKLE_MEDIAL"], landmarks["RIGHT_ANKLE_LATERAL"])
    if segment == "LeftTibia":
        return _tibia_frame(landmarks["LEFT_KNEE_LATERAL"], landmarks["LEFT_KNEE_MEDIAL"], landmarks["LEFT_ANKLE_LATERAL"], landmarks["LEFT_ANKLE_MEDIAL"])
    if segment == "RightFoot":
        return _foot_frame(landmarks["RIGHT_HEEL"], landmarks["RIGHT_FOOT_INDEX_MEDIAL"], landmarks["RIGHT_FOOT_INDEX_LATERAL"], landmarks["RIGHT_ANKLE_LATERAL"], landmarks["RIGHT_ANKLE_MEDIAL"])
    if segment == "LeftFoot":
        return _foot_frame(landmarks["LEFT_HEEL"], landmarks["LEFT_FOOT_INDEX_LATERAL"], landmarks["LEFT_FOOT_INDEX_MEDIAL"], landmarks["LEFT_ANKLE_LATERAL"], landmarks["LEFT_ANKLE_MEDIAL"])
    raise KeyError(segment)


@dataclass
class TechnicalCalibration:
    """Per-segment technical->anatomical offset from a static trial,
    plus the VSK data needed to refit each segment's technical frame on
    a dynamic trial. See :func:`calibrate_technical_frames`."""

    vsk: VSKData
    offsets: dict[str, np.ndarray]  # segment -> 4x4
    segments: tuple[str, ...]


def calibrate_technical_frames(
    vsk: VSKData,
    static_markers_raw: dict[str, np.ndarray],
    static_landmarks_3d: dict[str, np.ndarray],
    segment_landmarks: dict[str, tuple[str, ...]] = DEFAULT_SEGMENT_LANDMARKS,
    hjc_joint_balls: dict[str, str] = DEFAULT_HJC_JOINT_BALLS,
) -> TechnicalCalibration:
    """Compute the technical->anatomical offset for every VSK segment
    from one static trial (tier 3's one-time calibration step).

    Parameters
    ----------
    vsk : VSKData
        From :func:`parse_vsk`.
    static_markers_raw : dict
        Every marker in the static trial's C3D, keyed by its *original*
        label (not a resolved landmark name) and mean-position-only
        (shape (3,)) -- the VSK's technical clusters can reference any
        marker in the file, not just the canonical ISB ones.
    static_landmarks_3d : dict
        The canonical-landmark-keyed mean positions (ISB_REQUIRED_LANDMARKS)
        for this same static trial -- used to build the *anatomical*
        frames the technical clusters are calibrated against.
    """
    hjc = {}
    # Joint balls are keyed by segment-pair name (e.g. "pelvis_RightThigh"),
    # never by the bare segment name "pelvis" -- check for the *specific*
    # ball names this expects, not a key that never actually occurs.
    if "pelvis" in vsk.segment_markers_local and any(
        name in vsk.joint_balls_local for name in hjc_joint_balls.values()
    ):
        pelvis_tech, _ = technical_frame_from_vsk("pelvis", vsk.segment_markers_local, static_markers_raw)
        for side, ball_name in hjc_joint_balls.items():
            if ball_name in vsk.joint_balls_local:
                hjc[side] = (pelvis_tech @ np.append(vsk.joint_balls_local[ball_name], 1.0))[:3]
    if "R" not in hjc or "L" not in hjc:
        # No VSK joint ball for one or both hips -- fall back to the
        # Harrington regression (tier 2's estimator) rather than failing
        # tier 3 outright over a missing virtual point.
        r_hjc, l_hjc = estimate_hjc_harrington(
            static_landmarks_3d["RIGHT_ASIS"], static_landmarks_3d["LEFT_ASIS"],
            static_landmarks_3d["RIGHT_PSIS"], static_landmarks_3d["LEFT_PSIS"],
        )
        hjc.setdefault("R", r_hjc)
        hjc.setdefault("L", l_hjc)

    offsets: dict[str, np.ndarray] = {}
    segments_calibrated: list[str] = []
    for segment in segment_landmarks:
        if segment not in vsk.segment_markers_local:
            continue
        try:
            technical, _ = technical_frame_from_vsk(segment, vsk.segment_markers_local, static_markers_raw)
        except (KeyError, ValueError):
            continue
        landmarks = {lm: static_landmarks_3d[lm] for lm in segment_landmarks[segment] if lm in static_landmarks_3d}
        if len(landmarks) < len(segment_landmarks[segment]):
            continue
        anatomical = _anatomical_frame(segment, landmarks, hjc)
        offsets[segment] = np.linalg.inv(technical) @ anatomical
        segments_calibrated.append(segment)

    return TechnicalCalibration(vsk=vsk, offsets=offsets, segments=tuple(segments_calibrated))


def apply_technical_calibration(calibration: TechnicalCalibration, world_markers_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One dynamic frame: refit each calibrated segment's technical
    cluster and apply its static offset to get that frame's anatomical
    pose. Raises the same (KeyError, ValueError) as
    :func:`technical_frame_from_vsk` for a segment whose markers are
    unavailable this frame -- callers should catch and skip the frame,
    same convention as :func:`myogait.isb.reconstruct_isb_angles`."""
    out: dict[str, np.ndarray] = {}
    for segment in calibration.segments:
        technical, _ = technical_frame_from_vsk(segment, calibration.vsk.segment_markers_local, world_markers_raw)
        out[segment] = technical @ calibration.offsets[segment]
    return out


# ── Raw C3D marker access (needed for tier 3's technical clusters) ────
#
# myogait.experimental_vicon.load_c3d only keeps the landmarks a
# marker_mapping resolves, discarding every other marker in the file --
# correct for tier 1/2 (canonical landmarks only), but tier 3's VSK
# technical clusters can reference *any* marker (e.g. thigh wands with
# no anatomical-landmark role at all), so this needs the full raw set.


def load_raw_c3d_markers(c3d_path) -> tuple[dict[str, np.ndarray], float]:
    """Every POINT marker in a C3D file, by its original label.

    Same "(0,0,0) is occlusion" convention load_c3d itself applies
    (some exporters, seen on at least one real open dataset, mark a gap
    with an exact-zero triplet rather than a negative POINT residual).
    """
    import ezc3d

    c3d = ezc3d.c3d(str(c3d_path))
    labels = [lbl.strip() for lbl in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    points = c3d["data"]["points"]  # (4, n_markers, n_frames)
    fps = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])

    markers: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        xyz = points[:3, i, :].T.astype(float, copy=True)  # (n_frames, 3)
        residual = points[3, i, :]
        zero = np.all(xyz == 0.0, axis=1)
        xyz[(residual < 0) | zero, :] = np.nan
        markers[label] = xyz
    return markers, fps


def _mean_markers(markers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for name, arr in markers.items():
        with np.errstate(invalid="ignore"):
            out[name] = np.nanmean(arr, axis=0)
    return out


# ── Tier 2/3 top-level drivers ──────────────────────────────────────


def reconstruct_isb_angles_tier2(
    data: dict,
    static_markers_3d: dict[str, np.ndarray],
    joints=("hip", "knee", "ankle"),
    leg_length_mm: Optional[float] = None,
) -> dict:
    """Tier 2: :func:`myogait.isb.reconstruct_isb_angles`, with the hip
    joint centre from a Harrington regression calibrated on a static
    trial instead of the tier-1 fixed-ratio proxy.

    Parameters
    ----------
    data : dict
        A dynamic-trial pivot, exactly as :func:`myogait.isb.
        reconstruct_isb_angles` expects (``compute_angles`` already run).
    static_markers_3d : dict
        The *static* trial's ``c3d_markers_3d`` (i.e. run ``load_c3d``
        and ``compute_angles`` on the static C3D too, with the same
        enriched ``marker_mapping``, and pass its ``c3d_markers_3d``
        here -- only the pelvis landmarks are actually read).
    leg_length_mm : float, optional
        See :func:`estimate_hjc_harrington`.
    """
    calibration = calibrate_hjc_from_static(static_markers_3d, leg_length_mm=leg_length_mm)

    def _hjc_fn(RASIS, LASIS, RPSIS, LPSIS, pelvis_frame):
        return hjc_from_calibration(calibration, pelvis_frame)

    data = reconstruct_isb_angles(data, joints=joints, hjc_fn=_hjc_fn)
    data["angles"]["isb_reference"] = "isb_3d_tier2_static_hjc"
    return data


def reconstruct_isb_angles_tier3(
    data: dict,
    dynamic_markers_raw: dict[str, np.ndarray],
    calibration: TechnicalCalibration,
    joints=("hip", "knee", "ankle"),
) -> dict:
    """Tier 3: full VSK+static-calibrated technical-cluster reconstruction.

    Unlike tiers 1/2, this does not reuse
    :func:`myogait.isb.reconstruct_isb_angles`'s loop -- every segment
    frame (not just the hip joint centre) comes from the calibrated
    technical cluster, not from the landmark-based frame builders
    directly -- but it shares their anatomical-frame math via
    :func:`myogait.isb._joint_angles_zxy` for the final decomposition,
    so the two tiers report angles in exactly the same convention.

    Parameters
    ----------
    data : dict
        A dynamic-trial pivot with ``compute_angles`` already run
        (only ``data["angles"]["frames"]`` is written to; the ISB
        reconstruction here works entirely from *dynamic_markers_raw*,
        not from ``data["c3d_markers_3d"]``).
    dynamic_markers_raw : dict
        Every marker in the dynamic trial's C3D, by original label --
        see :func:`load_raw_c3d_markers`. Must be frame-aligned with
        *data* (same C3D file, same frame count).
    calibration : TechnicalCalibration
        From :func:`calibrate_technical_frames`.
    """
    if "angles" not in data or "frames" not in data["angles"]:
        raise ValueError("Run compute_angles() before reconstruct_isb_angles_tier3().")
    missing_segments = {"pelvis", "RightThigh", "LeftThigh", "RightTibia", "LeftTibia", "RightFoot", "LeftFoot"} - set(calibration.segments)
    if missing_segments:
        raise InsufficientLandmarksForISBError(
            f"This VSK/static calibration did not resolve: {sorted(missing_segments)}. "
            "Fall back to a lower tier."
        )

    frames = data["angles"]["frames"]
    n_frames = len(frames)
    segment_names = {"R_hip": ("pelvis", "RightThigh"), "L_hip": ("pelvis", "LeftThigh"),
                      "R_knee": ("RightThigh", "RightTibia"), "L_knee": ("LeftThigh", "LeftTibia"),
                      "R_ankle": ("RightTibia", "RightFoot"), "L_ankle": ("LeftTibia", "LeftFoot")}
    invert_flexion = {"R_knee", "L_knee"}

    from .isb import _write_joint, _write_none_side

    for i in range(n_frames):
        world_frame = {}
        for name, arr in dynamic_markers_raw.items():
            if i < len(arr):
                world_frame[name] = arr[i]
        try:
            anatomical = apply_technical_calibration(calibration, world_frame)
        except (KeyError, ValueError):
            for side in ("L", "R"):
                _write_none_side(frames[i], side, joints)
            continue

        for key, (prox, dist) in segment_names.items():
            side, joint = key.split("_")
            if joint not in joints:
                continue
            if prox not in anatomical or dist not in anatomical:
                _write_none_side(frames[i], side, (joint,))
                continue
            angles = _joint_angles_zxy(anatomical[prox], anatomical[dist], invert_flexion=key in invert_flexion)
            _write_joint(frames[i], joint, side, angles)

    data["angles"]["isb_reference"] = "isb_3d_tier3_calibrated"
    return data
