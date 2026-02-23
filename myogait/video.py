"""Video rendering utilities for skeleton overlay and stick-figure animation.

Provides functions to overlay pose landmarks on video frames,
render full skeleton videos, and generate anonymized stick-figure
animations (GIF or MP4).

Functions
---------
render_skeleton_frame
    Draw landmarks and skeleton connections on a single image.
render_skeleton_video
    Overlay skeleton on every frame of a source video.
render_stickfigure_animation
    Generate an anonymized stick-figure animation (GIF/MP4).
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    GOLIATH_LANDMARK_NAMES,
    GOLIATH_SKELETON_CONNECTIONS,
    GOLIATH_FACE_START,
)

logger = logging.getLogger(__name__)

# Skeleton connections defined as pairs of landmark names
SKELETON_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    ("NOSE", "LEFT_EYE"), ("NOSE", "RIGHT_EYE"),
    ("LEFT_EAR", "LEFT_EYE"), ("RIGHT_EAR", "RIGHT_EYE"),
]

# Color constants (BGR for OpenCV)
_COLOR_LEFT = (255, 100, 0)     # Blue (left side)
_COLOR_RIGHT = (0, 0, 255)     # Red (right side)
_COLOR_CENTER = (0, 200, 0)    # Green (center)
_COLOR_WHITE = (255, 255, 255)


def _side_color(name: str) -> Tuple[int, int, int]:
    """Return BGR color based on landmark side."""
    if name.startswith("LEFT_"):
        return _COLOR_LEFT
    elif name.startswith("RIGHT_"):
        return _COLOR_RIGHT
    return _COLOR_CENTER


def _connection_color(name_a: str, name_b: str) -> Tuple[int, int, int]:
    """Return BGR color for a skeleton connection based on its endpoints."""
    if name_a.startswith("LEFT_") and name_b.startswith("LEFT_"):
        return _COLOR_LEFT
    elif name_a.startswith("RIGHT_") and name_b.startswith("RIGHT_"):
        return _COLOR_RIGHT
    return _COLOR_CENTER


def _goliath_side_color(idx: int) -> Tuple[int, int, int]:
    """Return BGR color for a Goliath landmark based on its name prefix."""
    name = GOLIATH_LANDMARK_NAMES[idx]
    if name.startswith("left_") or name.startswith("l_"):
        return _COLOR_LEFT
    elif name.startswith("right_") or name.startswith("r_"):
        return _COLOR_RIGHT
    return _COLOR_CENTER


def _goliath_conn_color(idx_a: int, idx_b: int) -> Tuple[int, int, int]:
    """Return BGR color for a Goliath connection based on endpoint names."""
    na = GOLIATH_LANDMARK_NAMES[idx_a]
    nb = GOLIATH_LANDMARK_NAMES[idx_b]
    a_left = na.startswith("left_") or na.startswith("l_")
    b_left = nb.startswith("left_") or nb.startswith("l_")
    a_right = na.startswith("right_") or na.startswith("r_")
    b_right = nb.startswith("right_") or nb.startswith("r_")
    if a_left and b_left:
        return _COLOR_LEFT
    if a_right and b_right:
        return _COLOR_RIGHT
    return _COLOR_CENTER


def render_skeleton_frame(
    frame_image: np.ndarray,
    landmarks: dict,
    angles: Optional[dict] = None,
    events: Optional[dict] = None,
    skeleton_color: str = "auto",
    goliath308: Optional[list] = None,
) -> np.ndarray:
    """Draw landmarks and skeleton connections on an image.

    Parameters
    ----------
    frame_image : np.ndarray
        BGR image (H, W, 3).
    landmarks : dict
        Mapping of landmark name to dict with ``'x'``, ``'y'`` (normalised
        0-1) and optionally ``'visibility'``.
    angles : dict, optional
        Angle values keyed by joint name (e.g. ``'hip_L'``, ``'knee_R'``).
        When provided, angle values are annotated next to the joints.
    events : dict, optional
        Event information for this frame. Expected keys:
        ``'type'`` (``'HS'`` or ``'TO'``), ``'side'`` (``'left'``/``'right'``).
    skeleton_color : str
        ``'auto'`` colours by side (left=blue, right=red, centre=green).
        Any other value is interpreted as a single BGR tuple string
        (not currently used -- falls back to auto).
    goliath308 : list, optional
        List of 308 ``[x, y, confidence]`` triplets (Goliath keypoints).
        When provided, renders using all 308 keypoints and
        ``GOLIATH_SKELETON_CONNECTIONS`` instead of the MediaPipe skeleton.

    Returns
    -------
    np.ndarray
        Copy of *frame_image* with skeleton drawn on it.
    """
    frame = frame_image.copy()
    h, w = frame.shape[:2]

    if goliath308 is not None:
        return _render_goliath_on_frame(
            frame, goliath308, h, w, skeleton_color, angles, events,
        )

    # Convert normalised coords to pixel coords
    pts: Dict[str, Tuple[int, int]] = {}
    vis: Dict[str, float] = {}
    for name, lm in landmarks.items():
        x = lm.get("x")
        y = lm.get("y")
        if x is None or y is None:
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        px = int(x * w)
        py = int(y * h)
        pts[name] = (px, py)
        vis[name] = lm.get("visibility", 1.0)

    # Draw connections
    for name_a, name_b in SKELETON_CONNECTIONS:
        if name_a in pts and name_b in pts:
            if skeleton_color == "auto":
                color = _connection_color(name_a, name_b)
            else:
                color = _COLOR_CENTER
            cv2.line(frame, pts[name_a], pts[name_b], color, 2, cv2.LINE_AA)

    # Draw landmarks (circles)
    for name, pt in pts.items():
        if skeleton_color == "auto":
            color = _side_color(name)
        else:
            color = _COLOR_CENTER
        radius = 4
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # Annotate angles next to joints
    if angles:
        _angle_joint_map = {
            "hip_L": "LEFT_HIP",
            "hip_R": "RIGHT_HIP",
            "knee_L": "LEFT_KNEE",
            "knee_R": "RIGHT_KNEE",
            "ankle_L": "LEFT_ANKLE",
            "ankle_R": "RIGHT_ANKLE",
        }
        for angle_name, joint_name in _angle_joint_map.items():
            val = angles.get(angle_name)
            if val is not None and joint_name in pts:
                px, py = pts[joint_name]
                label = f"{val:.0f} deg"
                cv2.putText(
                    frame, label, (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_WHITE, 1, cv2.LINE_AA,
                )

    # Show event indicator
    if events:
        ev_type = events.get("type", "")
        ev_side = events.get("side", "")
        label = f"{ev_type} ({ev_side})"
        color = _COLOR_LEFT if ev_side == "left" else _COLOR_RIGHT
        cv2.putText(
            frame, label, (w // 2 - 40, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )

    return frame


def _render_goliath_on_frame(
    frame: np.ndarray,
    goliath308: list,
    h: int,
    w: int,
    skeleton_color: str,
    angles: Optional[dict],
    events: Optional[dict],
) -> np.ndarray:
    """Draw Goliath 308 keypoints and connections on a frame."""
    n_pts = len(goliath308)

    # Build pixel coordinate array
    px_pts: Dict[int, Tuple[int, int]] = {}
    for idx in range(n_pts):
        pt = goliath308[idx]
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = float(pt[0]), float(pt[1])
        conf = float(pt[2]) if len(pt) >= 3 else 0.0
        if np.isnan(x) or np.isnan(y) or conf < 0.1:
            continue
        px_pts[idx] = (int(x * w), int(y * h))

    # Draw connections
    for idx_a, idx_b in GOLIATH_SKELETON_CONNECTIONS:
        if idx_a in px_pts and idx_b in px_pts:
            if skeleton_color == "auto":
                color = _goliath_conn_color(idx_a, idx_b)
            else:
                color = _COLOR_CENTER
            cv2.line(frame, px_pts[idx_a], px_pts[idx_b], color, 2, cv2.LINE_AA)

    # Draw landmarks — body/hands/feet (0-69) larger, face (70+) smaller
    for idx, pt in px_pts.items():
        if skeleton_color == "auto":
            color = _goliath_side_color(idx)
        else:
            color = _COLOR_CENTER
        radius = 3 if idx < GOLIATH_FACE_START else 1
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # Angles (reuse COCO-style joint map via Goliath indices)
    if angles:
        _goliath_angle_map = {
            "hip_L": 9, "hip_R": 10,
            "knee_L": 11, "knee_R": 12,
            "ankle_L": 13, "ankle_R": 14,
        }
        for angle_name, gidx in _goliath_angle_map.items():
            val = angles.get(angle_name)
            if val is not None and gidx in px_pts:
                px, py = px_pts[gidx]
                label = f"{val:.0f} deg"
                cv2.putText(
                    frame, label, (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_WHITE, 1, cv2.LINE_AA,
                )

    # Events
    if events:
        ev_type = events.get("type", "")
        ev_side = events.get("side", "")
        label = f"{ev_type} ({ev_side})"
        color = _COLOR_LEFT if ev_side == "left" else _COLOR_RIGHT
        cv2.putText(
            frame, label, (w // 2 - 40, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )

    return frame


def render_skeleton_video(
    video_path: str,
    data: dict,
    output_path: str,
    show_angles: bool = False,
    show_events: bool = False,
    show_confidence: bool = False,
    skeleton_color: str = "auto",
    fps: Optional[float] = None,
    codec: str = "mp4v",
    use_goliath: bool = False,
    min_confidence: float = 0.0,
) -> str:
    """Overlay skeleton on every frame of a source video.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    data : dict
        Pivot JSON dict with ``frames`` populated (and optionally
        ``angles`` and ``events``).
    output_path : str
        Destination path for the rendered video.
    show_angles : bool
        Annotate joint angles on each frame.
    show_events : bool
        Show gait event indicators (HS/TO).
    show_confidence : bool
        Modulate landmark circle size / line thickness by visibility.
    skeleton_color : str
        ``'auto'`` or a fixed colour mode.
    fps : float, optional
        Output FPS. Defaults to the source video FPS.
    codec : str
        FourCC codec string (default ``'mp4v'``).
    use_goliath : bool
        When ``True``, render using the full Goliath 308 keypoints stored
        in ``frame["goliath308"]`` instead of the 33 MediaPipe landmarks.
    min_confidence : float
        Skip overlay on frames whose ``confidence`` is below this
        threshold (the raw video frame is still written).

    Returns
    -------
    str
        The *output_path* written.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    ValueError
        If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps if fps is not None else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    frames_data = data.get("frames", [])
    angles_data = data.get("angles", {})
    angle_frames = angles_data.get("frames", []) if angles_data else []

    # Build event lookup: frame_idx -> event info
    event_lookup: Dict[int, dict] = {}
    if show_events:
        events_dict = data.get("events", {})
        if events_dict:
            for key in ["left_hs", "right_hs", "left_to", "right_to"]:
                ev_list = events_dict.get(key, [])
                side = "left" if key.startswith("left") else "right"
                ev_type = "HS" if key.endswith("_hs") else "TO"
                for ev in ev_list:
                    fidx = ev.get("frame")
                    if fidx is not None:
                        event_lookup[fidx] = {"type": ev_type, "side": side}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(frames_data):
            fd = frames_data[frame_idx]

            # Skip low-confidence frames (write raw frame without overlay)
            if fd.get("confidence", 0.0) < min_confidence:
                writer.write(frame)
                frame_idx += 1
                continue

            lm = fd.get("landmarks", {})

            # Angles for this frame
            frame_angles = None
            if show_angles and frame_idx < len(angle_frames):
                frame_angles = angle_frames[frame_idx]

            # Events for this frame
            frame_events = event_lookup.get(frame_idx) if show_events else None

            # Optionally modulate by confidence
            if show_confidence:
                # Adjust visibility to modulate rendering
                lm_copy = {}
                for name, val in lm.items():
                    lm_copy[name] = dict(val)
                lm = lm_copy

            # Goliath 308 rendering
            goliath_data = fd.get("goliath308") if use_goliath else None

            frame = render_skeleton_frame(
                frame, lm,
                angles=frame_angles,
                events=frame_events,
                skeleton_color=skeleton_color,
                goliath308=goliath_data,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info(f"Skeleton video written to {output_path} ({frame_idx} frames)")
    return output_path


def render_stickfigure_animation(
    data: dict,
    output_path: str,
    format: str = "gif",
    figsize: Tuple[float, float] = (6, 8),
    fps: Optional[float] = None,
    show_angles: bool = False,
    show_trail: bool = False,
    background_color: str = "white",
    cycles: Optional[dict] = None,
    use_goliath: bool = False,
    min_confidence: float = 0.0,
) -> str:
    """Generate an anonymized stick-figure animation.

    Creates a clean stick-figure rendering on a plain background,
    suitable for publications and presentations where video privacy
    is a concern.

    Parameters
    ----------
    data : dict
        Pivot JSON dict with ``frames`` populated.
    output_path : str
        Destination file path.
    format : str
        ``'gif'`` or ``'mp4'`` (default ``'gif'``).
    figsize : tuple
        Matplotlib figure size ``(width, height)`` in inches.
    fps : float, optional
        Animation frame rate. Defaults to ``data['meta']['fps']`` or 30.
    show_angles : bool
        Annotate angle values next to joints.
    show_trail : bool
        Show trailing positions with decreasing opacity.
    background_color : str
        Background colour name (default ``'white'``).
    cycles : dict, optional
        Cycle data for colouring stance/swing phases differently.
    use_goliath : bool
        When ``True``, render using all Goliath 308 keypoints
        stored in ``frame["goliath308"]``.
    min_confidence : float
        Skip frames whose ``confidence`` is below this threshold.

    Returns
    -------
    str
        The *output_path* written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    format_lower = str(format).lower()
    if format_lower not in {"gif", "mp4"}:
        raise ValueError(f"Unsupported format: {format!r}. Use 'gif' or 'mp4'.")

    meta = data.get("meta", {})
    anim_fps = float(fps if fps is not None else meta.get("fps", 30))
    if anim_fps <= 0:
        raise ValueError(f"fps must be > 0, got {anim_fps}")
    frames_data = data.get("frames", [])

    # Filter frames by confidence
    if min_confidence > 0:
        render_indices = [
            i for i, fd in enumerate(frames_data)
            if fd.get("confidence", 0.0) >= min_confidence
        ]
    else:
        render_indices = list(range(len(frames_data)))

    n_render = len(render_indices)
    if n_render == 0:
        raise ValueError("No frames in data (or all below min_confidence)")

    logger.info(
        f"Rendering {n_render} frames"
        + (f" (skipping {len(frames_data) - n_render} empty)"
           if n_render < len(frames_data) else "")
    )

    # Build cycle phase lookup: frame_idx -> phase string
    phase_lookup: Dict[int, str] = {}
    if cycles:
        for c in cycles.get("cycles", []):
            hs_frame = c.get("hs_frame", 0)
            to_frame = c.get("to_frame")
            end_frame = c.get("end_frame", hs_frame)
            if to_frame is not None:
                for fi in range(hs_frame, to_frame + 1):
                    phase_lookup[fi] = "stance"
                for fi in range(to_frame + 1, end_frame + 1):
                    phase_lookup[fi] = "swing"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    trail_frames: List = []
    trail_max = 5

    def _draw_frame(render_idx):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Invert y so top of image is top of plot
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(background_color)

        if render_idx >= n_render:
            return

        orig_idx = render_indices[render_idx]
        fd = frames_data[orig_idx]

        # Determine phase colour
        phase = phase_lookup.get(fd.get("frame_idx", orig_idx))
        if phase == "stance":
            line_color = "#2196F3"
        elif phase == "swing":
            line_color = "#FF9800"
        else:
            line_color = "#333333"

        goliath_data = fd.get("goliath308") if use_goliath else None

        if goliath_data is not None:
            _plot_goliath_stickfigure(ax, goliath_data, line_color,
                                     alpha=1.0, lw=2)
        else:
            lm = fd.get("landmarks", {})
            # Draw trail
            if show_trail:
                trail_frames.append(dict(lm))
                if len(trail_frames) > trail_max:
                    trail_frames.pop(0)
                for ti, trail_lm in enumerate(trail_frames[:-1]):
                    alpha = 0.1 + 0.15 * ti / max(1, trail_max - 1)
                    _plot_skeleton(ax, trail_lm, color="#AAAAAA",
                                   alpha=alpha, lw=1)

            _plot_skeleton(ax, lm, color=line_color, alpha=1.0, lw=2)

            # Draw landmarks
            for name, val in lm.items():
                x = val.get("x")
                y = val.get("y")
                if x is None or y is None:
                    continue
                if np.isnan(x) or np.isnan(y):
                    continue
                ax.plot(x, y, "o", color=line_color, markersize=4, alpha=0.9)

        # Annotate angles
        if show_angles:
            angles_data_local = data.get("angles", {})
            aframes = (angles_data_local.get("frames", [])
                       if angles_data_local else [])
            if orig_idx < len(aframes):
                af = aframes[orig_idx]
                if goliath_data is not None:
                    _annotate_goliath_angles(ax, goliath_data, af)
                else:
                    lm = fd.get("landmarks", {})
                    _angle_map = {
                        "hip_L": "LEFT_HIP", "hip_R": "RIGHT_HIP",
                        "knee_L": "LEFT_KNEE", "knee_R": "RIGHT_KNEE",
                        "ankle_L": "LEFT_ANKLE", "ankle_R": "RIGHT_ANKLE",
                    }
                    for aname, jname in _angle_map.items():
                        aval = af.get(aname)
                        jlm = lm.get(jname)
                        if aval is not None and jlm is not None:
                            jx = jlm.get("x")
                            jy = jlm.get("y")
                            if (jx is not None and jy is not None
                                    and not (np.isnan(jx) or np.isnan(jy))):
                                ax.annotate(
                                    f"{aval:.0f}\u00b0",
                                    (jx, jy), fontsize=7,
                                    textcoords="offset points",
                                    xytext=(8, -3), color="#555555",
                                )

        time_s = fd.get("time_s", orig_idx / anim_fps)
        ax.set_title(f"t = {time_s:.2f} s", fontsize=10, color="#666666")

    def _plot_skeleton(ax_obj, lm, color="#333333", alpha=1.0, lw=2):
        """Plot MediaPipe skeleton connections on the axes."""
        for name_a, name_b in SKELETON_CONNECTIONS:
            va = lm.get(name_a)
            vb = lm.get(name_b)
            if va is None or vb is None:
                continue
            xa, ya = va.get("x"), va.get("y")
            xb, yb = vb.get("x"), vb.get("y")
            if xa is None or ya is None or xb is None or yb is None:
                continue
            if np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb):
                continue
            ax_obj.plot([xa, xb], [ya, yb], color=color, alpha=alpha, lw=lw)

    def _plot_goliath_stickfigure(ax_obj, g308, color, alpha=1.0, lw=2):
        """Plot Goliath 308 skeleton on matplotlib axes."""
        n = len(g308)
        pts = {}
        for idx in range(min(n, GOLIATH_FACE_START)):
            pt = g308[idx]
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            conf = float(pt[2]) if len(pt) >= 3 else 0.0
            if np.isnan(x) or np.isnan(y) or conf < 0.1:
                continue
            pts[idx] = (x, y)

        # Draw connections
        for idx_a, idx_b in GOLIATH_SKELETON_CONNECTIONS:
            if idx_a in pts and idx_b in pts:
                xa, ya = pts[idx_a]
                xb, yb = pts[idx_b]
                ax_obj.plot([xa, xb], [ya, yb], color=color,
                            alpha=alpha, lw=lw)

        # Draw landmarks
        for idx, (x, y) in pts.items():
            ms = 4 if idx < 21 else 2  # smaller for hand points
            ax_obj.plot(x, y, "o", color=color, markersize=ms, alpha=0.9)

    def _annotate_goliath_angles(ax_obj, g308, angle_frame):
        """Annotate angles on Goliath keypoints."""
        _goliath_angle_map = {
            "hip_L": 9, "hip_R": 10,
            "knee_L": 11, "knee_R": 12,
            "ankle_L": 13, "ankle_R": 14,
        }
        for aname, gidx in _goliath_angle_map.items():
            aval = angle_frame.get(aname)
            if aval is None or gidx >= len(g308):
                continue
            pt = g308[gidx]
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            if np.isnan(x) or np.isnan(y):
                continue
            ax_obj.annotate(
                f"{aval:.0f}\u00b0", (x, y), fontsize=7,
                textcoords="offset points",
                xytext=(8, -3), color="#555555",
            )

    interval = 1000.0 / anim_fps  # milliseconds per frame
    anim = animation.FuncAnimation(
        fig, _draw_frame, frames=n_render, interval=interval, blit=False,
    )

    if format_lower == "gif":
        writer_cls = animation.PillowWriter(fps=anim_fps)
        anim.save(output_path, writer=writer_cls)
    elif format_lower == "mp4":
        try:
            writer_cls = animation.FFMpegWriter(fps=anim_fps)
            anim.save(output_path, writer=writer_cls)
        except Exception as exc:
            logger.warning(
                "FFMpeg MP4 export failed; falling back to imageio (%s)",
                exc,
            )
            # Fallback: save frames with imageio
            try:
                import imageio
                frames_list = []
                for i in range(n_render):
                    _draw_frame(i)
                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames_list.append(img)
                imageio.mimwrite(output_path, frames_list, fps=anim_fps)
            except ImportError:
                raise RuntimeError(
                    "Neither FFMpeg nor imageio are available for MP4 export."
                )
    plt.close(fig)
    logger.info(f"Stick-figure animation saved to {output_path} ({n_render} frames)")
    return output_path
