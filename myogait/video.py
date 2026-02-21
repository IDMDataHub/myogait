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


def render_skeleton_frame(
    frame_image: np.ndarray,
    landmarks: dict,
    angles: Optional[dict] = None,
    events: Optional[dict] = None,
    skeleton_color: str = "auto",
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

    Returns
    -------
    np.ndarray
        Copy of *frame_image* with skeleton drawn on it.
    """
    frame = frame_image.copy()
    h, w = frame.shape[:2]

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

            frame = render_skeleton_frame(
                frame, lm,
                angles=frame_angles,
                events=frame_events,
                skeleton_color=skeleton_color,
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
    n_frames = len(frames_data)

    if n_frames == 0:
        raise ValueError("No frames in data")

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

    trail_frames: List[dict] = []
    trail_max = 5

    def _draw_frame(frame_idx):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Invert y so top of image is top of plot
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(background_color)

        if frame_idx >= n_frames:
            return

        fd = frames_data[frame_idx]
        lm = fd.get("landmarks", {})

        # Determine phase colour
        phase = phase_lookup.get(fd.get("frame_idx", frame_idx))
        if phase == "stance":
            line_color = "#2196F3"
        elif phase == "swing":
            line_color = "#FF9800"
        else:
            line_color = "#333333"

        # Draw trail
        if show_trail:
            trail_frames.append(dict(lm))
            if len(trail_frames) > trail_max:
                trail_frames.pop(0)
            for ti, trail_lm in enumerate(trail_frames[:-1]):
                alpha = 0.1 + 0.15 * ti / max(1, trail_max - 1)
                _plot_skeleton(ax, trail_lm, color="#AAAAAA", alpha=alpha, lw=1)

        # Draw current skeleton
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
            aframes = angles_data_local.get("frames", []) if angles_data_local else []
            if frame_idx < len(aframes):
                af = aframes[frame_idx]
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
                        if jx is not None and jy is not None and not (np.isnan(jx) or np.isnan(jy)):
                            ax.annotate(
                                f"{aval:.0f}\u00b0",
                                (jx, jy), fontsize=7,
                                textcoords="offset points",
                                xytext=(8, -3), color="#555555",
                            )

        time_s = fd.get("time_s", frame_idx / anim_fps)
        ax.set_title(f"t = {time_s:.2f} s", fontsize=10, color="#666666")

    def _plot_skeleton(ax_obj, lm, color="#333333", alpha=1.0, lw=2):
        """Plot skeleton connections on the axes."""
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

    interval = 1000.0 / anim_fps  # milliseconds per frame
    anim = animation.FuncAnimation(
        fig, _draw_frame, frames=n_frames, interval=interval, blit=False,
    )

    if format_lower == "gif":
        writer_cls = animation.PillowWriter(fps=anim_fps)
        anim.save(output_path, writer=writer_cls)
    elif format_lower == "mp4":
        try:
            writer_cls = animation.FFMpegWriter(fps=anim_fps)
            anim.save(output_path, writer=writer_cls)
        except Exception:
            # Fallback: save frames with imageio
            try:
                import imageio
                frames_list = []
                for i in range(n_frames):
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
    logger.info(f"Stick-figure animation saved to {output_path} ({n_frames} frames)")
    return output_path
