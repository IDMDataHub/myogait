"""End-to-end Sapiens 2 demo: compare 3 pose models on a video.

Walks a colleague through the full plug-and-play workflow:

    1. (one-shot setup, done once per machine)
       pip install myogait[sapiens2]
       myogait setup-sapiens2 --size 0.4b --cleanup-safetensors

    2. Run this script on any video:
       python examples/sapiens2_compare_pipeline.py path/to/video.mp4

The script writes everything into an ``output/`` directory (configurable):

    output/
      <stem>_mediapipe.json       <-- pivot data, never deleted
      <stem>_mediapipe.mp4        <-- skeleton overlay
      <stem>_sapiens-quick.json
      <stem>_sapiens-quick.mp4
      <stem>_sapiens2-quick.json
      <stem>_sapiens2-quick.mp4
      compare_3way_slow.mp4       <-- side-by-side, half-speed, with legends
      compare_angles.png          <-- per-cycle angle curves (hip/knee/ankle, L+R)

If the JSONs already exist they are reused — re-running the script after a
crash, or on a different ``--out-dir``, is cheap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Local checkout wins over any older installed myogait when we run as a
# script (Python only puts the script's directory on sys.path).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "myogait" / "__init__.py").exists():
    sys.path.insert(0, str(_ROOT))

import myogait as mg  # noqa: E402

import cv2  # noqa: E402


MODELS = [
    ("mediapipe",       "MediaPipe"),
    ("sapiens-quick",   "Sapiens v1 (0.3B)"),
    ("sapiens2-quick",  "Sapiens 2 (0.4B)"),
]

JOINTS = ("hip", "knee", "ankle")
SIDES = ("left", "right")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ── Step 1 — extract + save JSON + render overlay ─────────────────────

def extract_one(video: Path, model: str, out_dir: Path,
                line_thickness: int, dot_radius: int, darken: float) -> dict:
    json_path = out_dir / f"{video.stem}_{model}.json"
    overlay_path = out_dir / f"{video.stem}_{model}.mp4"

    if json_path.exists():
        print(f"  [{model}] reuse JSON {json_path.name}")
        data = mg.load_json(str(json_path))
    else:
        t0 = time.time()
        data = mg.extract(str(video), model=model)
        n = len(data["frames"])
        det = sum(1 for f in data["frames"] if f.get("landmarks"))
        print(
            f"  [{model}] extracted {det}/{n} frames in "
            f"{time.time() - t0:.1f}s"
        )
        mg.save_json(data, str(json_path))
        print(f"  [{model}] -> {json_path.name}")

    if not overlay_path.exists():
        mg.render_skeleton_video(
            str(video), data, str(overlay_path),
            show_confidence=True,
            line_thickness=line_thickness,
            dot_radius=dot_radius,
            darken=darken,
        )
        print(f"  [{model}] -> {overlay_path.name}")
    else:
        print(f"  [{model}] reuse overlay {overlay_path.name}")
    return data


# ── Step 2 — side-by-side video with legends, half speed ──────────────

def _title_bar(panel_w: int, label: str, bar_h: int = 64) -> np.ndarray:
    bar = np.full((bar_h, panel_w, 3), 40, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.putText(bar, label, ((panel_w - tw) // 2, (bar_h + th) // 2),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bar


def render_sidebyside(overlays, labels, output: Path,
                      panel_width: int = 700, slow: float = 2.0) -> None:
    caps = [cv2.VideoCapture(str(p)) for p in overlays]
    metas = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              c.get(cv2.CAP_PROP_FPS),
              int(c.get(cv2.CAP_PROP_FRAME_COUNT))) for c in caps]
    src_w, src_h, src_fps, _ = metas[0]
    n_min = min(m[3] for m in metas)
    panel_w = panel_width
    panel_h = int(round(src_h * panel_w / src_w))
    title_h = 64
    out_w = panel_w * len(overlays)
    out_h = panel_h + title_h
    out_fps = max(1.0, src_fps / max(slow, 0.001))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, out_fps, (out_w, out_h))
    bars = [_title_bar(panel_w, lbl, title_h) for lbl in labels]

    for i in range(n_min):
        panels = []
        for cap, bar in zip(caps, bars):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((src_h, src_w, 3), dtype=np.uint8)
            small = cv2.resize(frame, (panel_w, panel_h), cv2.INTER_AREA)
            panels.append(np.vstack([bar, small]))
        writer.write(np.hstack(panels))
        if (i + 1) % 100 == 0 or i + 1 == n_min:
            print(f"  side-by-side frame {i + 1}/{n_min}", flush=True)

    writer.release()
    for c in caps:
        c.release()
    print(f"  -> {output.name}")


# ── Step 3 — per-cycle joint angle plot ───────────────────────────────

def cycles_for(joint: str, side: str, cycles: dict):
    arrays = []
    for c in cycles["cycles"]:
        if c["side"] != side:
            continue
        v = c["angles_normalized"].get(joint)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        if arr.shape == (101,) and np.isfinite(arr).any():
            arrays.append(arr)
    summary = cycles.get("summary", {}).get(side, {})
    mean = summary.get(f"{joint}_mean")
    return arrays, np.asarray(mean) if mean is not None else None


def plot_angles(results, output: Path) -> None:
    fig, axes = plt.subplots(len(JOINTS), len(SIDES),
                             figsize=(11, 9), sharex=True)
    x = np.arange(101)
    for row, joint in enumerate(JOINTS):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            for (label, cycles, color) in results:
                arrays, mean = cycles_for(joint, side, cycles)
                for arr in arrays:
                    ax.plot(x, arr, color=color, alpha=0.18, linewidth=0.8)
                if mean is not None:
                    ax.plot(x, mean, color=color, linewidth=2.6,
                            label=f"{label}  (n={len(arrays)})")
            ax.set_title(f"{joint.capitalize()} - {side}", fontsize=11)
            ax.set_ylabel(f"{joint} (deg)")
            ax.grid(alpha=0.3)
            if row == len(JOINTS) - 1:
                ax.set_xlabel("Gait cycle (%)")
            ax.set_xlim(0, 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results),
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output.name}")


# ── Driver ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sapiens 2 plug-and-play demo: extract + side-by-side "
                    "overlay + per-cycle angle plot for 3 models.",
    )
    parser.add_argument("video", help="Input video path.")
    parser.add_argument(
        "--out-dir", default="output",
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--models", nargs="+", default=[m for m, _ in MODELS],
        help="Models to compare (default: mediapipe sapiens-quick sapiens2-quick).",
    )
    parser.add_argument("--panel-width", type=int, default=700)
    parser.add_argument("--slow", type=float, default=2.0,
                        help="Side-by-side speed factor (default 2.0 = half).")
    parser.add_argument("--line-thickness", type=int, default=6,
                        help="Skeleton line thickness (default 6).")
    parser.add_argument("--dot-radius", type=int, default=10,
                        help="Joint dot radius (default 10).")
    parser.add_argument("--darken", type=float, default=0.55,
                        help="Source frame darkening factor 0..1 (default 0.55).")
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = dict(MODELS)
    selected = [(m, label_map.get(m, m)) for m in args.models]

    print(f"\n=== Sapiens 2 plug-and-play demo ===")
    print(f"video : {video}")
    print(f"out   : {out_dir}")
    print(f"models: {', '.join(m for m, _ in selected)}\n")

    # Step 1: extract for each model
    datas = []
    for model, _ in selected:
        print(f"--- {model} ---")
        datas.append(extract_one(
            video, model, out_dir,
            args.line_thickness, args.dot_radius, args.darken,
        ))

    overlays = [out_dir / f"{video.stem}_{m}.mp4" for m, _ in selected]
    labels = [lbl for _, lbl in selected]

    # Step 2: side-by-side video
    print("\n--- side-by-side ---")
    sidebyside_path = out_dir / "compare_3way_slow.mp4"
    render_sidebyside(overlays, labels, sidebyside_path,
                      panel_width=args.panel_width, slow=args.slow)

    # Step 3: per-cycle angle plots — needs angles + events + cycles
    print("\n--- per-cycle angles ---")
    results = []
    for (model, label), data, color in zip(selected, datas, COLORS):
        d = mg.normalize(data, filters=["butterworth"])
        d = mg.compute_angles(
            d, method="sagittal_vertical_axis",
            correction_factor=1.0, calibrate=False,
        )
        d = mg.detect_events(d, method="zeni")
        cycles = mg.segment_cycles(d)
        n_l = sum(1 for c in cycles["cycles"] if c["side"] == "left")
        n_r = sum(1 for c in cycles["cycles"] if c["side"] == "right")
        print(f"  [{model}] cycles L={n_l} R={n_r}")
        results.append((label, cycles, color))

    plot_angles(results, out_dir / "compare_angles.png")

    print(f"\nAll deliverables written to {out_dir}")


if __name__ == "__main__":
    main()
