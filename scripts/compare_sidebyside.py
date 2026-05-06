"""Combine N overlay videos en un side-by-side avec légendes, ralenti optionnel.

Lit les .mp4 d'overlay produits par ``compare_models.py`` et les juxtapose
horizontalement, avec un bandeau-titre par modèle. Exporte un seul .mp4
(par défaut à demi-vitesse).

Usage :
    python scripts/compare_sidebyside.py overlay1.mp4 overlay2.mp4 overlay3.mp4 \\
        --labels MediaPipe "Sapiens v1" "Sapiens 2" \\
        --output compare_sidebyside.mp4 \\
        --slow 2.0 \\
        --panel-width 700
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_title(panel_w: int, label: str, bar_h: int = 64) -> np.ndarray:
    """Render a title bar above a panel."""
    bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = (panel_w - tw) // 2
    y = (bar_h + th) // 2
    cv2.putText(bar, label, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side de plusieurs vidéos overlay avec légendes."
    )
    parser.add_argument("videos", nargs="+", help="Vidéos overlay à juxtaposer.")
    parser.add_argument(
        "--labels", nargs="+",
        help="Légendes (une par vidéo). Si omis, utilise le stem du fichier.",
    )
    parser.add_argument(
        "--output", "-o", default="compare_sidebyside.mp4",
        help="Vidéo de sortie (default : compare_sidebyside.mp4).",
    )
    parser.add_argument(
        "--panel-width", type=int, default=700,
        help="Largeur de chaque panneau en pixels (default : 700).",
    )
    parser.add_argument(
        "--slow", type=float, default=2.0,
        help="Facteur de ralentissement (default : 2.0 = mi-vitesse). "
             "1.0 = vitesse normale.",
    )
    parser.add_argument(
        "--title-height", type=int, default=64,
        help="Hauteur du bandeau-titre en pixels (default : 64).",
    )
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    for v in videos:
        if not v.exists():
            raise FileNotFoundError(v)
    labels = args.labels or [v.stem for v in videos]
    if len(labels) != len(videos):
        raise ValueError("Nombre de --labels doit matcher le nombre de vidéos.")

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    metas = []
    for cap, v in zip(caps, videos):
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {v}")
        metas.append({
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        })

    # Sanity : on attend des vidéos de même résolution / fps / nb frames.
    src_fps = metas[0]["fps"]
    n_min = min(m["n"] for m in metas)
    src_w = metas[0]["w"]
    src_h = metas[0]["h"]

    # Dimensions panneau (préserve l'aspect ratio).
    panel_w = args.panel_width
    panel_h = int(round(src_h * panel_w / src_w))
    title_h = args.title_height
    out_w = panel_w * len(videos)
    out_h = panel_h + title_h
    out_fps = max(1.0, src_fps / max(args.slow, 0.001))

    print(f"Sources : {len(videos)} vidéos")
    for v, m, l in zip(videos, metas, labels):
        print(f"  - {l:20s} {v.name}  {m['w']}x{m['h']} @ {m['fps']:.1f}fps "
              f"({m['n']} frames)")
    print(f"Panneau : {panel_w}x{panel_h}")
    print(f"Canvas  : {out_w}x{out_h}")
    print(f"FPS out : {out_fps:.1f} (slow={args.slow}x)")
    print(f"Frames  : {n_min}")

    # Pre-render title bars (une fois par modèle).
    title_bars = [_draw_title(panel_w, l, title_h) for l in labels]

    # Writer.
    out_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        # Try avc1 fallback
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}.")

    for i in range(n_min):
        panels = []
        for cap, bar in zip(caps, title_bars):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((src_h, src_w, 3), dtype=np.uint8)
            small = cv2.resize(frame, (panel_w, panel_h), cv2.INTER_AREA)
            panels.append(np.vstack([bar, small]))
        canvas = np.hstack(panels)
        writer.write(canvas)
        if (i + 1) % 50 == 0 or i + 1 == n_min:
            print(f"  frame {i + 1}/{n_min}", flush=True)

    writer.release()
    for c in caps:
        c.release()
    print(f"\nÉcrit : {out_path.resolve()}")


if __name__ == "__main__":
    main()
