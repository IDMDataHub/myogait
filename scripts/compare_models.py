"""Comparaison N-way de modèles de pose sur la même vidéo.

Lance ``mg.extract`` avec chaque modèle, écrit un overlay squelette + un
JSON pivot ``data`` pour chacun, et imprime un résumé (taux de détection,
temps par frame).

Usage :
    python scripts/compare_models.py video.mp4
    python scripts/compare_models.py video.mp4 --max-frames 100
    python scripts/compare_models.py video.mp4 --models mediapipe sapiens2-quick
    python scripts/compare_models.py video.mp4 --out-dir runs/compare1
    python scripts/compare_models.py video.mp4 --no-json   # pas de JSON

Modèles par défaut : mediapipe, sapiens-quick (v1 0.3B), sapiens2-quick (v2 0.4B).
Les overlays et les JSON sont écrits dans ``--out-dir`` (default : ``comparison_out/``).
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure the local checkout wins over any older installed myogait when
# the script is invoked as ``python scripts/compare_models.py`` (Python
# only puts the *script's* directory on sys.path, not the project root).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "myogait" / "__init__.py").exists():
    sys.path.insert(0, str(_ROOT))

import myogait as mg  # noqa: E402


DEFAULT_MODELS = ["mediapipe", "sapiens-quick", "sapiens2-quick"]


def run_one(video: str, model: str, max_frames, out_dir: Path,
            save_json: bool) -> dict:
    print(f"\n=== {model} ===")
    t0 = time.time()
    data = mg.extract(video, model=model, max_frames=max_frames)
    dt = time.time() - t0
    n = len(data["frames"])
    det = sum(1 for f in data["frames"] if f.get("landmarks"))
    per_frame = dt / n if n else 0.0
    print(
        f"   {n} frames, detection {det}/{n} "
        f"({100 * det / max(n, 1):.0f}%), "
        f"{dt:.1f}s ({per_frame:.2f}s/frame)"
    )

    stem = Path(video).stem
    json_path = None
    if save_json:
        json_path = out_dir / f"{stem}_{model}.json"
        mg.save_json(data, str(json_path))
        print(f"   json    -> {json_path}")

    out_path = out_dir / f"{stem}_{model}.mp4"
    mg.render_skeleton_video(
        video, data, str(out_path),
        show_confidence=True,
    )
    print(f"   overlay -> {out_path}")

    return {
        "model": model,
        "frames": n,
        "detected": det,
        "time_s": dt,
        "per_frame_s": per_frame,
        "out": str(out_path),
        "json": str(json_path) if json_path else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison N-way de modèles de pose (overlay squelette)."
    )
    parser.add_argument("video", help="Chemin de la vidéo source.")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Modèles à comparer (default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="Nombre maximum de frames à traiter (0 = toutes ; default: 200).",
    )
    parser.add_argument(
        "--out-dir", default="comparison_out",
        help="Dossier de sortie pour les overlays (default: comparison_out).",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="N'écrit pas le JSON pivot par modèle (par défaut : écrit).",
    )
    args = parser.parse_args()
    save_json = not args.no_json

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_frames = args.max_frames if args.max_frames > 0 else None

    print(f"Vidéo : {args.video}")
    print(f"Modèles : {', '.join(args.models)}")
    print(f"max_frames : {max_frames if max_frames else 'all'}")
    print(f"Sortie : {out_dir.resolve()}")

    results = []
    for model in args.models:
        try:
            results.append(
                run_one(args.video, model, max_frames, out_dir, save_json)
            )
        except Exception as exc:
            print(f"   [{model}] ERREUR : {exc}")
            results.append({
                "model": model, "frames": 0, "detected": 0,
                "time_s": 0.0, "per_frame_s": 0.0,
                "out": None, "error": str(exc),
            })

    print("\n=== Résumé ===")
    print(f"{'Modèle':22s}  {'Détec.':>10s}  {'Temps':>8s}  {'Frame':>10s}  Sortie")
    for r in results:
        det_str = f"{r['detected']}/{r['frames']}" if r["frames"] else "ERR"
        print(
            f"{r['model']:22s}  {det_str:>10s}  "
            f"{r['time_s']:>7.1f}s  "
            f"{r['per_frame_s']:>9.2f}s  "
            f"{r['out'] or r.get('error', '')}"
        )


if __name__ == "__main__":
    main()
