"""Compare per-cycle joint angle curves across multiple pose models.

Pour chaque JSON pivot produit par ``compare_models.py`` :
  1. ``normalize`` (Butterworth)
  2. ``compute_angles`` (sagittal_vertical_axis, signed knee)
  3. ``detect_events`` (zeni)
  4. ``segment_cycles`` (HS-to-HS, normalisé 0-100%)

Puis trace pour hip / knee / ankle, côté gauche et droit :
  - tous les cycles individuels en couleur translucide (par modèle)
  - la moyenne par modèle (trait épais opaque)

Sortie : un PNG par côté + un PNG combiné L+R.

Usage :
    python scripts/compare_angles.py json1 json2 json3 \
        --labels MediaPipe "Sapiens v1" "Sapiens 2" \
        --output-dir myogait_compare/
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make local checkout win over installed myogait when run as a script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if (_ROOT / "myogait" / "__init__.py").exists():
    sys.path.insert(0, str(_ROOT))

import myogait as mg  # noqa: E402


JOINTS = ("hip", "knee", "ankle")
SIDES = ("left", "right")

# Distinct, color-blind-friendly colors per model (Tab10).
_DEFAULT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]


def prepare(data: dict) -> dict:
    """Run normalize + angles + events + cycles in place."""
    data = mg.normalize(data, filters=["butterworth"])
    data = mg.compute_angles(
        data,
        method="sagittal_vertical_axis",
        correction_factor=1.0,
        calibrate=False,
    )
    data = mg.detect_events(data, method="zeni")
    cycles = mg.segment_cycles(data)
    return cycles


def cycles_for(joint: str, side: str, cycles: dict):
    """Return (list of (101,) arrays for individual cycles, mean array or None)."""
    side_cycles = [c for c in cycles["cycles"] if c["side"] == side]
    arrays = []
    for c in side_cycles:
        vals = c["angles_normalized"].get(joint)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=float)
        if arr.shape == (101,) and np.isfinite(arr).any():
            arrays.append(arr)
    summary = cycles.get("summary", {}).get(side, {})
    mean = summary.get(f"{joint}_mean")
    mean_arr = np.asarray(mean, dtype=float) if mean is not None else None
    return arrays, mean_arr


def plot_grid(results: list, output: Path, title: str = ""):
    """Create a 3x2 grid (joints x sides) with per-model overlays."""
    fig, axes = plt.subplots(
        len(JOINTS), len(SIDES), figsize=(11, 9), sharex=True
    )
    x = np.arange(101)

    for row, joint in enumerate(JOINTS):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            for (label, cycles, color) in results:
                arrays, mean_arr = cycles_for(joint, side, cycles)
                # Individual cycles: thin translucent lines
                for arr in arrays:
                    ax.plot(x, arr, color=color, alpha=0.18, linewidth=0.8)
                # Mean: thick opaque line
                if mean_arr is not None:
                    n = len(arrays)
                    ax.plot(
                        x, mean_arr, color=color, linewidth=2.6,
                        label=f"{label}  (n={n})",
                    )
            ax.set_title(f"{joint.capitalize()} — {side}", fontsize=11)
            ax.set_ylabel(f"{joint} angle (deg)")
            ax.grid(alpha=0.3)
            if row == len(JOINTS) - 1:
                ax.set_xlabel("Gait cycle (%)")
            ax.set_xlim(0, 100)

    # One legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(results),
        bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=10,
    )
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-cycle joint-angle curves across multiple pose models."
        )
    )
    parser.add_argument(
        "jsons", nargs="+",
        help="JSON pivot files produced by compare_models.py.",
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="Labels (un par JSON). Défaut : stem du fichier.",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Dossier de sortie pour les PNG (default : .).",
    )
    parser.add_argument(
        "--prefix", default="compare_angles",
        help="Préfixe du nom de fichier de sortie (default: compare_angles).",
    )
    args = parser.parse_args()

    jsons = [Path(p) for p in args.jsons]
    for j in jsons:
        if not j.exists():
            raise FileNotFoundError(j)

    labels = args.labels or [j.stem for j in jsons]
    if len(labels) != len(jsons):
        raise ValueError("--labels doit avoir le même nombre d'entrées que les JSON.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for j, label, color in zip(jsons, labels, _DEFAULT_COLORS):
        print(f"\n=== {label} ({j.name}) ===")
        data = mg.load_json(str(j))
        cycles = prepare(data)
        n_l = sum(1 for c in cycles["cycles"] if c["side"] == "left")
        n_r = sum(1 for c in cycles["cycles"] if c["side"] == "right")
        print(f"  cycles : L={n_l}  R={n_r}")
        results.append((label, cycles, color))

    out_path = out_dir / f"{args.prefix}.png"
    print()
    plot_grid(results, out_path,
              title="Joint angles per gait cycle — multi-model comparison")

    # Per-model summary print
    print("\n=== Résumé ===")
    for label, cycles, _ in results:
        n_l = sum(1 for c in cycles["cycles"] if c["side"] == "left")
        n_r = sum(1 for c in cycles["cycles"] if c["side"] == "right")
        print(f"  {label:22s}  L={n_l}  R={n_r}")


if __name__ == "__main__":
    main()
