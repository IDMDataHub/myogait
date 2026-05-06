"""Export Sapiens 2 SafeTensors → TorchScript (.pt2).

Lance ce script depuis un environnement où le package sapiens2 est installé.
Les fichiers .pt2 sont sauvegardés dans ~/.myogait/models/ et myogait
les trouvera automatiquement.

Usage :
    python export_sapiens2_pt2.py                       # exporte 0.4b, 0.8b, 1b
    python export_sapiens2_pt2.py --sizes 0.4b 1b       # exporte seulement ceux-là
    python export_sapiens2_pt2.py --sizes 5b            # exporte le 5B (besoin 32 Go RAM)
    python export_sapiens2_pt2.py --cleanup-safetensors # supprime le .safetensors après trace
"""

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


# Modèles disponibles
MODELS = {
    "0.4b": "facebook/sapiens2-pose-0.4b",
    "0.8b": "facebook/sapiens2-pose-0.8b",
    "1b": "facebook/sapiens2-pose-1b",
    "5b": "facebook/sapiens2-pose-5b",
}

FILENAMES = {
    "0.4b": "sapiens2_0.4b_pose.safetensors",
    "0.8b": "sapiens2_0.8b_pose.safetensors",
    "1b": "sapiens2_1b_pose.safetensors",
    "5b": "sapiens2_5b_pose.safetensors",
}

# Configs pour reconstruire l'architecture (relative au package ``sapiens``).
CONFIG_RELPATHS = {
    "0.4b": "pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "0.8b": "pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "1b":   "pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "5b":   "pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py",
}


def _resolve_config_path(size: str) -> str:
    """Find the config file shipped with the installed sapiens package."""
    import sapiens
    pkg_root = Path(sapiens.__file__).parent
    cfg = pkg_root / CONFIG_RELPATHS[size]
    if not cfg.exists():
        raise FileNotFoundError(
            f"Config introuvable pour {size} : {cfg}\n"
            f"Vérifie que le repo sapiens2 est bien installé et que les configs "
            f"sont présentes."
        )
    return str(cfg)

OUTPUT_DIR = Path.home() / ".myogait" / "models"

# Résolution d'entrée Sapiens 2 (identique à v1)
INPUT_H, INPUT_W = 1024, 768


def _auto_device() -> str:
    """Pick the best device for tracing: cuda > xpu > cpu.

    Tracing on CPU is enough to *produce* a portable .pt2, but TorchScript
    bakes constants in at trace time — so a CPU-traced model can fail at
    runtime on XPU/CUDA with "Expected all tensors to be on the same
    device". Preferring an accelerator avoids that footgun.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def export_model(size: str, cleanup_safetensors: bool = False,
                 device: str = "auto"):
    """Télécharge, charge et exporte un modèle en TorchScript."""
    if device == "auto":
        device = _auto_device()
    repo_id = MODELS[size]
    filename = FILENAMES[size]
    output_path = OUTPUT_DIR / filename.replace(".safetensors", ".pt2")

    if output_path.exists():
        print(f"  [{size}] Déjà exporté : {output_path}")
        return output_path

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [{size}] Téléchargement depuis {repo_id}...")
    safetensors_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(OUTPUT_DIR),
    )
    print(f"  [{size}] SafeTensors : {safetensors_path}")

    # Charger le modèle via le package sapiens2
    print(f"  [{size}] Construction du modèle...")
    try:
        from sapiens.pose.models import init_model
    except ImportError:
        print(
            "\nERREUR : le package sapiens2 n'est pas installé.\n"
            "  git clone https://github.com/facebookresearch/sapiens2\n"
            "  cd sapiens2 && pip install -e .\n"
        )
        sys.exit(1)

    config_path = _resolve_config_path(size)
    print(f"  [{size}] Config : {config_path}")
    model = init_model(config_path, checkpoint=safetensors_path, device=device)
    model.eval()
    print(f"  [{size}] Modèle chargé sur {device}.")

    # Tracer avec un dummy input. Le tensor doit être sur le même device que
    # le modèle, sinon les constantes bakées par le trace (positional embeds,
    # masks…) seront sur le device du dummy et planteront à l'inférence.
    print(f"  [{size}] Tracing TorchScript ({device})...")
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Sauvegarder
    traced.save(str(output_path))
    print(f"  [{size}] Exporté : {output_path}")

    # Vérification rapide. On recharge sur le même device et on relance la
    # même forward — si ça marche ici, ça marchera côté myogait.
    loaded = torch.jit.load(str(output_path), map_location=device)
    with torch.no_grad():
        out = loaded(dummy)
    if isinstance(out, (tuple, list)):
        shapes = [tuple(t.shape) for t in out if hasattr(t, "shape")]
        print(f"  [{size}] Vérification OK — outputs : {shapes}")
    else:
        print(f"  [{size}] Vérification OK — output shape : {tuple(out.shape)}")

    if cleanup_safetensors:
        try:
            Path(safetensors_path).unlink()
            print(f"  [{size}] SafeTensors supprimé : {safetensors_path}")
        except OSError as exc:
            print(f"  [{size}] Impossible de supprimer le .safetensors : {exc}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Sapiens 2 -> TorchScript")
    parser.add_argument(
        "--sizes", nargs="+", default=["0.4b", "0.8b", "1b"],
        choices=list(MODELS.keys()),
        help="Tailles à exporter (default: 0.4b 0.8b 1b)",
    )
    parser.add_argument(
        "--cleanup-safetensors", action="store_true",
        help="Supprime le .safetensors après un trace réussi (libère 1.6–20 Go).",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=("auto", "cpu", "cuda", "xpu"),
        help="Device pour le tracing (default: auto = cuda > xpu > cpu). "
             "Tracer sur CPU peut produire un .pt2 qui plante au runtime sur "
             "GPU/XPU à cause des constantes bakées.",
    )
    args = parser.parse_args()

    device = _auto_device() if args.device == "auto" else args.device
    print(f"Export Sapiens 2 -> TorchScript (.pt2)")
    print(f"Destination : {OUTPUT_DIR}")
    print(f"Modèles : {', '.join(args.sizes)}")
    print(f"Device : {device}")
    if args.cleanup_safetensors:
        print("Cleanup .safetensors : ON")
    print()

    for size in args.sizes:
        try:
            export_model(
                size,
                cleanup_safetensors=args.cleanup_safetensors,
                device=device,
            )
        except Exception as e:
            print(f"  [{size}] ERREUR : {e}")
        print()

    print("Terminé. Les fichiers .pt2 sont dans :")
    print(f"  {OUTPUT_DIR}")
    print()
    print("myogait les trouvera automatiquement avec :")
    print('  mg.extract("video.mp4", model="sapiens2-top")')


if __name__ == "__main__":
    main()
