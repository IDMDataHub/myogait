# Export Sapiens 2 → TorchScript (.pt2)

## Prérequis

- Windows 10/11
- Python 3.12+ (python.org ou winget)
- ~16 Go de RAM (32 Go pour le modèle 5B)
- Connexion internet (téléchargement des poids depuis HuggingFace)

## Étapes

### 1. Créer un environnement propre

```powershell
python -m venv C:\sapiens2_export
C:\sapiens2_export\Scripts\Activate.ps1
```

### 2. Installer PyTorch 2.7+

```powershell
# Si tu vas inferer sur Intel Arc / Xe (recommandé) — IMPORTANT : tracer sur
# le même device que celui où le modèle tournera, sinon les constantes
# bakées par le trace plantent au runtime ("Expected all tensors to be on
# the same device").
pip install torch --index-url https://download.pytorch.org/whl/xpu

# Sinon CPU :
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Installer les dépendances

```powershell
pip install safetensors huggingface-hub
```

### 4. Cloner et installer le repo sapiens2

```powershell
cd C:\
git clone https://github.com/facebookresearch/sapiens2.git
cd sapiens2
pip install -e .
```

### 5. Lancer le script d'export

```powershell
cd C:\sapiens2
# Auto-detect du device (cuda > xpu > cpu) :
python <chemin\vers\myogait>\scripts\export_sapiens2_pt2.py --cleanup-safetensors
# Forcer un device :
python <chemin\vers\myogait>\scripts\export_sapiens2_pt2.py --device xpu --cleanup-safetensors
```

Le script va :
1. Télécharger chaque modèle depuis HuggingFace (0.4b, 0.8b, 1b)
2. Reconstruire l'architecture
3. Charger les poids SafeTensors
4. Tracer le modèle avec un dummy input (1, 3, 1024, 768)
5. Sauvegarder en .pt2 dans `~/.myogait/models/`
   (sur Windows : `$env:USERPROFILE\.myogait\models\`)

### 6. Vérifier

```powershell
python -c "import torch; m = torch.jit.load(\"$env:USERPROFILE/.myogait/models/sapiens2_0.4b_pose.pt2\"); print('OK')"
```

### 7. Nettoyer (optionnel)

```powershell
deactivate
Remove-Item -Recurse C:\sapiens2_export
Remove-Item -Recurse C:\sapiens2
```

Les .pt2 restent dans `~/.myogait/models/` et myogait les trouvera automatiquement —
le loader préfère le `.pt2` au `.safetensors` quand les deux sont présents, donc il n'y
a pas besoin du package `sapiens2` au runtime une fois l'export terminé.
