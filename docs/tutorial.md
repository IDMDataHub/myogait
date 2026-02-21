# myogait — Tutorial complet

Guide pratique couvrant tous les cas d'usage de myogait, de l'extraction vidéo
à l'export OpenSim, avec des exemples de code reproductibles.

---

## Table des matières

1. [Installation](#1-installation)
2. [Pipeline de base](#2-pipeline-de-base)
3. [Choix du backend de pose](#3-choix-du-backend-de-pose)
4. [Qualité des données](#4-qualité-des-données)
5. [Angles articulaires](#5-angles-articulaires)
6. [Détection des événements de marche](#6-détection-des-événements-de-marche)
7. [Segmentation en cycles](#7-segmentation-en-cycles)
8. [Analyse spatio-temporelle](#8-analyse-spatio-temporelle)
9. [Scores cliniques (GPS-2D, SDI, GVS)](#9-scores-cliniques)
10. [Comparaison normative](#10-comparaison-normative)
11. [Visualisation et plots](#11-visualisation-et-plots)
12. [Vidéo annotée et stick figure](#12-vidéo-annotée-et-stick-figure)
13. [Rapport PDF clinique](#13-rapport-pdf-clinique)
14. [Export vers OpenSim](#14-export-vers-opensim)
15. [Export vers Pose2Sim](#15-export-vers-pose2sim)
16. [Export multi-formats](#16-export-multi-formats)
17. [Analyse frontale (avec profondeur)](#17-analyse-frontale)
18. [Détection de pathologies](#18-détection-de-pathologies)
19. [Analyse longitudinale multi-sessions](#19-analyse-longitudinale)
20. [Utilisation en ligne de commande (CLI)](#20-cli)
21. [Configuration YAML](#21-configuration-yaml)
22. [Cas d'usage cliniques](#22-cas-dusage-cliniques)

---

## 1. Installation

### Installation de base

```bash
pip install myogait
```

Ceci installe myogait avec ses dépendances obligatoires : numpy, pandas, scipy,
opencv, matplotlib, et **gaitkit** (détection d'événements).

### Avec un backend de pose

```bash
# MediaPipe — léger, CPU uniquement, 33 landmarks
pip install myogait[mediapipe]

# YOLO — rapide, GPU supporté, 17 keypoints COCO
pip install myogait[yolo]

# Sapiens — Meta AI, profondeur + segmentation
pip install myogait[sapiens]

# ViTPose — state-of-the-art accuracy
pip install myogait[vitpose]

# RTMW — 133 keypoints (corps entier + mains + visage)
pip install myogait[rtmw]

# Tout installer
pip install myogait[all]
```

### Vérification

```python
import myogait
print(myogait.__version__)  # 0.3.0
print(len(myogait.__all__))  # 90+ fonctions publiques
```

---

## 2. Pipeline de base

Le pipeline complet en 6 étapes :

```python
from myogait import (
    extract, normalize, compute_angles,
    detect_events, segment_cycles, analyze_gait
)

# Étape 1 : Extraction des landmarks depuis la vidéo
data = extract("marche_sagittale.mp4", model="mediapipe")
print(f"{len(data['frames'])} frames extraites à {data['meta']['fps']} FPS")

# Étape 2 : Filtrage et normalisation
data = normalize(data, filters=["butterworth"])

# Étape 3 : Calcul des angles articulaires
data = compute_angles(data)
print(f"Angles : {list(data['angles']['frames'][0].keys())}")

# Étape 4 : Détection des événements (heel strike, toe off)
data = detect_events(data, method="gk_bike")
n_hs = len(data["events"]["left_hs"]) + len(data["events"]["right_hs"])
print(f"{n_hs} heel strikes détectés")

# Étape 5 : Segmentation en cycles de marche
cycles = segment_cycles(data)

# Étape 6 : Analyse spatio-temporelle
stats = analyze_gait(data, cycles)
print(f"Cadence : {stats['cadence']:.1f} pas/min")
print(f"Vitesse : {stats['speed']:.2f} m/s")
print(f"Temps de stance : {stats['stance_pct']:.1f}%")
```

### Le dictionnaire `data`

Toutes les fonctions opèrent sur un dictionnaire `data` qui s'enrichit à
chaque étape :

```python
data = {
    "meta": {"fps": 30.0, "width": 1920, "height": 1080, "model": "mediapipe"},
    "frames": [
        {
            "frame_idx": 0,
            "landmarks": {
                "NOSE": {"x": 0.52, "y": 0.10, "visibility": 0.99},
                "LEFT_HIP": {"x": 0.48, "y": 0.45, "visibility": 0.95},
                # ... 33 landmarks
            }
        },
        # ... N frames
    ],
    "angles": {"frames": [{"hip_L": 25.3, "knee_L": 5.2, ...}, ...]},
    "events": {"left_hs": [...], "right_hs": [...], "left_to": [...], "right_to": [...]},
}
```

---

## 3. Choix du backend de pose

### Comparaison rapide

```python
# MediaPipe — le plus simple, bon pour le prototypage
data = extract("video.mp4", model="mediapipe")

# YOLO — rapide et robuste
data = extract("video.mp4", model="yolo")

# Sapiens — le plus précis + profondeur monoculaire
data = extract("video.mp4", model="sapiens-top", with_depth=True, with_seg=True)

# ViTPose — excellent compromis précision/vitesse
data = extract("video.mp4", model="vitpose-large")

# RTMW — 133 keypoints (corps, mains, visage)
data = extract("video.mp4", model="rtmw")
```

### Informations sur la précision des modèles

```python
from myogait import model_accuracy_info

info = model_accuracy_info("mediapipe")
print(f"MAE : {info['mae_px']} px")
print(f"PCK@0.5 : {info['pck_05']}")
print(f"Référence : {info['reference']}")
```

---

## 4. Qualité des données

### Filtrage par confiance

```python
from myogait import confidence_filter, detect_outliers, data_quality_score

# Supprimer les landmarks avec confiance < 30%
data = confidence_filter(data, threshold=0.3)

# Détecter et interpoler les outliers (z-score > 3)
data = detect_outliers(data, z_thresh=3.0)

# Score de qualité global (0-100)
quality = data_quality_score(data)
print(f"Score qualité : {quality['score']}/100")
print(f"Taux de détection : {quality['detection_rate']:.1%}")
print(f"Jitter moyen : {quality['jitter']:.4f}")
```

### Comblement des gaps

```python
from myogait import fill_gaps

# Interpolation linéaire des gaps courts (max 10 frames)
data = fill_gaps(data, method="linear", max_gap_frames=10)

# Interpolation spline pour les gaps plus longs
data = fill_gaps(data, method="spline", max_gap_frames=20)
```

### Dashboard qualité

```python
from myogait import plot_quality_dashboard

fig = plot_quality_dashboard(data)
fig.savefig("quality_dashboard.png", dpi=150)
```

---

## 5. Angles articulaires

### Angles sagittaux (standard)

```python
from myogait import compute_angles

data = compute_angles(data)

# Accès aux angles par frame
frame_0 = data["angles"]["frames"][0]
print(f"Hip L : {frame_0['hip_L']:.1f}°")
print(f"Knee L : {frame_0['knee_L']:.1f}°")
print(f"Ankle L : {frame_0['ankle_L']:.1f}°")
print(f"Trunk : {frame_0['trunk_angle']:.1f}°")
print(f"Pelvis tilt : {frame_0['pelvis_tilt']:.1f}°")
```

**Convention ISB :**
- Hip : flexion (+), extension (-)
- Knee : 0° = extension complète, flexion (+)
- Ankle : dorsiflexion (+), plantarflexion (-)

### Angles étendus (bras, tête)

```python
from myogait import compute_extended_angles

data = compute_extended_angles(data)

frame_0 = data["angles"]["frames"][0]
print(f"Shoulder flex L : {frame_0['shoulder_flex_L']:.1f}°")
print(f"Elbow flex L : {frame_0['elbow_flex_L']:.1f}°")
print(f"Head angle : {frame_0['head_angle']:.1f}°")
```

### Angles frontaux (nécessite profondeur)

```python
from myogait import compute_frontal_angles

# Requires depth data (Sapiens with_depth=True)
data = extract("video.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = compute_frontal_angles(data)

# Angles frontaux disponibles
frame_0 = data["angles_frontal"]["frames"][0]
print(f"Pelvis obliquity : {frame_0.get('pelvis_list', 'N/A')}°")
print(f"Hip adduction L : {frame_0.get('hip_adduction_L', 'N/A')}°")
```

### Angle de progression du pied

```python
from myogait import foot_progression_angle

data = foot_progression_angle(data)
```

---

## 6. Détection des événements de marche

### Méthodes disponibles

```python
from myogait import list_event_methods

methods = list_event_methods()
print(methods)
# Méthodes intégrées : zeni, velocity, crossing, oconnor
# Méthodes gaitkit :   gk_bike, gk_zeni, gk_oconnor, gk_hreljac,
#                      gk_mickelborough, gk_ghoussayni, gk_vancanneyt,
#                      gk_dgei, gk_ensemble
```

### Détection simple

```python
from myogait import detect_events

# gk_bike : Bayesian BIS — meilleur F1 score (0.80)
data = detect_events(data, method="gk_bike")

# Méthode classique (Zeni 2008)
data = detect_events(data, method="zeni")

# O'Connor (vélocité du talon)
data = detect_events(data, method="oconnor")
```

### Consensus multi-méthodes

```python
from myogait import event_consensus

# Vote majoritaire entre 3 détecteurs
data = event_consensus(
    data,
    methods=["gk_bike", "gk_zeni", "gk_oconnor"],
    tolerance=3  # tolérance en frames
)
print(f"Méthode : {data['events']['method']}")  # "consensus"
print(f"Nombre de méthodes : {data['events']['n_methods']}")
```

### Ensemble gaitkit (pondéré par F1 benchmark)

```python
# Utilise les poids benchmark pour combiner les détecteurs
data = detect_events(data, method="gk_ensemble")
```

### Validation des événements

```python
from myogait import validate_events

# Vérification de plausibilité biomécanique
report = validate_events(data)
print(f"Événements valides : {report['valid']}")
```

---

## 7. Segmentation en cycles

```python
from myogait import segment_cycles

cycles = segment_cycles(data)

# Structure des cycles
print(f"Nombre de cycles : {len(cycles['cycles'])}")
for c in cycles["cycles"]:
    print(f"  Cycle {c['cycle_id']} ({c['side']}): "
          f"frames {c['start_frame']}-{c['end_frame']}, "
          f"stance {c['stance_pct']:.1f}%")
```

---

## 8. Analyse spatio-temporelle

### Paramètres de base

```python
from myogait import analyze_gait

stats = analyze_gait(data, cycles)
print(f"Cadence : {stats['cadence']:.1f} pas/min")
print(f"Vitesse : {stats['speed']:.2f} m/s")
print(f"Temps de stride : {stats['stride_time']:.3f} s")
print(f"Stance % : {stats['stance_pct']:.1f}%")
print(f"Indice de symétrie : {stats['symmetry_index']:.2f}")
```

### Paramètres avancés

```python
from myogait import (
    single_support_time, toe_clearance, stride_variability,
    arm_swing_analysis, speed_normalized_params, segment_lengths,
    instantaneous_cadence, compute_rom_summary
)

# Temps d'appui unipodal
sst = single_support_time(data, cycles)

# Clearance du pied pendant le swing
tc = toe_clearance(data, cycles)

# Variabilité (CV) des paramètres spatio-temporels
var = stride_variability(data, cycles)
print(f"CV stride time : {var['stride_time_cv']:.1%}")

# Analyse du balancement des bras
arms = arm_swing_analysis(data, cycles)
print(f"Amplitude bras L : {arms['amplitude_L']:.1f}°")
print(f"Asymétrie : {arms['asymmetry']:.2f}")

# Paramètres adimensionnels (Hof 1996)
norm = speed_normalized_params(data, cycles, height_m=1.75)
print(f"Froude : {norm['froude']:.3f}")

# Longueurs segmentaires
segs = segment_lengths(data)
print(f"Fémur L : {segs['left_femur']['mean']:.3f}")

# Cadence instantanée
cad = instantaneous_cadence(data)

# Résumé des ROM par articulation
rom = compute_rom_summary(data, cycles)
```

---

## 9. Scores cliniques

### GPS-2D (Gait Profile Score adapté 2D)

```python
from myogait import gait_profile_score_2d

gps = gait_profile_score_2d(cycles)
print(f"GPS-2D : {gps['gps']:.1f}°")
print(f"Note : {gps['note']}")
# GPS < 5° : marche normale
# GPS 5-10° : déviation légère
# GPS > 10° : déviation significative
```

### SDI (Sagittal Deviation Index)

```python
from myogait import sagittal_deviation_index

sdi = sagittal_deviation_index(cycles)
print(f"SDI : {sdi['sdi']:.1f}")
# SDI = 100 : marche normale
# SDI < 80 : déviation significative
# SDI > 100 : au-dessus de la normale (peu fréquent)
```

### GVS (Gait Variable Scores — par articulation)

```python
from myogait import gait_variable_scores

gvs = gait_variable_scores(cycles)
for joint, score in gvs["scores"].items():
    status = "OK" if score < 5.0 else "DÉVIÉ"
    print(f"  {joint}: {score:.1f}° [{status}]")
```

### MAP (Movement Analysis Profile)

```python
from myogait import movement_analysis_profile, plot_gvs_profile

map_data = movement_analysis_profile(cycles)

# Visualisation
fig = plot_gvs_profile(gvs)
fig.savefig("movement_analysis_profile.png", dpi=150)
```

### Avec variables frontales

```python
# Si les angles frontaux sont disponibles dans les cycles
gps = gait_profile_score_2d(cycles, include_frontal=True)
sdi = sagittal_deviation_index(cycles, include_frontal=True)
print(f"GPS (sagittal + frontal) : {gps['gps']:.1f}°")
```

---

## 10. Comparaison normative

### Courbes normatives disponibles

```python
from myogait import list_joints, list_strata, get_normative_curve, get_normative_band

# Articulations disponibles
print(list_joints())
# ['hip_flexion', 'knee_flexion', 'ankle_dorsiflexion',
#  'trunk_flexion', 'pelvis_tilt',
#  'pelvis_obliquity', 'hip_adduction', 'knee_valgus']

# Strates d'âge
print(list_strata())
# ['adult', 'elderly', 'pediatric']

# Courbe normative (101 points, 0-100% du cycle)
mean, sd = get_normative_curve("hip_flexion", stratum="adult")

# Bande normative (mean ± 1 SD)
mean, lower, upper = get_normative_band("knee_flexion", stratum="elderly")
```

### Plot de comparaison

```python
from myogait import plot_normative_comparison

# Plan sagittal uniquement
fig = plot_normative_comparison(data, cycles, plane="sagittal")
fig.savefig("sagittal_vs_normative.png", dpi=150)

# Plan frontal uniquement
fig = plot_normative_comparison(data, cycles, plane="frontal")

# Les deux plans
fig = plot_normative_comparison(data, cycles, plane="both", stratum="adult")
fig.savefig("full_normative_comparison.png", dpi=150)
```

---

## 11. Visualisation et plots

### Dashboard de synthèse

```python
from myogait import plot_summary

fig = plot_summary(data, cycles, stats)
fig.savefig("dashboard.png", dpi=150)
```

### Angles articulaires

```python
from myogait import plot_angles, plot_cycles

# Angles bruts sur tout l'enregistrement
fig = plot_angles(data)
fig.savefig("angles_bruts.png", dpi=150)

# Cycles superposés et moyennés
fig = plot_cycles(data, cycles)
fig.savefig("cycles.png", dpi=150)
```

### Événements de marche

```python
from myogait import plot_events

fig = plot_events(data)
fig.savefig("events.png", dpi=150)
```

### Diagramme de phase

```python
from myogait import plot_phase_plane

fig = plot_phase_plane(data, cycles)
fig.savefig("phase_plane.png", dpi=150)
```

### Profil de cadence

```python
from myogait import plot_cadence_profile, instantaneous_cadence

cad = instantaneous_cadence(data)
fig = plot_cadence_profile(data, cad)
fig.savefig("cadence_profile.png", dpi=150)
```

### ROM summary

```python
from myogait import plot_rom_summary, compute_rom_summary

rom = compute_rom_summary(data, cycles)
fig = plot_rom_summary(rom)
fig.savefig("rom_summary.png", dpi=150)
```

### Butterfly plot (symétrie L/R)

```python
from myogait import plot_butterfly

fig = plot_butterfly(data, cycles)
fig.savefig("butterfly.png", dpi=150)
```

### Balancement des bras

```python
from myogait import plot_arm_swing

fig = plot_arm_swing(data, cycles)
fig.savefig("arm_swing.png", dpi=150)
```

---

## 12. Vidéo annotée et stick figure

### Skeleton overlay sur la vidéo

```python
from myogait import render_skeleton_video

# Overlay avec angles et événements
render_skeleton_video(
    "marche.mp4", data, "marche_overlay.mp4",
    show_angles=True, show_events=True
)
```

### Stick figure anonymisée

```python
from myogait import render_stickfigure_animation

# GIF animé
render_stickfigure_animation(data, "stickfigure.gif")

# Avec tracé de trajectoire
render_stickfigure_animation(data, "stickfigure_trail.gif", show_trail=True)

# Avec segmentation en cycles
render_stickfigure_animation(data, "stickfigure_cycles.gif", cycles=cycles)
```

---

## 13. Rapport PDF clinique

### Rapport standard

```python
from myogait import generate_report

# Rapport en français
generate_report(data, cycles, stats, "rapport_marche.pdf", language="fr")

# Rapport en anglais
generate_report(data, cycles, stats, "gait_report.pdf", language="en")
```

Le rapport contient :
- Page de synthèse (paramètres spatio-temporels)
- Angles articulaires par cycle
- Comparaison normative (sagittal)
- Comparaison frontale (si données disponibles)
- GVS profile
- Dashboard qualité

### Rapport longitudinal (multi-sessions)

```python
from myogait import generate_longitudinal_report

# Comparer plusieurs sessions
sessions = {
    "2024-01-15": (data_1, cycles_1, stats_1),
    "2024-04-20": (data_2, cycles_2, stats_2),
    "2024-07-10": (data_3, cycles_3, stats_3),
}
generate_longitudinal_report(sessions, "evolution.pdf", language="fr")
```

---

## 14. Export vers OpenSim

### Fichier .trc (marqueurs)

```python
from myogait import export_trc

# Export avec conversion d'unités basée sur la taille
export_trc(data, "markers.trc", opensim_model="gait2392")

# Avec profondeur Sapiens pour les coordonnées Z
export_trc(data, "markers_3d.trc", use_depth=True, depth_scale=1.0)

# Sans taille → coordonnées normalisées
export_trc(data, "markers_norm.trc")
```

### Fichier .mot (cinématique)

```python
from myogait import export_mot

# Angles articulaires + translations du pelvis
export_mot(data, "kinematics.mot")
```

### Setup Scale Tool

```python
from myogait import export_opensim_scale_setup

# Génère le XML pour OpenSim Scale Tool
data["subject"] = {"weight_kg": 75.0, "height_m": 1.75, "name": "Patient01"}
export_opensim_scale_setup(
    data, "scale_setup.xml",
    model_file="gait2392_simbody.osim",
    output_model="scaled_model.osim",
    static_frames=(0, 30)  # frames pour la pose statique
)
```

### Setup Inverse Kinematics

```python
from myogait import export_ik_setup

export_ik_setup(
    "markers.trc", "ik_setup.xml",
    model_file="scaled_model.osim",
    output_motion="ik_results.mot",
    start_time=0.0, end_time=5.0
)
```

### Setup MocoTrack

```python
from myogait import export_moco_setup

export_moco_setup(
    "ik_results.mot", "moco_setup.xml",
    model_file="scaled_model.osim",
    start_time=0.0, end_time=2.0
)
```

### Pipeline OpenSim complet

```python
from myogait import (
    extract, normalize, compute_angles, detect_events,
    export_trc, export_mot,
    export_opensim_scale_setup, export_ik_setup
)

# 1. Pipeline myogait
data = extract("marche.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = detect_events(data, method="gk_bike")
data["subject"] = {"weight_kg": 72.0, "height_m": 1.78, "name": "Subject01"}

# 2. Export fichiers OpenSim
export_trc(data, "subject01.trc", opensim_model="gait2392", use_depth=True)
export_mot(data, "subject01.mot")

# 3. Setup files
export_opensim_scale_setup(data, "scale.xml", model_file="gait2392.osim",
                           output_model="subject01_scaled.osim")
export_ik_setup("subject01.trc", "ik.xml", model_file="subject01_scaled.osim",
                output_motion="subject01_ik.mot")

# 4. Lancer dans OpenSim (via opensim-cmd ou l'API Python opensim)
# opensim-cmd run-tool scale.xml
# opensim-cmd run-tool ik.xml
```

---

## 15. Export vers Pose2Sim

Pose2Sim utilise des fichiers JSON au format OpenPose pour la triangulation
multi-caméras. myogait peut servir de front-end d'extraction.

```python
from myogait import extract, export_openpose_json

# Extraction depuis chaque caméra
for cam in ["cam1.mp4", "cam2.mp4", "cam3.mp4", "cam4.mp4"]:
    data = extract(cam, model="mediapipe")
    cam_name = cam.replace(".mp4", "")
    export_openpose_json(data, f"./pose2sim/{cam_name}/", model="BODY_25")

# Résultat : 1 fichier JSON par frame par caméra
# Structure compatible Pose2Sim pour triangulation → OpenSim
```

### Formats supportés

```python
# COCO 17 keypoints
export_openpose_json(data, "./output/", model="COCO")

# BODY_25 (25 keypoints, standard OpenPose)
export_openpose_json(data, "./output/", model="BODY_25")

# HALPE_26 (26 keypoints, AlphaPose)
export_openpose_json(data, "./output/", model="HALPE_26")
```

---

## 16. Export multi-formats

### CSV

```python
from myogait import export_csv

export_csv(data, "./csv_output/", cycles, stats)
# Crée : angles.csv, events.csv, spatiotemporal.csv, landmarks.csv
```

### Excel

```python
from myogait import export_excel

export_excel(data, "analyse_marche.xlsx", cycles, stats)
# Un onglet par type de données
```

### Pandas DataFrame

```python
from myogait import to_dataframe

# Angles articulaires
df_angles = to_dataframe(data, what="angles")
print(df_angles.head())

# Landmarks bruts
df_lm = to_dataframe(data, what="landmarks")

# Événements
df_ev = to_dataframe(data, what="events")

# Tout
df_all = to_dataframe(data, what="all")
```

### JSON compact

```python
from myogait import export_summary_json

export_summary_json(data, cycles, stats, "summary.json")
```

### C3D (optionnel)

```python
from myogait import export_c3d

# Nécessite : pip install myogait[c3d]
export_c3d(data, "markers.c3d")
```

---

## 17. Analyse frontale

L'analyse frontale nécessite des données de profondeur (Sapiens).

```python
from myogait import (
    extract, normalize, compute_angles, compute_frontal_angles,
    detect_events, segment_cycles,
    gait_profile_score_2d, plot_normative_comparison
)

# Extraction avec profondeur
data = extract("marche.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])

# Angles sagittaux ET frontaux
data = compute_angles(data)
data = compute_frontal_angles(data)

# Pipeline standard
data = detect_events(data, method="gk_bike")
cycles = segment_cycles(data)

# GPS incluant les variables frontales
gps = gait_profile_score_2d(cycles, include_frontal=True)
print(f"GPS (sagittal + frontal) : {gps['gps']:.1f}°")

# Visualisation des deux plans
fig = plot_normative_comparison(data, cycles, plane="both")
fig.savefig("sagittal_frontal_comparison.png", dpi=150)
```

---

## 18. Détection de pathologies

```python
from myogait import (
    detect_pathologies, detect_equinus,
    detect_antalgic, detect_parkinsonian
)

# Détection automatique multi-patterns
pathologies = detect_pathologies(data, cycles)
for name, detected in pathologies.items():
    if detected:
        print(f"  ⚠ {name} détecté")

# Détections spécifiques
equinus = detect_equinus(cycles)
# → True si dorsiflexion ≤ 0° pendant le stance

antalgic = detect_antalgic(cycles)
# → True si asymétrie stance > 55% d'un côté

parkinsonian = detect_parkinsonian(data, cycles)
# → True si stride court + arm swing réduit + cadence élevée
```

---

## 19. Analyse longitudinale

### Comparaison de sessions

```python
from myogait import plot_session_comparison, plot_longitudinal

# Comparer deux sessions (avant / après traitement)
fig = plot_session_comparison(cycles_avant, cycles_apres,
                               labels=["Avant", "Après"])
fig.savefig("comparison.png", dpi=150)

# Évolution longitudinale (GPS, cadence, symétrie)
sessions_data = [
    {"date": "2024-01", "gps": 8.2, "cadence": 95, "symmetry": 0.85},
    {"date": "2024-04", "gps": 6.5, "cadence": 102, "symmetry": 0.91},
    {"date": "2024-07", "gps": 5.1, "cadence": 108, "symmetry": 0.95},
]
fig = plot_longitudinal(sessions_data)
fig.savefig("longitudinal.png", dpi=150)
```

---

## 20. CLI

### Pipeline complet

```bash
# MediaPipe (défaut)
myogait run marche.mp4

# Sapiens avec profondeur
myogait run marche.mp4 -m sapiens-top --with-depth

# ViTPose
myogait run marche.mp4 -m vitpose-large
```

### Extraction seule

```bash
myogait extract marche.mp4 -m sapiens-quick --with-depth --with-seg
```

### Analyse d'un JSON existant

```bash
myogait analyze resultat.json --csv --pdf --language fr
```

### Batch processing

```bash
myogait batch *.mp4 -o resultats/ -m mediapipe
```

### Téléchargement de modèles

```bash
myogait download --list
myogait download sapiens-0.3b
myogait download sapiens-depth-1b
```

---

## 21. Configuration YAML

```python
from myogait import load_config, save_config

config = load_config("pipeline.yaml")
```

Exemple de fichier `pipeline.yaml` :

```yaml
extraction:
  model: mediapipe

preprocessing:
  filters:
    - butterworth
  cutoff_freq: 6.0
  confidence_threshold: 0.3

angles:
  method: sagittal_vertical_axis
  calibrate: true

events:
  method: gk_bike

analysis:
  height_m: 1.75
  weight_kg: 72.0

export:
  csv: true
  pdf: true
  mot: true
  trc: true
  language: fr
```

---

## 22. Cas d'usage cliniques

### Cas 1 : Suivi post-opératoire

Un patient opéré du genou. Évaluation de la récupération à J+30, J+90, J+180.

```python
from myogait import *

dates = ["J+30", "J+90", "J+180"]
videos = ["j30.mp4", "j90.mp4", "j180.mp4"]
results = {}

for date, video in zip(dates, videos):
    data = extract(video, model="mediapipe")
    data = normalize(data, filters=["butterworth"])
    data = compute_angles(data)
    data = detect_events(data, method="gk_bike")
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    gps = gait_profile_score_2d(cycles)
    sdi = sagittal_deviation_index(cycles)

    results[date] = {
        "gps": gps["gps"],
        "sdi": sdi["sdi"],
        "cadence": stats["cadence"],
        "speed": stats["speed"],
    }
    generate_report(data, cycles, stats, f"rapport_{date}.pdf", language="fr")

# Tableau de suivi
for date, r in results.items():
    print(f"{date}: GPS={r['gps']:.1f}° SDI={r['sdi']:.0f} "
          f"Cadence={r['cadence']:.0f} Vitesse={r['speed']:.2f}")
```

### Cas 2 : Screening neuromusculaire (Duchenne)

```python
data = extract("duchenne_patient.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = compute_extended_angles(data)  # bras, tête
data = detect_events(data, method="gk_ensemble")
cycles = segment_cycles(data)
stats = analyze_gait(data, cycles)

# Score GPS avec strate pédiatrique
from myogait import select_stratum
stratum = select_stratum(age=12)  # "pediatric"
gps = gait_profile_score_2d(cycles)

# Détection patterns
pathologies = detect_pathologies(data, cycles)
arms = arm_swing_analysis(data, cycles)

# Rapport complet
generate_report(data, cycles, stats, "rapport_duchenne.pdf", language="fr")
```

### Cas 3 : Recherche — export pour OpenSim et analyse statistique

```python
import pandas as pd
from myogait import *

subjects = ["s01.mp4", "s02.mp4", "s03.mp4"]
all_stats = []

for video in subjects:
    data = extract(video, model="vitpose-large")
    data = normalize(data, filters=["butterworth"])
    data = compute_angles(data)
    data = detect_events(data, method="gk_bike")
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    gps = gait_profile_score_2d(cycles)

    # Export OpenSim
    name = video.replace(".mp4", "")
    export_trc(data, f"{name}.trc", opensim_model="gait2392")
    export_mot(data, f"{name}.mot")

    # Collecter stats
    stats["subject"] = name
    stats["gps"] = gps["gps"]
    all_stats.append(stats)

# DataFrame pour analyse statistique
df = pd.DataFrame(all_stats)
df.to_csv("group_stats.csv", index=False)
print(df[["subject", "cadence", "speed", "gps"]].to_string())
```

---

## Ressources

- **GitHub** : https://github.com/IDMDataHub/myogait
- **PyPI** : https://pypi.org/project/myogait/
- **gaitkit** : https://github.com/IDMDataHub/gaitkit
- **Institut de Myologie** : https://www.institut-myologie.org/
- **Fondation Myologie** : https://www.fondation-myologie.org/
- **AFM-Téléthon** : https://www.afm-telethon.fr/
- **Téléthon** : https://www.telethon.fr/

---

*myogait est développé par Frederic Fer au sein du PhysioEvalLab,
[Institut de Myologie](https://www.institut-myologie.org/), Paris, avec le
soutien de l'[AFM-Téléthon](https://www.afm-telethon.fr/), de la
[Fondation Myologie](https://www.fondation-myologie.org/) et du
[Téléthon](https://www.telethon.fr/).*
