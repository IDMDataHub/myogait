# myogait — Complete Tutorial

A practical guide covering all myogait use cases, from video extraction
to OpenSim export, with reproducible code examples.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Basic Pipeline](#2-basic-pipeline)
3. [Choosing a Pose Backend](#3-choosing-a-pose-backend)
4. [Data Quality](#4-data-quality)
5. [Joint Angles](#5-joint-angles)
6. [Gait Event Detection](#6-gait-event-detection)
7. [Cycle Segmentation](#7-cycle-segmentation)
8. [Spatiotemporal Analysis](#8-spatiotemporal-analysis)
9. [Clinical Scores (GPS-2D, SDI, GVS)](#9-clinical-scores)
10. [Normative Comparison](#10-normative-comparison)
11. [Visualization and Plots](#11-visualization-and-plots)
12. [Annotated Video and Stick Figure](#12-annotated-video-and-stick-figure)
13. [Clinical PDF Report](#13-clinical-pdf-report)
14. [Export to OpenSim](#14-export-to-opensim)
15. [Export to Pose2Sim](#15-export-to-pose2sim)
16. [Multi-Format Export](#16-multi-format-export)
17. [Frontal Plane Analysis (with Depth)](#17-frontal-plane-analysis)
18. [Pathology Detection](#18-pathology-detection)
19. [Longitudinal Multi-Session Analysis](#19-longitudinal-analysis)
20. [Command-Line Interface (CLI)](#20-cli)
21. [YAML Configuration](#21-yaml-configuration)
22. [Clinical Use Cases](#22-clinical-use-cases)

---

## 1. Installation

### Basic Installation

```bash
pip install myogait
```

This installs myogait with its required dependencies: numpy, pandas, scipy,
opencv, matplotlib, and **gaitkit** (event detection).

### With a Pose Backend

```bash
# MediaPipe — lightweight, CPU only, 33 landmarks
pip install myogait[mediapipe]

# YOLO — fast, GPU supported, 17 COCO keypoints
pip install myogait[yolo]

# Sapiens — Meta AI, depth + segmentation
pip install myogait[sapiens]

# ViTPose — state-of-the-art accuracy
pip install myogait[vitpose]

# RTMW — 133 keypoints (full body + hands + face)
pip install myogait[rtmw]

# Install everything
pip install myogait[all]
```

### Verification

```python
import myogait
print(myogait.__version__)  # 0.3.0
print(len(myogait.__all__))  # 90+ public functions
```

---

## 2. Basic Pipeline

The complete pipeline in 6 steps:

```python
from myogait import (
    extract, normalize, compute_angles,
    detect_events, segment_cycles, analyze_gait
)

# Step 1: Extract landmarks from the video
data = extract("sagittal_walk.mp4", model="mediapipe")
print(f"{len(data['frames'])} frames extracted at {data['meta']['fps']} FPS")

# Step 2: Filtering and normalization
data = normalize(data, filters=["butterworth"])

# Step 3: Compute joint angles
data = compute_angles(data)
print(f"Angles: {list(data['angles']['frames'][0].keys())}")

# Step 4: Detect events (heel strike, toe off)
data = detect_events(data, method="gk_bike")
n_hs = len(data["events"]["left_hs"]) + len(data["events"]["right_hs"])
print(f"{n_hs} heel strikes detected")

# Step 5: Segment into gait cycles
cycles = segment_cycles(data)

# Step 6: Spatiotemporal analysis
stats = analyze_gait(data, cycles)
print(f"Cadence: {stats['cadence']:.1f} steps/min")
print(f"Speed: {stats['speed']:.2f} m/s")
print(f"Stance time: {stats['stance_pct']:.1f}%")
```

### The `data` Dictionary

All functions operate on a `data` dictionary that is progressively enriched
at each step:

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

## 3. Choosing a Pose Backend

### Experimental Input Degradation (AIM Benchmark Only)

Use this only for robustness benchmarking. By default, all values are neutral
and no degradation is applied.

```python
data = extract(
    "video.mp4",
    model="mediapipe",
    experimental={
        "enabled": True,
        "target_fps": 15.0,
        "downscale": 0.6,
        "contrast": 0.7,
        "aspect_ratio": 1.2,
        "perspective_x": 0.2,
        "perspective_y": 0.1,
    },
)
```

### Experimental VICON Alignment (AIM, Single Video)

```python
from myogait import run_single_trial_vicon_benchmark

data = run_single_trial_vicon_benchmark(
    data,
    trial_dir="/path/to/trial_01_1",
    vicon_fps=200.0,
    max_lag_seconds=10.0,
)

print(data["experimental"]["vicon_benchmark"]["metrics"].keys())
```

### Experimental Single-Pair Benchmark Runner (AIM Only)

```python
from myogait import run_single_pair_benchmark

manifest = run_single_pair_benchmark(
    video_path="video.mp4",
    vicon_trial_dir="/path/to/trial_01_1",
    output_dir="./benchmark_out",
    benchmark_config={
        "models": ["mediapipe", "yolo"],     # or "all"
        "event_methods": "all",
        "normalization_variants": [
            {"name": "none", "enabled": False, "kwargs": {}},
            {"name": "bw", "enabled": True, "kwargs": {"filters": ["butterworth"]}},
        ],
        "degradation_variants": [
            {"name": "none", "experimental": {"enabled": False}},
            {"name": "robust_1", "experimental": {"enabled": True, "downscale": 0.7, "contrast": 0.8}},
        ],
        "continue_on_error": True,
    },
)

print(manifest["summary_csv"])
```

This helper is experimental and should only be used for AIM benchmark studies.

### Quick Comparison

```python
# MediaPipe — simplest option, good for prototyping
data = extract("video.mp4", model="mediapipe")

# YOLO — fast and robust
data = extract("video.mp4", model="yolo")

# Sapiens — most accurate + monocular depth
data = extract("video.mp4", model="sapiens-top", with_depth=True, with_seg=True)

# ViTPose — excellent accuracy/speed trade-off
data = extract("video.mp4", model="vitpose-large")

# RTMW — 133 keypoints (body, hands, face)
data = extract("video.mp4", model="rtmw")
```

### Model Accuracy Information

```python
from myogait import model_accuracy_info

info = model_accuracy_info("mediapipe")
print(f"MAE: {info['mae_px']} px")
print(f"PCK@0.5: {info['pck_05']}")
print(f"Reference: {info['reference']}")
```

---

## 4. Data Quality

### Confidence Filtering

```python
from myogait import confidence_filter, detect_outliers, data_quality_score

# Remove landmarks with confidence < 30%
data = confidence_filter(data, threshold=0.3)

# Detect and interpolate outliers (z-score > 3)
data = detect_outliers(data, z_thresh=3.0)

# Overall quality score (0-100)
quality = data_quality_score(data)
print(f"Quality score: {quality['score']}/100")
print(f"Detection rate: {quality['detection_rate']:.1%}")
print(f"Mean jitter: {quality['jitter']:.4f}")
```

### Gap Filling

```python
from myogait import fill_gaps

# Linear interpolation for short gaps (max 10 frames)
data = fill_gaps(data, method="linear", max_gap_frames=10)

# Spline interpolation for longer gaps
data = fill_gaps(data, method="spline", max_gap_frames=20)
```

### Quality Dashboard

```python
from myogait import plot_quality_dashboard

fig = plot_quality_dashboard(data)
fig.savefig("quality_dashboard.png", dpi=150)
```

---

## 5. Joint Angles

### Sagittal Angles (Standard)

```python
from myogait import compute_angles

data = compute_angles(data)

# Access angles per frame
frame_0 = data["angles"]["frames"][0]
print(f"Hip L: {frame_0['hip_L']:.1f}°")
print(f"Knee L: {frame_0['knee_L']:.1f}°")
print(f"Ankle L: {frame_0['ankle_L']:.1f}°")
print(f"Trunk: {frame_0['trunk_angle']:.1f}°")
print(f"Pelvis tilt: {frame_0['pelvis_tilt']:.1f}°")
```

**ISB Convention:**
- Hip: flexion (+), extension (-)
- Knee: 0° = full extension, flexion (+)
- Ankle: dorsiflexion (+), plantarflexion (-)

### Extended Angles (Arms, Head)

```python
from myogait import compute_extended_angles

data = compute_extended_angles(data)

frame_0 = data["angles"]["frames"][0]
print(f"Shoulder flex L: {frame_0['shoulder_flex_L']:.1f}°")
print(f"Elbow flex L: {frame_0['elbow_flex_L']:.1f}°")
print(f"Head angle: {frame_0['head_angle']:.1f}°")
```

### Frontal Angles (Requires Depth)

```python
from myogait import compute_frontal_angles

# Requires depth data (Sapiens with_depth=True)
data = extract("video.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = compute_frontal_angles(data)

# Available frontal angles
frame_0 = data["angles_frontal"]["frames"][0]
print(f"Pelvis obliquity: {frame_0.get('pelvis_list', 'N/A')}°")
print(f"Hip adduction L: {frame_0.get('hip_adduction_L', 'N/A')}°")
```

### Foot Progression Angle

```python
from myogait import foot_progression_angle

data = foot_progression_angle(data)
```

---

## 6. Gait Event Detection

### Available Methods

```python
from myogait import list_event_methods

methods = list_event_methods()
print(methods)
# Built-in methods: zeni, velocity, crossing, oconnor
# gaitkit methods:   gk_bike, gk_zeni, gk_oconnor, gk_hreljac,
#                    gk_mickelborough, gk_ghoussayni, gk_vancanneyt,
#                    gk_dgei, gk_ensemble
```

### Simple Detection

```python
from myogait import detect_events

# gk_bike: Bayesian BIS — best F1 score (0.80)
data = detect_events(data, method="gk_bike")

# Classic method (Zeni 2008)
data = detect_events(data, method="zeni")

# O'Connor (heel velocity)
data = detect_events(data, method="oconnor")
```

### Multi-Method Consensus

```python
from myogait import event_consensus

# Majority vote across 3 detectors
data = event_consensus(
    data,
    methods=["gk_bike", "gk_zeni", "gk_oconnor"],
    tolerance=3  # tolerance in frames
)
print(f"Method: {data['events']['method']}")  # "consensus"
print(f"Number of methods: {data['events']['n_methods']}")
```

### gaitkit Ensemble (Weighted by F1 Benchmark)

```python
# Uses benchmark weights to combine detectors
data = detect_events(data, method="gk_ensemble")
```

### Event Validation

```python
from myogait import validate_events

# Biomechanical plausibility check
report = validate_events(data)
print(f"Valid events: {report['valid']}")
```

---

## 7. Cycle Segmentation

```python
from myogait import segment_cycles

cycles = segment_cycles(data)

# Cycle structure
print(f"Number of cycles: {len(cycles['cycles'])}")
for c in cycles["cycles"]:
    print(f"  Cycle {c['cycle_id']} ({c['side']}): "
          f"frames {c['start_frame']}-{c['end_frame']}, "
          f"stance {c['stance_pct']:.1f}%")
```

---

## 8. Spatiotemporal Analysis

### Basic Parameters

```python
from myogait import analyze_gait

stats = analyze_gait(data, cycles)
print(f"Cadence: {stats['cadence']:.1f} steps/min")
print(f"Speed: {stats['speed']:.2f} m/s")
print(f"Stride time: {stats['stride_time']:.3f} s")
print(f"Stance %: {stats['stance_pct']:.1f}%")
print(f"Symmetry index: {stats['symmetry_index']:.2f}")
```

### Advanced Parameters

```python
from myogait import (
    single_support_time, toe_clearance, stride_variability,
    arm_swing_analysis, speed_normalized_params, segment_lengths,
    instantaneous_cadence, compute_rom_summary
)

# Single support time
sst = single_support_time(data, cycles)

# Foot clearance during swing
tc = toe_clearance(data, cycles)

# Variability (CV) of spatiotemporal parameters
var = stride_variability(data, cycles)
print(f"CV stride time: {var['stride_time_cv']:.1%}")

# Arm swing analysis
arms = arm_swing_analysis(data, cycles)
print(f"Arm amplitude L: {arms['amplitude_L']:.1f}°")
print(f"Asymmetry: {arms['asymmetry']:.2f}")

# Dimensionless parameters (Hof 1996)
norm = speed_normalized_params(data, cycles, height_m=1.75)
print(f"Froude: {norm['froude']:.3f}")

# Segment lengths
segs = segment_lengths(data)
print(f"Femur L: {segs['left_femur']['mean']:.3f}")

# Instantaneous cadence
cad = instantaneous_cadence(data)

# ROM summary by joint
rom = compute_rom_summary(data, cycles)
```

---

## 9. Clinical Scores

### GPS-2D (2D-Adapted Gait Profile Score)

```python
from myogait import gait_profile_score_2d

gps = gait_profile_score_2d(cycles)
print(f"GPS-2D: {gps['gps']:.1f}°")
print(f"Rating: {gps['note']}")
# GPS < 5°: normal gait
# GPS 5-10°: mild deviation
# GPS > 10°: significant deviation
```

### SDI (Sagittal Deviation Index)

The SDI is a simplified, z-score-based deviation index using sagittal-plane
data only (hip, knee, ankle, trunk). It is **not** the GDI of Schwartz &
Rozumalski (2008), which requires 9 kinematic variables from 3D motion
capture and uses a PCA-based algorithm. For the standard 3D GDI, use a full
3D motion capture system and the original PCA-based algorithm.

```python
from myogait import sagittal_deviation_index

sdi = sagittal_deviation_index(cycles)
print(f"SDI: {sdi['sdi']:.1f}")
# SDI = 100: normal gait
# SDI < 80: significant deviation
# SDI > 100: above normal (uncommon)
```

### GVS (Gait Variable Scores — Per Joint)

```python
from myogait import gait_variable_scores

gvs = gait_variable_scores(cycles)
for joint, score in gvs["scores"].items():
    status = "OK" if score < 5.0 else "DEVIATED"
    print(f"  {joint}: {score:.1f}° [{status}]")
```

### MAP (Movement Analysis Profile)

```python
from myogait import movement_analysis_profile, plot_gvs_profile

map_data = movement_analysis_profile(cycles)

# Visualization
fig = plot_gvs_profile(gvs)
fig.savefig("movement_analysis_profile.png", dpi=150)
```

### With Frontal Variables

```python
# If frontal angles are available in the cycles
gps = gait_profile_score_2d(cycles, include_frontal=True)
sdi = sagittal_deviation_index(cycles, include_frontal=True)
print(f"GPS (sagittal + frontal): {gps['gps']:.1f}°")
```

---

## 10. Normative Comparison

### Available Normative Curves

```python
from myogait import list_joints, list_strata, get_normative_curve, get_normative_band

# Available joints
print(list_joints())
# ['hip_flexion', 'knee_flexion', 'ankle_dorsiflexion',
#  'trunk_flexion', 'pelvis_tilt',
#  'pelvis_obliquity', 'hip_adduction', 'knee_valgus']

# Age strata
print(list_strata())
# ['adult', 'elderly', 'pediatric']

# Normative curve (101 points, 0-100% of the cycle)
mean, sd = get_normative_curve("hip_flexion", stratum="adult")

# Normative band (mean +/- 1 SD)
mean, lower, upper = get_normative_band("knee_flexion", stratum="elderly")
```

### Comparison Plot

```python
from myogait import plot_normative_comparison

# Sagittal plane only
fig = plot_normative_comparison(data, cycles, plane="sagittal")
fig.savefig("sagittal_vs_normative.png", dpi=150)

# Frontal plane only
fig = plot_normative_comparison(data, cycles, plane="frontal")

# Both planes
fig = plot_normative_comparison(data, cycles, plane="both", stratum="adult")
fig.savefig("full_normative_comparison.png", dpi=150)
```

---

## 11. Visualization and Plots

### Summary Dashboard

```python
from myogait import plot_summary

fig = plot_summary(data, cycles, stats)
fig.savefig("dashboard.png", dpi=150)
```

### Joint Angles

```python
from myogait import plot_angles, plot_cycles

# Raw angles across the entire recording
fig = plot_angles(data)
fig.savefig("raw_angles.png", dpi=150)

# Overlaid and averaged cycles
fig = plot_cycles(data, cycles)
fig.savefig("cycles.png", dpi=150)
```

### Gait Events

```python
from myogait import plot_events

fig = plot_events(data)
fig.savefig("events.png", dpi=150)
```

### Phase Diagram

```python
from myogait import plot_phase_plane

fig = plot_phase_plane(data, cycles)
fig.savefig("phase_plane.png", dpi=150)
```

### Cadence Profile

```python
from myogait import plot_cadence_profile, instantaneous_cadence

cad = instantaneous_cadence(data)
fig = plot_cadence_profile(data, cad)
fig.savefig("cadence_profile.png", dpi=150)
```

### ROM Summary

```python
from myogait import plot_rom_summary, compute_rom_summary

rom = compute_rom_summary(data, cycles)
fig = plot_rom_summary(rom)
fig.savefig("rom_summary.png", dpi=150)
```

### Butterfly Plot (L/R Symmetry)

```python
from myogait import plot_butterfly

fig = plot_butterfly(data, cycles)
fig.savefig("butterfly.png", dpi=150)
```

### Arm Swing

```python
from myogait import plot_arm_swing

fig = plot_arm_swing(data, cycles)
fig.savefig("arm_swing.png", dpi=150)
```

---

## 12. Annotated Video and Stick Figure

### Skeleton Overlay on Video

```python
from myogait import render_skeleton_video

# Overlay with angles and events
render_skeleton_video(
    "walk.mp4", data, "walk_overlay.mp4",
    show_angles=True, show_events=True
)
```

### Anonymized Stick Figure

```python
from myogait import render_stickfigure_animation

# Animated GIF
render_stickfigure_animation(data, "stickfigure.gif")

# With trajectory trail
render_stickfigure_animation(data, "stickfigure_trail.gif", show_trail=True)

# With cycle segmentation
render_stickfigure_animation(data, "stickfigure_cycles.gif", cycles=cycles)
```

---

## 13. Clinical PDF Report

### Standard Report

```python
from myogait import generate_report

# Report in French
generate_report(data, cycles, stats, "rapport_marche.pdf", language="fr")

# Report in English
generate_report(data, cycles, stats, "gait_report.pdf", language="en")
```

The report includes:
- Summary page (spatiotemporal parameters)
- Joint angles per cycle
- Normative comparison (sagittal)
- Frontal comparison (if data available)
- GVS profile
- Quality dashboard

### Longitudinal Report (Multi-Session)

```python
from myogait import generate_longitudinal_report

# Compare multiple sessions
sessions = {
    "2024-01-15": (data_1, cycles_1, stats_1),
    "2024-04-20": (data_2, cycles_2, stats_2),
    "2024-07-10": (data_3, cycles_3, stats_3),
}
generate_longitudinal_report(sessions, "evolution.pdf", language="fr")
```

---

## 14. Export to OpenSim

### .trc File (Markers)

```python
from myogait import export_trc

# Export with unit conversion based on height
export_trc(data, "markers.trc", opensim_model="gait2392")

# With Sapiens depth for Z coordinates
export_trc(data, "markers_3d.trc", use_depth=True, depth_scale=1.0)

# Without height -> normalized coordinates
export_trc(data, "markers_norm.trc")
```

### .mot File (Kinematics)

```python
from myogait import export_mot

# Joint angles + pelvis translations
export_mot(data, "kinematics.mot")
```

### Scale Tool Setup

```python
from myogait import export_opensim_scale_setup

# Generate the XML for the OpenSim Scale Tool
data["subject"] = {"weight_kg": 75.0, "height_m": 1.75, "name": "Patient01"}
export_opensim_scale_setup(
    data, "scale_setup.xml",
    model_file="gait2392_simbody.osim",
    output_model="scaled_model.osim",
    static_frames=(0, 30)  # frames for the static pose
)
```

### Inverse Kinematics Setup

```python
from myogait import export_ik_setup

export_ik_setup(
    "markers.trc", "ik_setup.xml",
    model_file="scaled_model.osim",
    output_motion="ik_results.mot",
    start_time=0.0, end_time=5.0
)
```

### MocoTrack Setup

```python
from myogait import export_moco_setup

export_moco_setup(
    "ik_results.mot", "moco_setup.xml",
    model_file="scaled_model.osim",
    start_time=0.0, end_time=2.0
)
```

### Full OpenSim Pipeline

```python
from myogait import (
    extract, normalize, compute_angles, detect_events,
    export_trc, export_mot,
    export_opensim_scale_setup, export_ik_setup
)

# 1. myogait pipeline
data = extract("walk.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = detect_events(data, method="gk_bike")
data["subject"] = {"weight_kg": 72.0, "height_m": 1.78, "name": "Subject01"}

# 2. Export OpenSim files
export_trc(data, "subject01.trc", opensim_model="gait2392", use_depth=True)
export_mot(data, "subject01.mot")

# 3. Setup files
export_opensim_scale_setup(data, "scale.xml", model_file="gait2392.osim",
                           output_model="subject01_scaled.osim")
export_ik_setup("subject01.trc", "ik.xml", model_file="subject01_scaled.osim",
                output_motion="subject01_ik.mot")

# 4. Run in OpenSim (via opensim-cmd or the opensim Python API)
# opensim-cmd run-tool scale.xml
# opensim-cmd run-tool ik.xml
```

---

## 15. Export to Pose2Sim

Pose2Sim uses OpenPose-format JSON files for multi-camera triangulation.
myogait can serve as an extraction front-end.

```python
from myogait import extract, export_openpose_json

# Extract from each camera
for cam in ["cam1.mp4", "cam2.mp4", "cam3.mp4", "cam4.mp4"]:
    data = extract(cam, model="mediapipe")
    cam_name = cam.replace(".mp4", "")
    export_openpose_json(data, f"./pose2sim/{cam_name}/", model="BODY_25")

# Result: 1 JSON file per frame per camera
# Pose2Sim-compatible structure for triangulation -> OpenSim
```

### Supported Formats

```python
# COCO 17 keypoints
export_openpose_json(data, "./output/", model="COCO")

# BODY_25 (25 keypoints, standard OpenPose)
export_openpose_json(data, "./output/", model="BODY_25")

# HALPE_26 (26 keypoints, AlphaPose)
export_openpose_json(data, "./output/", model="HALPE_26")
```

---

## 16. Multi-Format Export

### CSV

```python
from myogait import export_csv

export_csv(data, "./csv_output/", cycles, stats)
# Creates: angles.csv, events.csv, spatiotemporal.csv, landmarks.csv
```

### Excel

```python
from myogait import export_excel

export_excel(data, "gait_analysis.xlsx", cycles, stats)
# One sheet per data type
```

### Pandas DataFrame

```python
from myogait import to_dataframe

# Joint angles
df_angles = to_dataframe(data, what="angles")
print(df_angles.head())

# Raw landmarks
df_lm = to_dataframe(data, what="landmarks")

# Events
df_ev = to_dataframe(data, what="events")

# Everything
df_all = to_dataframe(data, what="all")
```

### Compact JSON

```python
from myogait import export_summary_json

export_summary_json(data, cycles, stats, "summary.json")
```

### C3D (Optional)

```python
from myogait import export_c3d

# Requires: pip install myogait[c3d]
export_c3d(data, "markers.c3d")
```

---

## 17. Frontal Plane Analysis

Frontal plane analysis requires depth data (Sapiens).

```python
from myogait import (
    extract, normalize, compute_angles, compute_frontal_angles,
    detect_events, segment_cycles,
    gait_profile_score_2d, plot_normative_comparison
)

# Extraction with depth
data = extract("walk.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])

# Sagittal AND frontal angles
data = compute_angles(data)
data = compute_frontal_angles(data)

# Standard pipeline
data = detect_events(data, method="gk_bike")
cycles = segment_cycles(data)

# GPS including frontal variables
gps = gait_profile_score_2d(cycles, include_frontal=True)
print(f"GPS (sagittal + frontal): {gps['gps']:.1f}°")

# Visualization of both planes
fig = plot_normative_comparison(data, cycles, plane="both")
fig.savefig("sagittal_frontal_comparison.png", dpi=150)
```

---

## 18. Pathology Detection

```python
from myogait import (
    detect_pathologies, detect_equinus,
    detect_antalgic, detect_parkinsonian
)

# Automatic multi-pattern detection
pathologies = detect_pathologies(data, cycles)
for name, detected in pathologies.items():
    if detected:
        print(f"  Warning: {name} detected")

# Specific detections
equinus = detect_equinus(cycles)
# -> True if dorsiflexion <= 0 degrees during stance

antalgic = detect_antalgic(cycles)
# -> True if stance asymmetry > 55% on one side

parkinsonian = detect_parkinsonian(data, cycles)
# -> True if short stride + reduced arm swing + high cadence
```

---

## 19. Longitudinal Analysis

### Session Comparison

```python
from myogait import plot_session_comparison, plot_longitudinal

# Compare two sessions (before / after treatment)
fig = plot_session_comparison(cycles_before, cycles_after,
                               labels=["Before", "After"])
fig.savefig("comparison.png", dpi=150)

# Longitudinal trends (GPS, cadence, symmetry)
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

### Full Pipeline

```bash
# MediaPipe (default)
myogait run walk.mp4

# Sapiens with depth
myogait run walk.mp4 -m sapiens-top --with-depth

# ViTPose
myogait run walk.mp4 -m vitpose-large
```

### Extraction Only

```bash
myogait extract walk.mp4 -m sapiens-quick --with-depth --with-seg
```

### Analyze an Existing JSON

```bash
myogait analyze result.json --csv --pdf --language fr
```

### Batch Processing

```bash
myogait batch *.mp4 -o results/ -m mediapipe
```

### Model Download

```bash
myogait download --list
myogait download sapiens-0.3b
myogait download sapiens-depth-1b
```

---

## 21. YAML Configuration

```python
from myogait import load_config, save_config

config = load_config("pipeline.yaml")
```

Example `pipeline.yaml` file:

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

## 22. Clinical Use Cases

### Case 1: Post-Operative Follow-Up

A patient who underwent knee surgery. Recovery assessment at D+30, D+90, D+180.

```python
from myogait import *

dates = ["D+30", "D+90", "D+180"]
videos = ["d30.mp4", "d90.mp4", "d180.mp4"]
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
    generate_report(data, cycles, stats, f"report_{date}.pdf", language="fr")

# Follow-up table
for date, r in results.items():
    print(f"{date}: GPS={r['gps']:.1f}° SDI={r['sdi']:.0f} "
          f"Cadence={r['cadence']:.0f} Speed={r['speed']:.2f}")
```

### Case 2: Neuromuscular Screening (Duchenne)

```python
data = extract("duchenne_patient.mp4", model="sapiens-top", with_depth=True)
data = normalize(data, filters=["butterworth"])
data = compute_angles(data)
data = compute_extended_angles(data)  # arms, head
data = detect_events(data, method="gk_ensemble")
cycles = segment_cycles(data)
stats = analyze_gait(data, cycles)

# GPS score with pediatric stratum
from myogait import select_stratum
stratum = select_stratum(age=12)  # "pediatric"
gps = gait_profile_score_2d(cycles)

# Pattern detection
pathologies = detect_pathologies(data, cycles)
arms = arm_swing_analysis(data, cycles)

# Full report
generate_report(data, cycles, stats, "duchenne_report.pdf", language="fr")
```

### Case 3: Research — Export for OpenSim and Statistical Analysis

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

    # OpenSim export
    name = video.replace(".mp4", "")
    export_trc(data, f"{name}.trc", opensim_model="gait2392")
    export_mot(data, f"{name}.mot")

    # Collect stats
    stats["subject"] = name
    stats["gps"] = gps["gps"]
    all_stats.append(stats)

# DataFrame for statistical analysis
df = pd.DataFrame(all_stats)
df.to_csv("group_stats.csv", index=False)
print(df[["subject", "cadence", "speed", "gps"]].to_string())
```

---

## Resources

- **GitHub**: https://github.com/IDMDataHub/myogait
- **PyPI**: https://pypi.org/project/myogait/
- **gaitkit**: https://github.com/IDMDataHub/gaitkit
- **Institut de Myologie**: https://www.institut-myologie.org/
- **Fondation Myologie**: https://www.fondation-myologie.org/
- **AFM-Téléthon**: https://www.afm-telethon.fr/
- **Téléthon**: https://www.telethon.fr/

---

*myogait is developed by Frederic Fer at PhysioEvalLab,
[Institut de Myologie](https://www.institut-myologie.org/), Paris, with the
support of [AFM-Téléthon](https://www.afm-telethon.fr/), the
[Fondation Myologie](https://www.fondation-myologie.org/), and
[Téléthon](https://www.telethon.fr/).*
