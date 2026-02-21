# myogait

Markerless video-based gait analysis toolkit.

[![CI](https://github.com/IDMDataHub/myogait/actions/workflows/ci.yml/badge.svg)](https://github.com/IDMDataHub/myogait/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/myogait)](https://pypi.org/project/myogait/)
[![Python 3.10|3.11|3.12](https://img.shields.io/pypi/pyversions/myogait)](https://pypi.org/project/myogait/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-749%20passed-brightgreen)]()
[![Downloads](https://img.shields.io/pypi/dm/myogait)](https://pypi.org/project/myogait/)

**Author:** Frederic Fer, Institut de Myologie ([f.fer@institut-myologie.org](mailto:f.fer@institut-myologie.org))

---

## Features

- **Multi-model pose extraction** — MediaPipe, YOLO, Sapiens (3 sizes), ViTPose, RTMW, HRNet, RTMPose
- **Sapiens depth estimation** — monocular relative depth per landmark
- **Sapiens body segmentation** — 28-class body-part labels per landmark
- Butterworth, Savitzky-Golay, and Kalman filtering
- Joint angle computation (sagittal vertical axis, sagittal classic)
- 4 event detection methods (Zeni, crossing, velocity, O'Connor)
- Gait cycle segmentation and normalization
- Spatio-temporal analysis (cadence, stride time, stance%)
- Symmetry and variability indices
- Step length and walking speed estimation
- Harmonic ratio and regularity index
- Advanced pathology detection (Trendelenburg, spastic, steppage, crouch)
- Biomechanical validation against physiological ranges
- Publication-quality matplotlib plots
- Multi-page PDF clinical report
- Export to CSV, OpenSim (.mot/.trc), Excel
- YAML/JSON pipeline configuration
- CLI with `extract`, `run`, `analyze`, `batch`, `download`, and `info` commands

## Installation

```bash
pip install myogait
```

Install with a specific pose estimation backend:

```bash
pip install myogait[mediapipe]   # MediaPipe (lightweight, CPU)
pip install myogait[yolo]        # YOLO via Ultralytics
pip install myogait[sapiens]     # Sapiens (Meta AI) + Intel Arc GPU support
pip install myogait[vitpose]     # ViTPose via HuggingFace Transformers
pip install myogait[rtmw]        # RTMW 133-keypoint whole-body
pip install myogait[mmpose]      # HRNet / RTMPose via MMPose
pip install myogait[all]         # All backends
```

## Supported Pose Estimation Backends

| Backend | Install | Notes |
|---------|---------|-------|
| MediaPipe | `pip install myogait[mediapipe]` | Fast, good for real-time. 33 landmarks |
| YOLOv8-Pose | `pip install myogait[yolo]` | Fast, robust. 17 COCO keypoints |
| ViTPose | `pip install myogait[vitpose]` | State-of-the-art accuracy |
| Sapiens | `pip install myogait[sapiens]` | Meta model, depth estimation |
| RTMPose/RTMLib | `pip install myogait[rtmw]` | Real-time, ONNX optimized |
| MMPose | `pip install myogait[mmpose]` | Academic reference, many models |

## GPU Support

All models support **NVIDIA CUDA** GPUs.  Sapiens and ViTPose also support
**Intel Arc / Xe GPUs** (via `intel-extension-for-pytorch`).

| Model | CUDA | Intel Arc (XPU) | CPU |
|-------|------|-----------------|-----|
| MediaPipe | — | — | yes |
| YOLO | yes | — | yes |
| Sapiens (pose, depth, seg) | yes | yes | yes |
| ViTPose | yes | yes | yes |
| RTMW | yes (onnxruntime) | — | yes |
| HRNet / RTMPose | yes | — | yes |

## Supported Pose Models

| Name | Keypoints | Format | Backend | Install |
|------|-----------|--------|---------|---------|
| `mediapipe` | 33 | MediaPipe | Google MediaPipe Tasks (heavy) | `pip install myogait[mediapipe]` |
| `yolo` | 17 | COCO | Ultralytics YOLOv8-Pose | `pip install myogait[yolo]` |
| `sapiens-quick` | 17 COCO + 308 Goliath | COCO | Meta Sapiens 0.3B (336M params) | `pip install myogait[sapiens]` |
| `sapiens-mid` | 17 COCO + 308 Goliath | COCO | Meta Sapiens 0.6B (664M params) | `pip install myogait[sapiens]` |
| `sapiens-top` | 17 COCO + 308 Goliath | COCO | Meta Sapiens 1B (1.1B params) | `pip install myogait[sapiens]` |
| `vitpose` | 17 | COCO | ViTPose-base (HuggingFace) | `pip install myogait[vitpose]` |
| `vitpose-large` | 17 | COCO | ViTPose+-large (HuggingFace) | `pip install myogait[vitpose]` |
| `vitpose-huge` | 17 | COCO | ViTPose+-huge (HuggingFace) | `pip install myogait[vitpose]` |
| `rtmw` | 17 COCO + 133 whole-body | COCO | RTMW via rtmlib | `pip install myogait[rtmw]` |
| `hrnet` | 17 | COCO | HRNet-W48 via MMPose | `pip install myogait[mmpose]` |
| `mmpose` | 17 | COCO | RTMPose-m via MMPose | `pip install myogait[mmpose]` |

## Sapiens Auxiliary Models

In addition to pose, Sapiens provides **depth estimation** and **body-part
segmentation**.  These run alongside any Sapiens pose model to enrich
per-landmark data.

### Depth Estimation

Monocular relative depth.  Per-landmark depth values (closer = higher)
are stored in each frame as `landmark_depths`.

| Size | HuggingFace repo |
|------|------------------|
| 0.3b | `facebook/sapiens-depth-0.3b-torchscript` |
| 0.6b | `facebook/sapiens-depth-0.6b-torchscript` |
| 1b | `facebook/sapiens-depth-1b-torchscript` |
| 2b | `facebook/sapiens-depth-2b-torchscript` |

### Body-Part Segmentation

28-class segmentation (face, torso, arms, legs, hands, feet, clothing...).
Per-landmark body-part labels are stored in each frame as `landmark_body_parts`.

| Size | mIoU | HuggingFace repo |
|------|------|------------------|
| 0.3b | 76.7 | `facebook/sapiens-seg-0.3b-torchscript` |
| 0.6b | 77.8 | `facebook/sapiens-seg-0.6b-torchscript` |
| 1b | 79.9 | `facebook/sapiens-seg-1b-torchscript` |

### Usage

```bash
# Pose + depth + segmentation in one pass
myogait extract video.mp4 -m sapiens-quick --with-depth --with-seg

# Or via Python
from myogait import extract
data = extract("video.mp4", model="sapiens-top", with_depth=True, with_seg=True)
```

### References

- **Paper:** Rawal et al., *Sapiens: Foundation for Human Vision Models*, ECCV 2024 — [arXiv:2408.12569](https://arxiv.org/abs/2408.12569)
- **Code:** [github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens)
- **Models:** [HuggingFace collection](https://huggingface.co/collections/facebook/sapiens-66d22047daa6402d565cb2fc)

## ViTPose

Vision Transformer for pose estimation (NeurIPS 2022).  Top-down
architecture with RT-DETR person detector.  Fully pip-installable
via HuggingFace Transformers.

| Variant | Size | HuggingFace repo |
|---------|------|------------------|
| `vitpose` (base) | 90M | `usyd-community/vitpose-base-simple` |
| `vitpose-large` | 400M | `usyd-community/vitpose-plus-large` |
| `vitpose-huge` | 900M | `usyd-community/vitpose-plus-huge` |

- **Paper:** Xu et al., *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*, NeurIPS 2022 — [arXiv:2204.12484](https://arxiv.org/abs/2204.12484)
- **Paper:** Xu et al., *ViTPose++: Vision Transformer for Generic Body Pose Estimation*, TPAMI 2024 — [arXiv:2212.04246](https://arxiv.org/abs/2212.04246)

## RTMW (Whole-Body 133 Keypoints)

Real-Time Multi-person Whole-body estimation: body (17) + feet (6) +
face (68) + hands (42) = 133 keypoints.  Uses rtmlib (lightweight
ONNX inference, no MMPose required).

| Mode | Pose Model | Speed |
|------|-----------|-------|
| `performance` | RTMW-x-l 384x288 | Slower, most accurate |
| `balanced` | RTMW-x-l 256x192 | Default |
| `lightweight` | RTMW-l-m 256x192 | Fastest |

- **Paper:** Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023
- **RTMW configs:** [MMPose cocktail14](https://github.com/open-mmlab/mmpose/tree/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14)
- **rtmlib:** [github.com/Tau-J/rtmlib](https://github.com/Tau-J/rtmlib)

## Other Pose Models

### MediaPipe

Google's MediaPipe PoseLandmarker — 33 landmarks with full-body coverage.
Uses the **heavy** model variant (most accurate). Auto-downloaded on first use.

- **Docs:** [developers.google.com/mediapipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

### YOLO Pose

Ultralytics YOLOv8-Pose — 17 COCO keypoints, fast single-shot detection.

- **Docs:** [docs.ultralytics.com](https://docs.ultralytics.com/tasks/pose/)

### HRNet / RTMPose (MMPose)

HRNet-W48 and RTMPose-m via the OpenMMLab MMPose framework — 17 COCO keypoints.

- **MMPose:** [github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
- **HRNet:** Sun et al., *Deep High-Resolution Representation Learning for Visual Recognition*, TPAMI 2019
- **RTMPose:** Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023

## Tutorials & Examples

### Basic Pipeline

The core workflow extracts pose landmarks from video, preprocesses the signal,
computes joint kinematics, detects gait events, and derives spatio-temporal
parameters.

```python
from myogait import extract, normalize, compute_angles, detect_events
from myogait import segment_cycles, analyze_gait

# 1. Extract landmarks from video
data = extract("walking_video.mp4", model="mediapipe")

# 2. Preprocess: filter noise, handle gaps
data = normalize(data, filters=["butterworth"])

# 3. Compute joint angles (sagittal + frontal if depth available)
data = compute_angles(data)

# 4. Detect gait events (heel strikes, toe offs)
data = detect_events(data, method="gk_bike")  # uses gaitkit Bayesian detector

# 5. Segment into gait cycles
cycles = segment_cycles(data)

# 6. Compute spatio-temporal parameters
stats = analyze_gait(data, cycles)
print(f"Cadence: {stats['cadence']:.1f} steps/min")
print(f"Walking speed: {stats['speed']:.2f} m/s")
```

### Clinical Scores

Compute standardized gait quality indices used in clinical gait analysis
laboratories worldwide.

```python
from myogait import gait_profile_score_2d, sagittal_deviation_index
from myogait import gait_variable_scores, movement_analysis_profile

# GPS-2D: overall gait quality score (RMS deviation from normative)
gps = gait_profile_score_2d(cycles)
print(f"GPS-2D: {gps['gps']:.1f}")

# SDI: Sagittal Deviation Index (0-120, 100 = normal)
sdi = sagittal_deviation_index(cycles)
print(f"SDI: {sdi['sdi']:.1f}")

# Per-joint deviation scores
gvs = gait_variable_scores(cycles)
for joint, score in gvs["scores"].items():
    print(f"  {joint}: {score:.1f}")
```

### Data Quality Assessment

Evaluate landmark detection confidence and flag outliers before running
the analysis pipeline.

```python
from myogait import confidence_filter, detect_outliers, data_quality_score

# Filter low-confidence landmarks
data = confidence_filter(data, threshold=0.3)

# Detect and interpolate outliers
data = detect_outliers(data, z_thresh=3.0)

# Quality report
quality = data_quality_score(data)
print(f"Quality score: {quality['score']}/100")
```

### Normative Comparison

Compare patient kinematics against published normative reference bands
(Perry & Burnfield).

```python
from myogait import plot_normative_comparison, get_normative_band

# Plot patient vs normative bands (Perry & Burnfield)
fig = plot_normative_comparison(data, cycles, plane="both")
fig.savefig("normative_comparison.png", dpi=150)

# Access raw normative data
mean, lower, upper = get_normative_band("hip_flexion", stratum="adult")
```

### Event Detection with gaitkit

myogait integrates multiple gait event detection algorithms, including
Bayesian and ensemble methods from the gaitkit library.

```python
from myogait import detect_events, event_consensus, list_event_methods

# List all available methods
print(list_event_methods())
# ['zeni', 'velocity', 'crossing', 'oconnor',
#  'gk_bike', 'gk_zeni', 'gk_ensemble', ...]

# Single method (gaitkit Bayesian BIS -- best F1 score)
data = detect_events(data, method="gk_bike")

# Consensus: vote across multiple detectors
data = event_consensus(data, methods=["gk_bike", "gk_zeni", "gk_oconnor"])
```

### Export to OpenSim / Pose2Sim

Export landmarks and kinematics in formats compatible with OpenSim
musculoskeletal modeling and Pose2Sim multi-camera triangulation.

```python
from myogait import export_trc, export_mot, export_openpose_json
from myogait import export_opensim_scale_setup, export_ik_setup

# OpenSim .trc markers file (with height-based unit conversion)
export_trc(data, "markers.trc", opensim_model="gait2392")

# OpenSim .mot kinematics
export_mot(data, "kinematics.mot")

# OpenSim Scale Tool setup XML
export_opensim_scale_setup(data, "scale_setup.xml", model_file="gait2392.osim")

# OpenSim IK setup XML
export_ik_setup("markers.trc", "ik_setup.xml", model_file="scaled_model.osim")

# Pose2Sim: export OpenPose-format JSON for triangulation
export_openpose_json(data, "./openpose_output/", model="BODY_25")
```

### Video Overlay & Visualization

Generate annotated videos, anonymized stick-figure animations, and
publication-ready dashboards.

```python
from myogait import render_skeleton_video, render_stickfigure_animation
from myogait import plot_summary, plot_gvs_profile

# Skeleton overlay on original video
render_skeleton_video("video.mp4", data, "overlay.mp4", show_angles=True)

# Anonymized stick figure GIF
render_stickfigure_animation(data, "stickfigure.gif")

# Summary dashboard
fig = plot_summary(data, cycles, stats)
fig.savefig("dashboard.png", dpi=150)

# MAP barplot (Movement Analysis Profile)
fig = plot_gvs_profile(gvs)
fig.savefig("gvs_profile.png", dpi=150)
```

### PDF Report

Generate a multi-page clinical report with kinematic plots, spatio-temporal
tables, and normative comparisons.

```python
from myogait import generate_report

# Generate clinical PDF report (French or English)
generate_report(data, cycles, stats, "rapport_marche.pdf", language="fr")
```

### Multiple Export Formats

Export analysis results to a variety of tabular and structured formats for
downstream processing and archival.

```python
from myogait import export_csv, export_excel, to_dataframe, export_summary_json

# CSV files (one per data type)
export_csv(data, "./csv_output/", cycles, stats)

# Excel workbook
export_excel(data, "gait_analysis.xlsx", cycles, stats)

# Pandas DataFrame for custom analysis
df = to_dataframe(data, what="angles")
print(df.head())

# Compact summary JSON
export_summary_json(data, cycles, stats, "summary.json")
```

## CLI Usage

Run the full pipeline on a video:

```bash
myogait run video.mp4                                    # MediaPipe (default)
myogait run video.mp4 -m sapiens-quick                   # Sapiens 0.3B
myogait run video.mp4 -m sapiens-top --with-depth        # Sapiens 1B + depth
myogait run video.mp4 -m vitpose                         # ViTPose
myogait run video.mp4 -m rtmw                            # RTMW 133 keypoints
```

Extract landmarks only:

```bash
myogait extract video.mp4 -m sapiens-top --with-depth --with-seg
```

Analyze previously extracted results:

```bash
myogait analyze result.json --csv --pdf
```

Batch process multiple videos:

```bash
myogait batch *.mp4 -o results/
```

Download models:

```bash
myogait download --list                 # list all available models
myogait download sapiens-0.3b           # Sapiens pose 0.3B
myogait download sapiens-depth-1b       # Sapiens depth 1B
myogait download sapiens-seg-0.6b       # Sapiens seg 0.6B
```

Inspect a result file:

```bash
myogait info result.json
```

## API Reference

All functions operate on a single `data` dict that flows through the pipeline.

| Function | Description |
|---|---|
| **Core pipeline** | |
| `extract(video, model)` | Extract pose landmarks from video |
| `normalize(data, filters)` | Filter and normalize landmark trajectories |
| `compute_angles(data)` | Compute sagittal joint angles |
| `compute_frontal_angles(data)` | Compute frontal plane angles (requires depth) |
| `detect_events(data, method)` | Detect gait events (15 methods incl. gaitkit) |
| `event_consensus(data, methods)` | Multi-method voting for robust event detection |
| `segment_cycles(data)` | Segment into individual gait cycles |
| `analyze_gait(data, cycles)` | Compute spatio-temporal parameters |
| **Clinical scores** | |
| `gait_profile_score_2d(cycles)` | GPS-2D: overall gait deviation (sagittal + frontal) |
| `sagittal_deviation_index(cycles)` | SDI: 0-120 index (100 = normal) |
| `gait_variable_scores(cycles)` | Per-joint deviation vs normative |
| `movement_analysis_profile(cycles)` | MAP barplot data |
| **Quality** | |
| `confidence_filter(data)` | Remove low-confidence landmarks |
| `detect_outliers(data)` | Detect and interpolate spikes |
| `data_quality_score(data)` | Composite quality score 0-100 |
| `fill_gaps(data)` | Interpolate missing landmark gaps |
| **Analysis** | |
| `walking_speed(data, cycles)` | Estimated walking speed |
| `step_length(data, cycles)` | Step/stride length estimation |
| `stride_variability(data, cycles)` | CV of spatio-temporal parameters |
| `arm_swing_analysis(data, cycles)` | Arm swing amplitude and asymmetry |
| `segment_lengths(data)` | Anthropometric segment lengths |
| `detect_pathologies(data, cycles)` | Pattern detection (equinus, antalgic, etc.) |
| **Visualization** | |
| `plot_summary(data, cycles, stats)` | Summary dashboard |
| `plot_normative_comparison(data, cycles)` | Patient vs normative bands |
| `plot_gvs_profile(gvs)` | Movement Analysis Profile barplot |
| `render_skeleton_video(video, data, out)` | Skeleton overlay on video |
| `render_stickfigure_animation(data, out)` | Anonymized stick figure GIF |
| **Export** | |
| `export_csv(data, dir, cycles, stats)` | CSV files |
| `export_mot(data, path)` | OpenSim .mot kinematics |
| `export_trc(data, path)` | OpenSim .trc markers |
| `export_openpose_json(data, dir)` | OpenPose JSON for Pose2Sim |
| `export_opensim_scale_setup(data, path)` | OpenSim Scale Tool XML |
| `export_ik_setup(trc, path)` | OpenSim IK setup XML |
| `to_dataframe(data, what)` | Pandas DataFrame |
| `generate_report(data, cycles, stats, path)` | Multi-page clinical PDF report |
| `validate_biomechanical(data, cycles)` | Validate against physiological ranges |

## Configuration

myogait supports YAML and JSON pipeline configuration files:

```python
from myogait import load_config, save_config

config = load_config("pipeline.yaml")
config["filter"]["method"] = "butterworth"
config["filter"]["cutoff"] = 6.0
save_config(config, "pipeline_updated.yaml")
```

## JSON Output Format

When using Sapiens with depth and segmentation:

```json
{
  "extraction": {
    "model": "sapiens-top",
    "depth_model": "sapiens-depth-1b",
    "seg_model": "sapiens-seg-1b",
    "auxiliary_format": "goliath308",
    "seg_classes": ["Background", "Apparel", "Face_Neck", "..."]
  },
  "frames": [
    {
      "frame_idx": 0,
      "landmarks": { "NOSE": {"x": 0.52, "y": 0.31, "visibility": 0.95} },
      "goliath308": [[0.52, 0.31, 0.95], "..."],
      "landmark_depths": { "NOSE": 0.73, "LEFT_HIP": 0.45, "..." : 0.0 },
      "landmark_body_parts": { "NOSE": "Face_Neck", "LEFT_HIP": "Left_Upper_Leg" }
    }
  ]
}
```

## Acknowledgments

myogait is developed at the [Institut de Myologie](https://www.institut-myologie.org/) (Paris, France) by the **PhysioEvalLab / IDMDataHub** team.

This work is supported by:
- [AFM-Téléthon](https://www.afm-telethon.fr/) — French Muscular Dystrophy Association
- [Fondation Myologie](https://www.fondation-myologie.org/) — Research foundation for muscle diseases
- [Téléthon](https://www.telethon.fr/) — Annual fundraising event for rare disease research

<p align="center">
  <a href="https://www.institut-myologie.org/">
    <img src="https://img.shields.io/badge/Institut_de_Myologie-Paris-0055A4?style=for-the-badge" alt="Institut de Myologie">
  </a>
  &nbsp;
  <a href="https://www.afm-telethon.fr/">
    <img src="https://img.shields.io/badge/AFM--Téléthon-Supporter-E30613?style=for-the-badge" alt="AFM-Téléthon">
  </a>
  &nbsp;
  <a href="https://www.fondation-myologie.org/">
    <img src="https://img.shields.io/badge/Fondation_Myologie-Research-00A651?style=for-the-badge" alt="Fondation Myologie">
  </a>
  &nbsp;
  <a href="https://www.telethon.fr/">
    <img src="https://img.shields.io/badge/Téléthon-Funding-FFC107?style=for-the-badge" alt="Téléthon">
  </a>
</p>

## Citation

If you use myogait in your research, please cite:

```bibtex
@software{myogait,
  author = {Fer, Frederic},
  title = {myogait: Markerless video-based gait analysis toolkit},
  year = {2025},
  institution = {Institut de Myologie, Paris, France},
  publisher = {GitHub},
  url = {https://github.com/IDMDataHub/myogait}
}
```

A peer-reviewed publication is in preparation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
