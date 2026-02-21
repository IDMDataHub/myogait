# myogait

Markerless video-based gait analysis toolkit.

![Python](https://img.shields.io/pypi/pyversions/myogait)
![License](https://img.shields.io/pypi/l/myogait)

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

## Quick Start

```python
from myogait import (
    extract, normalize, compute_angles, detect_events,
    segment_cycles, analyze_gait, plot_summary, export_csv,
)

# Extract pose landmarks from video
data = extract("video.mp4", model="mediapipe")

# With Sapiens depth + segmentation
data = extract("video.mp4", model="sapiens-top", with_depth=True, with_seg=True)

# Filter and normalize
data = normalize(data, filters=["butterworth"])

# Compute joint angles
data = compute_angles(data, method="sagittal_vertical_axis")

# Detect gait events (heel strikes, toe-offs)
data = detect_events(data, method="zeni")

# Segment into gait cycles
cycles = segment_cycles(data)

# Run spatio-temporal analysis
stats = analyze_gait(data, cycles)

# Visualize
fig = plot_summary(data, cycles, stats)
fig.savefig("summary.png")

# Export
export_csv(data, "./output", cycles, stats)
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
| `extract(video_path, model)` | Extract pose landmarks from a video file |
| `normalize(data, filters)` | Filter and normalize landmark trajectories |
| `compute_angles(data, method)` | Compute joint angles (sagittal_vertical_axis, sagittal_classic) |
| `detect_events(data, method)` | Detect gait events (zeni, crossing, velocity, oconnor) |
| `segment_cycles(data)` | Segment data into individual gait cycles |
| `analyze_gait(data, cycles)` | Compute spatio-temporal and symmetry metrics |
| `plot_summary(data, cycles, stats)` | Generate publication-quality summary plots |
| `export_csv(data, output_dir, cycles, stats)` | Export results to CSV files |
| `export_mot(data, path)` | Export to OpenSim .mot format |
| `export_trc(data, path)` | Export to OpenSim .trc format |
| `generate_report(data, cycles, stats, path)` | Generate a multi-page PDF clinical report |
| `validate_biomechanical(data, cycles)` | Validate angles against physiological ranges |

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use myogait in your research, please cite:

```
Fer, F. (2025). myogait: Markerless video-based gait analysis toolkit.
Institut de Myologie. https://pypi.org/project/myogait/
```

A peer-reviewed publication is in preparation.
