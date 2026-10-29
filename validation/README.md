# Validation of myogait against marker-based motion capture

This folder holds the code and the aggregate results of the validation of
**myogait 0.8.9** (single video + Sapiens2 pose model + myogait pipeline)
against simultaneous marker-based motion capture on the BioCV dataset of the
University of Bath ([doi:10.15125/BATH-01258](https://doi.org/10.15125/BATH-01258)):
9 healthy adults (P03, P04, P06, P08, P09, P10, P13, P16, P17) walking
overground, filmed by fixed synchronised machine-vision cameras (200 fps,
analysed at 60 fps) from four sides, with **Qualisys** motion capture
(200 Hz, Visual3D joint centres) as reference — 194 video–marker pairs
(89 trials). The marker trials are processed by the same myogait angle, event
and cycle algorithms, so the comparison isolates the pose-estimation stage.
Gait events of the marker side are detected by myogait on the marker data,
not taken from the Visual3D event files (some of which are wrong on a few
BioCV trials). In the tables, "vicon" denotes the marker side.

A clinical validation in patients (hand-held smartphone vs Vicon) is ongoing
and will be reported separately.

## Main results

**Kinematics with the myogait 0.9.3 defaults** (landmark filter 8 Hz, no angle
scaling), left lateral camera, 9 subjects, subject-mean curves; marker angles
computed directly from the markers and filtered at 15 Hz
(`code/python/kinematics_093.py`):

| | Hip | Knee | Ankle |
|---|---|---|---|
| Waveform *r* | 0.99 | 1.00 | 0.98 |
| Shape RMSE (offset removed) | 2.9° | 2.9° | 1.8° |
| Absolute RMSE | 8.7° | 5.5° | 3.6° |
| Mean offset (video − markers) | −6.1° | −0.6° | −2.7° |
| Range-of-motion error | −3.5° | −7.7° | +0.8° |

Up to myogait 0.9.2 the defaults (4 Hz filter, angles scaled by 0.8)
under-estimated knee and ankle range of motion by about 10-17° and flattened
the ankle push-off and knee swing peaks; the 0.8.9 results below the line are
kept for traceability.

**Spatio-temporal parameters and events** were computed with myogait 0.8.9;
the 0.9.3 filter change does not affect them (64 clinic walks: cadence bias
vs Vicon −0.65 vs −0.62 steps/min, 758 vs 759 cycles detected). Parameters
per trial: ICC(2,1) (two-way random, absolute agreement, single measure) and
Bland–Altman limits of agreement (LoA). Events: relative timing after
kinematic synchronisation of the two systems.

| myogait 0.8.9 | Left lateral (cam01) | Right lateral (cam05) |
|---|---|---|
| Hip / knee / ankle *r* | 0.98 / 0.96 / 0.88 | 0.96 / 0.90 / 0.80 |
| Centred RMSE hip / knee / ankle | 2.9° / 4.9° / 4.8° | 3.2° / 6.2° / 5.7° |
| Cadence ICC (bias, LoA) | 0.93 (−0.9, −5.5 to +3.8 steps/min) | 0.91 |
| Stride time ICC | 0.93 | 0.89 |
| Stride length ICC (bias, LoA) | 0.71 (−0.04 m, −0.28 to +0.20) | 0.41 |
| Walking speed ICC | 0.83 | 0.56 |
| Initial contact / toe-off timing | −6 ± 26 / −3 ± 29 ms | −5 ± 35 / +1 ± 46 ms |
| Synchronised trials (sync *r* ≥ 0.9) | 62 / 83 | 11 / 79 |

Frontal (cam03) and rear (cam07) views do not give usable sagittal angles.

**Limits.** Waveform shape, peaks, timing and cadence from a lateral view are
the validated outputs. The hip angle is measured between thigh and trunk
(thigh–pelvis for markers), hence a subject-specific offset driven by pelvic
tilt, which pose models do not see. The camera on the far side of the subject
gives lower agreement.

The ankle restoration (`myogait.restore_ankle_dynamics`) compensated the
former 4 Hz / ×0.8 defaults; it is deprecated since 0.9.3.

## Contents

| Path | What |
|---|---|
| `code/python/build_validation_dataset.py` | pairs video and marker trials, runs myogait on both, synchronises them and writes the tidy tables |
| `code/python/finalise_quality.py` | quality flags (synchronisation, side mapping) |
| `code/R/` | tables and figures; `run_all.R` runs them in order |
| `data_dictionary.csv` | every column, unit and definition of the prepared tables |
| `results/tables/` | aggregate results by camera view (`L3_*.csv`) |
| `results/figures/` | main figures |

Per-subject and per-cycle tables are not included here.

## Reproducing

The raw data are not redistributed. BioCV is available from the University of
Bath Research Data Archive under its own terms (all rights reserved; cite
Evans et al. 2024, doi:10.15125/BATH-01258). With the data in place, from the
package root:

```bash
pip install myogait==0.8.9
python code/python/build_validation_dataset.py   # ~20 min
Rscript code/R/run_all.R                          # ~4 min, R >= 4.3
```
