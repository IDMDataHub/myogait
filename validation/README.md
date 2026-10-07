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

Curves: per subject, Pearson *r*, RMSE and centred RMSE (RMSE after removing
the mean offset). Parameters: per trial, ICC(2,1) (two-way random, absolute
agreement, single measure) and Bland–Altman limits of agreement (LoA).
Events: relative timing after kinematic synchronisation of the two systems.

| | Left lateral (cam01) | Right lateral (cam05) |
|---|---|---|
| Hip / knee / ankle *r* | 0.98 / 0.96 / 0.88 | 0.96 / 0.90 / 0.80 |
| Centred RMSE hip / knee / ankle | 2.9° / 4.9° / 4.8° | 3.2° / 6.2° / 5.7° |
| Absolute RMSE hip / knee / ankle | 8.6° / 7.8° / 6.6° | 10.0° / 9.0° / 7.7° |
| Mean bias hip / knee / ankle | −6.9° / −5.0° / +3.2° | −8.4° / −4.8° / +4.3° |
| Cadence ICC (bias, LoA) | 0.93 (−0.9, −5.5 to +3.8 steps/min) | 0.91 |
| Stride time ICC | 0.93 | 0.89 |
| Stride length ICC (bias, LoA) | 0.71 (−0.04 m, −0.28 to +0.20) | 0.41 |
| Walking speed ICC | 0.83 | 0.56 |
| Initial contact / toe-off timing | −6 ± 26 / −3 ± 29 ms | −5 ± 35 / +1 ± 46 ms |
| Synchronised trials (sync *r* ≥ 0.9) | 62 / 83 | 11 / 79 |

Frontal (cam03) and rear (cam07) views do not give usable sagittal angles.

**Limits.** Waveform shape, timing and cadence from a lateral view are the
validated outputs. Absolute angles are biased: the hip is offset by about −7°,
partly because the video hip angle is measured between thigh and trunk
(thigh–pelvis for markers). Knee and ankle ranges of motion are
underestimated by about 9°, and peak angular velocities are underestimated.
The camera on the far side of the subject gives lower agreement.

The optional ankle restoration (`myogait.restore_ankle_dynamics`, disabled by
default and not applied here) was calibrated on BioCV for Sapiens2 and reduced
the ankle range-of-motion bias from −10.6° to −5.3° in leave-one-subject-out
validation.

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
