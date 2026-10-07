# Validation of myogait against marker-based motion capture

This folder holds the code and the aggregate results of the validation of
**myogait 0.8.9** (single video + Sapiens2 pose model + myogait pipeline)
against simultaneous marker-based motion capture (**Qualisys** at Bath,
**Vicon** at the Institut de Myologie; both 200 Hz, called "Vicon"/"marker"
side in the tables). The marker trials are processed by the same myogait
angle, event and cycle algorithms, so the comparison isolates
the pose-estimation stage. Events on both sides come from kinematic detection,
not from force plates.

| Dataset | Setting | Participants | Video–marker pairs |
|---|---|---|---|
| BioCV, University of Bath ([doi:10.15125/BATH-01258](https://doi.org/10.15125/BATH-01258)) | laboratory, fixed synchronised machine-vision cameras (200 fps, analysed at 60 fps), 4 views; Qualisys | 9 healthy adults (P03, P04, P06, P08, P09, P10, P13, P16, P17) | 194 (89 trials) |
| Myokinesis, Institut de Myologie (**preliminary**) | clinic, hand-held iPhone (60 Hz) following the subject; Vicon | 15 (10 healthy, 5 neuromuscular disease) | 64 |

## Main results

Curves: per subject, Pearson *r* and centred RMSE (RMSE after removing the mean
offset). Parameters: per trial, ICC(2,1) (two-way random, absolute agreement,
single measure) and Bland–Altman limits of agreement (LoA). Events: relative
timing after kinematic synchronisation of the two systems.

| | Bath, left lateral (cam01) | Bath, right lateral (cam05) | Myokinesis, hand-held |
|---|---|---|---|
| Hip / knee / ankle *r* | 0.98 / 0.96 / 0.88 | 0.96 / 0.90 / 0.80 | 0.92 / 0.98 / 0.91 |
| Centred RMSE hip / knee / ankle | 2.9° / 4.9° / 4.8° | 3.2° / 6.2° / 5.7° | 3.1° / 5.3° / 4.8° |
| Absolute RMSE hip / knee / ankle | 8.6° / 7.8° / 6.6° | 10.0° / 9.0° / 7.7° | 13.2° / 6.2° / 8.0° |
| Mean bias hip / knee / ankle | −6.9° / −5.0° / +3.2° | −8.4° / −4.8° / +4.3° | −12.0° / −2.1° / +5.3° |
| Cadence ICC (bias, LoA) | 0.93 (−0.9, −5.5 to +3.8 steps/min) | 0.91 | 0.97 (−0.8, −4.2 to +2.6) |
| Stride time ICC | 0.93 | 0.89 | 0.97 |
| Stride length ICC (bias, LoA) | 0.71 (−0.04 m, −0.28 to +0.20) | 0.41 | 0.80 (−0.01 m, −0.25 to +0.23) |
| Walking speed ICC | 0.83 | 0.56 | 0.86 |
| Initial contact / toe-off timing | −6 ± 26 / −3 ± 29 ms | −5 ± 35 / +1 ± 46 ms | −8 ± 8 / −2 ± 9 ms |
| Synchronised trials (sync *r* ≥ 0.9) | 62 / 83 | 11 / 79 | 60 / 64 |

Frontal (cam03) and rear (cam07) views do not give usable sagittal angles.

**Limits.** Waveform shape and timing are the validated outputs. Absolute
angles are biased: the hip is offset by −7° to −12°, partly because the video
hip angle is measured between thigh and trunk (thigh–pelvis for markers). Knee
and ankle ranges of motion are underestimated by about 9° in the laboratory and
up to 17° in the clinic, and peak angular velocities are underestimated. Pose
estimation failed for one Myokinesis patient (left/right confusions). Healthy
vs patient group differences have the same direction with both systems but are
attenuated by video. The Myokinesis cohort is still being recruited: these
numbers are a demonstration of use, not a clinical validation.

The optional ankle restoration (`myogait.restore_ankle_dynamics`, disabled by
default and not applied here) was calibrated on BioCV for Sapiens2 and reduced
the ankle range-of-motion bias from −10.6° to −5.3° in leave-one-subject-out
validation.

## Contents

| Path | What |
|---|---|
| `code/python/build_validation_dataset.py` | pairs video and marker trials, runs myogait on both, synchronises them and writes the tidy tables |
| `code/python/finalise_quality.py` | quality flags (synchronisation, side mapping) |
| `code/R/` | every table and figure; `run_all.R` runs them in order |
| `data_dictionary.csv` | every column, unit and definition of the prepared tables |
| `results/tables/` | aggregate results by setting (`L3_*.csv`) |
| `results/figures/` | main figures |

Per-subject and per-cycle tables are not included here.

## Reproducing

The raw data are not redistributed. BioCV is available from the University of
Bath Research Data Archive under its own terms (all rights reserved; cite
Evans et al. 2024, doi:10.15125/BATH-01258).
Gait events of the marker side are detected by myogait on the marker data,
not taken from the Visual3D event files (some of which are known to be
wrong on a few BioCV trials). Myokinesis data are clinical and available on
reasonable request to the authors, subject to the study's ethics approval.
With the data in place, from the package root:

```bash
pip install myogait==0.8.9
python code/python/build_validation_dataset.py   # ~20 min
Rscript code/R/run_all.R                          # ~4 min, R >= 4.3
```

A complete package (original pivots, prepared data, all outputs) is archived
separately (Zenodo DOI to be added).
