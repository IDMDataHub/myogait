---
title: 'myogait: markerless gait analysis from a single video'
tags:
  - Python
  - gait analysis
  - biomechanics
  - markerless motion capture
  - pose estimation
  - neuromuscular disease
authors:
  - name: Frédéric Fer
    orcid: 0000-0000-0000-0000   # TODO ORCID
    corresponding: true
    affiliation: 1
  - name: Romain Feigean
    orcid: 0000-0000-0000-0000   # TODO ORCID
    affiliation: 2
affiliations:
  - name: Institut de Myologie, Myodata team, Paris, France
    index: 1
  - name: Institut de Myologie, Neuromuscular Physiology and Evaluation Laboratory, Neuromuscular Exploration and Evaluation Center, Paris, France
    index: 2
date: 7 October 2026
bibliography: paper.bib
---

# Summary

`myogait` is an open-source Python package that turns an ordinary video of a
person walking into a quantitative gait analysis: sagittal hip, knee and ankle
angles, gait events, time-normalised gait cycles, spatio-temporal parameters,
summary clinical parameters and a PDF report. It wraps several interchangeable
human pose estimators, stores every processing stage in a single pivot JSON
document, and exports to the formats used in motion laboratories (C3D,
TRC/MOT, OpenSim scaling and inverse-kinematics set-ups). A separate Streamlit
application, `myogait-app` [@myogaitapp], builds a no-code clinical interface
on top of it. `myogait` is distributed on PyPI under the MIT licence; it is a
research tool, not a certified medical device.

# Statement of need

Instrumented gait analysis with optoelectronic markers is the reference for
quantifying walking, but it requires a dedicated laboratory, trained staff and
lengthy preparation and processing. Many patients, including most people with
neuromuscular diseases, are therefore followed with clinical scales and timed
tests, which describe performance rather than *how* they walk. Pose
estimators now detect body landmarks in ordinary video, and in healthy adults
a single video can recover sagittal kinematics with moderate-to-good agreement
with marker-based systems [@Stenum2021; @Washabaugh2022; @Wade2022]. Turning
landmarks into trustworthy gait parameters still requires substantial work:
choosing and running a pose model, filtering, computing joint angles with
clinical sign conventions, detecting gait events, segmenting cycles, scaling
distances without a calibration object, rejecting implausible frames, and
checking the result against a reference.

`myogait` packages these steps into one reproducible pipeline that runs on a
single video with only the subject's stature as calibration, including video
taken by a clinician who walks alongside the patient with a hand-held phone.
It targets researchers who need a scriptable, versioned pipeline with exports
to their biomechanics tools, and it provides the processing layer for
clinician-facing software. It was developed for the follow-up of patients
with neuromuscular diseases at the Institut de Myologie.

# State of the field

Multi-camera systems, commercial (Theia3D, @Kanko2021) or open source
(OpenCap, @Uhlrich2023; Pose2Sim, @Pagnon2022), estimate 3-D kinematics and
feed OpenSim [@Delp2007; @Seth2018], but need several synchronised, calibrated
cameras, which is often impractical in a clinic corridor. Single-camera
approaches range from pose estimators that output landmarks only — MediaPipe
[@Lugaresi2019], ViTPose [@Xu2022], RTMPose [@Jiang2023], Sapiens
[@Khirodkar2024] — to gait pipelines evaluated in clinical populations
[@Stenum2021; @Kidzinski2020; @Stenum2024; @Liu2026]. Closest in spirit, the
Portable Biomechanics Laboratory [@Peiffer2026] fits a 3-D biomechanical
model to a hand-held smartphone video; its kinematics were validated against
markers in healthy adults and against multi-camera markerless capture in
patients. OpenCap-based metrics separate neuromuscular diseases [@Ruth2025],
with two fixed phones. Kinovea [@Kinovea] supports manual and semi-automatic 2-D
measurement without pose-estimation-based landmark detection. Sports2D
[@Pagnon2024] is the closest open-source tool: it computes 2-D joint and
segment angles from a single video for sport and general movement. `myogait`
is complementary rather than a replacement: it is gait-specific (gait events,
cycle normalisation, spatio-temporal and clinical parameters, normative
comparison, minimal detectable change, reports), it is designed for a moving,
hand-held camera, and it processes marker-based recordings (C3D) with the same
code so that video can be benchmarked against a laboratory reference. These
gait-specific layers, rather than pose extraction, are the core of the
package, which is why it was built as a separate library that can consume
landmarks from any backend.

# Software design

**Pivot data model.** Every stage reads and enriches one JSON document
(metadata, per-frame landmarks, angles, events, cycles, parameters,
provenance). Any intermediate state can be saved, shared and reloaded, and the
document can also be built from marker data (C3D), so video and reference
recordings go through the same angle, event and cycle algorithms.

**Interchangeable pose backends.** MediaPipe, YOLO, ViTPose, RTMW, MMPose/HRNet
and Sapiens/Sapiens2 [@Khirodkar2024; @Khirodkar2026], among others, are
available behind a common interface, with optional dependencies and a
one-command installer for the heavier stacks (`myogait setup-mmpose`).
Changing the pose model does not change the rest of the analysis.

**Gait pipeline.** Landmarks are gap-filled and filtered; sagittal angles are
2-D projection angles with clinical sign conventions (hip and knee flexion and
ankle dorsiflexion positive; hip angle measured between thigh and trunk).
Initial contact and toe-off are detected with kinematic methods [@Zeni2008] or
with the detectors of the `gaitkit` library [@gaitkit], which were benchmarked
on sixteen marker-based datasets and on real markerless video [@Fer2026]; on
BioCV video, the myogait preprocessing raised the heel-strike F1 of the BIKE
detector from 0.13 (raw landmarks) to 0.83 [@Fer2026].
Cycles are normalised to 101 points. Distances are scaled from the subject's
stature over the whole recording, and step, stride and speed are computed
from within-frame quantities so that they remain valid when the camera pans
to follow the subject. `run_auto` inspects the recording (standing start,
there-and-back walkway, marker source) to choose the processing recipe and
falls back to the validated overground recipe if no gait cycle is found.
Plausibility guards flag frames, events, cycles and
parameters outside physiological ranges instead of silently reporting them.

**Reference and biomechanics tools.** For marker data, `myogait` reconstructs
3-D joint angles from ISB anatomical frames [@Wu2002]. An optional ankle
restoration step, disabled by default, compensates the low-pass behaviour of
pose estimators at the ankle with a cadence-adaptive Wiener deconvolution; its
transfer function was calibrated once against marker data on the BioCV dataset for
the Sapiens2 backend and reduced the ankle range-of-motion bias from −10.6° to
−5.3° in leave-one-subject-out validation. Results can be compared with
normative data, summarised with minimal detectable changes (MDC95), and
exported to C3D, TRC/MOT, OpenSim (scaling, inverse kinematics, Moco
templates), OpenPose JSON, Excel/CSV and PDF reports.

**Companion application.** `myogait-app` builds on the public `myogait`
API, including the same recipe detection, so graphical and scripted analyses
share the same processing code.

**Quality.** About 1,400 automated tests run in continuous integration on
Linux and Windows for Python 3.10 to 3.14; each release is documented in a
changelog.

# Research impact statement

**Agreement with marker-based motion capture.** `myogait` 0.8.9 with the
Sapiens2 backend was compared with simultaneous marker-based recordings
(Qualisys or Vicon, 200 Hz) processed by the same `myogait` angle, event and
cycle algorithms, so the comparison
isolates the pose-estimation stage; events on both sides come from kinematic
detection, not force plates. Curve agreement is reported per subject as
Pearson *r* and centred RMSE (RMSE after removing the mean offset); parameter
agreement per trial as ICC(2,1) (two-way random, absolute agreement, single
measure, @Koo2016) and Bland–Altman limits of agreement [@Bland1986]; event
timing is relative, after kinematic synchronisation of the two systems.

*Laboratory* (BioCV dataset [@Evans2024; @Needham2021]: 9 healthy adults,
fixed machine-vision cameras at 200 fps analysed at 60 fps, Qualisys, 194
video–marker pairs over four views). With the left lateral
camera, hip, knee and ankle curves reached *r* = 0.98, 0.96 and 0.88 (centred
RMSE 2.9°, 4.9° and 4.8°); cadence ICC was 0.93, stride-length ICC 0.71 and
speed ICC 0.83; initial-contact and toe-off timing differed by −6 ± 26 ms and
−3 ± 29 ms (62 of 83 synchronised trials). The right lateral camera, filming
the same trials from the other side, gave lower agreement (ankle *r* 0.80,
stride-length and speed ICC 0.41–0.56); frontal and rear views do not give
usable sagittal angles.

*Clinic, preliminary use* (Myokinesis study: 64 walks of 15 participants,
5 with a neuromuscular disease, filmed with a hand-held iPhone at 60 Hz
following the subject, Vicon). Curves reached *r* = 0.92, 0.98 and 0.91 (centred
RMSE 3.1°, 5.3°, 4.8°); cadence ICC 0.97, stride-length ICC 0.80 (bias
−0.01 m, limits −0.25 to +0.23 m), speed ICC 0.86; event timing −8 ± 8 ms and
−2 ± 9 ms over short overlapping recordings. Pose estimation failed for one
patient (left/right confusions). Group differences between healthy and
patient participants had the same direction with both systems but were
attenuated by video. This is a demonstration of use, not a clinical
validation, which will be reported separately on the complete cohort.

Across both settings, waveform shape and timing are the validated outputs.
Absolute angles are biased (hip −7° to −12°, partly because of the
trunk-based hip definition), giving absolute RMSE of 8.6°, 7.8° and 6.6° (hip,
knee, ankle) in the laboratory and 13.2°, 6.2° and 8.0° in the clinic, and
knee and ankle ranges of motion are underestimated by about 9° in the
laboratory and up to 17° in the clinic. To our knowledge this is the first
comparison of hand-held smartphone gait kinematics against marker-based
capture that includes patients.

**Reproducibility and use.** The validation is distributed as a
self-contained package (prepared data, Python build scripts and R analyses)
that regenerates every table and figure (TODO: DOI Zenodo). `myogait` has been
released on PyPI since February 2026 (about 340 downloads per month) and
archived on Zenodo [@myogait]; the repository had 12 stars and 6 external forks
in October 2026. It is
used in the Myokinesis clinical study at the Institut de Myologie, provided
the markerless video chain of a gait-event benchmark [@Fer2026], and is the
processing layer of `myogait-app` [@myogaitapp].

# Ethics

TODO : Myokinesis was approved by [CPP …, IDRCB/NCT …]; all participants gave
written informed consent. The BioCV data were collected under the ethics
approval of the University of Bath and are used under the dataset licence;
they are not redistributed.

# AI usage disclosure

Large language models (Claude, Anthropic; and Codex, OpenAI) were used as
development tools for the software, its tests and documentation, the
validation analyses, and during manuscript preparation. The software design,
the validation protocol, the interpretation of results and the final content
of the software and of this paper are entirely the responsibility of the
authors, who reviewed and accepted every contribution. No large language model
is listed as an author or meets authorship criteria. Correctness is checked by
the automated test suite and by the comparison with marker-based motion
capture reported above.

# Acknowledgements

We thank the participants and the clinical team of the Myokinesis study,
W. Legendre for code contributions, T. Marques, J.-Y. Hogrel and M. Jacoupy
for discussions on the validation, the University of Bath group (L. Needham,
L. Wade, M. Evans, S. Colyer, P. McGuigan and colleagues) for access to the
BioCV synchronised video, motion-capture and force-plate dataset, and Ersin
Metin for setting up the computational tools and infrastructure. This work was
funded by Crédit Agricole CIB-LCL and AFM-Téléthon. The authors declare no
competing interests.

# References
