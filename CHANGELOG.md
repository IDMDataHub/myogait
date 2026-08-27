# Changelog

All notable changes to myogait are documented here. The project follows
semantic versioning: breaking API changes only occur in major releases.

## [0.8.5] — 2026-08-27

Bug-fix release: correct metric spatiotemporal parameters on a C3D marker
source, and finish the walking-direction robustness work. Validated on the
Bath BioCV dataset (markerless video vs Vicon C3D) and a panning-camera
clip.

### Fixed
- `step_length` / `walking_speed` on a **C3D pivot**: step, stride and speed
  are now read directly from the real 3-D markers (``c3d_markers_3d``, in mm)
  instead of the pixel-to-metre scale path. A C3D pivot's 2-D landmark
  projection squashes the real-world capture volume anisotropically into the
  image box, so a femur/height pixel scale (derived from the vertical femur)
  did not apply to the horizontal step and produced values ~100x too large
  (e.g. 82 m steps, 168 m/s). Markerless-video vs Vicon-C3D walking speed now
  agree to ~0.1% on Bath BioCV (both 1.35 m/s).
- Walking-direction detection falls back to a displacement heuristic when the
  feet are missing/occluded, instead of silently defaulting to
  left-to-right. `detect_walking_direction_from_feet` gained a ``default``
  parameter (``"right"`` for backward compatibility; the event detectors pass
  ``"unknown"`` to trigger the pelvis/ankle-displacement fallback). Previously
  a right-to-left walk with occluded feet had its heel-strike and toe-off
  events swapped.
- `toe_clearance`: minimum toe clearance is now referenced against the toe
  marker's own vertical floor (not the heel's) and searched over mid-swing
  only. It previously mixed the heel ground with the toe marker and included
  the toe-off / terminal-swing ground contacts, so the MTC came out ≈0 or
  slightly negative; it is now the physiological mid-swing clearance (~1–2 cm).
- Dropped Python 3.9 from the supported set (``requires-python >= 3.10``): it
  reached end-of-life in Oct 2025 and the numeric dependency floors no longer
  install on it.

### Added
- App (`myogait_app`): the pipeline now drops the against-direction cycle
  group on a there-and-back walkway (``CyclesConfig.filter_direction``,
  enabled automatically by ``autoconfig.detect_config`` on a detected
  reversal), matching library ``run_pipeline`` behaviour. The mirrored
  return-pass cycles no longer pollute the ROM / symmetry averages.

## [0.8.4] — 2026-08-27

Bug-fix release: a codebase-wide audit for the same defect classes behind
the 0.8.3 fix — positional frame indexing, camera-motion-corrupted
metrics, and inconsistent calibration flags.

### Fixed
- `stride_variability`: step-length CV now measures the same-frame
  inter-ankle separation resolved through the frame-index map, instead of
  a single ankle's cross-frame displacement read positionally. The old
  formulation was corrupted both by the frame-index mismatch (analysed
  window starting late / shorter than the source) and by a tracking
  camera, producing impossible CVs (>100%); it now returns
  physiologically plausible values.
- `toe_clearance`: read the correct cycle key (`toe_off_frame`, not the
  non-existent `to_frame`) and resolve cycle frames through the
  frame-index map. Every minimum-toe-clearance value was previously
  `None` because the swing loop never ran.
- `to_opensim_mot` (`.mot` export): pelvis translations are read through
  a frame-index → position map; positional indexing corrupted every
  translation row when the analysed window did not start at frame 0.
- `detect_parkinsonian`, `harmonic_ratio`, `postural_sway`: reformulated
  to pelvis-relative (same-frame) quantities so a tracking/panning camera
  can no longer distort them. In particular the parkinsonian "short
  stride" screen used the absolute ankle-x range, which a panning camera
  collapses toward zero — falsely flagging a healthy subject; it now uses
  the ankle excursion relative to the pelvis and is camera-motion immune.
- `step_length` / `walking_speed` / `analyze_gait`: anthropometric
  references (height, femur, foot) are resolved once and back-filled from
  the pivot's `subject` block consistently, so the two functions can no
  longer disagree on whether a trial is calibrated (metres vs normalised
  units).
- `detect_events` / `normalize`: acquisition rate is sanitised via a
  shared `safe_frame_rate`, so a malformed `meta.fps` (`0`, negative,
  `None`, or non-numeric) no longer raises `ZeroDivisionError` / crashes
  or silently yields zero cycles.
- `trim_standstill`: read the event `frame` key (events are still in
  array-index space at this stage) instead of the non-existent
  `frame_idx` / `index` keys, which collapsed every event to index 0 and
  silently trimmed the entire gait bout to zero cycles on any clip that
  began with the subject standing still.

### Added
- `analyze_gait` populates `stats["warnings"]`: physiological plausibility
  guards on metric step/stride/speed record any breach machine-readably
  and set `valid_for_progression = False`, so a grossly mis-scaled
  calibration can no longer be read as trustworthy.

## [0.8.3] — 2026-08-27

Bug-fix release: make spatiotemporal metrics reliable under a tracking
(panning) camera.

### Fixed
- `step_length` / `walking_speed`: step length is now the antero-posterior
  separation between the two ankles *within* the heel-strike frame, stride
  is the sum of the two step lengths, and speed is step × cadence. A
  subject-following/panning camera is optically identical to a treadmill —
  it cancels the forward image translation and corrupts every cross-frame
  displacement metric — so the previous single-ankle progression
  under-estimated step, stride and speed several-fold. The new
  formulation is same-frame / event-timing based and therefore immune to
  camera panning, validated across walking direction, pan sign, and
  there-and-back trials.
- Event/cycle frames are resolved through a frame-index → array-position
  map, fixing wrong-frame reads and silently dropped heel strikes when the
  analysed window does not start at video frame 0.

## [0.8.2] — 2026-08-25

Bug-fix release: correct spatial scaling of gait distances.

### Fixed
- `step_length` / `walking_speed`: distances are now de-normalised to
  source pixels before the anthropometric scale is applied. Landmarks
  are normalised per axis (`x / width`, `y / height`), so on a
  non-square frame one x-unit and one y-unit span different real
  distances. The metric scale is derived mostly from the (vertical)
  femur but a step is a (horizontal) antero-posterior distance, so the
  previous code under-estimated step/stride length by roughly the image
  aspect ratio on landscape footage (e.g. ~1.78× on 16:9). The scale is
  now isotropic (metres per source pixel), fixing step length, stride
  length and walking speed. When frame dimensions are unavailable the
  behaviour is unchanged (unit scale).

## [0.8.1] — 2026-08-21

Technical release: post-audit hardening and documentation pass.

### Added
- `available_models()` — non-destructive backend discovery via
  `importlib.util.find_spec` (including secondary requirements), so a
  UI can grey out unavailable backends without importing anything.
- `myogait.exceptions` — dedicated error hierarchy (`MyogaitError`
  root) that also inherits the historical builtins, so existing
  `except ValueError` handlers keep working. Adopted at the
  unreadable-video and missing-backend sites.
- `segment_cycles(min_confidence=…, min_coherence=…)` — optional
  per-cycle quality gates on landmark confidence / frame coherence;
  rejections reported in `summary["n_rejected_quality"]`.
- `femur_ratio` parameter on `analyze_gait`, `step_length` and
  `walking_speed` (defaults to the documented `FEMUR_HEIGHT_RATIO`).
- CI: `backend-mediapipe` job running a real end-to-end extraction,
  so the badge finally covers an actual pose backend.

### Changed
- `ensure_xpu_torch()` is strictly opt-in: by default it only warns
  with the manual install command; the pip-reinstall + process-restart
  path requires `auto_upgrade=True` or `MYOGAIT_AUTO_XPU=1`.
- GPU extractors explicitly release CUDA/XPU cached memory in
  `teardown()` (`BasePoseExtractor.release_gpu_memory()`).
- README: validation figure (video vs optical capture, three healthy
  adults) and feature list regrouped by theme; tutorial trimmed of
  deprecated workflows.

## [0.8.0] — 2026-08-21

The accuracy & validation release. Every change below came out of a
systematic benchmarking campaign against marker-based optical motion
capture (two laboratories, two camera types, two pose-model
generations, healthy adults and neuromuscular patients — see the
Validation section of the README).

### Added
- `run_pipeline()` — the validated end-to-end pipeline in one call
  (video, `.myogait.json`, or `.c3d` input), with built-in quality
  diagnostics: tracking coverage, out-of-sagittal-plane distortion,
  standing-prelude detection, direction-inconsistent cycle counts,
  implausible ankle range. Warnings are returned and logged instead of
  failing silently.
- `canonicalize_angle_signs()` — enforces a flexion-positive sagittal
  convention independent of walking direction, with the hip verified
  against the knee-flexion peak. Without it, two recordings of the
  same subject walking in opposite directions (or a video and a C3D
  reference) can disagree in sign.
- `compute_c3d_reference_angles()` — recomputes the ankle of a C3D
  reference directly from the 3-D marker positions. The sagittal 2-D
  projection is faithful for hip and knee (r ≥ 0.99 vs a Vicon 3-D
  model) but collapses the ankle (r ≈ 0.4, ROM halved); the 3-D ankle
  restores r = 0.99.
- Landmark bias-correction family (experimental):
  `fit_landmark_bias_by_phase`, `merge_landmark_biases`,
  `smooth_landmark_bias`, `apply_landmark_bias_correction` —
  phase-binned, walking-direction-aware landmark calibration against
  an optical reference. Benchmarks show the *uncorrected* pipeline is
  already at reference level with a strong pose backbone; reserve
  these for weak backbones or per-patient calibration setups (see
  docstrings for measured caveats).
- `bath_biocv` C3D marker convention; convention autodetection
  covers Plug-in Gait, ISB medial+lateral, Helen Hayes, underscore
  variants, and BioCV-style explicit labels.
- Per-cycle biomarker sheet in `export_excel`.
- `calibration_max_offset_deg` guard in `compute_angles` (default
  25°): neutral calibration is skipped, with a warning, when the
  "neutral" window clearly caught mid-gait motion — previously this
  silently shifted the ankle by tens of degrees on clips that start
  mid-stride.

### Changed / Deprecated (following an independent two-track audit)
- **LASSO bias corrections deprecated** (`apply_hip/knee/ankle_bias_correction`,
  removal planned for 1.0) with an *executable* guardrail: every call
  now emits a `DeprecationWarning` stating the healthy-adult prior
  (n ≤ 12 training subjects) and the benchmark result that the raw
  pipeline outperforms the corrected one with modern pose models. The
  protection is no longer documentation-only.
- **`pelvis_obliquity`** exposed: the historical `pelvis_tilt` key
  actually measures frontal-plane pelvic obliquity, not sagittal tilt.
  The honest name now carries the same value; the legacy key remains
  for compatibility and will be redefined in 1.0.
- **`FEMUR_HEIGHT_RATIO`** (0.245) promoted to a documented, sourced
  (Winter 2009) and overridable module constant, with an explicit
  caveat for neuromuscular populations; used by the height-based
  metric scaling fallback.
- **CMC (Kadaba 1989)** added to the Vicon benchmark angle metrics —
  the standard waveform-similarity index of the gait literature.
- **Empirical normative curves.** The adult sagittal reference curves
  (hip, knee, ankle, pelvis) are now empirical mean ± SD waveforms
  derived from an instrumented optical motion-capture dataset of
  healthy adults, replacing curves digitised from textbook figures.
  Per-phase SD replaces the previous constant bands.
- `ROADMAP.md` added: audit items not addressed in this release
  (pathological event detection, cycle-quality gating, pediatric
  norms, GDI-2D concordance, engineering hardening for long-lived
  services) are tracked there.

### Fixed
- **Shared-config mutation bug.** `load_config` merged user configs
  onto a *shallow* copy of `DEFAULT_CONFIG`: any sub-dict not
  overridden stayed shared, so mutating one loaded config silently
  corrupted the process-wide defaults. Now deep-copied.
- **`load_c3d` isotropic normalisation (critical).** Each axis was
  normalised by its own range, so a 6 m walkway × 2 m height scene
  squashed the walking axis ~3:1 and every joint angle computed from
  the projection was geometrically wrong (hip ROM collapsed to ~6°,
  ankle inflated to ~90°). Any benchmark run against `load_c3d`
  output on a non-square scene should be re-run.
- **`load_c3d` occlusion handling.** ezc3d encodes missing samples as
  exact `(0,0,0)`, which corrupted the normalisation bounds; they are
  now treated as NaN.
- Walking-direction-dependent hip/knee sign inversions (see
  `canonicalize_angle_signs` above).
- Hip sign convention now flexion-positive (was extension-positive:
  signed r = −1.0 against a Vicon 3-D reference).

### Validation
- Sagittal waveforms vs optical capture: r = 0.99 (hip) / 0.98 (knee)
  / 0.90 (ankle); curve RMSE 2.4–4.0° after zero-offset removal.
- Spatiotemporal: stride-time bias 0.00 ± 0.03 s; cadence 0.9 ± 3.3
  steps/min; stance 0.5 ± 2.4 %.
- Peak knee angular velocity: ≈ 1 % bias.
- Repeatability supports MDC < 5° on hip/knee parameters from ~5–10
  averaged cycles.

## [0.7.0] — 2026-08-20
- `femur_mm` / `foot_mm` metric scaling for step length and speed.
- C3D marker-convention autodetection (`detect_c3d_convention`).
- Per-cycle biomarkers in Excel export.
- `apply_linear_detrend` for slow sagittal drift.
- Hip inversion fix on flipped videos; `min_cycle_duration` fix.

## [0.6.x] and earlier
See the git history.
