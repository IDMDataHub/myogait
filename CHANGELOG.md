# Changelog

All notable changes to myogait are documented here. The project follows
semantic versioning: breaking API changes only occur in major releases.

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
