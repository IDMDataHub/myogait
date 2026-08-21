# Roadmap

Priorities distilled from an independent two-track audit of the
package (software engineering + clinical biomechanics) and from the
optical-capture validation campaign. Items already addressed in v0.8.0
are listed at the bottom for traceability.

## Biomechanics / scientific validity

- **Pathological event detection.** All four event detectors (Zeni,
  velocity, crossing, O'Connor) assume a clear heel-strike kinematic
  signature; on severe steppage, antalgic or fixed-equinus gait they
  may converge on the same systematic error, so the consensus vote can
  be falsely reassuring. Evaluate degradation on annotated pathological
  gait, and consider a learned detector trained on pathological
  reference data as a complement.
- **Cycle-quality gating.** Cycle exclusion is duration-window only;
  `frame_coherence_score` exists but is not wired into
  `segment_cycles`. Add an optional per-cycle quality gate (landmark
  confidence / coherence / waveform-shape score).
- **Pediatric & population-specific norms.** The "pediatric" and
  "elderly" normative strata are scale factors applied to the adult
  curves, not independently digitised cohorts. Replace the pediatric
  stratum with curves digitised from speed- and age-stratified
  published data; add walking-speed as a covariate.
- **GDI-2D vs GDI-3D concordance study.** The 4-variable sagittal
  GPS/GDI adaptation is explicitly screening-only; a concordance study
  on subjects with both 2-D video and 3-D optical capture would
  establish (or bound) its clinical sensitivity.
- **Uncertainty propagation.** Anthropometric scaling
  (`FEMUR_HEIGHT_RATIO`) and per-landmark confidence are point
  estimates; propagate uncertainty to spatiotemporal outputs instead
  of a single scalar.
- **Multi-camera / sensor fusion (long-term).** Multi-view
  triangulation for fixed installations; IMU/video fusion for events
  and metric scale in consultation settings.
- **v1.0 removals.** The LASSO bias-correction family
  (`apply_{hip,knee,ankle}_bias_correction`, deprecated in 0.8.0) and
  the redefinition of the legacy `pelvis_tilt` key (the honest
  `pelvis_obliquity` alias ships since 0.8.0).

## Engineering (mainly for embedding in long-lived services / UIs)

- `inplace: bool` on every pipeline stage (pure-function option) so
  callers no longer need defensive deepcopies.
- `available_models() -> dict[str, bool]` via `importlib.util.find_spec`
  (non-destructive backend discovery), including secondary deps
  (e.g. the Sapiens weights package).
- A `myogait.exceptions` hierarchy (root `MyogaitError`, also
  inheriting the current builtins to stay non-breaking).
- Make `ensure_xpu_torch()` strictly opt-in (it can re-exec the
  process); never triggered implicitly from an extractor `setup()`.
- Explicit GPU teardown (`torch.cuda.empty_cache()` / XPU equivalent)
  in every GPU extractor.
- `cancel_check` callback in extraction loops; route model-download
  progress through `progress_callback`; silence the stdout progress
  bar when a callback is provided.
- CI: at least one end-to-end CPU backend job (mediapipe or yolo) so
  the green badge covers a real extraction path.
