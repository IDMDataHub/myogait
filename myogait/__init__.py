"""myogait -- Markerless video-based gait analysis toolkit.

Quick start::

    from myogait import extract, normalize, compute_angles, detect_events
    data = extract("video.mp4", model="mediapipe")
    data = normalize(data, filters=["butterworth"])
    data = compute_angles(data)
    data = detect_events(data)

Full pipeline with cycle analysis::

    from myogait import segment_cycles, analyze_gait, plot_summary
    cycles = segment_cycles(data)
    stats = analyze_gait(data, cycles)
    fig = plot_summary(data, cycles, stats)
    fig.savefig("summary.png")
"""

from importlib import import_module

__version__ = "0.5.27"

from .extract import extract, detect_sagittal_alignment, auto_crop_roi, select_person
from .normalize import (
    normalize,
    filter_median,
    filter_wavelet,
    confidence_filter,
    detect_outliers,
    data_quality_score,
    fill_gaps,
    residual_analysis,
    auto_cutoff_frequency,
    cross_correlation_lag,
    align_signals,
    procrustes_align,
    correct_lateral_labels,
)
from .angles import (
    compute_angles,
    compute_extended_angles,
    compute_frontal_angles,
    foot_progression_angle,
)
from .events import detect_events, list_event_methods, event_consensus, validate_events
from .cycles import segment_cycles, ensemble_average
from .analysis import (
    analyze_gait,
    regularity_index,
    harmonic_ratio,
    step_length,
    walking_speed,
    detect_pathologies,
    single_support_time,
    toe_clearance,
    stride_variability,
    arm_swing_analysis,
    speed_normalized_params,
    detect_equinus,
    detect_antalgic,
    detect_parkinsonian,
    segment_lengths,
    instantaneous_cadence,
    compute_rom_summary,
    estimate_center_of_mass,
    postural_sway,
    pca_waveform_analysis,
    compute_derivatives,
    time_frequency_analysis,
)
from .normative import (
    get_normative_curve,
    get_normative_band,
    select_stratum,
    list_joints,
    list_strata,
)
from .scores import (
    gait_variable_scores,
    gait_profile_score_2d,
    gait_deviation_index_2d,
    sagittal_deviation_index,
    movement_analysis_profile,
)
from .schema import load_json, save_json, set_subject
from .export import (
    export_csv,
    export_mot,
    export_trc,
    export_excel,
    export_c3d,
    to_dataframe,
    export_json,
    export_summary_json,
    export_openpose_json,
    export_landmarks_excel,
)
from .opensim import (
    export_opensim_scale_setup,
    export_ik_setup,
    export_moco_setup,
    get_opensim_marker_names,
)
from .validation import (
    validate_biomechanical,
    stratified_ranges,
    model_accuracy_info,
    validate_biomechanical_stratified,
)
from .config import load_config, save_config, DEFAULT_CONFIG
from .axis_utils import (
    detect_walking_direction_from_feet,
    detect_walking_direction_from_feet_arrays,
)
from .experimental import (
    VIDEO_DEGRADATION_DEFAULTS,
    build_video_degradation_config,
    apply_video_degradation,
)
from .experimental_vicon import (
    load_vicon_trial_mat,
    load_c3d,
    estimate_vicon_offset_seconds,
    align_vicon_to_myogait,
    compute_single_trial_benchmark_metrics,
    attach_vicon_experimental_block,
    run_single_trial_vicon_benchmark,
)
from .experimental_benchmark import (
    DEFAULT_SINGLE_PAIR_BENCHMARK_CONFIG,
    build_single_pair_benchmark_config,
    run_single_pair_benchmark,
)

_LAZY_EXPORT_MAP = {
    "plot_angles": ".plotting",
    "plot_cycles": ".plotting",
    "plot_events": ".plotting",
    "plot_summary": ".plotting",
    "plot_phase_plane": ".plotting",
    "plot_normative_comparison": ".plotting",
    "plot_gvs_profile": ".plotting",
    "plot_quality_dashboard": ".plotting",
    "plot_longitudinal": ".plotting",
    "plot_arm_swing": ".plotting",
    "plot_session_comparison": ".plotting",
    "plot_cadence_profile": ".plotting",
    "plot_rom_summary": ".plotting",
    "plot_butterfly": ".plotting",
    "animate_normative_comparison": ".plotting",
    "generate_report": ".report",
    "generate_longitudinal_report": ".report",
    "render_skeleton_video": ".video",
    "render_skeleton_frame": ".video",
    "render_stickfigure_animation": ".video",
}

__all__ = [
    "extract",
    "normalize",
    "compute_angles",
    "compute_extended_angles",
    "compute_frontal_angles",
    "foot_progression_angle",
    "detect_events",
    "list_event_methods",
    "event_consensus",
    "validate_events",
    "segment_cycles",
    "ensemble_average",
    "analyze_gait",
    "filter_median",
    "filter_wavelet",
    "confidence_filter",
    "detect_outliers",
    "data_quality_score",
    "fill_gaps",
    "residual_analysis",
    "auto_cutoff_frequency",
    "cross_correlation_lag",
    "align_signals",
    "procrustes_align",
    "correct_lateral_labels",
    "regularity_index",
    "harmonic_ratio",
    "step_length",
    "walking_speed",
    "detect_pathologies",
    "single_support_time",
    "toe_clearance",
    "stride_variability",
    "arm_swing_analysis",
    "speed_normalized_params",
    "detect_equinus",
    "detect_antalgic",
    "detect_parkinsonian",
    "segment_lengths",
    "instantaneous_cadence",
    "compute_rom_summary",
    "estimate_center_of_mass",
    "postural_sway",
    "pca_waveform_analysis",
    "compute_derivatives",
    "time_frequency_analysis",
    "get_normative_curve",
    "get_normative_band",
    "select_stratum",
    "list_joints",
    "list_strata",
    "gait_variable_scores",
    "gait_profile_score_2d",
    "gait_deviation_index_2d",
    "sagittal_deviation_index",
    "movement_analysis_profile",
    "load_json",
    "save_json",
    "set_subject",
    "plot_angles",
    "plot_cycles",
    "plot_events",
    "plot_summary",
    "plot_phase_plane",
    "plot_normative_comparison",
    "plot_gvs_profile",
    "plot_quality_dashboard",
    "plot_longitudinal",
    "plot_arm_swing",
    "plot_session_comparison",
    "plot_cadence_profile",
    "plot_rom_summary",
    "plot_butterfly",
    "animate_normative_comparison",
    "render_skeleton_video",
    "render_skeleton_frame",
    "render_stickfigure_animation",
    "generate_report",
    "generate_longitudinal_report",
    "export_csv",
    "export_json",
    "export_mot",
    "export_trc",
    "export_excel",
    "export_c3d",
    "to_dataframe",
    "export_summary_json",
    "export_openpose_json",
    "export_landmarks_excel",
    "export_opensim_scale_setup",
    "export_ik_setup",
    "export_moco_setup",
    "get_opensim_marker_names",
    "detect_sagittal_alignment",
    "auto_crop_roi",
    "select_person",
    "validate_biomechanical",
    "stratified_ranges",
    "model_accuracy_info",
    "validate_biomechanical_stratified",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG",
    "VIDEO_DEGRADATION_DEFAULTS",
    "build_video_degradation_config",
    "apply_video_degradation",
    "load_vicon_trial_mat",
    "load_c3d",
    "estimate_vicon_offset_seconds",
    "align_vicon_to_myogait",
    "compute_single_trial_benchmark_metrics",
    "attach_vicon_experimental_block",
    "run_single_trial_vicon_benchmark",
    "DEFAULT_SINGLE_PAIR_BENCHMARK_CONFIG",
    "build_single_pair_benchmark_config",
    "run_single_pair_benchmark",
    "detect_walking_direction_from_feet",
    "detect_walking_direction_from_feet_arrays",
    "__version__",
]


def __getattr__(name):
    """Lazily import plotting, video, and report helpers on demand."""
    if name == "__version__":
        return __version__
    module_name = _LAZY_EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Expose public API symbols for interactive help."""
    return sorted(__all__)
