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

Clinical scores::

    from myogait import gait_profile_score_2d, gait_deviation_index_2d
    gps = gait_profile_score_2d(cycles)
    gdi = gait_deviation_index_2d(cycles)

Export::

    from myogait import export_csv, export_mot, export_trc
    export_csv(data, "./output", cycles, stats)
    export_mot(data, "kinematics.mot")

Validation::

    from myogait import validate_biomechanical
    report = validate_biomechanical(data, cycles)
"""

__version__ = "0.3.0"

from .extract import extract
from .normalize import (
    normalize,
    confidence_filter,
    detect_outliers,
    data_quality_score,
)
from .angles import compute_angles, compute_extended_angles, compute_frontal_angles
from .events import detect_events, list_event_methods, event_consensus, validate_events
from .cycles import segment_cycles
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
    movement_analysis_profile,
)
from .schema import load_json, save_json, set_subject
from .plotting import plot_angles, plot_cycles, plot_events, plot_summary, plot_phase_plane
from .report import generate_report
from .export import export_csv, export_mot, export_trc, export_excel
from .validation import validate_biomechanical
from .config import load_config, save_config, DEFAULT_CONFIG

__all__ = [
    # Core pipeline
    "extract",
    "normalize",
    "compute_angles",
    "compute_extended_angles",
    "compute_frontal_angles",
    "detect_events",
    "list_event_methods",
    "event_consensus",
    "validate_events",
    "segment_cycles",
    "analyze_gait",
    # Quality
    "confidence_filter",
    "detect_outliers",
    "data_quality_score",
    # Analysis functions
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
    # Normative
    "get_normative_curve",
    "get_normative_band",
    "select_stratum",
    "list_joints",
    "list_strata",
    # Clinical scores
    "gait_variable_scores",
    "gait_profile_score_2d",
    "gait_deviation_index_2d",
    "movement_analysis_profile",
    # Schema
    "load_json",
    "save_json",
    "set_subject",
    # Visualization
    "plot_angles",
    "plot_cycles",
    "plot_events",
    "plot_summary",
    "plot_phase_plane",
    # Report
    "generate_report",
    # Export
    "export_csv",
    "export_mot",
    "export_trc",
    "export_excel",
    # Validation
    "validate_biomechanical",
    # Config
    "load_config",
    "save_config",
    "DEFAULT_CONFIG",
    # Meta
    "__version__",
]
