"""Tests for export functions with full pipeline data.

Verifies that export_csv and to_dataframe work correctly when
called after the full pipeline (extract, angles, events, cycles).
Regression test for Bug 3: export returning empty results.
"""

import pytest
from pathlib import Path

from conftest import make_walking_data, walking_data_with_angles, run_full_pipeline


def test_export_csv_after_pipeline(tmp_path):
    """export_csv should produce files when called with pipeline data."""
    from myogait.export import export_csv
    data, cycles, stats = run_full_pipeline()
    created = export_csv(data, str(tmp_path), cycles=cycles, stats=stats)
    # Should create at least angles.csv and events.csv
    assert len(created) >= 2
    names = [Path(p).name for p in created]
    assert "angles.csv" in names
    assert "events.csv" in names


def test_export_csv_auto_detects_cycles(tmp_path):
    """export_csv should auto-detect cycles from data['cycles_data']."""
    from myogait.export import export_csv
    data, cycles, stats = run_full_pipeline()
    # Don't pass cycles explicitly — should auto-detect from data["cycles_data"]
    created = export_csv(data, str(tmp_path))
    names = [Path(p).name for p in created]
    assert "angles.csv" in names
    assert "events.csv" in names
    # cycles.csv should also be created from auto-detected cycles_data
    assert "cycles.csv" in names


def test_to_dataframe_after_pipeline():
    """to_dataframe should return non-empty DataFrame after pipeline."""
    from myogait.export import to_dataframe
    data, cycles, stats = run_full_pipeline()
    df = to_dataframe(data, what="angles")
    assert len(df) > 0
    assert "hip_L" in df.columns
    assert "knee_L" in df.columns


def test_to_dataframe_all_after_pipeline():
    """to_dataframe(what='all') should return non-empty DataFrames."""
    from myogait.export import to_dataframe
    data, cycles, stats = run_full_pipeline()
    result = to_dataframe(data, what="all")
    assert len(result["angles"]) > 0
    assert len(result["events"]) > 0


def test_export_csv_without_cycles(tmp_path):
    """export_csv should still export angles/events even without cycles."""
    from myogait import compute_angles, detect_events
    from myogait.export import export_csv
    data = make_walking_data(300, fps=30.0)
    compute_angles(data, correction_factor=1.0, calibrate=False)
    detect_events(data)
    # No segment_cycles call — just angles + events
    created = export_csv(data, str(tmp_path))
    names = [Path(p).name for p in created]
    assert "angles.csv" in names
    assert "events.csv" in names


def test_export_csv_angles_only(tmp_path):
    """export_csv should export angles even without events."""
    from myogait.export import export_csv
    data = walking_data_with_angles()
    created = export_csv(data, str(tmp_path))
    names = [Path(p).name for p in created]
    assert "angles.csv" in names
