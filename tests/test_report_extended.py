"""Extended tests for report i18n, new report pages, and C3D export."""

import os
import pytest

from conftest import run_full_pipeline

from myogait.report import generate_report, generate_longitudinal_report, _STRINGS
from myogait.export import export_c3d


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_data():
    """Run the full pipeline once for the module."""
    data, cycles, stats = run_full_pipeline()
    return data, cycles, stats


# ── Report i18n tests ─────────────────────────────────────────────────

def test_generate_report_english(pipeline_data, tmp_path):
    """generate_report with language='en' creates a PDF with English metadata."""
    data, cycles, stats = pipeline_data
    out = str(tmp_path / "report_en.pdf")
    result = generate_report(data, cycles, stats, out, language="en")
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_generate_report_french(pipeline_data, tmp_path):
    """generate_report with language='fr' (default) creates a PDF."""
    data, cycles, stats = pipeline_data
    out = str(tmp_path / "report_fr.pdf")
    result = generate_report(data, cycles, stats, out)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_strings_dict_complete():
    """All keys present in both 'fr' and 'en' translations."""
    fr_keys = set(_STRINGS["fr"].keys())
    en_keys = set(_STRINGS["en"].keys())
    missing_in_en = fr_keys - en_keys
    missing_in_fr = en_keys - fr_keys
    assert not missing_in_en, f"Keys in FR but not EN: {missing_in_en}"
    assert not missing_in_fr, f"Keys in EN but not FR: {missing_in_fr}"
    # Verify both have at least the core keys
    core_keys = {"title", "patient", "hip", "knee", "ankle", "left", "right",
                 "overview_title", "bilateral_title", "stats_title",
                 "normative_title", "gvs_title", "quality_title"}
    assert core_keys.issubset(fr_keys), f"Missing core keys in FR: {core_keys - fr_keys}"
    assert core_keys.issubset(en_keys), f"Missing core keys in EN: {core_keys - en_keys}"


def test_report_has_normative_page(pipeline_data, tmp_path):
    """Report with normative comparison page produces a valid PDF."""
    data, cycles, stats = pipeline_data
    out = str(tmp_path / "report_norm.pdf")
    result = generate_report(data, cycles, stats, out, language="en")
    assert os.path.isfile(result)
    # The file should be larger than a minimal PDF (~a few KB at minimum)
    assert os.path.getsize(result) > 1000


def test_longitudinal_report(pipeline_data, tmp_path):
    """generate_longitudinal_report creates a multi-session PDF."""
    data, cycles, stats = pipeline_data
    sessions = [
        {"data": data, "cycles": cycles, "stats": stats, "label": "Session 1"},
        {"data": data, "cycles": cycles, "stats": stats, "label": "Session 2"},
    ]
    out = str(tmp_path / "longitudinal.pdf")
    result = generate_longitudinal_report(sessions, out, language="fr")
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_export_c3d_import_error():
    """export_c3d raises ImportError when c3d package is not installed."""
    import sys
    # Temporarily make c3d unimportable
    original = sys.modules.get("c3d", None)
    sys.modules["c3d"] = None
    try:
        data = {"frames": [{"frame_idx": 0, "landmarks": {"NOSE": {"x": 0.5, "y": 0.5}}}],
                "meta": {"fps": 30.0}}
        with pytest.raises(ImportError, match="c3d is required"):
            export_c3d(data, "/tmp/test.c3d")
    finally:
        if original is None:
            sys.modules.pop("c3d", None)
        else:
            sys.modules["c3d"] = original
