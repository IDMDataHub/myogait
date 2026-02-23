"""Robustness tests for report page fallback behavior."""

import matplotlib.pyplot as plt


class _DummyPdf:
    def __init__(self):
        self.calls = 0

    def savefig(self, fig, dpi=None):
        assert fig is not None
        assert dpi is not None
        self.calls += 1


def _strings():
    return {
        "normative_title": "Normative",
        "frontal_title": "Frontal",
        "gvs_title": "GVS",
        "quality_title": "Quality",
        "no_data": "No data",
    }


def _raise(*_args, **_kwargs):
    raise RuntimeError("forced plotting error")


def test_save_fallback_page_writes_single_pdf_page():
    from myogait.report import _save_fallback_page

    pdf = _DummyPdf()
    _save_fallback_page(pdf, "Title", "Message")

    assert pdf.calls == 1


def test_page_normative_uses_fallback_on_plot_error(monkeypatch):
    from myogait import report as report_mod

    called = {}
    monkeypatch.setattr("myogait.plotting.plot_normative_comparison", _raise)
    monkeypatch.setattr(
        report_mod,
        "_save_fallback_page",
        lambda _pdf, title, msg: called.setdefault("args", (title, msg)),
    )

    report_mod._page_normative(_DummyPdf(), {}, {}, _strings())

    assert called["args"] == ("Normative", "No data")


def test_page_frontal_uses_fallback_on_plot_error(monkeypatch):
    from myogait import report as report_mod

    called = {}
    monkeypatch.setattr("myogait.plotting.plot_normative_comparison", _raise)
    monkeypatch.setattr(
        report_mod,
        "_save_fallback_page",
        lambda _pdf, title, msg: called.setdefault("args", (title, msg)),
    )

    report_mod._page_frontal(_DummyPdf(), {}, {}, _strings())

    assert called["args"] == ("Frontal", "No data")


def test_page_gvs_uses_fallback_on_plot_error(monkeypatch):
    from myogait import report as report_mod

    called = {}
    monkeypatch.setattr("myogait.plotting.plot_gvs_profile", _raise)
    monkeypatch.setattr(
        report_mod,
        "_save_fallback_page",
        lambda _pdf, title, msg: called.setdefault("args", (title, msg)),
    )

    report_mod._page_gvs(_DummyPdf(), {}, {}, _strings())

    assert called["args"] == ("GVS", "No data")


def test_page_quality_uses_fallback_on_plot_error(monkeypatch):
    from myogait import report as report_mod

    called = {}
    monkeypatch.setattr("myogait.plotting.plot_quality_dashboard", _raise)
    monkeypatch.setattr(
        report_mod,
        "_save_fallback_page",
        lambda _pdf, title, msg: called.setdefault("args", (title, msg)),
    )

    report_mod._page_quality(_DummyPdf(), {}, _strings())

    assert called["args"] == ("Quality", "No data")


def test_fallback_page_closes_figure(monkeypatch):
    from myogait.report import _save_fallback_page

    pdf = _DummyPdf()
    before = len(plt.get_fignums())
    _save_fallback_page(pdf, "T", "M")
    after = len(plt.get_fignums())

    assert after == before
