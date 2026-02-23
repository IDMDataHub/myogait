"""Edge-case tests for export, video rendering, and CLI validation paths."""

import argparse
import builtins
import sys

import pytest


def test_export_functions_raise_type_error_on_non_dict(tmp_path):
    from myogait.export import export_csv, export_excel, export_mot, export_trc, export_c3d

    with pytest.raises(TypeError, match="data must be a dict"):
        export_csv([], str(tmp_path))
    with pytest.raises(TypeError, match="data must be a dict"):
        export_mot([], str(tmp_path / "a.mot"))
    with pytest.raises(TypeError, match="data must be a dict"):
        export_trc([], str(tmp_path / "a.trc"))
    with pytest.raises(TypeError, match="data must be a dict"):
        export_excel([], str(tmp_path / "a.xlsx"))
    with pytest.raises(TypeError, match="data must be a dict"):
        export_c3d([], str(tmp_path / "a.c3d"))


def test_export_excel_raises_clear_import_error_when_openpyxl_missing(monkeypatch, tmp_path):
    from myogait.export import export_excel

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "openpyxl":
            raise ImportError("blocked openpyxl")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("openpyxl", None)

    with pytest.raises(ImportError, match="openpyxl is required"):
        export_excel({"frames": []}, str(tmp_path / "a.xlsx"))


def test_render_stickfigure_animation_rejects_non_positive_fps(tmp_path):
    from myogait.video import render_stickfigure_animation

    data = {
        "meta": {"fps": 30.0},
        "frames": [{"frame_idx": 0, "time_s": 0.0, "landmarks": {}, "confidence": 1.0}],
    }

    with pytest.raises(ValueError, match="fps must be > 0"):
        render_stickfigure_animation(data, str(tmp_path / "out.gif"), fps=0)


def test_render_stickfigure_mp4_raises_runtime_error_without_ffmpeg_and_imageio(monkeypatch, tmp_path):
    from myogait import video as video_mod
    import matplotlib.animation as mpl_animation

    data = {
        "meta": {"fps": 30.0},
        "frames": [{"frame_idx": 0, "time_s": 0.0, "landmarks": {}, "confidence": 1.0}],
    }

    class DummyAnim:
        def save(self, *_args, **_kwargs):
            raise RuntimeError("ffmpeg unavailable")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "imageio":
            raise ImportError("blocked imageio")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(mpl_animation, "FuncAnimation", lambda *a, **k: DummyAnim())
    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("imageio", None)

    with pytest.raises(RuntimeError, match="Neither FFMpeg nor imageio"):
        video_mod.render_stickfigure_animation(
            data,
            str(tmp_path / "out.mp4"),
            format="mp4",
        )


def test_cli_cmd_analyze_exits_when_angles_missing(monkeypatch):
    from myogait import cli

    monkeypatch.setattr("myogait.load_json", lambda _p: {})

    args = argparse.Namespace(
        json_file="dummy.json",
        output_dir=".",
        no_plots=True,
        pdf=False,
        csv=False,
        mot=False,
        trc=False,
        excel=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_analyze(args)
    assert exc.value.code == 1


def test_cli_cmd_download_exits_when_model_missing(capsys):
    from myogait import cli

    args = argparse.Namespace(model="", list=False, dest=None)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_download(args)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "specify a model name" in out
