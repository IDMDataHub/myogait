"""Coverage-focused tests for CLI parsing and error handling."""

import argparse

import pytest


def test_main_without_command_exits_1(monkeypatch):
    from myogait import cli

    monkeypatch.setattr("sys.argv", ["myogait"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_main_handles_filenotfounderror(monkeypatch):
    from myogait import cli

    def _boom(_):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(cli, "cmd_extract", _boom)
    monkeypatch.setattr("sys.argv", ["myogait", "extract", "video.mp4"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_main_handles_valueerror(monkeypatch):
    from myogait import cli

    def _boom(_):
        raise ValueError("bad value")

    monkeypatch.setattr(cli, "cmd_extract", _boom)
    monkeypatch.setattr("sys.argv", ["myogait", "extract", "video.mp4"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_main_handles_importerror(monkeypatch):
    from myogait import cli

    def _boom(_):
        raise ImportError("missing dep")

    monkeypatch.setattr(cli, "cmd_extract", _boom)
    monkeypatch.setattr("sys.argv", ["myogait", "extract", "video.mp4"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_main_handles_keyboard_interrupt(monkeypatch):
    from myogait import cli

    def _boom(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "cmd_extract", _boom)
    monkeypatch.setattr("sys.argv", ["myogait", "extract", "video.mp4"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 130


def test_main_dispatches_subcommand(monkeypatch):
    from myogait import cli

    called = {"ok": False}

    def _fake(args):
        assert isinstance(args, argparse.Namespace)
        called["ok"] = True

    monkeypatch.setattr(cli, "cmd_extract", _fake)
    monkeypatch.setattr("sys.argv", ["myogait", "extract", "video.mp4"])
    cli.main()
    assert called["ok"] is True


def test_experimental_from_args_defaults():
    from myogait.cli import _experimental_from_args

    args = argparse.Namespace()
    cfg = _experimental_from_args(args)
    assert cfg["enabled"] is False
    assert cfg["downscale"] == 1.0
    assert cfg["contrast"] == 1.0


def test_get_version_returns_string():
    from myogait.cli import _get_version

    assert isinstance(_get_version(), str)


def test_cmd_download_list(monkeypatch, capsys):
    from myogait import cli

    monkeypatch.setattr(
        "myogait.models.sapiens._MODELS",
        {"0.3b": ("m.pt2", "facebook/sapiens-pose-0.3b-torchscript")},
    )
    monkeypatch.setattr(
        "myogait.models.sapiens_depth._DEPTH_MODELS",
        {"0.3b": ("d.pt2", "facebook/sapiens-depth-0.3b-torchscript")},
    )
    monkeypatch.setattr(
        "myogait.models.sapiens_seg._SEG_MODELS",
        {"0.3b": ("s.pt2", "facebook/sapiens-seg-0.3b-torchscript")},
    )

    args = argparse.Namespace(list=True, model="", dest=None)
    cli.cmd_download(args)
    out = capsys.readouterr().out
    assert "Available models" in out
    assert "sapiens-0.3b" in out


def test_cmd_download_unknown_model_exits(monkeypatch):
    from myogait import cli

    monkeypatch.setattr("myogait.models.sapiens._MODELS", {})
    monkeypatch.setattr("myogait.models.sapiens_depth._DEPTH_MODELS", {})
    monkeypatch.setattr("myogait.models.sapiens_seg._SEG_MODELS", {})

    args = argparse.Namespace(list=False, model="unknown", dest=None)
    with pytest.raises(SystemExit):
        cli.cmd_download(args)


def test_cmd_info_no_frames(monkeypatch, capsys):
    from myogait import cli

    monkeypatch.setattr(
        cli,
        "cmd_info",
        cli.cmd_info,
    )
    monkeypatch.setattr(
        "myogait.load_json",
        lambda _: {
            "myogait_version": "0.4.1",
            "meta": {"video_path": "x.mp4", "fps": 30.0, "width": 100, "height": 100, "n_frames": 0, "duration_s": 0.0},
            "frames": [],
        },
    )
    args = argparse.Namespace(json_file="dummy.json")
    cli.cmd_info(args)
    out = capsys.readouterr().out
    assert "No frames" in out


def test_cmd_batch_no_match_exits():
    from myogait import cli

    args = argparse.Namespace(inputs=["/no/match/*.mp4"], output_dir="out", config=None, model="mediapipe", csv=False, pdf=False)
    with pytest.raises(SystemExit):
        cli.cmd_batch(args)
