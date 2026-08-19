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


# ── analyze --detrend (opt-in sagittal drift correction) ─────────────

# Injected drift (deg/frame) for the CLI detrend tests: 0.2 deg/frame
# over 40 frames is an 8 deg slide, within the 10-30 deg range quoted
# in the apply_linear_detrend docstring for real recordings.
_DETREND_DRIFT = 0.2
_DETREND_BASE = 10.0
_DETREND_N_FRAMES = 40  # >= 20 valid samples required by the OLS fit


def _write_detrend_fixture(tmp_path):
    """Write a pivot JSON with a linear ramp on hip_L and events present.

    Events are pre-populated (shape copied from the benchmark fakes) so
    cmd_analyze takes the "events already present" branch and does not
    re-run detection.
    """
    from myogait.schema import create_empty, save_json

    data = create_empty(video_path="x.mp4", fps=30.0, n_frames=_DETREND_N_FRAMES)
    frames = []
    for i in range(_DETREND_N_FRAMES):
        frames.append({
            "frame_idx": i,
            "hip_L": float(_DETREND_BASE + _DETREND_DRIFT * i),
        })
    data["angles"] = {"frames": frames}
    data["events"] = {
        "method": "test",
        "left_hs": [{"frame": 1}],
        "right_hs": [{"frame": 2}],
        "left_to": [{"frame": 1}],
        "right_to": [{"frame": 2}],
    }
    path = tmp_path / "detrend.json"
    save_json(data, str(path))
    return str(path)


def _analyze_args(json_file, detrend):
    return argparse.Namespace(
        json_file=json_file,
        output_dir=".",
        no_plots=True,
        pdf=False,
        csv=False,
        mot=False,
        trc=False,
        excel=False,
        detrend=detrend,
    )


def _stub_analyze_pipeline(monkeypatch):
    """Stub the heavy steps cmd_analyze runs after detrending."""
    monkeypatch.setattr("myogait.segment_cycles", lambda data: {"cycles": []})
    monkeypatch.setattr(
        "myogait.analyze_gait", lambda data, cycles: {"spatiotemporal": {}}
    )


def test_analyze_detrend_flag_registered(monkeypatch):
    """The analyze subparser exposes --detrend, defaulting to False.

    Detrending must stay opt-in (validations exist only for healthy
    adults), so the flag must be absent-by-default and store_true.
    """
    from myogait import cli

    captured = {}

    def _fake(args):
        captured["args"] = args

    monkeypatch.setattr(cli, "cmd_analyze", _fake)

    monkeypatch.setattr(
        "sys.argv", ["myogait", "analyze", "x.json", "--detrend", "--no-plots"]
    )
    cli.main()
    assert captured["args"].detrend is True

    monkeypatch.setattr("sys.argv", ["myogait", "analyze", "x.json", "--no-plots"])
    cli.main()
    assert captured["args"].detrend is False


def test_cmd_analyze_detrend_applies_and_persists(monkeypatch, tmp_path, capsys):
    """--detrend removes the drift AND saves the corrected JSON back."""
    import numpy as np

    from myogait import cli
    from myogait.schema import load_json

    path = _write_detrend_fixture(tmp_path)
    before = load_json(path)
    before_vals = np.array(
        [f["hip_L"] for f in before["angles"]["frames"]], dtype=float
    )
    before_mean = float(np.mean(before_vals))
    idx = np.arange(len(before_vals))
    before_slope = float(np.polyfit(idx, before_vals, 1)[0])
    assert abs(before_slope - _DETREND_DRIFT) < 1e-9  # drift present

    _stub_analyze_pipeline(monkeypatch)
    cli.cmd_analyze(_analyze_args(path, detrend=True))

    out = capsys.readouterr().out
    assert "Detrend" in out

    # Persisted to disk, not just in memory.
    saved = load_json(path)
    assert saved["angles"].get("linear_detrended") is True
    saved_vals = np.array(
        [f["hip_L"] for f in saved["angles"]["frames"]], dtype=float
    )
    assert abs(float(np.polyfit(idx, saved_vals, 1)[0])) < 1e-9  # drift gone
    assert abs(float(np.mean(saved_vals)) - before_mean) < 1e-9  # mean kept


def test_cmd_analyze_without_detrend_flag_leaves_angles_untouched(
    monkeypatch, tmp_path
):
    """Without --detrend the angles on disk keep their drift."""
    import numpy as np

    from myogait import cli
    from myogait.schema import load_json

    path = _write_detrend_fixture(tmp_path)

    _stub_analyze_pipeline(monkeypatch)
    cli.cmd_analyze(_analyze_args(path, detrend=False))

    saved = load_json(path)
    assert "linear_detrended" not in saved["angles"]
    vals = np.array([f["hip_L"] for f in saved["angles"]["frames"]], dtype=float)
    idx = np.arange(len(vals))
    assert abs(float(np.polyfit(idx, vals, 1)[0]) - _DETREND_DRIFT) < 1e-9
