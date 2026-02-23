"""Security and I/O validation tests for model download and lookup paths."""

import os
import types
from pathlib import Path

import pytest

from myogait.models import mediapipe as mp_mod
from myogait.models import sapiens as sapiens_mod


def test_ensure_model_returns_explicit_existing_path(tmp_path):
    path = tmp_path / "custom.task"
    path.write_bytes(b"x")

    resolved = mp_mod._ensure_model(str(path))

    assert resolved == str(path)


def test_ensure_model_prefers_existing_default_file(monkeypatch, tmp_path):
    default_dir = tmp_path / "models"
    default_file = default_dir / "pose_landmarker_heavy.task"
    default_dir.mkdir()
    default_file.write_bytes(b"ok")

    monkeypatch.setattr(mp_mod, "_DEFAULT_MODEL_DIR", str(default_dir))

    resolved = mp_mod._ensure_model()

    assert resolved == str(default_file)


def test_ensure_model_downloads_when_missing(monkeypatch, tmp_path):
    default_dir = tmp_path / "models"
    default_file = default_dir / "pose_landmarker_heavy.task"
    monkeypatch.setattr(mp_mod, "_DEFAULT_MODEL_DIR", str(default_dir))

    called = {"makedirs": False, "url": None, "dest": None}

    def _fake_makedirs(path, exist_ok=False):
        called["makedirs"] = True
        Path(path).mkdir(parents=True, exist_ok=True)

    def _fake_urlretrieve(url, dest):
        called["url"] = url
        called["dest"] = dest
        Path(dest).write_bytes(b"model-bytes")
        return dest, None

    monkeypatch.setattr(os, "makedirs", _fake_makedirs)
    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)

    resolved = mp_mod._ensure_model()

    assert resolved == str(default_file)
    assert called["makedirs"] is True
    assert called["url"] == mp_mod._MODEL_URL
    assert called["dest"] == str(default_file)


def test_download_model_unknown_size_rejected(monkeypatch):
    fake_hub = types.SimpleNamespace(hf_hub_download=lambda **_kwargs: "/tmp/model.pt2")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    with pytest.raises(ValueError, match="Unknown model size"):
        sapiens_mod.download_model("bogus")


def test_download_model_success_invokes_integrity_check(monkeypatch, tmp_path):
    calls = {}

    def _fake_hf_hub_download(**kwargs):
        calls["hub_kwargs"] = kwargs
        return str(tmp_path / "a.pt2")

    fake_hub = types.SimpleNamespace(hf_hub_download=_fake_hf_hub_download)

    monkeypatch.setattr(sapiens_mod, "_MODEL_SHA256", {"0.3b": "abc", "0.6b": None, "1b": None})
    monkeypatch.setattr(sapiens_mod, "_MODEL_REVISIONS", {"0.3b": "rev123", "0.6b": "main", "1b": "main"})
    monkeypatch.setattr(
        sapiens_mod,
        "_verify_model_integrity",
        lambda path, expected, label: calls.setdefault("verify", (path, expected, label)),
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)

    path = sapiens_mod.download_model("0.3b", dest=str(tmp_path))

    assert path == str(tmp_path / "a.pt2")
    assert calls["hub_kwargs"]["revision"] == "rev123"
    assert calls["hub_kwargs"]["local_dir"] == str(tmp_path)
    assert calls["verify"][0] == path
    assert calls["verify"][1] == "abc"
    assert calls["verify"][2] == "sapiens-0.3b"


def test_find_model_returns_explicit_path_if_present(tmp_path):
    path = tmp_path / "model.pt2"
    path.write_bytes(b"x")

    assert sapiens_mod._find_model("0.3b", model_path=str(path)) == str(path)


def test_find_model_raises_for_unknown_size():
    with pytest.raises(ValueError, match="Unknown Sapiens model size"):
        sapiens_mod._find_model("9b")


def test_find_model_uses_local_path_and_verifies(monkeypatch, tmp_path):
    filename, _ = sapiens_mod._MODELS["0.3b"]
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    candidate = model_dir / filename
    candidate.write_bytes(b"payload")

    verify_calls = []
    monkeypatch.setattr(sapiens_mod, "_DEFAULT_MODEL_PATHS", [model_dir])
    monkeypatch.setattr(sapiens_mod, "_MODEL_SHA256", {"0.3b": "abcd", "0.6b": None, "1b": None})
    monkeypatch.setattr(
        sapiens_mod,
        "_verify_model_integrity",
        lambda path, expected, label: verify_calls.append((path, expected, label)),
    )

    resolved = sapiens_mod._find_model("0.3b")

    assert resolved == str(candidate)
    assert verify_calls == [(str(candidate), "abcd", "sapiens-0.3b")]


def test_find_model_import_error_from_download_is_wrapped(monkeypatch, tmp_path):
    filename, _ = sapiens_mod._MODELS["0.3b"]
    monkeypatch.setattr(sapiens_mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
    monkeypatch.setattr(
        sapiens_mod,
        "download_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("no hub")),
    )

    with pytest.raises(FileNotFoundError) as exc:
        sapiens_mod._find_model("0.3b")

    msg = str(exc.value)
    assert "huggingface-hub" in msg
    assert "Searched:" in msg
    assert filename in msg


def test_find_model_generic_download_error_is_wrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(sapiens_mod, "_DEFAULT_MODEL_PATHS", [tmp_path])
    monkeypatch.setattr(
        sapiens_mod,
        "download_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    with pytest.raises(FileNotFoundError, match="Auto-download failed"):
        sapiens_mod._find_model("0.3b")
