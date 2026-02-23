"""Coverage-focused tests for config/schema I/O branches."""

import json

import pytest


def test_load_config_yaml_roundtrip(tmp_path):
    from myogait.config import save_config, load_config

    cfg = {
        "extract": {"model": "mediapipe"},
        "events": {"method": "zeni"},
    }
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded["extract"]["model"] == "mediapipe"
    assert loaded["events"]["method"] == "zeni"
    # merged defaults present
    assert "normalize" in loaded


def test_load_config_yaml_non_dict_raises(tmp_path):
    from myogait.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Config must be a dict"):
        load_config(path)


def test_schema_load_json_root_non_dict_raises(tmp_path):
    from myogait.schema import load_json

    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root must be a dict"):
        load_json(path)


def test_schema_save_load_unicode_content(tmp_path):
    from myogait.schema import save_json, load_json

    payload = {
        "myogait_version": "0.0.0",
        "meta": {"fps": 30.0},
        "frames": [],
        "subject": {"notes": "marche régulière côté gauche"},
    }
    path = tmp_path / "unicode.json"
    save_json(payload, path)
    loaded = load_json(path)
    assert loaded["subject"]["notes"] == payload["subject"]["notes"]
