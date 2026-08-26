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


def test_load_config_does_not_share_subdicts_with_defaults(tmp_path):
    """Regression: a loaded config must never alias DEFAULT_CONFIG's
    sub-dicts — mutating one user's config corrupted the process-wide
    defaults (audit finding, fixed in 0.8.0)."""
    import json
    from myogait.config import load_config, DEFAULT_CONFIG

    p = tmp_path / "user.json"
    p.write_text(json.dumps({"extract": {"model": "mediapipe"}}))
    merged = load_config(str(p))

    for key, val in DEFAULT_CONFIG.items():
        if isinstance(val, dict):
            assert merged[key] is not val, f"sub-dict '{key}' is shared"

    # Mutating the merged config must leave the defaults untouched.
    probe_key = next(k for k, v in DEFAULT_CONFIG.items() if isinstance(v, dict))
    probe_sub = next(iter(DEFAULT_CONFIG[probe_key]))
    original = DEFAULT_CONFIG[probe_key][probe_sub]
    merged[probe_key][probe_sub] = "corrupted"
    assert DEFAULT_CONFIG[probe_key][probe_sub] == original


def test_deep_merge_does_not_alias_the_override_values():
    from myogait.config import deep_merge

    base = {"nested": {"defaults": [1]}, "other": 1}
    override = {"nested": {"provided": [2]}}
    merged = deep_merge(base, override)
    merged["nested"]["defaults"].append(3)
    merged["nested"]["provided"].append(4)

    assert base == {"nested": {"defaults": [1]}, "other": 1}
    assert override == {"nested": {"provided": [2]}}
