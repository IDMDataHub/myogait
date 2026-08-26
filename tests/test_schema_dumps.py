"""Tests for dumps_json_safe, the single JSON serialization entry point.

Regression coverage for upstream issue #35: JSON dumps crashed on numpy
scalars, pathlib.Path and enum.Enum values embedded in user-provided dicts.
"""

import enum
import io
import json
from pathlib import Path

import numpy as np
import pytest

from myogait.schema import dumps_json_safe, load_json, save_json


class _Side(enum.Enum):
    """Sample enum used to exercise Enum serialization."""

    LEFT = "left"


def test_numpy_scalars_and_arrays():
    data = {
        "f64": np.float64(1.5),
        "i64": np.int64(7),
        "flag": np.bool_(True),
        "arr": np.array([1.0, 2.0, 3.0]),
    }
    out = json.loads(dumps_json_safe(data))
    assert out["f64"] == 1.5
    assert out["i64"] == 7
    assert out["flag"] is True
    assert out["arr"] == [1.0, 2.0, 3.0]


def test_path_serialized_as_str():
    video = Path("data") / "video.mp4"
    out = json.loads(dumps_json_safe({"video": video}))
    assert out["video"] == str(video)


def test_enum_serialized_as_value():
    out = json.loads(dumps_json_safe({"side": _Side.LEFT}))
    assert out["side"] == _Side.LEFT.value


def test_nested_containers():
    data = {
        "outer": [
            np.int64(1),
            (np.float64(2.5), Path("nested.txt")),
            {"deep": np.array([np.int64(3)])},
        ]
    }
    out = json.loads(dumps_json_safe(data))
    assert out["outer"][0] == 1
    assert out["outer"][1] == [2.5, "nested.txt"]
    assert out["outer"][2]["deep"] == [3]


def test_indent_kwarg_forwarded():
    data = {"a": 1}
    assert dumps_json_safe(data) == json.dumps(
        {"a": 1}, indent=2, ensure_ascii=False
    )
    assert dumps_json_safe(data, indent=4) == json.dumps(
        {"a": 1}, indent=4, ensure_ascii=False
    )


def test_unicode_preserved():
    # ensure_ascii=False must keep human-readable unicode in the output.
    notes = "marche régulière côté gauche"
    assert notes in dumps_json_safe({"notes": notes})


def test_unserializable_object_raises_type_error():
    with pytest.raises(TypeError):
        dumps_json_safe({"bad": object()})


def test_load_json_accepts_a_binary_stream():
    data = load_json(io.BytesIO(b'{"meta": {"fps": 30}, "frames": []}'))

    assert data["meta"]["fps"] == 30
    assert data["frames"] == []


def test_save_json_preserves_an_existing_pivot_when_serialization_fails(tmp_path):
    target = tmp_path / "pivot.json"
    target.write_text('{"meta": {}, "frames": []}', encoding="utf-8")

    with pytest.raises(TypeError):
        save_json({"meta": {}, "frames": [], "unsupported": object()}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"meta": {}, "frames": []}
    assert not list(tmp_path.glob(".pivot.json.*.tmp"))
