"""Tests for model integrity helpers (revision/checksum enforcement)."""

import hashlib

import pytest

from myogait.models.sapiens import _sha256_file, _verify_model_integrity


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "model.pt2"
    payload = b"test-model-bytes"
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert _sha256_file(str(path)) == expected


def test_verify_model_integrity_accepts_matching_checksum(tmp_path):
    path = tmp_path / "model.pt2"
    payload = b"abc123"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    _verify_model_integrity(str(path), expected, "sapiens-0.3b")


def test_verify_model_integrity_raises_on_mismatch(tmp_path):
    path = tmp_path / "model.pt2"
    path.write_bytes(b"payload")

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        _verify_model_integrity(str(path), "0" * 64, "sapiens-0.3b")


def test_verify_model_integrity_strict_missing_checksum_raises(tmp_path, monkeypatch):
    from myogait.models import sapiens as sapiens_mod

    path = tmp_path / "model.pt2"
    path.write_bytes(b"payload")

    monkeypatch.setattr(sapiens_mod, "_STRICT_MODEL_CHECKSUM", True)
    with pytest.raises(RuntimeError, match="No SHA256 configured"):
        _verify_model_integrity(str(path), None, "sapiens-0.3b")
