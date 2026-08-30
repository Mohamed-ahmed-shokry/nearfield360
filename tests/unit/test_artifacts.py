from pathlib import Path

import pytest

from nearfield360.utils.artifacts import ArtifactError, read_json, write_json


def test_json_round_trip_is_stable_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "nested/result.json"
    value = {"z": [2, 1], "a": "مسار"}

    write_json(path, value)

    assert read_json(path) == value
    assert path.read_text(encoding="utf-8").startswith('{\n  "a":')
    assert list(path.parent.iterdir()) == [path]


def test_json_write_never_clobbers_without_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"original": True})

    with pytest.raises(ArtifactError, match="Unable to write"):
        write_json(path, {"replacement": True})
    assert read_json(path) == {"original": True}
    assert list(tmp_path.iterdir()) == [path]

    write_json(path, {"replacement": True}, overwrite=True)
    assert read_json(path) == {"replacement": True}


@pytest.mark.parametrize("payload", [b'{"a":0,"a":1}', b"NaN", b"Infinity", b"\xff", b"{"])
def test_json_reader_rejects_ambiguous_or_invalid_documents(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)

    with pytest.raises(ArtifactError, match="Unable to read"):
        read_json(path)


def test_json_reader_bounds_size_and_nesting(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
    with pytest.raises(ArtifactError, match="byte limit"):
        read_json(path, max_bytes=100)
    with pytest.raises(ArtifactError, match="Unable to read"):
        read_json(path)
    with pytest.raises(ArtifactError, match="positive integer"):
        read_json(path, max_bytes=0)


def test_json_writer_rejects_non_finite_results_without_modifying_original(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"original": True})
    with pytest.raises(ArtifactError, match="Unable to write"):
        write_json(path, {"bad_metric": float("nan")}, overwrite=True)
    assert read_json(path) == {"original": True}
