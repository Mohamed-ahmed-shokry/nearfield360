"""Bounded JSON inputs and atomic, non-clobbering local result publication."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Never

DEFAULT_JSON_BYTES = 8 * 1024 * 1024


class ArtifactError(ValueError):
    """A local artifact is malformed, unavailable, oversized, or would be overwritten."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _non_finite(value: str) -> Never:
    raise ArtifactError(f"Non-finite JSON constant: {value}")


def read_json(path: Path, *, max_bytes: int = DEFAULT_JSON_BYTES) -> Any:
    """Read one bounded UTF-8 document, rejecting duplicate keys and NaN/Infinity."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ArtifactError("max_bytes must be a positive integer")
    try:
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ArtifactError(f"JSON exceeds the {max_bytes}-byte limit")
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_non_finite
        )
    except (OSError, ValueError, RecursionError) as exc:
        raise ArtifactError(f"Unable to read JSON artifact {path}: {exc}") from exc


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    """Publish complete JSON or leave the previous file untouched on failure.

    The default uses an atomic hard-link create to avoid a check-then-overwrite race.
    A filesystem without hard-link support fails explicitly. ``overwrite=True`` uses
    an atomic replace. Temporary files are created on the destination filesystem.
    """
    temporary: Path | None = None
    try:
        document = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ArtifactError(f"Unable to write JSON artifact {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["DEFAULT_JSON_BYTES", "ArtifactError", "read_json", "write_json"]
