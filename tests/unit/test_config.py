from pathlib import Path

import pytest
from pydantic import ValidationError

from nearfield360.config import MAX_CONFIG_BYTES, ConfigurationError, load_config


def test_load_config_uses_typed_defaults() -> None:
    config = load_config()

    assert config.paths.dataset_root is None
    assert config.paths.output_root == Path("outputs")
    assert config.runtime.device == "auto"
    assert config.runtime.seed == 42
    assert config.logging.level == "INFO"
    assert config.logging.structured is False


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "paths:\n  dataset_root: fixtures/woodscape\n"
        "runtime:\n  device: cpu\n  seed: 7\n"
        "logging:\n  level: DEBUG\n  structured: true\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.dataset_root == Path("fixtures/woodscape")
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 7
    assert config.logging.level == "DEBUG"
    assert config.logging.structured is True


def test_environment_overrides_nested_file_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("runtime:\n  device: cpu\n  seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_RUNTIME__DEVICE", "cuda")

    config = load_config(config_path)

    assert config.runtime.device == "cuda"
    assert config.runtime.seed == 7


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("- not\n- a\n- mapping\n", "root must be a mapping"),
        ("1: value\n", "keys must be strings"),
    ],
)
def test_load_config_rejects_invalid_document_roots(
    tmp_path: Path, contents: str, message: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text("surprise: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="surprise"):
        load_config(config_path)


def test_load_config_wraps_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("paths: [unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unable to read configuration file"):
        load_config(config_path)


def test_load_config_validates_seed_range(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-seed.yaml"
    config_path.write_text("runtime:\n  seed: -1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_config(config_path)


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_bounds_file_size(tmp_path: Path) -> None:
    config_path = tmp_path / "oversized.yaml"
    config_path.write_bytes(b"x" * (MAX_CONFIG_BYTES + 1))

    with pytest.raises(ConfigurationError, match="safety limit"):
        load_config(config_path)
