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
    assert config.geometry.theta_max == 2.2
    assert config.geometry.ground_z == 0.0
    assert config.geometry.max_distance == 15.0
    assert (config.bev.x_min, config.bev.x_max) == (-6.0, 10.0)
    assert (config.bev.y_min, config.bev.y_max) == (-6.0, 6.0)
    assert config.bev.resolution == 0.05
    assert config.risk.front_length == 3.0
    assert config.risk.rear_length == 3.0
    assert config.risk.half_width == 0.9
    assert config.risk.start_x == 0.0
    assert config.risk.rear_start_x == 0.0
    assert config.risk.lateral_width == 0.8
    assert (config.risk.vehicle_x_min, config.risk.vehicle_x_max) == (-2.0, 2.0)
    assert config.risk.near_radius == 0.5
    assert config.risk.warning_radius == 1.5
    assert config.risk.danger_occupancy == 0.5


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


def test_legacy_config_without_geometry_sections_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("runtime:\n  seed: 7\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.runtime.seed == 7
    assert config.geometry.theta_max == 2.2
    assert config.bev.resolution == 0.05


def test_geometry_and_bev_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("geometry:\n  theta_max: 2.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_GEOMETRY__THETA_MAX", "2.5")
    monkeypatch.setenv("NEARFIELD360_BEV__RESOLUTION", "0.1")

    config = load_config(config_path)

    assert config.geometry.theta_max == 2.5
    assert config.bev.resolution == 0.1


def test_geometry_config_rejects_nonpositive_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-bev.yaml"
    config_path.write_text("bev:\n  resolution: 0.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="resolution"):
        load_config(config_path)


def test_risk_config_validates_bounds(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-risk.yaml"
    config_path.write_text("risk:\n  rear_length: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="rear_length"):
        load_config(config_path)


def test_robustness_config_defaults() -> None:
    config = load_config()

    assert config.robustness.severities == (1, 2, 3, 4, 5)
    assert config.robustness.rotation_perturbations_deg == (0.5, 1.0, 2.0, 3.0, 5.0)
    assert config.robustness.translation_perturbations_m == (0.02, 0.05, 0.10)
    assert config.robustness.seed == 42


def test_robustness_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("robustness:\n  seed: 99\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_ROBUSTNESS__SEED", "123")

    config = load_config(config_path)

    assert config.robustness.seed == 123


@pytest.mark.parametrize(
    ("field", "bad_value", "pattern"),
    [
        ("severities", "[0, 1]", "severity 0"),
        ("severities", "[1, 6]", "severity 6"),
        ("severities", "[]", "severities must not be empty"),
        ("rotation_perturbations_deg", "[-0.5]", "must be non-negative"),
        ("translation_perturbations_m", "[-0.01]", "must be non-negative"),
    ],
)
def test_robustness_config_validates_fields(
    tmp_path: Path, field: str, bad_value: str, pattern: str
) -> None:
    config_path = tmp_path / "invalid-robustness.yaml"
    config_path.write_text(f"robustness:\n  {field}: {bad_value}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=pattern):
        load_config(config_path)

