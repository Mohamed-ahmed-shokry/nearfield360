"""Typed project configuration with safe YAML loading and environment overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

MAX_CONFIG_BYTES = 1024 * 1024


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be loaded safely."""


class PathsConfig(BaseModel):
    """Filesystem locations used by NearField360."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_root: Path | None = None
    output_root: Path = Path("outputs")

    @field_validator("dataset_root", "output_root", mode="before")
    @classmethod
    def expand_user_path(cls, value: object) -> object:
        """Expand a leading home-directory marker without resolving the path."""
        if isinstance(value, str):
            return Path(value).expanduser()
        return value


class RuntimeConfig(BaseModel):
    """Cross-cutting deterministic runtime options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class LoggingConfig(BaseModel):
    """Application logging behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    structured: bool = False


class ProjectConfig(BaseSettings):
    """Top-level NearField360 settings.

    Environment variables use ``NEARFIELD360_`` plus ``__`` for nesting. For
    example, ``NEARFIELD360_PATHS__DATASET_ROOT`` overrides the YAML value.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEARFIELD360_",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="forbid",
        frozen=True,
        nested_model_default_partial_update=True,
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Give process environment values precedence over file values."""
        del settings_cls, dotenv_settings
        return env_settings, init_settings, file_secret_settings


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigurationError(
            f"Configuration file exceeds the {MAX_CONFIG_BYTES}-byte safety limit: {path}"
        )

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to read configuration file {path}: {exc}") from exc

    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in document):
        raise ConfigurationError(f"Configuration keys must be strings: {path}")
    return document


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load validated settings from YAML, then apply environment overrides."""
    values = {} if path is None else dict(_read_yaml_mapping(path))
    return ProjectConfig(**values)


__all__ = [
    "MAX_CONFIG_BYTES",
    "ConfigurationError",
    "LoggingConfig",
    "PathsConfig",
    "ProjectConfig",
    "RuntimeConfig",
    "load_config",
]
