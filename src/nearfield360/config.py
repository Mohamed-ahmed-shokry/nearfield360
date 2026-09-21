"""Typed project configuration with safe YAML loading and environment overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


class GeometryConfig(BaseModel):
    """Fisheye angular domain and vehicle ground-plane defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    theta_max: float = Field(default=2.2, gt=0.0, lt=3.141592653589793, allow_inf_nan=False)
    ground_z: float = Field(default=0.0, allow_inf_nan=False)
    max_distance: float = Field(default=15.0, gt=0.0, allow_inf_nan=False)


class BevConfig(BaseModel):
    """Local bird's-eye-view grid extent and resolution in metres."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float = Field(default=-6.0, allow_inf_nan=False)
    x_max: float = Field(default=10.0, allow_inf_nan=False)
    y_min: float = Field(default=-6.0, allow_inf_nan=False)
    y_max: float = Field(default=6.0, allow_inf_nan=False)
    resolution: float = Field(default=0.05, gt=0.0, allow_inf_nan=False)


class OccupancyConfig(BaseModel):
    """Which semantic classes vote free/occupied and how evidence is weighted.

    Class lists are resolved against the WoodScape semantic palette by name
    and must stay disjoint. Weight decay reduces the contribution of distant,
    uncertainty-dominated measurements.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    free_classes: tuple[str, ...] = ("road", "lanemarks", "curb")
    occupied_classes: tuple[str, ...] = (
        "person",
        "rider",
        "vehicles",
        "bicycle",
        "motorcycle",
        "traffic_sign",
    )
    confidence_slope: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    min_evidence: int = Field(default=1, ge=1)


class RiskConfig(BaseModel):
    """Spatial risk filtering over the fused occupancy layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    front_length: float = Field(default=3.0, gt=0.0, allow_inf_nan=False)
    rear_length: float = Field(default=3.0, gt=0.0, allow_inf_nan=False)
    half_width: float = Field(default=0.9, ge=0.0, allow_inf_nan=False)
    start_x: float = Field(default=0.0, allow_inf_nan=False)
    rear_start_x: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    lateral_width: float = Field(default=0.8, gt=0.0, allow_inf_nan=False)
    vehicle_x_min: float = Field(default=-2.0, allow_inf_nan=False)
    vehicle_x_max: float = Field(default=2.0, allow_inf_nan=False)
    near_radius: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    warning_radius: float = Field(default=1.5, gt=0.0, allow_inf_nan=False)
    danger_occupancy: float = Field(default=0.5, ge=0.0, le=1.0)


class RobustnessConfig(BaseModel):
    """Controlled perturbation and corruption benchmark options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severities: tuple[int, ...] = (1, 2, 3, 4, 5)
    rotation_perturbations_deg: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    translation_perturbations_m: tuple[float, ...] = (0.02, 0.05, 0.10)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("severities")
    @classmethod
    def validate_severities(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("severities must not be empty")
        for s in values:
            if s < 1 or s > 5:
                raise ValueError(f"severity {s} out of bounds [1, 5]")
        return values

    @field_validator("rotation_perturbations_deg", "translation_perturbations_m")
    @classmethod
    def validate_non_negative_floats(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("perturbation list must not be empty")
        for v in values:
            if v < 0.0:
                raise ValueError(f"perturbation value {v} must be non-negative")
        return values


class CameraHealthConfig(BaseModel):
    """Thresholds and parameters for fisheye camera health assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    soiling_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    blur_threshold: float = Field(default=100.0, gt=0.0)
    min_brightness: float = Field(default=15.0, ge=0.0, le=255.0)
    max_brightness: float = Field(default=240.0, ge=0.0, le=255.0)
    blockage_ratio_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    discount_factor: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_brightness_range(self) -> Self:
        if self.min_brightness >= self.max_brightness:
            raise ValueError(
                f"min_brightness ({self.min_brightness}) must be less than "
                f"max_brightness ({self.max_brightness})"
            )
        return self


class TrackingConfig(BaseModel):
    """Parameters for multi-camera 2D metric Kalman tracking on the BEV plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dt: float = Field(default=0.1, gt=0.0)
    max_distance: float = Field(default=2.5, gt=0.0)
    min_hits: int = Field(default=3, ge=1)
    max_age: int = Field(default=5, ge=1)
    process_noise_pos: float = Field(default=0.5, gt=0.0)
    process_noise_vel: float = Field(default=1.0, gt=0.0)
    measurement_noise: float = Field(default=0.5, gt=0.0)
    forecast_horizon_s: float = Field(default=3.0, gt=0.0)


class InferenceConfig(BaseModel):
    """Neural network perception inference and runtime settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["opencv", "onnxruntime"] = "opencv"
    device: Literal["cpu", "cuda", "directml"] = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    input_height: int = Field(default=480, gt=0, le=4096)
    input_width: int = Field(default=640, gt=0, le=4096)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, gt=0, le=64)

    @field_validator("std")
    @classmethod
    def validate_std_positive(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s <= 0.0 for s in value):
            raise ValueError("Normalization standard deviations must be strictly positive")
        return value


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
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    bev: BevConfig = Field(default_factory=BevConfig)
    occupancy: OccupancyConfig = Field(default_factory=OccupancyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    health: CameraHealthConfig = Field(default_factory=CameraHealthConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

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
    "BevConfig",
    "CameraHealthConfig",
    "ConfigurationError",
    "GeometryConfig",
    "InferenceConfig",
    "LoggingConfig",
    "OccupancyConfig",
    "PathsConfig",
    "ProjectConfig",
    "RiskConfig",
    "RobustnessConfig",
    "RuntimeConfig",
    "TrackingConfig",
    "load_config",
]
