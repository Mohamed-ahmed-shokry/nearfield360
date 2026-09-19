"""Camera health, lens soiling, and degradation monitoring models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CameraHealthStatus(StrEnum):
    """Discrete operational status of an automotive fisheye camera."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class HealthAnomaly(StrEnum):
    """Specific sensor or optical anomalies detected in camera imagery."""

    LENS_SOILING = "lens_soiling"
    BLUR = "blur"
    UNDEREXPOSURE = "underexposure"
    OVEREXPOSURE = "overexposure"
    BLOCKAGE = "blockage"


class CameraHealthMetrics(BaseModel):
    """Quantitative photometric and sharpness metrics for camera health."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    soiling_score: float = Field(
        ge=0.0, le=1.0, description="Estimated fraction of soiled lens surface."
    )
    blur_score: float = Field(ge=0.0, description="Spatial Laplacian variance (sharpness metric).")
    mean_brightness: float = Field(ge=0.0, le=255.0, description="Mean luminance across image.")
    contrast: float = Field(ge=0.0, description="Standard deviation of luminance.")
    blockage_ratio: float = Field(
        ge=0.0, le=1.0, description="Fraction of spatial blocks lacking valid optical structure."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall health confidence multiplier for perception fusion."
    )


class CameraHealthReport(BaseModel):
    """Comprehensive health assessment report for a single camera frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera: str = Field(description="Camera identifier (e.g. FV, RV, MVL, MVR).")
    status: CameraHealthStatus = Field(description="Overall camera operational status.")
    anomalies: tuple[HealthAnomaly, ...] = Field(
        default=(), description="Detected optical or sensor anomalies."
    )
    metrics: CameraHealthMetrics = Field(description="Quantitative health metrics.")
    discount_weight: float = Field(
        ge=0.0, le=1.0, description="Recommended evidence weight multiplier for BEV fusion."
    )


__all__ = [
    "CameraHealthMetrics",
    "CameraHealthReport",
    "CameraHealthStatus",
    "HealthAnomaly",
]
