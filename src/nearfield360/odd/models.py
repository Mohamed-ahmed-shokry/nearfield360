"""Domain models, classifications, and telemetry schemas for Dynamic ODD Monitoring."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ODDState(StrEnum):
    """Hierarchical operational design domain operating tiers under ISO 21448 / ISO 23374."""

    NOMINAL = "nominal"
    ADVISORY_DEGRADED = "advisory_degraded"
    CRITICAL_RESTRICTED = "critical_restricted"
    ODD_BREACH_MRM = "odd_breach_mrm"


def odd_state_severity(state: ODDState) -> int:
    """Return integer severity rank for ODD state (0=nominal, 3=breach)."""
    severity_map = {
        ODDState.NOMINAL: 0,
        ODDState.ADVISORY_DEGRADED: 1,
        ODDState.CRITICAL_RESTRICTED: 2,
        ODDState.ODD_BREACH_MRM: 3,
    }
    return severity_map[state]


class IlluminationCondition(StrEnum):
    """Ambient lighting conditions derived from surround camera histograms."""

    DAYLIGHT = "daylight"
    TWILIGHT = "twilight"
    NIGHT = "night"
    GLARE_DAZZLE = "glare_dazzle"


class PrecipitationCondition(StrEnum):
    """Atmospheric weather conditions affecting optical propagation."""

    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG_HAZE = "fog_haze"
    SNOW = "snow"


class SurfaceCondition(StrEnum):
    """Roadway/parking pavement condition and estimated friction category."""

    DRY = "dry"
    WET = "wet"
    PUDDLE_AQUAPLANING = "puddle_aquaplaning"
    ICY_SNOW = "icy_snow"


class SensorHealthStatus(BaseModel):
    """Per-sensor health and occlusion diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensor_id: str
    is_healthy: bool = True
    occlusion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    lens_dirt_score: float = Field(default=0.0, ge=0.0, le=1.0)
    snr_db: float = Field(default=25.0, ge=0.0, le=60.0)
    timestamp_s: float = 0.0


class ODDMetrics(BaseModel):
    """Aggregated environmental and system envelope metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambient_lux: float = Field(default=500.0, ge=0.0)
    glare_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    friction_coeff: float = Field(default=0.85, ge=0.0, le=1.2)
    precipitation_rate_mmph: float = Field(default=0.0, ge=0.0)
    v2x_latency_ms: float = Field(default=45.0, ge=0.0)
    v2x_packet_loss_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    max_camera_occlusion: float = Field(default=0.0, ge=0.0, le=1.0)


class ODDEvaluationResult(BaseModel):
    """Instantaneous evaluation outcome across all ODD dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_state: ODDState = ODDState.NOMINAL
    prior_state: ODDState = ODDState.NOMINAL
    illumination: IlluminationCondition = IlluminationCondition.DAYLIGHT
    precipitation: PrecipitationCondition = PrecipitationCondition.CLEAR
    surface: SurfaceCondition = SurfaceCondition.DRY
    active_limiters: list[str] = Field(default_factory=list)
    governor_speed_limit_mps: float = 4.5
    safety_headway_multiplier: float = 1.0
    mrm_required: bool = False
    recommendations: list[str] = Field(default_factory=list)
    metrics: ODDMetrics = Field(default_factory=ODDMetrics)
    timestamp_s: float = 0.0


class ODDReport(BaseModel):
    """Comprehensive ODD telemetry and audit report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str
    vehicle_id: str = "AVP_EGO_01"
    timestamp_iso: str
    latest_evaluation: ODDEvaluationResult
    sensor_statuses: dict[str, SensorHealthStatus] = Field(default_factory=dict)
    history: list[ODDEvaluationResult] = Field(default_factory=list)
    total_samples: int = 0
    mrm_triggered_count: int = 0
