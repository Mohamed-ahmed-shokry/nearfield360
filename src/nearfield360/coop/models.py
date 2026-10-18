"""Pydantic domain models and ETSI CPM protocol schemas for cooperative perception."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RPUSensorType(StrEnum):
    """Sensing modality deployed at a roadside perception unit station."""

    LIDAR = "lidar"
    FISHEYE_CAMERA = "fisheye_camera"
    RADAR = "radar"
    FUSED_STATION = "fused_station"


class CPMStationType(StrEnum):
    """ETSI ITS station type transmitting or receiving perception messages."""

    INFRASTRUCTURE_RPU = "infrastructure_rpu"
    VEHICLE_ONBOARD = "vehicle_onboard"
    SPECIAL_EQUIPMENT = "special_equipment"


class CoopTrackSource(StrEnum):
    """Origin of a perceptual track within the facility fusion supervisor."""

    ONBOARD_ONLY = "onboard_only"
    INFRASTRUCTURE_ONLY = "infrastructure_only"
    FUSED_COOPERATIVE = "fused_cooperative"


class BlindSpotRiskLevel(StrEnum):
    """Categorized collision risk for an occluded hazard approaching conflict."""

    SAFE = "safe"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL_BRAKE = "critical_brake"


class PerceivedObjectClass(StrEnum):
    """Classification of an object perceived by infrastructure or onboard sensors."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    SHOPPING_CART = "shopping_cart"
    STATIC_OBSTACLE = "static_obstacle"
    UNKNOWN = "unknown"


class OcclusionState(StrEnum):
    """Line-of-sight visibility of an object from a given observer vantage."""

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    UNKNOWN = "unknown"


class WarningAction(StrEnum):
    """Action recommended to automated vehicle executive upon blind-spot hazard detection."""

    MAINTAIN_SPEED = "maintain_speed"
    DECELERATE = "decelerate"
    STOP_AND_YIELD = "stop_and_yield"
    EMERGENCY_BRAKE = "emergency_brake"


class RPUSensorState(BaseModel):
    """Configuration and telemetry for a stationary Roadside Perception Unit (RPU)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str = Field(min_length=1)
    sensor_type: RPUSensorType
    position_m: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    range_max_m: float = Field(default=35.0, gt=0.0, le=200.0, allow_inf_nan=False)
    fov_azimuth_deg: float = Field(default=120.0, gt=0.0, le=360.0, allow_inf_nan=False)
    is_operational: bool = True


class CPMPerceivedObject(BaseModel):
    """Perceived object telemetry standardized in ETSI TR 103 562 / SAE J3224 CPM containers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: str = Field(min_length=1)
    station_id: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    category: PerceivedObjectClass = PerceivedObjectClass.UNKNOWN
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(default=0.0, allow_inf_nan=False)
    vx_mps: float = Field(default=0.0, allow_inf_nan=False)
    vy_mps: float = Field(default=0.0, allow_inf_nan=False)
    length_m: float = Field(default=4.5, gt=0.0, le=25.0, allow_inf_nan=False)
    width_m: float = Field(default=2.0, gt=0.0, le=10.0, allow_inf_nan=False)
    height_m: float = Field(default=1.6, gt=0.0, le=6.0, allow_inf_nan=False)
    cov_xx: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    cov_yy: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    cov_xy: float = Field(default=0.0, allow_inf_nan=False)
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, allow_inf_nan=False)
    occlusion_state: OcclusionState = OcclusionState.VISIBLE


class CollectivePerceptionMessage(BaseModel):
    """ETSI Collective Perception Message (CPM) broadcast envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str = Field(min_length=1)
    station_type: CPMStationType = CPMStationType.INFRASTRUCTURE_RPU
    sequence_number: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    reference_x_m: float = Field(default=0.0, allow_inf_nan=False)
    reference_y_m: float = Field(default=0.0, allow_inf_nan=False)
    reference_z_m: float = Field(default=0.0, allow_inf_nan=False)
    perceived_objects: list[CPMPerceivedObject] = Field(default_factory=list)


class CooperativeTrack(BaseModel):
    """Fused track maintaining spatial state from onboard and/or infrastructure sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str = Field(min_length=1)
    source: CoopTrackSource
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    vx_mps: float = Field(default=0.0, allow_inf_nan=False)
    vy_mps: float = Field(default=0.0, allow_inf_nan=False)
    cov_xx: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    cov_yy: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    cov_xy: float = Field(default=0.0, allow_inf_nan=False)
    category: PerceivedObjectClass = PerceivedObjectClass.UNKNOWN
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, allow_inf_nan=False)
    age_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    hits: int = Field(default=1, ge=1)
    onboard_track_id: str | None = None
    rpu_station_id: str | None = None
    rpu_object_id: str | None = None
