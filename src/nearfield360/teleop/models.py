"""Pydantic domain models for remote teleoperation assistance, heartbeats, and MRM."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TeleopMode(StrEnum):
    """Operational mode of the remote teleoperation assistance interface."""

    IDLE_AUTONOMOUS = "idle_autonomous"
    TELE_ASSISTANCE = "tele_assistance"
    TELE_GUIDANCE = "tele_guidance"
    DIRECT_DRIVING = "direct_driving"
    MRM_SAFE_STOP = "mrm_safe_stop"


class MRMStage(StrEnum):
    """Deterministic stages of a Minimum Risk Maneuver execution."""

    INACTIVE = "inactive"
    DECELERATING = "decelerating"
    PULLING_TO_SHOULDER = "pulling_to_shoulder"
    STANDSTILL_LOCKED = "standstill_locked"
    EMERGENCY_BEACON = "emergency_beacon"


class NetworkQuality(StrEnum):
    """Categorized health of the bidirectional vehicle-to-operator communication link."""

    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    SEVERED = "severed"


class TeleopWaypoint(BaseModel):
    """2D guidance waypoint provided by the remote operator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    target_speed_mps: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)


class TeleopCommand(BaseModel):
    """Actuation or guidance command transmitted from the remote operator desk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    mode: TeleopMode = TeleopMode.TELE_GUIDANCE
    steering_angle_rad: float = Field(default=0.0, allow_inf_nan=False)
    target_velocity_mps: float = Field(default=0.0, allow_inf_nan=False)
    emergency_stop: bool = False
    suggested_waypoints: list[TeleopWaypoint] = Field(default_factory=list)


class TeleopHeartbeat(BaseModel):
    """Cyclic telemetry beacon between vehicle and remote operations center."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    rtt_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    network_quality: NetworkQuality = NetworkQuality.OPTIMAL
    battery_soc: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    current_speed_mps: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    active_mrm: bool = False


class TeleopSession(BaseModel):
    """Active or archived teleoperation engagement between an operator and vehicle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)
    active_mode: TeleopMode = TeleopMode.IDLE_AUTONOMOUS
    started_at_s: float = Field(ge=0.0, allow_inf_nan=False)
    ended_at_s: float | None = None
    commands_processed_count: int = Field(default=0, ge=0)
    mrm_stage: MRMStage = MRMStage.INACTIVE
    is_active: bool = True


class TeleopReport(BaseModel):
    """Facility-wide teleoperation interventions, latency statistics, and MRM audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    active_sessions_count: int = Field(default=0, ge=0)
    completed_sessions_count: int = Field(default=0, ge=0)
    total_interventions_count: int = Field(default=0, ge=0)
    average_rtt_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    mrm_executions_count: int = Field(default=0, ge=0)
    sessions: list[TeleopSession] = Field(default_factory=list)

    @field_validator("average_rtt_ms")
    @classmethod
    def round_kpis(cls, value: float) -> float:
        return round(value, 2)
