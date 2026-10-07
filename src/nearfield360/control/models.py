"""Domain models for closed-loop parking trajectory tracking and vehicle execution."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nearfield360.planning.models import ManeuverGear, ManeuverPhase


class ExecutionStatus(StrEnum):
    """Execution status of the closed-loop parking maneuver."""

    ACTIVE = "active"
    SWITCHING_GEARS = "switching_gears"
    COMPLETED = "completed"
    EMERGENCY_STOPPED = "emergency_stopped"
    ABORTED_DEVIATION = "aborted_deviation"


class ControlCommand(BaseModel):
    """Low-level vehicle actuator control command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steering_angle_rad: float = Field(..., allow_inf_nan=False)
    steering_rate_rad_s: float = Field(default=0.0, allow_inf_nan=False)
    target_velocity: float = Field(default=0.0, allow_inf_nan=False)
    acceleration_cmd: float = Field(default=0.0, allow_inf_nan=False)
    gear: ManeuverGear = ManeuverGear.FORWARD
    emergency_brake: bool = False


class TrackingErrorState(BaseModel):
    """Instantaneous trajectory tracking error metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cross_track_error_m: float = Field(..., allow_inf_nan=False)
    heading_error_rad: float = Field(..., allow_inf_nan=False)
    longitudinal_error_m: float = Field(default=0.0, allow_inf_nan=False)
    speed_error_m_s: float = Field(default=0.0, allow_inf_nan=False)
    closest_waypoint_idx: int = Field(default=0, ge=0)


class VehicleSimState(BaseModel):
    """Dynamic state of the simulated ego vehicle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float = Field(..., allow_inf_nan=False)
    velocity: float = Field(default=0.0, allow_inf_nan=False)
    acceleration: float = Field(default=0.0, allow_inf_nan=False)
    steer_angle_rad: float = Field(default=0.0, allow_inf_nan=False)
    gear: ManeuverGear = ManeuverGear.FORWARD

    @property
    def pose(self) -> tuple[float, float, float]:
        """Return 2D pose (x, y, heading_rad)."""
        return (self.x, self.y, self.heading_rad)

    @property
    def xy(self) -> tuple[float, float]:
        """Return 2D position (x, y)."""
        return (self.x, self.y)


class ManeuverExecutionStep(BaseModel):
    """Single discrete-time simulation step of the parking maneuver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t: float = Field(..., ge=0.0, allow_inf_nan=False)
    step_index: int = Field(..., ge=0)
    segment_index: int = Field(..., ge=0)
    phase: ManeuverPhase
    vehicle_state: VehicleSimState
    command: ControlCommand
    error: TrackingErrorState
    nearest_obstacle_distance_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    status: ExecutionStatus


class ControlPerformanceKPIs(BaseModel):
    """Quantitative performance and comfort metrics for trajectory tracking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_cross_track_error_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    mean_cross_track_error_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    rmse_cross_track_error_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_heading_error_rad: float = Field(..., ge=0.0, allow_inf_nan=False)
    mean_heading_error_rad: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_lateral_accel_m_s2: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_jerk_m_s3: float = Field(..., ge=0.0, allow_inf_nan=False)
    docking_error_x_m: float = Field(..., allow_inf_nan=False)
    docking_error_y_m: float = Field(..., allow_inf_nan=False)
    docking_error_heading_rad: float = Field(..., allow_inf_nan=False)
    docking_distance_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    is_docked_successfully: bool = False


class ManeuverExecutionReport(BaseModel):
    """Comprehensive summary record of closed-loop parking maneuver execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    slot_id: str
    status: ExecutionStatus
    total_steps: int = Field(..., ge=0)
    duration_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    kpis: ControlPerformanceKPIs
    steps: list[ManeuverExecutionStep] = Field(default_factory=list)
    message: str = ""

    @field_validator("duration_s")
    @classmethod
    def validate_duration_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("duration_s must be finite")
        return v
