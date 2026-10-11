"""Domain data models for V2X communications, cloud fleet coordination, and conflict management."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nearfield360.mapping.models import PoseEstimate


class V2XMessageType(StrEnum):
    """Enumeration of standardized V2I/V2V communication message types."""

    HEARTBEAT = "heartbeat"
    DISPATCH_ASSIGNMENT = "dispatch_assignment"
    DISPATCH_ACK = "dispatch_ack"
    OCCUPANCY_STATE = "occupancy_state"
    YIELD_REQUEST = "yield_request"
    YIELD_GRANT = "yield_grant"
    CONFLICT_ADVISORY = "conflict_advisory"


class VehicleFleetStatus(StrEnum):
    """Operational status of an individual vehicle within the facility fleet."""

    UNASSIGNED = "unassigned"
    EN_ROUTE_SLOT = "en_route_slot"
    PARKING_MANEUVER = "parking_maneuver"
    PARKED = "parked"
    UNPARKING_MANEUVER = "unparking_maneuver"
    EN_ROUTE_EXIT = "en_route_exit"
    YIELDING = "yielding"
    ERROR = "error"


class ConflictResolutionAction(StrEnum):
    """Cooperative action commanded to resolve a spatiotemporal trajectory conflict."""

    PROCEED = "proceed"
    SLOW_DOWN = "slow_down"
    YIELD_HOLD = "yield_hold"
    REROUTE = "reroute"


class V2XMessage(BaseModel):
    """Envelope containing standard V2X protocol header metadata and payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(default="1.0")
    message_type: V2XMessageType
    sender_id: str
    recipient_id: str = Field(default="cloud_dispatcher")
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    payload: dict[str, Any] = Field(default_factory=dict)


class VehicleHeartbeat(BaseModel):
    """Telematics heartbeat transmitted periodically by each autonomous vehicle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    pose: PoseEstimate
    speed_mps: float = Field(..., ge=0.0, allow_inf_nan=False)
    steering_angle_rad: float = Field(default=0.0, allow_inf_nan=False)
    status: VehicleFleetStatus = Field(default=VehicleFleetStatus.UNASSIGNED)
    target_slot_id: str | None = None
    battery_pct: float = Field(default=100.0, ge=0.0, le=100.0, allow_inf_nan=False)


class DispatchAssignment(BaseModel):
    """Target bay assignment and navigation plan issued by the cloud facility manager."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    vehicle_id: str
    assigned_slot_id: str
    route_waypoint_ids: list[str] = Field(default_factory=list)
    issued_at_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    lease_expires_at_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    target_arrival_s: float | None = None
    priority_level: int = Field(default=1, ge=1, le=10)


class SpatiotemporalWaypoint(BaseModel):
    """4D space-time trajectory waypoint defining coordinate, heading, and arrival window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float = Field(default=0.0, allow_inf_nan=False)
    t_expected_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    t_min_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    t_max_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    speed_target_mps: float = Field(default=1.5, ge=0.0, allow_inf_nan=False)

    @field_validator("t_max_s")
    @classmethod
    def validate_time_window(cls, v: float, info: Any) -> float:
        t_min = info.data.get("t_min_s", 0.0)
        if v < t_min:
            raise ValueError(f"t_max_s ({v}) cannot be earlier than t_min_s ({t_min})")
        return v


class TrajectoryProfile(BaseModel):
    """Predicted 4D spatiotemporal path submitted by a vehicle for deconfliction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str
    waypoints: list[SpatiotemporalWaypoint] = Field(default_factory=list)
    total_duration_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    safety_margin_m: float = Field(default=1.2, gt=0.0, allow_inf_nan=False)


class ConflictPoint(BaseModel):
    """Detected spatiotemporal intersection between two or more vehicle trajectories."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    conflict_id: str
    primary_vehicle_id: str
    secondary_vehicle_id: str
    conflict_location: tuple[float, float]
    earliest_arrival_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    latest_departure_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    distance_between_trajectories_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    action_primary: ConflictResolutionAction = Field(default=ConflictResolutionAction.PROCEED)
    action_secondary: ConflictResolutionAction = Field(default=ConflictResolutionAction.YIELD_HOLD)
    resolved: bool = False


class FacilityOccupancySummary(BaseModel):
    """Real-time facility bay availability breakdown transmitted over V2I."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_slots: int = Field(..., ge=0)
    vacant_slots: int = Field(..., ge=0)
    reserved_slots: int = Field(..., ge=0)
    occupied_slots: int = Field(..., ge=0)
    blocked_slots: int = Field(..., ge=0)
    slot_allocations: dict[str, str] = Field(default_factory=dict)  # slot_id -> vehicle_id


class FleetDispatchReport(BaseModel):
    """Comprehensive facility fleet coordination telemetry summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_vehicles_active: int = Field(..., ge=0)
    occupancy_summary: FacilityOccupancySummary
    active_assignments: list[DispatchAssignment] = Field(default_factory=list)
    active_conflicts: list[ConflictPoint] = Field(default_factory=list)
    resolved_conflicts_count: int = Field(default=0, ge=0)
    average_wait_time_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def to_json(self, indent: int = 2) -> str:
        """Export report to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> FleetDispatchReport:
        """Deserialize report from JSON string."""
        return cls.model_validate(json.loads(json_str))
