"""Domain models for multi-floor facility topology, vertical ramps, and elevators."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PortalType(StrEnum):
    """Categorization of inter-floor vertical transit conduits."""

    STRAIGHT_RAMP = "straight_ramp"
    SPIRAL_RAMP = "spiral_ramp"
    VEHICLE_ELEVATOR = "vehicle_elevator"


class ElevatorState(StrEnum):
    """Operational states of an automated vehicle hoist elevator."""

    IDLE = "idle"
    CALLED = "called"
    TRANSITING = "transiting"
    DOORS_OPENING = "doors_opening"
    LOADING = "loading"
    LEVELING = "leveling"
    HOISTING = "hoisting"
    UNLOADING = "unloading"
    HOLD = "hold"


class RampDirection(StrEnum):
    """Traversing orientation on an inclined vertical ramp."""

    ASCENDING = "ascending"
    DESCENDING = "descending"
    BIDIRECTIONAL = "bidirectional"


class RampTokenPriority(StrEnum):
    """Conflict resolution policy for single-lane bidirectional ramp access."""

    ASCENDING_FIRST = "ascending_first"
    DESCENDING_FIRST = "descending_first"
    FIFO = "fifo"


class FloorLevel(BaseModel):
    """A single planar level in a multi-story or underground parking garage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    floor_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    elevation_m: float = Field(allow_inf_nan=False)
    bays_count: int = Field(ge=0)
    available_bays: int = Field(ge=0)
    is_rooftop: bool = False
    is_basement: bool = False


class VerticalPortal(BaseModel):
    """Physical vertical transition conduit between two adjacent floors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    portal_id: str = Field(min_length=1)
    portal_type: PortalType
    from_floor_id: str = Field(min_length=1)
    to_floor_id: str = Field(min_length=1)
    length_m: float = Field(gt=0.0, allow_inf_nan=False)
    grade_pct: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    curvature_radius_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    is_bidirectional: bool = True
    speed_limit_mps: float = Field(default=2.5, gt=0.0, allow_inf_nan=False)


class VehiclePitchState(BaseModel):
    """Kinematic and dynamic vehicle state on an inclined or spiral ramp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(min_length=1)
    pitch_angle_deg: float = Field(allow_inf_nan=False)
    grade_pct: float = Field(allow_inf_nan=False)
    elevation_m: float = Field(allow_inf_nan=False)
    gravity_force_n: float = Field(allow_inf_nan=False)
    normal_force_n: float = Field(allow_inf_nan=False)
    traction_margin: float = Field(allow_inf_nan=False)
    breakover_clearance_m: float = Field(allow_inf_nan=False)
    is_clearance_safe: bool = True


class Waypoint3D(BaseModel):
    """3D spatial waypoint with elevation and pitch gradient."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(allow_inf_nan=False)
    heading_rad: float = Field(default=0.0, allow_inf_nan=False)
    pitch_deg: float = Field(default=0.0, allow_inf_nan=False)
    target_speed_mps: float = Field(default=2.0, ge=0.0, allow_inf_nan=False)


class ElevatorCabin(BaseModel):
    """Physical automated car hoist cabin and its real-time elevation state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    elevator_id: str = Field(min_length=1)
    current_floor_id: str = Field(min_length=1)
    target_floor_id: str | None = None
    state: ElevatorState = ElevatorState.IDLE
    current_elevation_m: float = Field(default=0.0, allow_inf_nan=False)
    max_weight_kg: float = Field(default=3500.0, gt=0.0, allow_inf_nan=False)
    occupied_vehicle_id: str | None = None
    leveling_offset_m: float = Field(default=0.0, allow_inf_nan=False)


class RampToken(BaseModel):
    """Exclusive access token granted to a vehicle traversing a single-lane ramp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_id: str = Field(min_length=1)
    portal_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    direction: RampDirection
    granted_at_s: float = Field(ge=0.0, allow_inf_nan=False)
    expires_at_s: float = Field(ge=0.0, allow_inf_nan=False)
    is_active: bool = True


class MultiFloorRoute(BaseModel):
    """End-to-end topological route traversing multiple floor levels and portals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    origin_floor_id: str = Field(min_length=1)
    destination_floor_id: str = Field(min_length=1)
    portals_sequence: list[str] = Field(default_factory=list)
    waypoints: list[Waypoint3D] = Field(default_factory=list)
    total_distance_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    estimated_transit_time_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class MultiFloorFacilityReport(BaseModel):
    """Aggregated structural and operational telemetry for a multi-level facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    facility_name: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    floors_count: int = Field(ge=1)
    total_bays: int = Field(ge=0)
    available_bays: int = Field(ge=0)
    active_vehicles_count: int = Field(default=0, ge=0)
    ramp_token_events_count: int = Field(default=0, ge=0)
    elevator_cycles_count: int = Field(default=0, ge=0)
    average_vertical_transit_time_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    floors: list[FloorLevel] = Field(default_factory=list)
    portals: list[VerticalPortal] = Field(default_factory=list)
    elevators: list[ElevatorCabin] = Field(default_factory=list)

