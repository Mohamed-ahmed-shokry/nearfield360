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
