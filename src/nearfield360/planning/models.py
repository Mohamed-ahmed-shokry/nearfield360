"""Domain models for autonomous parking trajectory planning and maneuver generation."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nearfield360.slots.models import ParkingSlotType


class ManeuverGear(StrEnum):
    """Transmission driving gear direction for a trajectory maneuver."""

    FORWARD = "forward"
    REVERSE = "reverse"


class ManeuverPhase(StrEnum):
    """Functional phase of a multi-stage parking maneuver."""

    APPROACH = "approach"
    STEER_IN = "steer_in"
    ALIGN = "align"
    DOCK = "dock"
    FINAL_ALIGN = "final_align"


class PlanStatus(StrEnum):
    """Outcome status of parking motion trajectory planning."""

    SUCCESS = "success"
    NO_VACANT_SLOT = "no_vacant_slot"
    COLLISION_DETECTED = "collision_detected"
    KINEMATIC_LIMIT_EXCEEDED = "kinematic_limit_exceeded"
    UNREACHABLE = "unreachable"


class TrajectoryWaypoint(BaseModel):
    """A single spatial, kinematic, and temporal trajectory waypoint in the vehicle frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float = Field(..., allow_inf_nan=False)
    curvature: float = Field(default=0.0, allow_inf_nan=False)
    velocity: float = Field(default=0.0, allow_inf_nan=False)
    acceleration: float = Field(default=0.0, allow_inf_nan=False)
    gear: ManeuverGear = ManeuverGear.FORWARD
    t: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    distance_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @property
    def pose(self) -> tuple[float, float, float]:
        """Return 2D pose (x, y, heading_rad)."""
        return (self.x, self.y, self.heading_rad)

    @property
    def xy(self) -> tuple[float, float]:
        """Return 2D position (x, y)."""
        return (self.x, self.y)


class ManeuverSegment(BaseModel):
    """A contiguous trajectory segment executed in a single gear and maneuver phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_index: int = Field(..., ge=0)
    phase: ManeuverPhase
    gear: ManeuverGear
    length_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    duration_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    waypoints: list[TrajectoryWaypoint] = Field(default_factory=list)


class ParkingTrajectoryPlan(BaseModel):
    """A complete multi-stage collision-free parking trajectory plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    slot_id: str
    slot_type: ParkingSlotType
    status: PlanStatus
    total_length_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_duration_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    gear_switches: int = Field(..., ge=0)
    max_curvature: float = Field(..., ge=0.0, allow_inf_nan=False)
    min_clearance_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    start_pose: tuple[float, float, float]
    target_pose: tuple[float, float, float]
    is_executable: bool
    segments: list[ManeuverSegment] = Field(default_factory=list)

    @field_validator("start_pose", "target_pose")
    @classmethod
    def validate_pose(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("pose must be a 3-tuple of (x, y, heading_rad)")
        if not (math.isfinite(value[0]) and math.isfinite(value[1]) and math.isfinite(value[2])):
            raise ValueError("pose elements must be finite numbers")
        return value

    @property
    def all_waypoints(self) -> list[TrajectoryWaypoint]:
        """Return a flattened list of all waypoints across segments."""
        pts: list[TrajectoryWaypoint] = []
        for seg in self.segments:
            pts.extend(seg.waypoints)
        return pts


class ParkingPlanReport(BaseModel):
    """Summary record of parking motion planning execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_slot_id: str | None = None
    candidate_slots_evaluated: int = Field(default=0, ge=0)
    status: PlanStatus
    total_length_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    total_duration_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    gear_switches: int = Field(default=0, ge=0)
    min_clearance_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    is_executable: bool = False
    plan: ParkingTrajectoryPlan | None = None
