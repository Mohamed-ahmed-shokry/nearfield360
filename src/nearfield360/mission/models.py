"""Domain data models for Autonomous Valet Parking (AVP) mission management."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nearfield360.control.models import ControlPerformanceKPIs
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.slots.models import ParkingSlot


class MissionState(StrEnum):
    """Lifecycle states for Autonomous Valet Parking (AVP) execution."""

    STANDBY = "standby"
    SEARCHING = "searching"
    SLOT_SELECTED = "slot_selected"
    APPROACH = "approach"
    PARKING_MANEUVER = "parking_maneuver"
    OBSTACLE_HOLD = "obstacle_hold"
    REPLANNING = "replanning"
    FINAL_ALIGNMENT = "final_alignment"
    COMPLETED = "completed"
    ABORTED = "aborted"


class MissionTrigger(StrEnum):
    """Transition events that advance the mission lifecycle state machine."""

    ACTIVATE = "activate"
    SLOT_DISCOVERED = "slot_discovered"
    TARGET_LOCKED = "target_locked"
    APPROACH_START = "approach_start"
    APPROACH_COMPLETE = "approach_complete"
    MANEUVER_START = "maneuver_start"
    OBSTACLE_DETECTED = "obstacle_detected"
    OBSTACLE_CLEARED = "obstacle_cleared"
    OBSTACLE_TIMEOUT = "obstacle_timeout"
    REPLAN_SUCCESS = "replan_success"
    REPLAN_FAILED = "replan_failed"
    TARGET_REACHED = "target_reached"
    ALIGNMENT_COMPLETE = "alignment_complete"
    ABORT = "abort"


class MissionEvent(BaseModel):
    """Recorded lifecycle state transition event with timestamp and rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t: float = Field(..., ge=0.0, allow_inf_nan=False)
    source_state: MissionState
    target_state: MissionState
    trigger: MissionTrigger
    description: str


class TrackedParkingSlot(BaseModel):
    """A parking slot tracked across multi-frame sequences with temporal filtering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str
    slot: ParkingSlot
    first_observed_frame: int = Field(default=0, ge=0)
    last_observed_frame: int = Field(default=0, ge=0)
    hit_count: int = Field(default=1, ge=1)
    miss_count: int = Field(default=0, ge=0)
    stability_score: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    is_confirmed: bool = Field(default=False)


class RecoveryManeuver(BaseModel):
    """Dynamic recovery maneuver generated when an obstacle blocks execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    replan_id: str
    trigger_reason: str
    start_pose: tuple[float, float, float]
    target_slot_id: str
    is_successful: bool
    plan: ParkingTrajectoryPlan | None = None
    clearance_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @field_validator("start_pose")
    @classmethod
    def validate_pose(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or not all(math.isfinite(v) for v in value):
            raise ValueError("start_pose must be a 3-tuple of finite floats (x, y, heading_rad)")
        return value


class MissionSummaryReport(BaseModel):
    """Comprehensive summary artifact reporting mission outcomes and KPIs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mission_id: str
    final_state: MissionState
    target_slot_id: str | None = None
    total_duration_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_steps: int = Field(..., ge=0)
    replan_count: int = Field(default=0, ge=0)
    events: list[MissionEvent] = Field(default_factory=list)
    final_kpis: ControlPerformanceKPIs | None = None
    is_success: bool
    message: str
