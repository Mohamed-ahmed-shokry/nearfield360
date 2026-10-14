"""Domain data models for curbside drop-off/pick-up staging, dwell, and safety interlocks."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HandoverType(StrEnum):
    """Classification of handover operation at the curbside staging zone."""

    DROPOFF = "dropoff"
    PICKUP = "pickup"


class BayStatus(StrEnum):
    """Operational occupancy and readiness state of a curbside staging bay."""

    VACANT = "vacant"
    OCCUPIED_UNLOADING = "occupied_unloading"
    OCCUPIED_BOARDING = "occupied_boarding"
    HANDOVER_READY = "handover_ready"
    CLEARING = "clearing"


class SafetyInterlockStatus(StrEnum):
    """State of the motion inhibitor safety interlock."""

    ENGAGED_DOOR_OPEN = "engaged_door_open"
    ENGAGED_PEDESTRIAN_PROXIMITY = "engaged_pedestrian_proximity"
    ENGAGED_TRUNK_OPEN = "engaged_trunk_open"
    DISENGAGED_CLEAR = "disengaged_clear"


class PedestrianDetection(BaseModel):
    """Proximity perception observation of a pedestrian near the vehicle envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pedestrian_id: str = Field(min_length=1)
    distance_m: float = Field(ge=0.0, allow_inf_nan=False)
    relative_bearing_rad: float = Field(default=0.0, allow_inf_nan=False)
    in_critical_zone: bool = False


class CurbsideBay(BaseModel):
    """Dedicated curbside staging bay for vehicle-to-human handover."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bay_id: str = Field(min_length=1)
    bay_type: HandoverType = HandoverType.DROPOFF
    status: BayStatus = BayStatus.VACANT
    occupied_vehicle_id: str | None = None
    occupancy_start_s: float | None = None
    current_dwell_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    interlock_status: SafetyInterlockStatus = SafetyInterlockStatus.DISENGAGED_CLEAR
    slot_id_ref: str | None = None

    @property
    def is_occupied(self) -> bool:
        """Check whether bay is currently occupied by a vehicle."""
        return self.status != BayStatus.VACANT


class CurbsideMission(BaseModel):
    """Lifecycle record of a passenger drop-off or pick-up summon mission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mission_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    handover_type: HandoverType = HandoverType.DROPOFF
    bay_id: str = Field(min_length=1)
    requested_at_s: float = Field(ge=0.0, allow_inf_nan=False)
    completed_at_s: float | None = None
    dwell_duration_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    interlock_triggers_count: int = Field(default=0, ge=0)
    overstay_penalty_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    is_completed: bool = False


class CurbsideReport(BaseModel):
    """Comprehensive facility curbside staging, queue, and safety audit report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    total_bays_count: int = Field(ge=0)
    vacant_bays_count: int = Field(ge=0)
    bays: list[CurbsideBay] = Field(default_factory=list)
    active_missions: list[CurbsideMission] = Field(default_factory=list)
    completed_missions_count: int = Field(default=0, ge=0)
    active_interlocks_count: int = Field(default=0, ge=0)
    total_overstay_penalties_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    average_dwell_duration_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @field_validator("total_overstay_penalties_usd", "average_dwell_duration_s")
    @classmethod
    def round_kpis(cls, value: float) -> float:
        return round(value, 2)
