"""Pydantic domain models for emergency preemption, evacuation, and life safety."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EmergencyType(StrEnum):
    """Categorized type of facility emergency event."""

    FIRE_ALARM = "fire_alarm"
    EV_THERMAL_RUNAWAY = "ev_thermal_runaway"
    GAS_LEAK = "gas_leak"
    STRUCTURAL_HAZARD = "structural_hazard"
    EMERGENCY_VEHICLE_APPROACH = "emergency_vehicle_approach"


class EmergencySeverity(StrEnum):
    """Incident escalation severity level."""

    ADVISORY = "advisory"
    EVP_PREEMPTION = "evp_preemption"
    PARTIAL_EVACUATION = "partial_evacuation"
    FULL_FACILITY_EVACUATION = "full_facility_evacuation"


class VehicleEvacuationStatus(StrEnum):
    """Evacuation and safety state of a vehicle in the facility."""

    STATIONARY_SAFE = "stationary_safe"
    HOLD_YIELDING = "hold_yielding"
    EVACUATING = "evacuating"
    EVACUATED = "evacuated"
    TRAPPED_IN_HAZARD = "trapped_in_hazard"


class FireDoorStatus(StrEnum):
    """Operational status of automatic facility fire barrier doors / rolling shutters."""

    OPEN = "open"
    CLOSING = "closing"
    LOCKED_SHUT = "locked_shut"


class ResponderType(StrEnum):
    """Classification of first responder emergency vehicle."""

    FIRE_ENGINE = "fire_engine"
    AMBULANCE = "ambulance"
    POLICE = "police"


class HazardZone(BaseModel):
    """Dynamic geofenced danger zone within the facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hazard_id: str = Field(min_length=1)
    emergency_type: EmergencyType
    severity: EmergencySeverity
    center_x: float = Field(allow_inf_nan=False)
    center_y: float = Field(allow_inf_nan=False)
    center_z: float = Field(default=0.0, allow_inf_nan=False)
    radius_m: float = Field(gt=0.0, allow_inf_nan=False)
    floor_id: str = Field(default="L1", min_length=1)
    active: bool = True
    reported_at_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    description: str = Field(default="", max_length=256)
