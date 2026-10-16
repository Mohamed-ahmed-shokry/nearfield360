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


class FireDoorInterlock(BaseModel):
    """Automatic fire barrier rolling shutter with timed closure interlock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    door_id: str = Field(min_length=1)
    floor_id: str = Field(default="L1", min_length=1)
    location_x: float = Field(allow_inf_nan=False)
    location_y: float = Field(allow_inf_nan=False)
    width_m: float = Field(default=4.0, gt=0.0, allow_inf_nan=False)
    status: FireDoorStatus = FireDoorStatus.OPEN
    time_to_close_s: float = Field(default=15.0, ge=0.0, allow_inf_nan=False)
    is_blocked: bool = False


class EmergencyVehicle(BaseModel):
    """First responder emergency apparatus entering or operating in the facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(min_length=1)
    responder_type: ResponderType = ResponderType.FIRE_ENGINE
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(default=0.0, allow_inf_nan=False)
    yaw_rad: float = Field(default=0.0, allow_inf_nan=False)
    velocity_mps: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    target_x: float = Field(default=0.0, allow_inf_nan=False)
    target_y: float = Field(default=0.0, allow_inf_nan=False)
    active_siren: bool = True


class EvacuationWaypoint(BaseModel):
    """Spatiotemporal waypoint on an emergency egress path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(default=0.0, allow_inf_nan=False)
    speed_limit_mps: float = Field(default=4.0, gt=0.0, allow_inf_nan=False)


class EvacuationRoute(BaseModel):
    """Calculated emergency egress trajectory from a bay to an exit portal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    origin_bay_id: str = Field(min_length=1)
    exit_portal_id: str = Field(min_length=1)
    waypoints: list[EvacuationWaypoint] = Field(min_length=1)
    total_length_m: float = Field(ge=0.0, allow_inf_nan=False)
    estimated_transit_time_s: float = Field(ge=0.0, allow_inf_nan=False)
    min_hazard_distance_m: float = Field(default=999.0, ge=0.0, allow_inf_nan=False)
    is_cleared: bool = True


class EvacuationWave(BaseModel):
    """Batched wave of departing vehicles to avoid exit portal congestion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wave_index: int = Field(ge=0)
    vehicle_ids: list[str] = Field(min_length=1)
    dispatch_time_s: float = Field(ge=0.0, allow_inf_nan=False)
    exit_portal_id: str = Field(min_length=1)


class LifeSafetyReport(BaseModel):
    """Comprehensive facility life safety and tactical situational report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_name: str = Field(default="Main Garage", min_length=1)
    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    emergency_active: bool = False
    incident_type: EmergencyType = EmergencyType.FIRE_ALARM
    severity: EmergencySeverity = EmergencySeverity.ADVISORY
    active_hazards: list[HazardZone] = Field(default_factory=list)
    fire_doors: list[FireDoorInterlock] = Field(default_factory=list)
    emergency_responders: list[EmergencyVehicle] = Field(default_factory=list)
    total_vehicles: int = Field(default=0, ge=0)
    evacuated_count: int = Field(default=0, ge=0)
    evacuating_count: int = Field(default=0, ge=0)
    yielding_count: int = Field(default=0, ge=0)
    trapped_count: int = Field(default=0, ge=0)
    evp_active: bool = False

