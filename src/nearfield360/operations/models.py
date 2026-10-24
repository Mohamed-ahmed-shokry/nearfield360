"""Domain data models for the facility operations control tower."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_VEHICLE_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


class ServiceLevel(StrEnum):
    """Operational service level experienced by a completed vehicle journey."""

    NOMINAL = "nominal"
    DEGRADED_DISPATCH = "degraded_dispatch"
    RELOCATED = "relocated"
    CURBSIDE_HANDOVER = "curbside_handover"
    BLOCKED = "blocked"


class JourneyStage(StrEnum):
    """Lifecycle stage of a vehicle inside the facility day-run."""

    ARRIVED = "arrived"
    DISPATCHED = "dispatched"
    PARKED = "parked"
    CHARGING_WINDOW = "charging_window"
    UNPARKING = "unparking"
    SETTLED = "settled"
    DEPARTED = "departed"


class VehicleArrival(BaseModel):
    """A single customer vehicle arrival event captured by facility gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(..., pattern=_VEHICLE_ID_PATTERN)
    arrival_time_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    plate_confidence: float = Field(default=0.92, ge=0.0, le=1.0, allow_inf_nan=False)
    requested_dwell_hours: float = Field(default=3.0, gt=0.0, le=24.0, allow_inf_nan=False)
    requests_charging: bool = False

    @field_validator("vehicle_id")
    @classmethod
    def vehicle_id_not_blank(cls, value: str) -> str:
        """Reject empty vehicle identifiers."""
        if not value.strip():
            raise ValueError("vehicle_id must not be blank")
        return value


class VehicleJourneyRecord(BaseModel):
    """End-to-end trace of one vehicle through the coordinated day-run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(..., pattern=_VEHICLE_ID_PATTERN)
    arrival_time_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    requested_dwell_hours: float = Field(..., gt=0.0, le=24.0, allow_inf_nan=False)
    stage: JourneyStage = JourneyStage.ARRIVED
    service_level: ServiceLevel = ServiceLevel.NOMINAL
    dispatched_slot_id: str | None = None
    dispatch_wait_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    parked_time_s: float | None = None
    unpark_time_s: float | None = None
    departed_time_s: float | None = None
    charging_windows: int = Field(default=0, ge=0)
    energy_charged_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    energy_exported_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    parking_fee_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    energy_fee_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class HourlySnapshot(BaseModel):
    """Facility activity captured at the end of one simulated hour."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hour: int = Field(..., ge=0)
    arrivals: int = Field(..., ge=0)
    departures: int = Field(..., ge=0)
    blocked: int = Field(..., ge=0)
    occupied_slots: int = Field(..., ge=0)
    queue_length: int = Field(..., ge=0)
    charging_vehicles: int = Field(..., ge=0)
    facility_load_kw: float = Field(..., ge=0.0, allow_inf_nan=False)
    export_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    revenue_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
