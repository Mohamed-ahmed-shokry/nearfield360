"""Domain data models for EV charging management, dynamic grid power, and bay scheduling."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChargerType(StrEnum):
    """Classification of charging hardware capability and connector type."""

    AC_LEVEL_2 = "ac_level_2"
    DC_FAST_50KW = "dc_fast_50kw"
    DC_FAST_150KW = "dc_fast_150kw"
    ROBOTIC_UNDERBODY_PDU = "robotic_underbody_pdu"


class ChargerPortStatus(StrEnum):
    """Operational status of an individual charging port or station bay."""

    AVAILABLE = "available"
    RESERVED = "reserved"
    OCCUPIED_CHARGING = "occupied_charging"
    OCCUPIED_COMPLETE_WAITING_RELOCATION = "occupied_complete_waiting_relocation"
    OUT_OF_SERVICE = "out_of_service"


class ChargingSessionStatus(StrEnum):
    """Lifecycle state of an EV charging session."""

    PENDING_DOCKING = "pending_docking"
    ACTIVE_CHARGING = "active_charging"
    TOPPING_OFF = "topping_off"
    COMPLETED = "completed"
    IDLE_OVERSTAY = "idle_overstay"
    ABORTED = "aborted"


class ChargerBay(BaseModel):
    """Specification and real-time state of an automated or robotic charging station bay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bay_id: str
    slot_id: str
    charger_type: ChargerType = ChargerType.DC_FAST_50KW
    max_power_rating_kw: float = Field(default=50.0, gt=0.0, allow_inf_nan=False)
    current_power_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    status: ChargerPortStatus = ChargerPortStatus.AVAILABLE
    efficiency_rating: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    docking_dwell_s: float = Field(default=15.0, gt=0.0, allow_inf_nan=False)
    occupied_by_vehicle_id: str | None = None

    @field_validator("current_power_kw")
    @classmethod
    def validate_current_power(cls, value: float, info: Any) -> float:
        # Pydantic v2 field validator
        return value


class EVBatteryState(BaseModel):
    """Electric vehicle battery telematics and energy state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str
    battery_capacity_kwh: float = Field(default=75.0, gt=0.0, allow_inf_nan=False)
    current_soc: float = Field(default=0.30, ge=0.0, le=1.0, allow_inf_nan=False)
    target_soc: float = Field(default=0.80, gt=0.0, le=1.0, allow_inf_nan=False)
    max_charge_acceptance_kw: float = Field(default=100.0, gt=0.0, allow_inf_nan=False)
    degradation_factor: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)

    @property
    def current_energy_kwh(self) -> float:
        """Calculate current energy stored in kilowatt-hours."""
        return self.battery_capacity_kwh * self.current_soc * self.degradation_factor

    @property
    def energy_needed_kwh(self) -> float:
        """Calculate remaining energy needed to reach target state of charge."""
        if self.current_soc >= self.target_soc:
            return 0.0
        return (
            self.battery_capacity_kwh
            * (self.target_soc - self.current_soc)
            * self.degradation_factor
        )

    @property
    def is_critical_low(self) -> bool:
        """Flag whether battery state-of-charge is critically depleted."""
        return self.current_soc < 0.20


class ChargingSession(BaseModel):
    """Managed EV charging session records and progress tracking."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    session_id: str
    vehicle_id: str
    bay_id: str
    start_time_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    last_update_time_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    completion_time_s: float | None = None
    start_soc: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    current_soc: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    target_soc: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    allocated_power_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    energy_delivered_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    peak_power_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    status: ChargingSessionStatus = Field(default=ChargingSessionStatus.PENDING_DOCKING)
    estimated_time_to_target_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    idle_dwell_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize charging session record to formatted JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class RelocationOrder(BaseModel):
    """Relocation dispatch order issued when charging is complete or to vacate a bay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str
    vehicle_id: str
    source_bay_id: str
    target_slot_id: str
    created_at_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    reason: str = Field(default="target_soc_reached")
    urgency_priority: int = Field(default=1, ge=1, le=10)


class FacilityPowerSummary(BaseModel):
    """Real-time aggregated power grid balance across all charging bays."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str
    grid_capacity_kw: float = Field(..., gt=0.0, allow_inf_nan=False)
    total_requested_power_kw: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_allocated_power_kw: float = Field(..., ge=0.0, allow_inf_nan=False)
    available_headroom_kw: float = Field(..., ge=0.0, allow_inf_nan=False)
    active_charging_sessions: int = Field(default=0, ge=0)
    throttled_sessions: int = Field(default=0, ge=0)
    grid_utilization_ratio: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    is_curtailed: bool = False


class ChargingDispatchReport(BaseModel):
    """Comprehensive facility EV charging dispatch, grid load, and relocation telemetry report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    power_summary: FacilityPowerSummary
    total_bays: int = Field(..., ge=0)
    available_bays: int = Field(..., ge=0)
    occupied_charging_bays: int = Field(..., ge=0)
    waiting_relocation_bays: int = Field(..., ge=0)
    active_sessions: list[ChargingSession] = Field(default_factory=list)
    pending_queue_count: int = Field(default=0, ge=0)
    relocation_orders: list[RelocationOrder] = Field(default_factory=list)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)
