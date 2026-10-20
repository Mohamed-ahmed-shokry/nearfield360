"""Pydantic domain models and telemetry protocol schemas for Vehicle-to-Grid (V2G) systems."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class V2GPowerMode(StrEnum):
    """Operational power flow mode for a bidirectional electric vehicle connection."""

    STANDBY = "standby"
    CHARGE = "charge"
    DISCHARGE_PEAK_SHAVE = "discharge_peak_shave"
    DISCHARGE_FREQUENCY_SUPPORT = "discharge_frequency_support"
    ISLANDED_BACKUP = "islanded_backup"


class InverterPhase(StrEnum):
    """Grid coupling phase topology for smart bidirectional inverters."""

    SINGLE_PHASE = "single_phase"
    THREE_PHASE = "three_phase"


class GridTariffWindow(BaseModel):
    """Time-of-Use (ToU) electricity pricing and emissions intensity schedule window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str = Field(min_length=1)
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)
    import_price_per_kwh: float = Field(ge=0.0, allow_inf_nan=False)
    export_price_per_kwh: float = Field(ge=0.0, allow_inf_nan=False)
    carbon_intensity_g_per_kwh: float = Field(default=250.0, ge=0.0, allow_inf_nan=False)


class VehicleBPTProfile(BaseModel):
    """Bidirectional Power Transfer (BPT) battery constraints and driver departure schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(min_length=1)
    battery_capacity_kwh: float = Field(gt=0.0, le=300.0, allow_inf_nan=False)
    current_soc: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    min_departure_soc: float = Field(default=0.60, ge=0.10, le=1.0, allow_inf_nan=False)
    max_discharge_kw: float = Field(default=22.0, gt=0.0, le=350.0, allow_inf_nan=False)
    max_charge_kw: float = Field(default=50.0, gt=0.0, le=350.0, allow_inf_nan=False)
    arrival_hour: float = Field(ge=0.0, le=24.0, allow_inf_nan=False)
    departure_hour: float = Field(gt=0.0, le=48.0, allow_inf_nan=False)
    degradation_cost_factor: float = Field(default=1.0, ge=0.5, le=3.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_departure_after_arrival(self) -> Self:
        if self.departure_hour <= self.arrival_hour:
            raise ValueError(
                f"departure_hour ({self.departure_hour}) must be strictly after "
                f"arrival_hour ({self.arrival_hour})"
            )
        return self


class InverterTelemetry(BaseModel):
    """High-frequency operational measurement from a bidirectional grid-tied inverter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inverter_id: str = Field(min_length=1)
    phase: InverterPhase = InverterPhase.THREE_PHASE
    active_power_kw: float = Field(allow_inf_nan=False)
    reactive_power_kvar: float = Field(default=0.0, allow_inf_nan=False)
    grid_frequency_hz: float = Field(default=50.0, gt=40.0, le=70.0, allow_inf_nan=False)
    voltage_rms: float = Field(default=400.0, gt=0.0, allow_inf_nan=False)
    dc_bus_voltage_v: float = Field(default=800.0, gt=0.0, allow_inf_nan=False)
    efficiency_percent: float = Field(default=96.5, gt=0.0, le=100.0, allow_inf_nan=False)
    power_factor: float = Field(default=1.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    timestamp: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class V2GTimeSlotDispatch(BaseModel):
    """Energy dispatch instruction and state for an individual time slot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_index: int = Field(ge=0)
    hour: float = Field(ge=0.0, le=48.0, allow_inf_nan=False)
    power_mode: V2GPowerMode
    active_power_kw: float = Field(allow_inf_nan=False)
    soc_end: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    import_tariff: float = Field(ge=0.0, allow_inf_nan=False)
    export_tariff: float = Field(ge=0.0, allow_inf_nan=False)
    cost_or_revenue_delta: float = Field(allow_inf_nan=False)


class V2GArbitrageSchedule(BaseModel):
    """Complete multi-interval charging and bidirectional discharge plan for a vehicle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = Field(min_length=1)
    dispatches: list[V2GTimeSlotDispatch] = Field(default_factory=list)
    total_energy_charged_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    total_energy_discharged_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    net_cost_dollars: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    net_revenue_dollars: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    net_profit_dollars: float = Field(default=0.0, allow_inf_nan=False)
    final_soc: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    min_departure_soc_met: bool = Field(default=True)


class MicrogridFacilityReport(BaseModel):
    """Aggregated microgrid power flow, peak shaving, and financial arbitrage telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_bays_connected: int = Field(default=0, ge=0)
    total_battery_capacity_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    available_flexible_energy_kwh: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    baseline_facility_peak_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    peak_shaved_demand_kw: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    transformer_utilization_percent: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    net_arbitrage_profit_dollars: float = Field(default=0.0, allow_inf_nan=False)
    carbon_offset_kg: float = Field(default=0.0, allow_inf_nan=False)
    schedules: list[V2GArbitrageSchedule] = Field(default_factory=list)
    is_islanded: bool = Field(default=False)
    mean_inverter_efficiency_percent: float = Field(
        default=96.5, ge=0.0, le=100.0, allow_inf_nan=False
    )
