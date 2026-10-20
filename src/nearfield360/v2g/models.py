"""Pydantic domain models and telemetry protocol schemas for Vehicle-to-Grid (V2G) systems."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
