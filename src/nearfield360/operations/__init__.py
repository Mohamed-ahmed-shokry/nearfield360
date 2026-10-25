"""Facility operations control tower: day-level simulation and cross-subsystem KPIs."""

from __future__ import annotations

from nearfield360.operations.demand import arrival_rate_per_hour, generate_arrivals
from nearfield360.operations.kpi import (
    BASE_FACILITY_LOAD_KW,
    aggregate_daily_kpis,
    compute_hourly_snapshots,
)
from nearfield360.operations.models import (
    DailyKPIReport,
    HourlySnapshot,
    JourneyStage,
    OperationsDayReport,
    ServiceLevel,
    VehicleArrival,
    VehicleJourneyRecord,
)
from nearfield360.operations.simulator import (
    DEFAULT_SLOT_CAPACITY,
    energy_fee_usd,
    parking_fee_usd,
    simulate_journeys,
)
from nearfield360.operations.viz import render_operations_dashboard

__all__ = [
    "BASE_FACILITY_LOAD_KW",
    "DEFAULT_SLOT_CAPACITY",
    "DailyKPIReport",
    "HourlySnapshot",
    "JourneyStage",
    "OperationsDayReport",
    "ServiceLevel",
    "VehicleArrival",
    "VehicleJourneyRecord",
    "aggregate_daily_kpis",
    "arrival_rate_per_hour",
    "compute_hourly_snapshots",
    "energy_fee_usd",
    "generate_arrivals",
    "parking_fee_usd",
    "render_operations_dashboard",
    "simulate_journeys",
]
