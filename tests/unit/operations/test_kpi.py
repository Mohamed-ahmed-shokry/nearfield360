"""Unit tests for operations KPI and snapshot aggregation."""

from __future__ import annotations

import pytest

from nearfield360.config import OperationsConfig
from nearfield360.operations.demand import generate_arrivals
from nearfield360.operations.kpi import (
    BASE_FACILITY_LOAD_KW,
    aggregate_daily_kpis,
    compute_hourly_snapshots,
)
from nearfield360.operations.models import (
    JourneyStage,
    ServiceLevel,
)
from nearfield360.operations.simulator import simulate_journeys


def test_empty_journeys_kpi() -> None:
    config = OperationsConfig(day_duration_hours=24.0)
    snaps = compute_hourly_snapshots([], config)
    assert len(snaps) == 24
    assert all(s.arrivals == 0 for s in snaps)
    assert all(s.occupied_slots == 0 for s in snaps)
    assert all(s.facility_load_kw == BASE_FACILITY_LOAD_KW for s in snaps)

    kpis = aggregate_daily_kpis([], snaps, config, slot_capacity=100)
    assert kpis.total_arrivals == 0
    assert kpis.completed_journeys == 0
    assert kpis.throughput_per_hour == 0.0
    assert kpis.peak_occupancy == 0
    assert kpis.sla_compliance_ratio == 1.0


def test_snapshots_and_kpis_conservation_and_consistency() -> None:
    config = OperationsConfig(day_duration_hours=24.0)
    arrivals = generate_arrivals(config, seed=42)
    slot_cap = 50
    journeys = simulate_journeys(arrivals, config, slot_capacity=slot_cap)

    snaps = compute_hourly_snapshots(journeys, config)
    assert len(snaps) == 24

    # Arrivals conserved across snapshots
    total_snap_arrivals = sum(s.arrivals for s in snaps)
    assert total_snap_arrivals == len(journeys)

    # Departures conserved
    total_snap_departures = sum(s.departures for s in snaps)
    completed_in_journeys = sum(
        1
        for j in journeys
        if j.stage is JourneyStage.DEPARTED and j.service_level is not ServiceLevel.BLOCKED
    )
    assert total_snap_departures == completed_in_journeys

    # Blocked conserved
    total_snap_blocked = sum(s.blocked for s in snaps)
    blocked_in_journeys = sum(1 for j in journeys if j.service_level is ServiceLevel.BLOCKED)
    assert total_snap_blocked == blocked_in_journeys

    kpis = aggregate_daily_kpis(journeys, snaps, config, slot_capacity=slot_cap)
    assert kpis.total_arrivals == len(journeys)
    assert kpis.completed_journeys == completed_in_journeys
    assert kpis.blocked_vehicles == blocked_in_journeys
    assert kpis.peak_occupancy <= slot_cap
    assert kpis.peak_occupancy_ratio <= 1.0
    assert kpis.total_revenue_usd == pytest.approx(
        kpis.parking_revenue_usd + kpis.energy_revenue_usd
    )
    assert kpis.net_grid_energy_kwh == pytest.approx(
        kpis.energy_charged_kwh - kpis.energy_exported_kwh
    )
    assert 0.0 <= kpis.sla_compliance_ratio <= 1.0
