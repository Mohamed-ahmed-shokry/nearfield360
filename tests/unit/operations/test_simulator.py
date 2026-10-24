"""Unit tests for the coordinated day-run journey simulator."""

from __future__ import annotations

import pytest

from nearfield360.config import OperationsConfig
from nearfield360.operations.demand import generate_arrivals
from nearfield360.operations.models import JourneyStage, ServiceLevel, VehicleArrival
from nearfield360.operations.simulator import (
    DISPATCH_LATENCY_S,
    MANUAL_PLATE_REVIEW_S,
    QUEUE_PATIENCE_S,
    simulate_journeys,
)


def _arrival(index: int, t: float, dwell: float = 1.0, plate: float = 0.95) -> VehicleArrival:
    return VehicleArrival(
        vehicle_id=f"EV-{index}",
        arrival_time_s=t,
        plate_confidence=plate,
        requested_dwell_hours=dwell,
    )


def test_single_vehicle_is_nominal_and_departs() -> None:
    config = OperationsConfig()
    (journey,) = simulate_journeys([_arrival(1, 100.0)], config, slot_capacity=2)
    assert journey.service_level is ServiceLevel.NOMINAL
    assert journey.stage is JourneyStage.DEPARTED
    assert journey.dispatch_wait_s == DISPATCH_LATENCY_S
    assert journey.parked_time_s == pytest.approx(100.0 + DISPATCH_LATENCY_S)
    assert journey.dispatched_slot_id == "S-0001"


def test_low_plate_confidence_degrades_dispatch() -> None:
    config = OperationsConfig()
    (journey,) = simulate_journeys([_arrival(1, 0.0, plate=0.5)], config, slot_capacity=2)
    assert journey.service_level is ServiceLevel.DEGRADED_DISPATCH
    assert journey.dispatch_wait_s == DISPATCH_LATENCY_S + MANUAL_PLATE_REVIEW_S


def test_full_facility_queues_then_balks() -> None:
    config = OperationsConfig()
    arrivals = [
        _arrival(1, 0.0, dwell=0.5),
        _arrival(2, 10.0, dwell=0.1),
        _arrival(3, 20.0, dwell=0.1),
    ]
    first, second, third = simulate_journeys(arrivals, config, slot_capacity=1)
    assert first.service_level is ServiceLevel.NOMINAL
    assert second.service_level is ServiceLevel.BLOCKED
    assert second.dispatch_wait_s == QUEUE_PATIENCE_S
    assert third.service_level is ServiceLevel.BLOCKED
    assert second.dispatched_slot_id is None


def test_short_wait_is_curbside_handover_and_reuses_slot() -> None:
    config = OperationsConfig()
    arrivals = [_arrival(1, 0.0, dwell=0.2), _arrival(2, 60.0, dwell=0.2)]
    first, second = simulate_journeys(arrivals, config, slot_capacity=1)
    assert second.service_level is ServiceLevel.CURBSIDE_HANDOVER
    assert second.dispatched_slot_id == first.dispatched_slot_id
    assert second.parked_time_s is not None and first.departed_time_s is not None
    assert second.parked_time_s >= first.departed_time_s


def test_vehicle_leaving_after_horizon_stays_parked() -> None:
    config = OperationsConfig(day_duration_hours=1.0)
    (journey,) = simulate_journeys([_arrival(1, 3000.0, dwell=2.0)], config, slot_capacity=1)
    assert journey.stage is JourneyStage.PARKED
    assert journey.departed_time_s is None


def test_invalid_capacity_rejected() -> None:
    with pytest.raises(ValueError, match="slot_capacity"):
        simulate_journeys([], OperationsConfig(), slot_capacity=0)


def test_vehicle_conservation_and_determinism() -> None:
    config = OperationsConfig()
    arrivals = generate_arrivals(config, seed=9)
    first = simulate_journeys(arrivals, config, slot_capacity=20)
    second = simulate_journeys(arrivals, config, slot_capacity=20)
    assert first == second
    assert len(first) == len(arrivals)
    assert {j.vehicle_id for j in first} == {a.vehicle_id for a in arrivals}
