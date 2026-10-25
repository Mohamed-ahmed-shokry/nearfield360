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
    energy_fee_usd,
    parking_fee_usd,
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


def test_parking_fee_applies_grace_period() -> None:
    assert parking_fee_usd(0.2, grace_minutes=15) == 0.0
    assert parking_fee_usd(0.5, grace_minutes=15) == 0.62
    assert parking_fee_usd(0.2, grace_minutes=30) == 0.0
    assert parking_fee_usd(3.0, grace_minutes=0) == 7.5


def test_energy_fee_nets_v2g_credit_and_never_negative() -> None:
    assert energy_fee_usd(10.0, 0.0) == 3.0
    assert energy_fee_usd(10.0, 5.0) == 2.4
    assert energy_fee_usd(1.0, 50.0) == 0.0


def test_simulation_with_charging_and_v2g() -> None:
    config = OperationsConfig()
    # Arrival during V2G window (16:00 - 20:00 -> 17.0h = 61200s)
    arrival_v2g = VehicleArrival(
        vehicle_id="EV-CHG-1",
        arrival_time_s=17.0 * 3600.0,
        requested_dwell_hours=2.0,
        requests_charging=True,
    )
    # Arrival outside V2G window (8:00 -> 28800s)
    arrival_no_v2g = VehicleArrival(
        vehicle_id="EV-CHG-2",
        arrival_time_s=8.0 * 3600.0,
        requested_dwell_hours=2.0,
        requests_charging=True,
    )
    # Arrival with no charging
    arrival_no_chg = VehicleArrival(
        vehicle_id="EV-NOCHG",
        arrival_time_s=8.0 * 3600.0,
        requested_dwell_hours=2.0,
        requests_charging=False,
    )

    journeys = simulate_journeys(
        [arrival_v2g, arrival_no_v2g, arrival_no_chg], config, slot_capacity=10
    )
    j_map = {j.vehicle_id: j for j in journeys}
    j_v2g = j_map["EV-CHG-1"]
    j_no_v2g = j_map["EV-CHG-2"]
    j_no_chg = j_map["EV-NOCHG"]

    assert j_no_chg.charging_windows == 0
    assert j_no_chg.energy_charged_kwh == 0.0
    assert j_no_chg.energy_exported_kwh == 0.0
    assert j_no_chg.energy_fee_usd == 0.0
    assert j_no_chg.parking_fee_usd > 0.0

    assert j_no_v2g.charging_windows == 1
    assert j_no_v2g.energy_charged_kwh > 0.0
    assert j_no_v2g.energy_exported_kwh == 0.0
    assert j_no_v2g.energy_fee_usd > 0.0

    assert j_v2g.charging_windows == 1
    assert j_v2g.energy_charged_kwh > 0.0
    assert j_v2g.energy_exported_kwh > 0.0
    assert j_v2g.energy_fee_usd < j_no_v2g.energy_fee_usd
