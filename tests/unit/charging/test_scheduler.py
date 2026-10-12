"""Unit tests for EV ChargingScheduler queue prioritization, bay allocation, and relocations."""

from __future__ import annotations

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargerType,
    ChargingSessionStatus,
)
from nearfield360.charging.scheduler import ChargingRequest, ChargingScheduler
from nearfield360.config import ChargingConfig
from nearfield360.mapping.builder import build_benchmark_garage


def test_scheduler_bay_registration_and_queue_prioritization() -> None:
    config = ChargingConfig(critical_soc_threshold=0.20)
    scheduler = ChargingScheduler(config=config)

    bay1 = ChargerBay(
        bay_id="BAY_01",
        slot_id="SLOT_CHG_01",
        charger_type=ChargerType.DC_FAST_150KW,
        max_power_rating_kw=150.0,
        status=ChargerPortStatus.AVAILABLE,
    )
    scheduler.register_charger_bays([bay1])

    # Enqueue normal request first
    scheduler.enqueue_request(
        ChargingRequest(
            vehicle_id="EV_NORM",
            arrival_time_s=100.0,
            current_soc=0.50,
            target_soc=0.80,
        )
    )

    # Enqueue critical request later
    scheduler.enqueue_request(
        ChargingRequest(
            vehicle_id="EV_CRIT",
            arrival_time_s=110.0,
            current_soc=0.10,  # Critical < 0.20
            target_soc=0.80,
        )
    )

    # Critical request should jump ahead and be assigned the available bay
    assignments = scheduler.schedule_assignments(current_time_s=120.0)
    assert len(assignments) == 1
    session = assignments[0]
    assert session.vehicle_id == "EV_CRIT"
    assert session.bay_id == "BAY_01"
    assert session.status == ChargingSessionStatus.PENDING_DOCKING

    # Normal vehicle remains in queue
    assert len(scheduler.queue) == 1
    assert scheduler.queue[0].vehicle_id == "EV_NORM"


def test_scheduler_completion_and_relocation_orders() -> None:
    garage = build_benchmark_garage()
    scheduler = ChargingScheduler(facility_map=garage)

    bay = ChargerBay(
        bay_id="BAY_01",
        slot_id="SLOT_CHG_01",
        status=ChargerPortStatus.AVAILABLE,
    )
    scheduler.register_charger_bays([bay])

    scheduler.enqueue_request(
        ChargingRequest(
            vehicle_id="EV_01",
            arrival_time_s=0.0,
            current_soc=0.75,
            target_soc=0.80,
        )
    )
    assignments = scheduler.schedule_assignments(current_time_s=10.0)
    session = assignments[0]
    session.status = ChargingSessionStatus.ACTIVE_CHARGING

    # Not yet complete
    relocations = scheduler.check_completions_and_relocations(current_time_s=100.0)
    assert len(relocations) == 0

    # Charge reaches target SoC
    session.current_soc = 0.82
    relocations = scheduler.check_completions_and_relocations(current_time_s=200.0)
    assert len(relocations) == 1
    order = relocations[0]
    assert order.vehicle_id == "EV_01"
    assert order.source_bay_id == "BAY_01"
    assert session.status == ChargingSessionStatus.COMPLETED

    # Complete relocation order to free up bay
    scheduler.complete_relocation(order.order_id)
    assert len(scheduler.relocation_orders) == 0
    assert bay.status == ChargerPortStatus.AVAILABLE
