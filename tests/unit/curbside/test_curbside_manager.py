"""Unit tests for CurbsideManager bay staging, queueing, and dwell overstay auditing."""

from __future__ import annotations

import pytest

from nearfield360.config import CurbsideConfig
from nearfield360.curbside.manager import CurbsideManager
from nearfield360.curbside.models import BayStatus, HandoverType
from nearfield360.curbside.safety import InterlockViolationError


def test_curbside_manager_initialization() -> None:
    cfg = CurbsideConfig(staging_capacity=4)
    mgr = CurbsideManager(facility_id="facility-test", config=cfg)

    assert len(mgr.bays) == 4
    dropoffs = [b for b in mgr.bays if b.bay_type == HandoverType.DROPOFF]
    pickups = [b for b in mgr.bays if b.bay_type == HandoverType.PICKUP]
    assert len(dropoffs) == 2
    assert len(pickups) == 2
    assert len(mgr.queue) == 0
    assert len(mgr.active_missions) == 0


def test_curbside_manager_dropoff_lifecycle() -> None:
    cfg = CurbsideConfig(
        max_dropoff_dwell_s=100.0,
        overstay_penalty_per_min=2.0,
        staging_capacity=2,
    )
    mgr = CurbsideManager(config=cfg)

    # 1. Request dropoff
    mission, bay = mgr.request_dropoff("veh-alpha", current_time_s=10.0)
    assert bay is not None
    assert bay.is_occupied is True
    assert bay.status == BayStatus.OCCUPIED_UNLOADING
    assert bay.occupied_vehicle_id == "veh-alpha"
    assert mission.vehicle_id == "veh-alpha"
    assert len(mgr.active_missions) == 1

    # 2. Complete dropoff at t=80.0s (dwell=70s <= 100s)
    completed = mgr.complete_dropoff("veh-alpha", current_time_s=80.0)
    assert completed.is_completed is True
    assert completed.dwell_duration_s == 70.0
    assert completed.overstay_penalty_usd == 0.0

    # Bay should be vacant now
    freed_bay = next(b for b in mgr.bays if b.bay_id == bay.bay_id)
    assert freed_bay.is_occupied is False
    assert freed_bay.status == BayStatus.VACANT


def test_curbside_manager_overstay_penalty() -> None:
    cfg = CurbsideConfig(
        max_dropoff_dwell_s=60.0,
        overstay_penalty_per_min=3.0,
        staging_capacity=2,
    )
    mgr = CurbsideManager(config=cfg)

    mgr.request_dropoff("veh-slow", current_time_s=0.0)
    # Dwell = 180s (120s excess = 2 min * $3/min = $6.0)
    completed = mgr.complete_dropoff("veh-slow", current_time_s=180.0)
    assert completed.dwell_duration_s == 180.0
    assert completed.overstay_penalty_usd == 6.0


def test_curbside_manager_interlock_inhibition() -> None:
    mgr = CurbsideManager()
    mgr.request_dropoff("veh-locked", current_time_s=0.0)

    # Simulate passenger door open
    mgr.interlock_engine.set_door_state("veh-locked", is_open=True)

    with pytest.raises(InterlockViolationError, match="engaged_door_open"):
        mgr.complete_dropoff("veh-locked", current_time_s=30.0)

    # Close door and complete successfully
    mgr.interlock_engine.set_door_state("veh-locked", is_open=False)
    completed = mgr.complete_dropoff("veh-locked", current_time_s=40.0)
    assert completed.is_completed is True


def test_curbside_manager_queue_and_dispatch() -> None:
    cfg = CurbsideConfig(staging_capacity=2)  # 1 dropoff bay, 1 pickup bay
    mgr = CurbsideManager(config=cfg)

    # First vehicle gets assigned to dropoff bay
    _m1, b1 = mgr.request_dropoff("veh-1", current_time_s=0.0)
    assert b1 is not None

    # Second vehicle enters queue because dropoff bay is busy
    m2, b2 = mgr.request_dropoff("veh-2", current_time_s=5.0)
    assert b2 is None
    assert m2.bay_id == "QUEUED"
    assert len(mgr.queue) == 1

    # First vehicle completes dropoff -> bay frees and automatically assigns to veh-2
    mgr.complete_dropoff("veh-1", current_time_s=20.0)
    assert len(mgr.queue) == 0
    assert len(mgr.active_missions) == 1
    assert "veh-2" in [m.vehicle_id for m in mgr.active_missions]


def test_curbside_manager_pickup_and_report() -> None:
    cfg = CurbsideConfig(staging_capacity=2)
    mgr = CurbsideManager(facility_id="garage-avp-alpha", config=cfg)

    _m, b = mgr.request_pickup("veh-pickup-01", current_time_s=10.0)
    assert b is not None
    assert b.status == BayStatus.OCCUPIED_BOARDING

    report = mgr.generate_report(current_time_s=25.0)
    assert report.facility_id == "garage-avp-alpha"
    assert report.total_bays_count == 2
    assert report.vacant_bays_count == 1
    assert len(report.active_missions) == 1

    # Complete pickup
    completed = mgr.complete_pickup("veh-pickup-01", current_time_s=40.0)
    assert completed.is_completed is True

    final_report = mgr.generate_report(current_time_s=50.0)
    assert final_report.completed_missions_count == 1
    assert final_report.vacant_bays_count == 2
