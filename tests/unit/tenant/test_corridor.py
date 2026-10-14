"""Unit tests for VIP express corridor reservations and traffic preemption."""

from __future__ import annotations

import pytest

from nearfield360.tenant.corridor import CorridorConflictError, VIPCorridorManager
from nearfield360.tenant.models import VIPExpressCorridor


def test_corridor_manager_registration() -> None:
    mgr = VIPCorridorManager()
    corridor = mgr.register_corridor(
        corridor_id="express-north",
        lane_ids=["lane_1", "lane_2"],
        default_duration_s=45.0,
    )
    assert corridor.corridor_id == "express-north"
    assert corridor.duration_s == 45.0
    assert len(mgr.corridors) == 1


def test_corridor_manager_reservation_and_checks() -> None:
    mgr = VIPCorridorManager()
    mgr.register_corridor("c1", ["lane_1", "lane_2"], default_duration_s=60.0)

    # Reserve corridor starting at t=10.0s for 30.0s
    res = mgr.reserve_corridor(
        corridor_id="c1",
        vehicle_id="vip-limo-1",
        start_time_s=10.0,
        duration_s=30.0,
        priority_level=2,
    )
    assert res.active_vehicle_id == "vip-limo-1"
    assert res.is_preempting is True

    # Check activity window
    assert res.is_active_at(5.0) is False
    assert res.is_active_at(10.0) is True
    assert res.is_active_at(25.0) is True
    assert res.is_active_at(40.0) is False

    # Check active corridor queries
    active = mgr.get_active_corridors(20.0)
    assert len(active) == 1
    assert active[0].corridor_id == "c1"

    # Check lane reservation
    is_res, veh = mgr.is_lane_reserved("lane_1", 20.0)
    assert is_res is True
    assert veh == "vip-limo-1"

    is_res2, veh2 = mgr.is_lane_reserved("lane_other", 20.0)
    assert is_res2 is False
    assert veh2 is None


def test_corridor_manager_preemption_and_conflicts() -> None:
    mgr = VIPCorridorManager()
    mgr.register_corridor("c1", ["lane_shared", "lane_A"])
    mgr.register_corridor("c2", ["lane_shared", "lane_B"])

    # c1 reserves with priority 1
    mgr.reserve_corridor("c1", "veh-standard", start_time_s=0.0, duration_s=100.0, priority_level=1)

    # c2 tries to reserve with priority 1 -> should raise conflict error
    with pytest.raises(CorridorConflictError, match="occupied by corridor 'c1'"):
        mgr.reserve_corridor(
            "c2", "veh-other", start_time_s=20.0, duration_s=50.0, priority_level=1
        )

    # c2 reserves with priority 3 -> preempts c1
    res2 = mgr.reserve_corridor(
        "c2", "veh-vip", start_time_s=20.0, duration_s=50.0, priority_level=3
    )
    assert res2.active_vehicle_id == "veh-vip"
    assert res2.is_preempting is True

    # c1 should be preempted
    c1 = next(c for c in mgr.corridors if c.corridor_id == "c1")
    assert c1.active_vehicle_id is None


def test_corridor_manager_release() -> None:
    c = VIPExpressCorridor(corridor_id="c1", lane_ids=["lane_1"], active_vehicle_id="veh-1")
    mgr = VIPCorridorManager([c])
    initial = mgr.corridors[0]
    assert initial.active_vehicle_id == "veh-1"

    mgr.release_corridor("c1")
    released = mgr.corridors[0]
    assert released.active_vehicle_id is None
    assert released.is_preempting is False


def test_corridor_manager_unknown_corridor() -> None:
    mgr = VIPCorridorManager()
    with pytest.raises(KeyError, match="not registered"):
        mgr.reserve_corridor("c-unknown", "veh-1", start_time_s=0.0)
