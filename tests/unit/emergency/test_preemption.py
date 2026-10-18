"""Unit tests for EmergencyPreemptionArbiter and emergency corridor clearance."""

from __future__ import annotations

import pytest

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.models import EmergencyVehicle, ResponderType, VehicleEvacuationStatus
from nearfield360.emergency.preemption import (
    EmergencyPreemptionArbiter,
    point_to_segment_distance,
)


def test_point_to_segment_distance() -> None:
    # Point perpendicular to segment (0,0) to (10,0) at (5, 3)
    d = point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(3.0)

    # Point beyond end point (10,0) at (14, 0)
    d_beyond = point_to_segment_distance(14.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert d_beyond == pytest.approx(4.0)

    # Point before start point (0,0) at (-3, 0)
    d_before = point_to_segment_distance(-3.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert d_before == pytest.approx(3.0)

    # Degenerate zero-length segment
    d_zero = point_to_segment_distance(2.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    assert d_zero == pytest.approx(1.4142, rel=1e-3)


def test_preemption_arbiter_empty() -> None:
    arbiter = EmergencyPreemptionArbiter()
    assert not arbiter.is_preemption_active()
    assert len(arbiter.responders) == 0

    status, reason = arbiter.evaluate_vehicle("veh-1", (10.0, 10.0))
    assert status == VehicleEvacuationStatus.STATIONARY_SAFE
    assert reason is None


def test_preemption_proximity_and_corridor() -> None:
    config = EmergencyConfig(preemption_radius_m=30.0)
    arbiter = EmergencyPreemptionArbiter(config)

    responder = EmergencyVehicle(
        vehicle_id="fd-1",
        responder_type=ResponderType.FIRE_ENGINE,
        x=0.0,
        y=0.0,
        target_x=100.0,
        target_y=0.0,
    )
    arbiter.register_responder(responder)
    assert arbiter.is_preemption_active()

    # Vehicle within preemption radius of responder position (15m <= 30m)
    status_near, reason_near = arbiter.evaluate_vehicle("veh-near", (10.0, 10.0))
    assert status_near == VehicleEvacuationStatus.HOLD_YIELDING
    assert reason_near is not None
    assert "Yielding right-of-way" in reason_near

    # Vehicle far from responder (x=60), but directly in the corridor path (y=5m <= 15m)
    status_corridor, reason_corridor = arbiter.evaluate_vehicle("veh-corridor", (60.0, 5.0))
    assert status_corridor == VehicleEvacuationStatus.HOLD_YIELDING
    assert reason_corridor is not None
    assert "In corridor path" in reason_corridor

    # Vehicle far away from responder and corridor (x=60, y=50)
    status_safe, reason_safe = arbiter.evaluate_vehicle("veh-safe", (60.0, 50.0))
    assert status_safe == VehicleEvacuationStatus.STATIONARY_SAFE
    assert reason_safe is None


def test_preemption_fleet_evaluation_and_lifecycle() -> None:
    arbiter = EmergencyPreemptionArbiter(EmergencyConfig(preemption_radius_m=25.0))
    responder = EmergencyVehicle(
        vehicle_id="amb-1",
        responder_type=ResponderType.AMBULANCE,
        x=20.0,
        y=20.0,
        target_x=20.0,
        target_y=60.0,
    )
    arbiter.register_responder(responder)

    fleet = {
        "v1": (22.0, 25.0),  # Close to responder
        "v2": (22.0, 45.0),  # Along corridor
        "v3": (80.0, 80.0),  # Safe
    }
    results = arbiter.evaluate_fleet(fleet)
    assert results["v1"][0] == VehicleEvacuationStatus.HOLD_YIELDING
    assert results["v2"][0] == VehicleEvacuationStatus.HOLD_YIELDING
    assert results["v3"][0] == VehicleEvacuationStatus.STATIONARY_SAFE

    # Remove responder
    removed = arbiter.remove_responder("amb-1")
    assert removed is True
    assert not arbiter.is_preemption_active()
    assert arbiter.remove_responder("amb-1") is False

    # Clear
    arbiter.register_responder(responder)
    assert arbiter.is_preemption_active()
    arbiter.clear()
    assert not arbiter.is_preemption_active()
