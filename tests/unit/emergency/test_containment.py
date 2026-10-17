"""Unit tests for HazardContainmentEngine and fire door interlocks."""

from __future__ import annotations

import pytest

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.containment import HazardContainmentEngine
from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    VehicleEvacuationStatus,
)


def test_hazard_containment_empty() -> None:
    engine = HazardContainmentEngine()
    assert len(engine.hazards) == 0
    assert len(engine.fire_doors) == 0

    assert not engine.is_point_inside_hazard(10.0, 10.0)
    status, reason, dist = engine.classify_vehicle_threat("v1", (10.0, 10.0))
    assert status == VehicleEvacuationStatus.STATIONARY_SAFE
    assert reason is None
    assert dist == float("inf")


def test_ev_thermal_runaway_detection() -> None:
    config = EmergencyConfig(thermal_runaway_temp_c=70.0, hazard_buffer_radius_m=12.0)
    engine = HazardContainmentEngine(config)

    # Below threshold -> no hazard created
    res_normal = engine.report_ev_temperature("ev-1", "bay-01", (10.0, 20.0), 45.0)
    assert res_normal is None
    assert len(engine.hazards) == 0

    # At or above threshold -> active thermal runaway hazard created
    hazard = engine.report_ev_temperature(
        "ev-1", "bay-01", (10.0, 20.0), 78.5, floor_id="L1", timestamp_s=50.0
    )
    assert hazard is not None
    assert hazard.emergency_type == EmergencyType.EV_THERMAL_RUNAWAY
    assert hazard.severity == EmergencySeverity.PARTIAL_EVACUATION
    assert hazard.radius_m == 12.0
    assert "ev-1" in hazard.description
    assert len(engine.hazards) == 1

    # Remove and clear
    assert engine.remove_hazard("thermal-ev-1") is True
    assert len(engine.hazards) == 0
    assert engine.remove_hazard("thermal-ev-1") is False


def test_hazard_threat_classification() -> None:
    config = EmergencyConfig(hazard_buffer_radius_m=10.0)
    engine = HazardContainmentEngine(config)

    hazard = HazardZone(
        hazard_id="fire-zone-1",
        emergency_type=EmergencyType.FIRE_ALARM,
        severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
        center_x=50.0,
        center_y=50.0,
        radius_m=15.0,
        floor_id="L1",
        active=True,
    )
    engine.add_hazard(hazard)

    # Point check
    assert engine.is_point_inside_hazard(55.0, 50.0, floor_id="L1")
    assert not engine.is_point_inside_hazard(55.0, 50.0, floor_id="L2")  # Different floor
    assert not engine.is_point_inside_hazard(80.0, 80.0, floor_id="L1")

    # Threat classification:
    # 1. Inside radius (center is 50,50, radius is 15 -> distance = 5m <= 15m)
    status_trapped, msg_trapped, d_trapped = engine.classify_vehicle_threat(
        "v-trapped", (50.0, 55.0), floor_id="L1"
    )
    assert status_trapped == VehicleEvacuationStatus.TRAPPED_IN_HAZARD
    assert msg_trapped is not None
    assert "inside hazard zone" in msg_trapped
    assert d_trapped == pytest.approx(5.0)

    # 2. In buffer (distance = 20m, 15m < 20m <= 25m)
    status_evac, msg_evac, d_evac = engine.classify_vehicle_threat(
        "v-evac", (50.0, 70.0), floor_id="L1"
    )
    assert status_evac == VehicleEvacuationStatus.EVACUATING
    assert msg_evac is not None
    assert "within hazard buffer" in msg_evac
    assert d_evac == pytest.approx(20.0)

    # 3. Stationary safe (distance = 50m > 25m)
    status_safe, msg_safe, d_safe = engine.classify_vehicle_threat(
        "v-safe", (100.0, 50.0), floor_id="L1"
    )
    assert status_safe == VehicleEvacuationStatus.STATIONARY_SAFE
    assert msg_safe is None
    assert d_safe == pytest.approx(50.0)


def test_fire_door_clearance_interlocks() -> None:
    engine = HazardContainmentEngine()
    door = FireDoorInterlock(
        door_id="door-sector-a",
        floor_id="L1",
        location_x=20.0,
        location_y=0.0,
        status=FireDoorStatus.OPEN,
        time_to_close_s=20.0,
    )
    engine.register_fire_door(door)

    # Open door -> can transit
    can_pass, reason = engine.can_transit_fire_door("door-sector-a", 10.0)
    assert can_pass is True
    assert "Clearance granted" in reason

    # Closing door, transit time 12s <= remaining 20s -> can transit
    engine.update_fire_door_status("door-sector-a", FireDoorStatus.CLOSING, time_to_close_s=20.0)
    can_pass_closing, _ = engine.can_transit_fire_door("door-sector-a", 12.0)
    assert can_pass_closing is True

    # Closing door, transit time 25s > remaining 20s -> cannot transit
    can_pass_late, reason_late = engine.can_transit_fire_door("door-sector-a", 25.0)
    assert can_pass_late is False
    assert "exceeds remaining closure countdown" in reason_late

    # Locked shut -> cannot transit
    engine.update_fire_door_status("door-sector-a", FireDoorStatus.LOCKED_SHUT)
    can_pass_locked, reason_locked = engine.can_transit_fire_door("door-sector-a", 1.0)
    assert can_pass_locked is False
    assert "locked shut" in reason_locked

    # Non-existent door
    can_pass_unknown, _ = engine.can_transit_fire_door("door-unknown", 5.0)
    assert can_pass_unknown is True
    assert engine.update_fire_door_status("door-unknown", FireDoorStatus.OPEN) is False
