"""Unit tests for emergency life safety domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    EmergencyVehicle,
    EvacuationRoute,
    EvacuationWave,
    EvacuationWaypoint,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    LifeSafetyReport,
    ResponderType,
    VehicleEvacuationStatus,
)


def test_emergency_enums() -> None:
    assert EmergencyType.FIRE_ALARM == "fire_alarm"
    assert EmergencyType.EV_THERMAL_RUNAWAY == "ev_thermal_runaway"
    assert EmergencySeverity.FULL_FACILITY_EVACUATION == "full_facility_evacuation"
    assert VehicleEvacuationStatus.HOLD_YIELDING == "hold_yielding"
    assert FireDoorStatus.LOCKED_SHUT == "locked_shut"
    assert ResponderType.FIRE_ENGINE == "fire_engine"


def test_hazard_zone_serialization() -> None:
    zone = HazardZone(
        hazard_id="hz-101",
        emergency_type=EmergencyType.EV_THERMAL_RUNAWAY,
        severity=EmergencySeverity.PARTIAL_EVACUATION,
        center_x=15.0,
        center_y=25.0,
        center_z=0.0,
        radius_m=12.5,
        floor_id="L1",
        active=True,
        reported_at_s=120.0,
        description="EV battery smoking in bay B-12",
    )
    raw = zone.model_dump()
    assert raw["hazard_id"] == "hz-101"
    assert raw["radius_m"] == 12.5
    roundtrip = HazardZone.model_validate(raw)
    assert roundtrip == zone


def test_hazard_zone_rejects_non_positive_radius() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        HazardZone(
            hazard_id="hz-001",
            emergency_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.ADVISORY,
            center_x=0.0,
            center_y=0.0,
            radius_m=0.0,
        )


def test_fire_door_interlock() -> None:
    door = FireDoorInterlock(
        door_id="door-f1-ramp",
        floor_id="L1",
        location_x=30.0,
        location_y=10.0,
        width_m=4.5,
        status=FireDoorStatus.CLOSING,
        time_to_close_s=8.5,
        is_blocked=False,
    )
    assert door.status == FireDoorStatus.CLOSING
    assert door.time_to_close_s == 8.5


def test_emergency_vehicle() -> None:
    ev = EmergencyVehicle(
        vehicle_id="fd-ladder-1",
        responder_type=ResponderType.FIRE_ENGINE,
        x=5.0,
        y=10.0,
        z=0.0,
        yaw_rad=1.57,
        velocity_mps=6.0,
        target_x=20.0,
        target_y=40.0,
        active_siren=True,
    )
    assert ev.responder_type == ResponderType.FIRE_ENGINE
    assert ev.active_siren is True


def test_evacuation_route_and_wave() -> None:
    wp1 = EvacuationWaypoint(x_m=10.0, y_m=20.0, z_m=0.0, speed_limit_mps=3.5)
    wp2 = EvacuationWaypoint(x_m=0.0, y_m=0.0, z_m=0.0, speed_limit_mps=4.0)
    route = EvacuationRoute(
        route_id="er-v1",
        vehicle_id="veh-001",
        origin_bay_id="bay-L1-05",
        exit_portal_id="portal-egress-north",
        waypoints=[wp1, wp2],
        total_length_m=22.36,
        estimated_transit_time_s=5.8,
        min_hazard_distance_m=18.4,
        is_cleared=True,
    )
    assert route.total_length_m == pytest.approx(22.36)
    assert len(route.waypoints) == 2

    wave = EvacuationWave(
        wave_index=0,
        vehicle_ids=["veh-001", "veh-002"],
        dispatch_time_s=0.0,
        exit_portal_id="portal-egress-north",
    )
    assert wave.wave_index == 0
    assert "veh-001" in wave.vehicle_ids


def test_life_safety_report() -> None:
    report = LifeSafetyReport(
        facility_name="Autonomous Valet Center 1",
        timestamp_s=150.0,
        emergency_active=True,
        incident_type=EmergencyType.FIRE_ALARM,
        severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
        total_vehicles=10,
        evacuated_count=4,
        evacuating_count=4,
        yielding_count=2,
        trapped_count=0,
        evp_active=True,
    )
    assert report.emergency_active is True
    assert report.total_vehicles == 10
    json_str = report.model_dump_json()
    assert "Autonomous Valet Center 1" in json_str
