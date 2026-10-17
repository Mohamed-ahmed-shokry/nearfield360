"""Unit tests for emergency life safety dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    EmergencyVehicle,
    EvacuationRoute,
    EvacuationWaypoint,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    LifeSafetyReport,
    ResponderType,
    VehicleEvacuationStatus,
)
from nearfield360.emergency.viz import render_emergency_dashboard


def test_render_emergency_dashboard(tmp_path: Path) -> None:
    report = LifeSafetyReport(
        facility_name="Central Multi-Level Deck",
        timestamp_s=120.0,
        emergency_active=True,
        incident_type=EmergencyType.FIRE_ALARM,
        severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
        total_vehicles=8,
        evacuated_count=3,
        evacuating_count=3,
        yielding_count=1,
        trapped_count=1,
        evp_active=True,
    )

    hazards = [
        HazardZone(
            hazard_id="fire-1",
            emergency_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            center_x=30.0,
            center_y=40.0,
            radius_m=12.0,
            floor_id="L1",
            active=True,
        )
    ]

    fire_doors = [
        FireDoorInterlock(
            door_id="fd-sector-1",
            floor_id="L1",
            location_x=20.0,
            location_y=20.0,
            status=FireDoorStatus.CLOSING,
            time_to_close_s=10.0,
        )
    ]

    responders = [
        EmergencyVehicle(
            vehicle_id="fd-ladder-10",
            responder_type=ResponderType.FIRE_ENGINE,
            x=0.0,
            y=10.0,
            yaw_rad=0.78,
            velocity_mps=5.0,
            target_x=30.0,
            target_y=40.0,
            active_siren=True,
        )
    ]

    routes = {
        "v1": EvacuationRoute(
            route_id="r1",
            vehicle_id="v1",
            origin_bay_id="b1",
            exit_portal_id="Exit-North",
            waypoints=[
                EvacuationWaypoint(x_m=10.0, y_m=10.0),
                EvacuationWaypoint(x_m=0.0, y_m=90.0),
            ],
            total_length_m=80.6,
            estimated_transit_time_s=20.1,
            is_cleared=True,
        )
    }

    vehicle_poses = {
        "v1": (10.0, 10.0),
        "v2": (32.0, 41.0),
    }

    vehicle_statuses = {
        "v1": VehicleEvacuationStatus.EVACUATING,
        "v2": VehicleEvacuationStatus.TRAPPED_IN_HAZARD,
    }

    out_png = tmp_path / "emergency_dashboard.png"
    img = render_emergency_dashboard(
        report,
        hazards=hazards,
        fire_doors=fire_doors,
        responders=responders,
        routes=routes,
        vehicle_poses=vehicle_poses,
        vehicle_statuses=vehicle_statuses,
        output_path=out_png,
    )

    assert isinstance(img, np.ndarray)
    assert img.shape == (900, 1340, 3)
    assert img.dtype == np.uint8

    assert out_png.is_file()
    data = out_png.read_bytes()
    assert len(data) > 0
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
