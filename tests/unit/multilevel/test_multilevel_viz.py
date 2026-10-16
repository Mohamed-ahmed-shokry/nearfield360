"""Unit tests for multi-floor BEV dashboard visualizer and PNG export."""

from pathlib import Path

import cv2
import numpy as np

from nearfield360.multilevel.graph import build_default_facility_graph
from nearfield360.multilevel.models import (
    ElevatorCabin,
    ElevatorState,
    MultiFloorFacilityReport,
    RampDirection,
    RampToken,
    VehiclePitchState,
)
from nearfield360.multilevel.viz import render_multilevel_dashboard


def test_render_multilevel_dashboard_in_memory() -> None:
    graph = build_default_facility_graph()
    route = graph.find_route("L1-entry", "L2-bay-42")

    canvas = render_multilevel_dashboard(graph, route=route, canvas_size=800)

    assert isinstance(canvas, np.ndarray)
    assert canvas.dtype == np.uint8
    # Height = 800, width = 800 + 440 = 1240, 3 BGR channels
    assert canvas.shape == (800, 1240, 3)
    # Ensure canvas is not blank
    assert canvas.sum() > 0


def test_render_multilevel_dashboard_with_all_telemetry_and_file_export(tmp_path: Path) -> None:
    graph = build_default_facility_graph()
    route = graph.find_route("L1-entry", "B1-bay-05")

    report = MultiFloorFacilityReport(
        facility_id="facility-multi-01",
        facility_name="Central Multi-Story Garage",
        timestamp_s=1700000000.0,
        floors_count=3,
        total_bays=74,
        available_bays=27,
        active_vehicles_count=4,
        ramp_token_events_count=12,
        elevator_cycles_count=8,
        average_vertical_transit_time_s=18.4,
        floors=list(graph.floors.values()),
        portals=list(graph.portals.values()),
        elevators=[],
    )

    token = RampToken(
        token_id="tok-ramp-100",  # noqa: S106
        portal_id="ramp-L1-to-L2",
        vehicle_id="veh-avp-01",
        direction=RampDirection.ASCENDING,
        granted_at_s=120.0,
        expires_at_s=150.0,
        is_active=True,
    )

    cabin = ElevatorCabin(
        elevator_id="lift-01",
        current_floor_id="L1",
        target_floor_id="L2",
        state=ElevatorState.HOISTING,
        current_elevation_m=2.1,
        max_weight_kg=3500.0,
        occupied_vehicle_id="veh-avp-02",
        leveling_offset_m=0.005,
    )

    pitch = VehiclePitchState(
        vehicle_id="veh-avp-01",
        pitch_angle_deg=6.84,
        grade_pct=12.0,
        elevation_m=1.8,
        gravity_force_n=2104.0,
        normal_force_n=17520.0,
        traction_margin=0.68,
        breakover_clearance_m=0.085,
        is_clearance_safe=True,
    )

    output_png = tmp_path / "multilevel_dashboard.png"

    canvas = render_multilevel_dashboard(
        graph=graph,
        report=report,
        route=route,
        active_token=token,
        active_cabin=cabin,
        pitch_state=pitch,
        canvas_size=900,
        output_path=output_png,
    )

    assert output_png.is_file()
    assert output_png.stat().st_size > 5000  # valid non-trivial PNG

    # Read back image using OpenCV
    loaded = cv2.imread(str(output_png))
    assert loaded is not None
    assert loaded.shape == canvas.shape
