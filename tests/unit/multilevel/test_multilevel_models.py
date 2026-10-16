"""Unit tests for multi-floor facility models, vertical portals, and elevator schemas."""

import pytest
from pydantic import ValidationError

from nearfield360.multilevel.models import (
    ElevatorCabin,
    ElevatorState,
    FloorLevel,
    MultiFloorFacilityReport,
    MultiFloorRoute,
    PortalType,
    RampDirection,
    RampToken,
    RampTokenPriority,
    VehiclePitchState,
    VerticalPortal,
    Waypoint3D,
)


def test_multilevel_enums() -> None:
    assert PortalType.STRAIGHT_RAMP.value == "straight_ramp"
    assert PortalType.SPIRAL_RAMP.value == "spiral_ramp"
    assert PortalType.VEHICLE_ELEVATOR.value == "vehicle_elevator"

    assert ElevatorState.IDLE.value == "idle"
    assert ElevatorState.HOISTING.value == "hoisting"
    assert ElevatorState.LEVELING.value == "leveling"

    assert RampDirection.ASCENDING.value == "ascending"
    assert RampDirection.DESCENDING.value == "descending"

    assert RampTokenPriority.ASCENDING_FIRST.value == "ascending_first"
    assert RampTokenPriority.FIFO.value == "fifo"


def test_floor_level_model() -> None:
    floor = FloorLevel(
        floor_id="L1",
        name="Ground Level",
        elevation_m=0.0,
        bays_count=50,
        available_bays=12,
        is_rooftop=False,
        is_basement=False,
    )
    assert floor.floor_id == "L1"
    assert floor.elevation_m == 0.0

    payload = floor.model_dump_json()
    restored = FloorLevel.model_validate_json(payload)
    assert restored == floor

    with pytest.raises(ValidationError):
        FloorLevel(
            floor_id="",
            name="Ground Level",
            elevation_m=0.0,
            bays_count=50,
            available_bays=12,
        )


def test_vertical_portal_model() -> None:
    portal = VerticalPortal(
        portal_id="ramp-L1-L2",
        portal_type=PortalType.SPIRAL_RAMP,
        from_floor_id="L1",
        to_floor_id="L2",
        length_m=35.0,
        grade_pct=12.5,
        curvature_radius_m=14.0,
        is_bidirectional=False,
        speed_limit_mps=2.5,
    )
    assert portal.portal_type == PortalType.SPIRAL_RAMP
    assert portal.curvature_radius_m == 14.0

    with pytest.raises(ValidationError):
        VerticalPortal(
            portal_id="ramp-bad",
            portal_type=PortalType.STRAIGHT_RAMP,
            from_floor_id="L1",
            to_floor_id="L2",
            length_m=-5.0,  # invalid negative length
        )


def test_vehicle_pitch_state_model() -> None:
    state = VehiclePitchState(
        vehicle_id="veh-01",
        pitch_angle_deg=7.12,
        grade_pct=12.5,
        elevation_m=1.8,
        gravity_force_n=2130.5,
        normal_force_n=16800.0,
        traction_margin=0.45,
        breakover_clearance_m=0.08,
        is_clearance_safe=True,
    )
    assert state.vehicle_id == "veh-01"
    assert state.is_clearance_safe is True

    payload = state.model_dump_json()
    assert "pitch_angle_deg" in payload


def test_elevator_cabin_and_token_models() -> None:
    cabin = ElevatorCabin(
        elevator_id="lift-01",
        current_floor_id="L1",
        target_floor_id="L3",
        state=ElevatorState.HOISTING,
        current_elevation_m=4.2,
        max_weight_kg=4000.0,
        occupied_vehicle_id="veh-02",
        leveling_offset_m=0.005,
    )
    assert cabin.state == ElevatorState.HOISTING
    assert cabin.occupied_vehicle_id == "veh-02"

    token = RampToken(
        token_id="tok-100",  # noqa: S106
        portal_id="ramp-L1-L2",
        vehicle_id="veh-01",
        direction=RampDirection.ASCENDING,
        granted_at_s=100.0,
        expires_at_s=130.0,
        is_active=True,
    )
    assert token.is_active is True
    assert token.direction == RampDirection.ASCENDING


def test_multifloor_route_and_facility_report() -> None:
    wp1 = Waypoint3D(x_m=10.0, y_m=20.0, z_m=0.0, heading_rad=0.0, pitch_deg=0.0)
    wp2 = Waypoint3D(x_m=15.0, y_m=25.0, z_m=3.5, heading_rad=0.5, pitch_deg=7.1)

    route = MultiFloorRoute(
        route_id="route-01",
        vehicle_id="veh-01",
        origin_floor_id="L1",
        destination_floor_id="L2",
        portals_sequence=["ramp-L1-L2"],
        waypoints=[wp1, wp2],
        total_distance_m=42.0,
        estimated_transit_time_s=18.5,
    )
    assert len(route.waypoints) == 2
    assert route.total_distance_m == 42.0

    floor1 = FloorLevel(
        floor_id="L1",
        name="Ground",
        elevation_m=0.0,
        bays_count=20,
        available_bays=5,
    )
    floor2 = FloorLevel(
        floor_id="L2",
        name="Upper Deck",
        elevation_m=3.5,
        bays_count=20,
        available_bays=8,
    )
    portal = VerticalPortal(
        portal_id="ramp-L1-L2",
        portal_type=PortalType.STRAIGHT_RAMP,
        from_floor_id="L1",
        to_floor_id="L2",
        length_m=30.0,
        grade_pct=11.6,
    )

    report = MultiFloorFacilityReport(
        facility_id="facility-central-01",
        facility_name="Central Multi-Story Garage",
        timestamp_s=1700000000.0,
        floors_count=2,
        total_bays=40,
        available_bays=13,
        active_vehicles_count=3,
        ramp_token_events_count=8,
        elevator_cycles_count=4,
        average_vertical_transit_time_s=22.4,
        floors=[floor1, floor2],
        portals=[portal],
        elevators=[],
    )
    assert report.floors_count == 2
    assert len(report.portals) == 1

    serialized = report.model_dump_json()
    restored = MultiFloorFacilityReport.model_validate_json(serialized)
    assert restored.facility_id == "facility-central-01"
    assert len(restored.floors) == 2
