"""Unit tests for facility map domain models, bounds, and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.mapping.models import (
    FacilityBounds,
    FacilityLane,
    FacilityMap,
    FacilityObstacle,
    FacilitySlot,
    FacilityWaypoint,
    GlobalRoute,
    LandmarkObservation,
    LaneDirection,
    LocalizationReport,
    PoseEstimate,
    RouteWaypoint,
    SlotBayType,
    SlotReservationStatus,
    WaypointType,
)


def test_facility_bounds_validation() -> None:
    bounds = FacilityBounds(x_min=-10.0, x_max=50.0, y_min=-20.0, y_max=30.0)
    assert bounds.contains(0.0, 0.0) is True
    assert bounds.contains(55.0, 0.0) is False
    assert bounds.contains(0.0, -25.0) is False

    with pytest.raises(ValidationError, match="x_max"):
        FacilityBounds(x_min=10.0, x_max=5.0, y_min=0.0, y_max=10.0)

    with pytest.raises(ValidationError, match="y_max"):
        FacilityBounds(x_min=0.0, x_max=10.0, y_min=10.0, y_max=5.0)


def test_facility_waypoint_and_lane_models() -> None:
    wp1 = FacilityWaypoint(
        waypoint_id="wp_1",
        x=0.0,
        y=0.0,
        heading_rad=0.0,
        waypoint_type=WaypointType.ENTRY,
        connected_waypoints=["wp_2"],
    )
    assert wp1.xy == (0.0, 0.0)

    lane = FacilityLane(
        lane_id="lane_1",
        name="Main Drive",
        centerline_points=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        width_m=3.5,
        speed_limit_mps=2.5,
        direction=LaneDirection.ONE_WAY,
        start_waypoint_id="wp_1",
        end_waypoint_id="wp_2",
    )
    assert lane.length_m == pytest.approx(20.0)

    with pytest.raises(ValidationError, match="at least 2 points"):
        FacilityLane(
            lane_id="lane_bad",
            centerline_points=[(0.0, 0.0)],
            start_waypoint_id="wp_1",
            end_waypoint_id="wp_2",
        )


def test_facility_slot_and_obstacle_models() -> None:
    corners = [(10.0, 2.0), (10.0, 7.0), (12.5, 7.0), (12.5, 2.0)]
    slot = FacilitySlot(
        slot_id="slot_A1",
        slot_type=SlotBayType.PERPENDICULAR,
        corners=corners,
        center=(11.25, 4.5),
        heading_rad=1.57,
        width_m=2.5,
        length_m=5.0,
        access_lane_id="lane_1",
        access_waypoint_id="wp_1",
        status=SlotReservationStatus.VACANT,
    )
    assert slot.slot_id == "slot_A1"
    assert slot.status == SlotReservationStatus.VACANT

    with pytest.raises(ValidationError, match="exactly 4 corner"):
        FacilitySlot(
            slot_id="bad",
            slot_type=SlotBayType.PARALLEL,
            corners=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            center=(0.5, 0.5),
            heading_rad=0.0,
            width_m=2.0,
            length_m=4.0,
            access_lane_id="lane_1",
            access_waypoint_id="wp_1",
        )

    obs = FacilityObstacle(
        obstacle_id="wall_north",
        obstacle_type="wall",
        polygon=[(0.0, 15.0), (30.0, 15.0)],
    )
    assert len(obs.polygon) == 2


def test_pose_estimate_and_uncertainty() -> None:
    pose = PoseEstimate(
        x=5.0,
        y=2.0,
        heading_rad=0.5,
        covariance=[[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.01]],
    )
    # sigma_pos = sqrt(0.04 + 0.09) = sqrt(0.13) ~ 0.360555
    assert pose.position_uncertainty_m == pytest.approx(0.360555, rel=1e-3)
    assert pose.heading_uncertainty_rad == pytest.approx(0.10, rel=1e-3)

    with pytest.raises(ValidationError, match="Covariance matrix must be 3x3"):
        PoseEstimate(x=0.0, y=0.0, heading_rad=0.0, covariance=[[0.1, 0.0], [0.0, 0.1]])

    with pytest.raises(ValidationError, match="non-negative"):
        PoseEstimate(
            x=0.0,
            y=0.0,
            heading_rad=0.0,
            covariance=[[-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.01]],
        )


def test_global_route_and_localization_report() -> None:
    wp = RouteWaypoint(
        x=1.0,
        y=2.0,
        heading_rad=0.0,
        target_speed_mps=2.0,
        curvature=0.05,
        lane_id="lane_1",
        s_m=1.0,
    )
    assert wp.xy == (1.0, 2.0)

    route = GlobalRoute(
        route_id="route_1",
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="slot_A1",
        total_length_m=15.0,
        estimated_duration_s=7.5,
        waypoints=[wp],
        lane_sequence=["lane_1"],
    )
    assert route.total_length_m == 15.0

    obs = LandmarkObservation(
        slot_id="slot_A1",
        observed_center=(11.2, 4.6),
        observed_heading=1.55,
        confidence=0.92,
    )
    assert obs.slot_id == "slot_A1"
    assert obs.confidence == 0.92

    report = LocalizationReport(
        trajectory_length_m=12.5,
        final_pose=PoseEstimate(x=10.0, y=5.0, heading_rad=0.1),
        max_position_uncertainty_m=0.45,
        mean_position_error_m=0.08,
        max_position_error_m=0.18,
        mean_heading_error_rad=0.02,
        max_heading_error_rad=0.05,
        total_landmark_updates=4,
        step_count=50,
    )
    assert report.total_landmark_updates == 4


def test_facility_map_integrity_and_json_roundtrip() -> None:
    bounds = FacilityBounds(x_min=-5.0, x_max=40.0, y_min=-5.0, y_max=30.0)
    wp1 = FacilityWaypoint(
        waypoint_id="wp_entry",
        x=0.0,
        y=0.0,
        heading_rad=0.0,
        waypoint_type=WaypointType.ENTRY,
        connected_waypoints=["wp_aisle"],
    )
    wp2 = FacilityWaypoint(
        waypoint_id="wp_aisle",
        x=20.0,
        y=0.0,
        heading_rad=0.0,
        waypoint_type=WaypointType.INTERSECTION,
        connected_waypoints=["wp_slot"],
    )
    wp3 = FacilityWaypoint(
        waypoint_id="wp_slot",
        x=20.0,
        y=10.0,
        heading_rad=1.57,
        waypoint_type=WaypointType.SLOT_ACCESS,
    )

    lane1 = FacilityLane(
        lane_id="lane_main",
        centerline_points=[(0.0, 0.0), (20.0, 0.0)],
        start_waypoint_id="wp_entry",
        end_waypoint_id="wp_aisle",
    )
    lane2 = FacilityLane(
        lane_id="lane_aisle",
        centerline_points=[(20.0, 0.0), (20.0, 10.0)],
        start_waypoint_id="wp_aisle",
        end_waypoint_id="wp_slot",
    )

    slot = FacilitySlot(
        slot_id="slot_01",
        slot_type=SlotBayType.PERPENDICULAR,
        corners=[(21.0, 10.0), (21.0, 15.0), (23.5, 15.0), (23.5, 10.0)],
        center=(22.25, 12.5),
        heading_rad=1.57,
        width_m=2.5,
        length_m=5.0,
        access_lane_id="lane_aisle",
        access_waypoint_id="wp_slot",
    )

    wall = FacilityObstacle(
        obstacle_id="wall_east",
        polygon=[(35.0, 0.0), (35.0, 25.0)],
    )

    facility = FacilityMap(
        map_id="map_garage_01",
        name="Benchmark Multi-Aisle Garage",
        facility_type="indoor_garage",
        bounds=bounds,
        lanes=[lane1, lane2],
        slots=[slot],
        obstacles=[wall],
        waypoints=[wp1, wp2, wp3],
    )

    # Valid integrity check
    issues = facility.validate_integrity()
    assert issues == []

    # Test lookups
    assert facility.get_slot("slot_01") is not None
    assert facility.get_slot("nonexistent") is None
    assert facility.get_waypoint("wp_entry") is not None
    assert facility.get_lane("lane_main") is not None

    # Test JSON serialization roundtrip
    json_data = facility.to_json()
    loaded = FacilityMap.from_json(json_data)
    assert loaded.map_id == facility.map_id
    assert len(loaded.lanes) == 2
    assert len(loaded.slots) == 1
    assert loaded.slots[0].slot_id == "slot_01"


def test_facility_map_detects_integrity_issues() -> None:
    bounds = FacilityBounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    # Broken lane references nonexistent waypoints and point out of bounds
    lane = FacilityLane(
        lane_id="lane_broken",
        centerline_points=[(0.0, 0.0), (15.0, 0.0)],  # 15.0 is outside bounds
        start_waypoint_id="missing_start",
        end_waypoint_id="missing_end",
    )
    # Broken slot references nonexistent lane and waypoint
    slot = FacilitySlot(
        slot_id="slot_broken",
        slot_type=SlotBayType.PERPENDICULAR,
        corners=[(1.0, 1.0), (1.0, 3.0), (2.0, 3.0), (2.0, 1.0)],
        center=(1.5, 2.0),
        heading_rad=0.0,
        width_m=2.0,
        length_m=4.0,
        access_lane_id="missing_lane",
        access_waypoint_id="missing_wp",
    )
    bad_map = FacilityMap(
        map_id="bad_map",
        name="Bad Map",
        bounds=bounds,
        lanes=[lane],
        slots=[slot],
    )
    issues = bad_map.validate_integrity()
    assert len(issues) >= 4
    assert any("missing_start" in i for i in issues)
    assert any("missing_end" in i for i in issues)
    assert any("exceeds facility bounds" in i for i in issues)
    assert any("missing_lane" in i for i in issues)
