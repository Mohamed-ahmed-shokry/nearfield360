"""Unit tests for Multi-Agent Spatiotemporal Deconfliction Engine."""

from __future__ import annotations

import pytest

from nearfield360.config import FleetConfig
from nearfield360.fleet.deconfliction import DeconflictionEngine, build_trajectory_profile
from nearfield360.fleet.models import ConflictResolutionAction
from nearfield360.mapping.models import GlobalRoute, RouteWaypoint


def _create_mock_route(waypoints: list[tuple[float, float]]) -> GlobalRoute:
    r_wps = [
        RouteWaypoint(
            x=x,
            y=y,
            heading_rad=0.0,
            target_speed_mps=1.0,
            lane_id="lane_0",
        )
        for x, y in waypoints
    ]
    return GlobalRoute(
        route_id="r_test",
        start_pose=(waypoints[0][0], waypoints[0][1], 0.0),
        total_length_m=float(len(waypoints)),
        estimated_duration_s=float(len(waypoints)),
        waypoints=r_wps,
    )


def test_build_trajectory_profile() -> None:
    route = _create_mock_route([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
    profile = build_trajectory_profile("veh_01", route, start_time_s=10.0, temporal_headway_s=2.0)

    assert profile.vehicle_id == "veh_01"
    assert len(profile.waypoints) == 3
    assert profile.waypoints[0].t_expected_s == 10.0
    assert profile.waypoints[1].t_expected_s == 12.0
    assert profile.waypoints[2].t_expected_s == 14.0
    assert profile.total_duration_s == 4.0


def test_deconfliction_detects_intersection() -> None:
    # Vehicle 1 moving East: (0, 5) -> (5, 5) -> (10, 5) at t=0
    route1 = _create_mock_route([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)])
    p1 = build_trajectory_profile("v1", route1, start_time_s=0.0)

    # Vehicle 2 moving North: (5, 0) -> (5, 5) -> (5, 10) at t=0
    route2 = _create_mock_route([(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)])
    p2 = build_trajectory_profile("v2", route2, start_time_s=0.0)

    engine = DeconflictionEngine(FleetConfig(conflict_radius_m=2.0, temporal_headway_s=3.0))
    conflicts = engine.detect_conflicts([p1, p2])

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.primary_vehicle_id == "v1"
    assert c.secondary_vehicle_id == "v2"
    assert pytest.approx(c.conflict_location[0], abs=0.1) == 5.0
    assert pytest.approx(c.conflict_location[1], abs=0.1) == 5.0
    # Both arrive at (5, 5) around t=5.0
    assert c.action_primary == ConflictResolutionAction.PROCEED
    assert c.action_secondary == ConflictResolutionAction.YIELD_HOLD


def test_deconfliction_no_conflict_when_temporally_separated() -> None:
    route1 = _create_mock_route([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)])
    p1 = build_trajectory_profile("v1", route1, start_time_s=0.0)

    # Vehicle 2 traverses (5, 5) much later (start_time=20.0s)
    route2 = _create_mock_route([(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)])
    p2 = build_trajectory_profile("v2", route2, start_time_s=20.0)

    engine = DeconflictionEngine(FleetConfig(conflict_radius_m=2.0, temporal_headway_s=2.0))
    conflicts = engine.detect_conflicts([p1, p2])

    assert len(conflicts) == 0


def test_adjust_trajectory_for_yield() -> None:
    route = _create_mock_route([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
    p = build_trajectory_profile("v_yield", route, start_time_s=0.0)

    engine = DeconflictionEngine(FleetConfig(conflict_radius_m=2.0, temporal_headway_s=3.0))
    # Manually create conflict at (5, 0)
    from nearfield360.fleet.models import ConflictPoint

    dummy_conflict = ConflictPoint(
        conflict_id="conf_test",
        primary_vehicle_id="v_other",
        secondary_vehicle_id="v_yield",
        conflict_location=(5.0, 0.0),
        earliest_arrival_s=5.0,
        latest_departure_s=8.0,
        distance_between_trajectories_m=0.0,
    )

    adjusted = engine.adjust_trajectory_for_yield(p, dummy_conflict, hold_duration_s=4.0)
    assert adjusted.total_duration_s == p.total_duration_s + 4.0
    # Waypoint past (5, 0) should be delayed by 4.0s
    assert adjusted.waypoints[-1].t_expected_s == p.waypoints[-1].t_expected_s + 4.0
