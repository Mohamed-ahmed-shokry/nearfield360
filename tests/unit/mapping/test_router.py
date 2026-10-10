"""Unit tests for RoutingGraph and GlobalRouter A* navigation planner."""

from __future__ import annotations

import pytest

from nearfield360.mapping.builder import build_benchmark_garage, build_surface_lot
from nearfield360.mapping.router import (
    GlobalRouter,
    RoutingError,
    RoutingGraph,
    find_nearest_waypoint,
)


def test_routing_graph_construction() -> None:
    garage = build_benchmark_garage()
    graph = RoutingGraph(garage)

    # Entry waypoint should have outgoing edges
    assert "wp_entry" in graph.adjacency
    assert len(graph.adjacency["wp_entry"]) >= 1

    # Exit waypoint has no outgoing edges in one-way garage
    assert len(graph.adjacency["wp_exit"]) == 0

    # Blocked lanes filter
    blocked_graph = RoutingGraph(garage, blocked_lanes={"lane_bottom_0"})
    assert len(blocked_graph.adjacency["wp_entry"]) == 0


def test_find_nearest_waypoint() -> None:
    garage = build_benchmark_garage()
    wp = find_nearest_waypoint(garage, 0.5, 0.2)
    assert wp.waypoint_id == "wp_entry"

    wp_exit = find_nearest_waypoint(garage, 0.2, 29.8)
    assert wp_exit.waypoint_id == "wp_exit"


def test_global_router_nominal_garage_route_to_slot() -> None:
    garage = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    router = GlobalRouter(garage)

    # Route from drop-off at (0, 0, 0) to bay in aisle 0
    route = router.plan(
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="bay_a0_s0",
    )

    assert route.route_id == "route_wp_entry_to_wp_aisle_0_start"
    assert route.total_length_m > 10.0
    assert route.estimated_duration_s > 0.0
    assert len(route.waypoints) > 10
    assert len(route.lane_sequence) >= 2
    assert "lane_bottom_0" in route.lane_sequence

    # Verify waypoint coordinates progress monotonically in s_m
    for i in range(len(route.waypoints) - 1):
        assert route.waypoints[i + 1].s_m >= route.waypoints[i].s_m


def test_global_router_surface_lot_two_way() -> None:
    lot = build_surface_lot()
    router = GlobalRouter(lot)

    # Route from exit back to entrance via two-way lane
    route = router.plan(
        start_pose=(30.0, 10.0, 3.14),
        target_waypoint_id="wp_lot_entry",
    )

    assert route.total_length_m == pytest.approx(30.0, abs=1.0)
    assert len(route.waypoints) > 5


def test_global_router_avoids_blocked_lanes() -> None:
    garage = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    router = GlobalRouter(garage)

    # Block aisle 0 entry
    route = router.plan(
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="bay_a1_s0",
        blocked_lanes={"lane_turn_in_0"},
    )
    # Should successfully route to aisle 1
    assert "lane_turn_in_1" in route.lane_sequence


def test_global_router_unreachable_target_raises_routing_error() -> None:
    garage = build_benchmark_garage(aisle_count=2)
    router = GlobalRouter(garage)

    # Exit waypoint has no outgoing lane back to entry in one-way garage
    with pytest.raises(RoutingError, match="No feasible route found"):
        router.plan(
            start_pose=(0.0, 30.0, 0.0),  # At exit
            target_waypoint_id="wp_entry",
        )


def test_global_router_validates_inputs() -> None:
    garage = build_benchmark_garage()
    router = GlobalRouter(garage)

    with pytest.raises(RoutingError, match="Must specify either"):
        router.plan(start_pose=(0.0, 0.0, 0.0))

    with pytest.raises(RoutingError, match="Target slot nonexistent not found"):
        router.plan(start_pose=(0.0, 0.0, 0.0), target_slot_id="nonexistent")
