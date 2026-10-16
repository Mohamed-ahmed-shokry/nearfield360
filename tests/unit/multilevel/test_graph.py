"""Unit tests for multi-floor topological graph and 3D routing engine."""

import pytest

from nearfield360.multilevel.graph import MultiFloorGraph, build_default_facility_graph
from nearfield360.multilevel.models import FloorLevel, PortalType, VerticalPortal


def test_multifloor_graph_construction_and_errors() -> None:
    graph = MultiFloorGraph()
    floor_l1 = FloorLevel(
        floor_id="L1",
        name="Ground",
        elevation_m=0.0,
        bays_count=20,
        available_bays=5,
    )
    floor_l2 = FloorLevel(
        floor_id="L2",
        name="Level 2",
        elevation_m=3.5,
        bays_count=20,
        available_bays=5,
    )

    with pytest.raises(ValueError, match="Floor L1 must be added before"):
        graph.add_node("n1", "L1", 0.0, 0.0)

    graph.add_floor(floor_l1)
    graph.add_floor(floor_l2)

    graph.add_node("n1", "L1", 0.0, 0.0)
    graph.add_node("n2", "L1", 10.0, 0.0)
    graph.add_node("n3", "L2", 0.0, 0.0)

    assert "L1" in graph.floors
    assert graph.nodes["n1"].z_m == 0.0
    assert graph.nodes["n3"].z_m == 3.5

    with pytest.raises(KeyError, match="Both nodes"):
        graph.add_intra_floor_edge("n1", "missing_node")

    with pytest.raises(ValueError, match="Intra-floor edges must connect nodes on same floor"):
        graph.add_intra_floor_edge("n1", "n3")

    graph.add_intra_floor_edge("n1", "n2", speed_mps=2.0)
    assert len(graph._adj["n1"]) == 1
    assert graph._adj["n1"][0].distance_m == pytest.approx(10.0)
    assert graph._adj["n1"][0].transit_time_s == pytest.approx(5.0)


def test_portal_mismatch_and_elevator_edge() -> None:
    graph = MultiFloorGraph()
    f1 = FloorLevel(
        floor_id="L1", name="Ground", elevation_m=0.0, bays_count=10, available_bays=2
    )
    f2 = FloorLevel(floor_id="L2", name="Deck", elevation_m=3.5, bays_count=10, available_bays=2)
    graph.add_floor(f1)
    graph.add_floor(f2)

    graph.add_node("L1-portal", "L1", 10.0, 10.0)
    graph.add_node("L2-portal", "L2", 10.0, 10.0)

    bad_portal = VerticalPortal(
        portal_id="p-mismatch",
        portal_type=PortalType.STRAIGHT_RAMP,
        from_floor_id="L2",
        to_floor_id="L1",
        length_m=20.0,
        grade_pct=10.0,
    )

    with pytest.raises(ValueError, match="mismatches node floors"):
        graph.add_portal(bad_portal, entry_node="L1-portal", exit_node="L2-portal")

    elevator_portal = VerticalPortal(
        portal_id="lift-01",
        portal_type=PortalType.VEHICLE_ELEVATOR,
        from_floor_id="L1",
        to_floor_id="L2",
        length_m=3.5,
        grade_pct=0.0,
        is_bidirectional=True,
    )
    graph.add_portal(
        elevator_portal,
        entry_node="L1-portal",
        exit_node="L2-portal",
        door_dwell_s=4.0,
        elevator_hoist_speed_mps=1.0,
    )

    assert "lift-01" in graph.portals
    # transit time: 2*4.0 + 3.5/1.0 = 8.0 + 3.5 = 11.5s
    edge = graph._adj["L1-portal"][0]
    assert edge.portal_type == PortalType.VEHICLE_ELEVATOR
    assert edge.transit_time_s == pytest.approx(11.5)


def test_find_route_and_waypoints() -> None:
    graph = build_default_facility_graph()

    route = graph.find_route("L1-entry", "L2-bay-42", vehicle_id="veh-test")
    assert route.origin_floor_id == "L1"
    assert route.destination_floor_id == "L2"
    assert "ramp-L1-to-L2" in route.portals_sequence
    assert len(route.waypoints) >= 4
    assert route.total_distance_m > 0.0
    assert route.estimated_transit_time_s > 0.0

    # Ensure waypoints have non-zero pitch during vertical ramp traversal
    ramp_waypoints = [wp for wp in route.waypoints if abs(wp.pitch_deg) > 0.1]
    assert len(ramp_waypoints) > 0

    # Test error cases
    with pytest.raises(KeyError, match="Origin node missing not found"):
        graph.find_route("missing", "L2-bay-42")

    with pytest.raises(KeyError, match="Destination node missing not found"):
        graph.find_route("L1-entry", "missing")


def test_disconnected_subgraph_raises() -> None:
    graph = MultiFloorGraph()
    f1 = FloorLevel(
        floor_id="L1", name="Ground", elevation_m=0.0, bays_count=10, available_bays=2
    )
    f2 = FloorLevel(floor_id="L2", name="Deck", elevation_m=3.5, bays_count=10, available_bays=2)
    graph.add_floor(f1)
    graph.add_floor(f2)

    graph.add_node("n1", "L1", 0.0, 0.0)
    graph.add_node("n2", "L2", 0.0, 0.0)

    with pytest.raises(ValueError, match="No navigable multi-floor route exists"):
        graph.find_route("n1", "n2")


def test_build_default_facility_graph_cross_floor_routes() -> None:
    graph = build_default_facility_graph()

    # Route from B1 basement to L2 upper deck
    route = graph.find_route("B1-bay-05", "L2-bay-42", vehicle_id="veh-cross")
    assert route.origin_floor_id == "B1"
    assert route.destination_floor_id == "L2"
    assert "ramp-L1-to-B1" in route.portals_sequence
    assert "ramp-L1-to-L2" in route.portals_sequence
    assert route.total_distance_m > 50.0
