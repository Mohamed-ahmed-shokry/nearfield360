"""Unit tests for HD vector map primitives, domain models, and schemas."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    LaneletType,
    LocalizationConvergence,
    LocalizationState,
    MapTile,
    ObstacleZone,
    ParkingSlotType,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
    PoseHypothesis,
    TurnDirection,
)


def test_point2d_distance_and_tuple() -> None:
    p1 = Point2D(x=3.0, y=4.0)
    p2 = Point2D(x=0.0, y=0.0)
    assert p1.distance_to(p2) == pytest.approx(5.0)
    assert p1.as_tuple() == (3.0, 4.0)


def test_polyline2d_length_and_distance() -> None:
    p1 = Point2D(x=0.0, y=0.0)
    p2 = Point2D(x=10.0, y=0.0)
    p3 = Point2D(x=10.0, y=10.0)
    line = Polyline2D(points=[p1, p2, p3])

    assert line.length_m == pytest.approx(20.0)

    # Point directly on the line
    pt_on = Point2D(x=5.0, y=0.0)
    assert line.distance_to_point(pt_on) == pytest.approx(0.0)

    # Point orthogonal to segment 1
    pt_ortho = Point2D(x=5.0, y=3.0)
    assert line.distance_to_point(pt_ortho) == pytest.approx(3.0)

    # Point past corner
    pt_corner = Point2D(x=13.0, y=0.0)
    assert line.distance_to_point(pt_corner) == pytest.approx(3.0)

    # Closed polyline
    closed = Polyline2D(points=[p1, p2, p3], is_closed=True)
    assert closed.length_m == pytest.approx(20.0 + math.hypot(10.0, 10.0))


def test_polyline2d_requires_minimum_points() -> None:
    with pytest.raises(ValidationError):
        Polyline2D(points=[Point2D(x=0.0, y=0.0)])


def test_lanelet_creation_and_topology() -> None:
    left = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=20.0, y=2.0)])
    right = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=20.0, y=-2.0)])
    center = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=20.0, y=0.0)])

    lane = Lanelet(
        lanelet_id="LANE-101",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
        speed_limit_mps=3.5,
        lanelet_type=LaneletType.PARKING_AISLE,
        turn_direction=TurnDirection.STRAIGHT,
        predecessor_ids=["LANE-100"],
        successor_ids=["LANE-102"],
    )

    assert lane.lanelet_id == "LANE-101"
    assert lane.speed_limit_mps == 3.5
    assert lane.predecessor_ids == ["LANE-100"]
    assert lane.successor_ids == ["LANE-102"]


def test_parking_space_feature() -> None:
    corners = [
        Point2D(x=0.0, y=0.0),
        Point2D(x=2.5, y=0.0),
        Point2D(x=2.5, y=5.0),
        Point2D(x=0.0, y=5.0),
    ]
    space = ParkingSpaceFeature(
        slot_id="SLOT-A01",
        boundary=Polyline2D(points=corners, is_closed=True),
        center=Point2D(x=1.25, y=2.5),
        heading_rad=math.pi / 2,
        slot_type=ParkingSlotType.WPT_INDUCTIVE,
        width_m=2.5,
        length_m=5.0,
    )
    assert space.slot_id == "SLOT-A01"
    assert space.slot_type == ParkingSlotType.WPT_INDUCTIVE
    assert space.boundary.is_closed is True


def test_map_tile_bounds_and_containment() -> None:
    tile = MapTile(
        tile_id="TILE-0-0",
        x_min=0.0,
        y_min=0.0,
        x_max=50.0,
        y_max=50.0,
    )

    assert tile.contains_point(Point2D(x=25.0, y=25.0)) is True
    assert tile.contains_point(Point2D(x=0.0, y=0.0)) is True
    assert tile.contains_point(Point2D(x=50.0, y=25.0)) is False
    assert tile.contains_point(Point2D(x=-1.0, y=25.0)) is False

    with pytest.raises(ValidationError, match="x_max must be strictly greater than x_min"):
        MapTile(tile_id="BAD", x_min=50.0, y_min=0.0, x_max=40.0, y_max=50.0)

    with pytest.raises(ValidationError, match="y_max must be strictly greater than y_min"):
        MapTile(tile_id="BAD", x_min=0.0, y_min=50.0, x_max=50.0, y_max=40.0)


def test_hd_vector_map_serialization() -> None:
    tile = MapTile(tile_id="T1", x_min=0.0, y_min=0.0, x_max=50.0, y_max=50.0)
    hd_map = HDVectorMap(map_id="MAP-AVP-01", facility_id="FACILITY-99", tiles=[tile])

    data = hd_map.model_dump(mode="json")
    assert data["map_id"] == "MAP-AVP-01"
    assert len(data["tiles"]) == 1
    assert "created_timestamp" in data


def test_obstacle_zone() -> None:
    poly = Polyline2D(
        points=[
            Point2D(x=10.0, y=10.0),
            Point2D(x=11.0, y=10.0),
            Point2D(x=11.0, y=11.0),
            Point2D(x=10.0, y=11.0),
        ],
        is_closed=True,
    )
    obs = ObstacleZone(zone_id="PIL-01", boundary=poly, zone_type="PILLAR")
    assert obs.zone_id == "PIL-01"
    assert obs.zone_type == "PILLAR"


def test_localization_state_and_particles() -> None:
    p = PoseHypothesis(x=12.5, y=8.0, yaw_rad=0.35, weight=0.98)
    assert p.x == 12.5
    assert p.weight == 0.98

    loc = LocalizationState(
        timestamp_s=100.5,
        estimated_x=12.45,
        estimated_y=8.02,
        estimated_yaw_rad=0.34,
        std_x=0.04,
        std_y=0.03,
        std_yaw_rad=0.015,
        status=LocalizationConvergence.CONVERGED,
        associated_features_count=8,
        active_tile_id="TILE-0-0",
    )
    assert loc.status == LocalizationConvergence.CONVERGED
    assert loc.std_x == 0.04
