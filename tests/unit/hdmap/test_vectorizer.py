"""Unit tests for BEV vectorizer, polyline simplification, and lanelet extraction."""

from __future__ import annotations

import math

import pytest

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import LaneletType, ParkingSlotType, Point2D, TurnDirection
from nearfield360.hdmap.vectorizer import BEVVectorizer, douglas_peucker


def test_douglas_peucker_collinear_points() -> None:
    # 5 collinear points along y = 0
    pts = [
        Point2D(x=0.0, y=0.0),
        Point2D(x=2.0, y=0.0),
        Point2D(x=5.0, y=0.0),
        Point2D(x=8.0, y=0.0),
        Point2D(x=10.0, y=0.0),
    ]
    simplified = douglas_peucker(pts, epsilon=0.1)
    assert len(simplified) == 2
    assert simplified[0] == pts[0]
    assert simplified[-1] == pts[-1]


def test_douglas_peucker_sharp_corner_preservation() -> None:
    # Corner at (5, 5) with intermediate collinear points on each side
    pts = [
        Point2D(x=0.0, y=0.0),
        Point2D(x=2.5, y=2.52),
        Point2D(x=5.0, y=5.0),
        Point2D(x=7.5, y=2.48),
        Point2D(x=10.0, y=0.0),
    ]
    simplified = douglas_peucker(pts, epsilon=0.2)
    assert len(simplified) == 3
    assert simplified[1].x == pytest.approx(5.0)
    assert simplified[1].y == pytest.approx(5.0)


def test_simplify_points_closed_loop() -> None:
    vectorizer = BEVVectorizer(HDMapConfig(douglas_peucker_epsilon_m=0.10))
    coords = [
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (10.0, 10.0),
        (5.0, 10.0),
        (0.0, 10.0),
        (0.0, 5.0),
        (0.0, 0.0),
    ]
    poly = vectorizer.simplify_points(coords, is_closed=True)
    assert poly.is_closed is True
    # Rectangular loop should simplify to 4 or 5 primary vertices
    assert len(poly.points) <= 5


def test_extract_centerline_parallel_boundaries() -> None:
    vectorizer = BEVVectorizer()
    left_coords = [(0.0, 2.0), (10.0, 2.0), (20.0, 2.0)]
    right_coords = [(0.0, -2.0), (10.0, -2.0), (20.0, -2.0)]

    left = vectorizer.simplify_points(left_coords)
    right = vectorizer.simplify_points(right_coords)

    center = vectorizer.extract_centerline(left, right, num_samples=10)
    assert center.length_m == pytest.approx(20.0)
    for p in center.points:
        assert p.y == pytest.approx(0.0, abs=1e-3)


def test_extract_centerline_tapered_corridor() -> None:
    vectorizer = BEVVectorizer()
    left_coords = [(0.0, 3.0), (20.0, 1.5)]
    right_coords = [(0.0, -3.0), (20.0, -1.5)]

    left = vectorizer.simplify_points(left_coords)
    right = vectorizer.simplify_points(right_coords)

    center = vectorizer.extract_centerline(left, right)
    for p in center.points:
        assert p.y == pytest.approx(0.0, abs=1e-3)


def test_build_lanelet_construction() -> None:
    vectorizer = BEVVectorizer(HDMapConfig(default_speed_limit_mps=2.5))
    left = [(0.0, 2.0), (25.0, 2.0)]
    right = [(0.0, -2.0), (25.0, -2.0)]

    lanelet = vectorizer.build_lanelet(
        lanelet_id="L-RAMP-01",
        left_pts=left,
        right_pts=right,
        speed_limit_mps=2.0,
        lanelet_type=LaneletType.RAMP,
        turn_direction=TurnDirection.LEFT,
        predecessor_ids=["L-ENTRY"],
        successor_ids=["L-AISLE-A"],
    )

    assert lanelet.lanelet_id == "L-RAMP-01"
    assert lanelet.speed_limit_mps == 2.0
    assert lanelet.lanelet_type == LaneletType.RAMP
    assert lanelet.turn_direction == TurnDirection.LEFT
    assert lanelet.predecessor_ids == ["L-ENTRY"]
    assert lanelet.successor_ids == ["L-AISLE-A"]


def test_extract_parking_space_dimensions_and_heading() -> None:
    vectorizer = BEVVectorizer()
    # 2.5m wide x 5.0m long rectangular slot oriented along Y axis
    corners = [
        (0.0, 0.0),
        (0.0, 5.0),
        (2.5, 5.0),
        (2.5, 0.0),
    ]
    space = vectorizer.extract_parking_space(
        slot_id="BAY-42",
        corners=corners,
        slot_type=ParkingSlotType.EV_CHARGING,
    )

    assert space.slot_id == "BAY-42"
    assert space.slot_type == ParkingSlotType.EV_CHARGING
    assert space.length_m == pytest.approx(5.0)
    assert space.width_m == pytest.approx(2.5)
    assert space.center.x == pytest.approx(1.25)
    assert space.center.y == pytest.approx(2.5)
    # Heading points along longer edge (Y direction: pi/2)
    assert space.heading_rad == pytest.approx(math.pi / 2, abs=1e-3)


def test_extract_parking_space_rejects_non_quad() -> None:
    vectorizer = BEVVectorizer()
    with pytest.raises(ValueError, match="requires exactly 4 corner vertices"):
        vectorizer.extract_parking_space(
            slot_id="BAD",
            corners=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        )
