from __future__ import annotations

import math

import pytest

from nearfield360.config import ParkingPlannerConfig
from nearfield360.planning.kinematics import (
    AckermannVehicle,
    normalize_angle,
    plan_two_circle_reeds_shepp,
)
from nearfield360.planning.models import ManeuverGear


def test_normalize_angle() -> None:
    assert normalize_angle(0.0) == 0.0
    assert abs(normalize_angle(math.pi) - math.pi) < 1e-6
    assert abs(normalize_angle(3.0 * math.pi) - math.pi) < 1e-6
    assert abs(normalize_angle(-3.0 * math.pi) - (-math.pi)) < 1e-6
    assert abs(normalize_angle(2.0 * math.pi) - 0.0) < 1e-6


def test_ackermann_vehicle_geometry_and_footprint() -> None:
    config = ParkingPlannerConfig(
        wheelbase=2.7,
        front_overhang=0.9,
        rear_overhang=0.9,
        vehicle_width=1.8,
        max_steer_angle_rad=0.65,
    )
    vehicle = AckermannVehicle(config)
    assert vehicle.wheelbase == 2.7
    assert vehicle.min_turn_radius > 3.0
    assert vehicle.max_curvature < 0.35

    # At origin facing along +X axis (heading = 0)
    # Rear axle is at (0, 0)
    # Front-left corner: x = 2.7 + 0.9 = 3.6, y = +0.9
    # Rear-right corner: x = -0.9, y = -0.9
    corners = vehicle.compute_footprint_polygon(0.0, 0.0, 0.0)
    assert len(corners) == 4
    fl, rl, rr, fr = corners
    assert fl == (3.6, 0.9)
    assert rl == (-0.9, 0.9)
    assert rr == (-0.9, -0.9)
    assert fr == (3.6, -0.9)


def test_footprint_rotation() -> None:
    config = ParkingPlannerConfig(
        wheelbase=2.0, front_overhang=0.5, rear_overhang=0.5, vehicle_width=2.0
    )
    vehicle = AckermannVehicle(config)
    # Heading = pi/2 (pointing along +Y)
    corners = vehicle.compute_footprint_polygon(0.0, 0.0, math.pi / 2)
    fl, _rl, _rr, _fr = corners
    # (x_body, y_body) -> (-y_body, x_body)
    # Front-left: body=(2.5, 1.0) -> (-1.0, 2.5)
    assert abs(fl[0] - (-1.0)) < 1e-3
    assert abs(fl[1] - 2.5) < 1e-3


def test_generate_straight_forward_and_reverse() -> None:
    vehicle = AckermannVehicle()
    wps_fwd = vehicle.generate_straight((0.0, 0.0, 0.0), length_m=2.0, gear=ManeuverGear.FORWARD)
    assert len(wps_fwd) > 10
    assert wps_fwd[0].x == 0.0
    assert abs(wps_fwd[-1].x - 2.0) < 1e-3
    assert wps_fwd[-1].distance_m == pytest.approx(2.0, abs=1e-3)

    wps_rev = vehicle.generate_straight((0.0, 0.0, 0.0), length_m=2.0, gear=ManeuverGear.REVERSE)
    assert wps_rev[0].x == 0.0
    assert abs(wps_rev[-1].x - (-2.0)) < 1e-3
    assert wps_rev[-1].distance_m == pytest.approx(2.0, abs=1e-3)


def test_generate_arc_turning() -> None:
    vehicle = AckermannVehicle()
    # Left turn in forward gear
    wps = vehicle.generate_arc(
        (0.0, 0.0, 0.0),
        curvature=0.2,
        arc_length_m=3.0,
        gear=ManeuverGear.FORWARD,
    )
    assert len(wps) > 10
    assert wps[0].heading_rad == 0.0
    assert wps[-1].heading_rad > 0.0  # Turned left
    assert wps[-1].y > 0.0

    # Right turn in reverse gear
    wps_rev = vehicle.generate_arc(
        (0.0, 0.0, 0.0),
        curvature=-0.2,
        arc_length_m=3.0,
        gear=ManeuverGear.REVERSE,
    )
    assert wps_rev[-1].x < 0.0


def test_two_circle_reeds_shepp_planner() -> None:
    vehicle = AckermannVehicle()
    # Start at (0, 0, 0), target with lateral offset -2.0m and parallel heading
    start = (0.0, 0.0, 0.0)
    target = (-4.0, -2.0, 0.0)
    result = plan_two_circle_reeds_shepp(start, target, radius=4.0, vehicle=vehicle)
    assert result is not None
    assert len(result) == 2
    g1, _wps1 = result[0]
    g2, wps2 = result[1]
    assert g1 == ManeuverGear.REVERSE
    assert g2 == ManeuverGear.REVERSE
    # Final waypoint of second arc should be close to target
    final_wp = wps2[-1]
    assert abs(final_wp.y - (-2.0)) < 0.2
    assert abs(final_wp.heading_rad - 0.0) < 0.1
