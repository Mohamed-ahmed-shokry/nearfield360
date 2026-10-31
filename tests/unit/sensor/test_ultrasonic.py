"""Unit tests for automotive ultrasonic acoustic array and time-of-flight simulation."""

from __future__ import annotations

import math

from nearfield360.sensor.ultrasonic import (
    UltrasonicArraySimulator,
    build_default_sensor_layout,
    calculate_speed_of_sound,
    normalize_angle_rad,
)


def test_speed_of_sound_temperature_scaling() -> None:
    c_20 = calculate_speed_of_sound(20.0)
    c_0 = calculate_speed_of_sound(0.0)
    c_neg20 = calculate_speed_of_sound(-20.0)

    assert abs(c_20 - 343.2) < 0.5
    assert abs(c_0 - 331.3) < 0.2
    assert c_neg20 < c_0 < c_20


def test_normalize_angle_rad() -> None:
    assert abs(normalize_angle_rad(0.0)) < 1e-6
    assert abs(normalize_angle_rad(math.pi * 3.0) - (-math.pi)) < 1e-6
    assert abs(normalize_angle_rad(-math.pi * 3.0) - (-math.pi)) < 1e-6


def test_default_sensor_layout_dimensions() -> None:
    layout12 = build_default_sensor_layout(12)
    assert len(layout12) == 12
    assert "uss_front_center_left" in layout12
    assert "uss_rear_center_right" in layout12
    assert "uss_side_front_left" in layout12

    layout8 = build_default_sensor_layout(8)
    assert len(layout8) == 8


def test_ultrasonic_detects_frontal_obstacle() -> None:
    sim = UltrasonicArraySimulator()

    # Place an obstacle directly in front of the vehicle center
    # front bumper is at x=3.85m, y=0.3m for FCL
    obstacle_x = 4.85
    obstacle_y = 0.30
    frame = sim.simulate_echoes(
        obstacles=[(obstacle_x, obstacle_y)],
        ambient_temp_c=20.0,
        timestamp_s=1.0,
    )

    assert len(frame.echoes) == 12

    # Find uss_front_center_left echo
    fcl_echo = next(e for e in frame.echoes if e.sensor_id == "uss_front_center_left")
    assert fcl_echo.in_detection_cone is True
    assert abs(fcl_echo.range_m - 1.0) < 0.05
    assert fcl_echo.echo_amplitude > 0.1

    # Rear sensors should not detect this frontal obstacle
    rr_echo = next(e for e in frame.echoes if e.sensor_id == "uss_rear_right")
    assert rr_echo.in_detection_cone is False


def test_ultrasonic_detects_rear_docking_wall() -> None:
    sim = UltrasonicArraySimulator()

    # Place a rear wall at x = -1.8m
    # rear bumper center is at x = -1.05m
    wall_obstacles = [(-1.8, y) for y in [-0.5, 0.0, 0.5]]
    frame = sim.simulate_echoes(
        obstacles=wall_obstacles,
        ambient_temp_c=15.0,
        timestamp_s=2.5,
    )

    rcl_echo = next(e for e in frame.echoes if e.sensor_id == "uss_rear_center_left")
    assert rcl_echo.in_detection_cone is True
    assert abs(rcl_echo.range_m - 0.75) < 0.1
