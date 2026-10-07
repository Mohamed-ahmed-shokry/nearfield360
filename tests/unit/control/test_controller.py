"""Unit tests for the PathTrackingController."""

from __future__ import annotations

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.controller import PathTrackingController
from nearfield360.control.models import TrackingErrorState
from nearfield360.planning.models import ManeuverGear, TrajectoryWaypoint


def test_find_target_waypoint_index() -> None:
    controller = PathTrackingController()
    waypoints = [TrajectoryWaypoint(x=float(i), y=0.0, heading_rad=0.0) for i in range(10)]
    idx = controller.find_target_waypoint_index(2.1, 0.1, waypoints, start_idx=0, search_window=5)
    assert idx == 2

    # Forward search window constraint
    idx2 = controller.find_target_waypoint_index(8.0, 0.0, waypoints, start_idx=1, search_window=3)
    # Search window [1, 4] means max index evaluated is 3
    assert idx2 == 3


def test_forward_cross_track_and_heading_error() -> None:
    planner_cfg = ParkingPlannerConfig(wheelbase=2.0)
    controller = PathTrackingController(planner_config=planner_cfg)

    # Reference path along +X (y=0, heading=0)
    target_wp = TrajectoryWaypoint(x=5.0, y=0.0, heading_rad=0.0, velocity=1.0)
    waypoints = [target_wp]

    # Vehicle at x=3.0, y=0.2 (left of path), heading=0.0
    # Front axle is at x=3.0 + 2.0 = 5.0, y=0.2
    error, wp = controller.compute_tracking_error(
        x=3.0,
        y=0.2,
        heading_rad=0.0,
        velocity=1.0,
        waypoints=waypoints,
        active_idx=0,
        gear=ManeuverGear.FORWARD,
    )
    assert error.cross_track_error_m == 0.2
    assert error.heading_error_rad == 0.0

    cmd = controller.compute_control_command(
        current_steer_rad=0.0,
        current_velocity=1.0,
        error_state=error,
        target_wp=wp,
        gear=ManeuverGear.FORWARD,
    )
    # Being left of path must command steering to the right (negative angle)
    assert cmd.steering_angle_rad < 0.0


def test_reverse_cross_track_and_heading_error() -> None:
    planner_cfg = ParkingPlannerConfig(wheelbase=2.0)
    controller = PathTrackingController(planner_config=planner_cfg)

    # Reference reverse path along -X from origin
    target_wp = TrajectoryWaypoint(x=-2.0, y=0.0, heading_rad=0.0, velocity=0.8)
    waypoints = [target_wp]

    # Vehicle rear axle at x=-2.0, y=0.15 (left of path), heading=0.0
    error, wp = controller.compute_tracking_error(
        x=-2.0,
        y=0.15,
        heading_rad=0.0,
        velocity=0.8,
        waypoints=waypoints,
        active_idx=0,
        gear=ManeuverGear.REVERSE,
    )
    assert error.cross_track_error_m == 0.15
    assert error.heading_error_rad == 0.0

    cmd = controller.compute_control_command(
        current_steer_rad=0.0,
        current_velocity=-0.8,
        error_state=error,
        target_wp=wp,
        gear=ManeuverGear.REVERSE,
    )
    # In reverse, turning wheels right (negative steer) steers rear right towards path
    assert cmd.steering_angle_rad < 0.0


def test_steering_rate_and_angle_saturation() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.05, max_steer_rate_rad_s=0.5)
    plan_cfg = ParkingPlannerConfig(max_steer_angle_rad=0.60)
    controller = PathTrackingController(control_config=ctl_cfg, planner_config=plan_cfg)

    target_wp = TrajectoryWaypoint(x=0.0, y=0.0, heading_rad=0.0, velocity=1.0)
    # Huge error requesting max steer
    huge_error = TrackingErrorState(
        cross_track_error_m=5.0,
        heading_error_rad=0.0,
        longitudinal_error_m=0.0,
        speed_error_m_s=0.0,
    )

    cmd = controller.compute_control_command(
        current_steer_rad=0.0,
        current_velocity=1.0,
        error_state=huge_error,
        target_wp=target_wp,
        gear=ManeuverGear.FORWARD,
    )
    # Max change per step: 0.5 * 0.05 = 0.025
    assert abs(cmd.steering_angle_rad - (-0.025)) < 1e-4
    assert cmd.steering_rate_rad_s == -0.5


def test_emergency_braking_command() -> None:
    controller = PathTrackingController()
    wp = TrajectoryWaypoint(x=1.0, y=1.0, heading_rad=0.0, velocity=1.0)
    err = TrackingErrorState(cross_track_error_m=0.0, heading_error_rad=0.0)

    cmd = controller.compute_control_command(
        current_steer_rad=0.1,
        current_velocity=1.0,
        error_state=err,
        target_wp=wp,
        gear=ManeuverGear.FORWARD,
        emergency_brake=True,
    )
    assert cmd.emergency_brake
    assert cmd.target_velocity == 0.0
    assert cmd.acceleration_cmd < -2.0
