"""Tests for SAE J2954 autonomous micro-docking inching controller."""

from __future__ import annotations

from nearfield360.config import WPTConfig
from nearfield360.wpt.controller import WPTInchingController
from nearfield360.wpt.models import WPTAlignmentStatus


def test_steering_trim_directionality() -> None:
    """Verify Ackermann steering trims counteract lateral offset and heading error."""
    controller = WPTInchingController()

    # Positive lateral error (vehicle is right of centerline) -> steer left (negative angle)
    steer_right = controller.compute_steering_command(delta_y_m=0.10, heading_err_deg=0.0)
    assert steer_right < 0.0

    # Negative lateral error (vehicle is left of centerline) -> steer right (positive angle)
    steer_left = controller.compute_steering_command(delta_y_m=-0.10, heading_err_deg=0.0)
    assert steer_left > 0.0

    # Heading error trim
    steer_heading = controller.compute_steering_command(delta_y_m=0.0, heading_err_deg=3.0)
    assert steer_heading < 0.0


def test_velocity_command_creep_and_stop() -> None:
    """Verify velocity commands respect inching speed limits and halt when docked."""
    config = WPTConfig(inching_speed_mps=0.08, docked_radius_threshold_mm=30.0)
    controller = WPTInchingController(config=config)

    # Far away -> max inching speed
    v_far = controller.compute_velocity_command(delta_x_m=0.30, delta_r_mm=300.0)
    assert v_far <= 0.08
    assert v_far > 0.05

    # Near target -> crept speed
    v_near = controller.compute_velocity_command(delta_x_m=0.05, delta_r_mm=50.0)
    assert 0.0 < v_near < v_far

    # Docked within 30 mm and 20 mm longitudinal -> zero speed
    v_docked = controller.compute_velocity_command(delta_x_m=0.015, delta_r_mm=20.0)
    assert v_docked == 0.0


def test_simulation_docking_convergence() -> None:
    """Verify vehicle starting within approach envelope converges to docked tolerance."""
    controller = WPTInchingController()

    steps = controller.simulate_docking(
        initial_x_m=0.20,
        initial_y_m=0.04,
        initial_heading_deg=1.5,
        air_gap_mm=170.0,
        max_steps=180,
    )

    assert len(steps) > 5
    last_step = steps[-1]
    assert last_step.docked is True
    assert last_step.coil_offset.alignment_status == WPTAlignmentStatus.DOCKED
    assert last_step.coil_offset.radial_offset_mm <= 30.0
    assert abs(last_step.coil_offset.delta_theta_deg) <= 1.5


def test_step_brake_lock_when_already_docked() -> None:
    """Verify step sets velocity and steering to 0 when starting already docked."""
    controller = WPTInchingController()
    step_record, _x, _y, _heading = controller.step(
        current_x_m=0.01,
        current_y_m=0.01,
        current_heading_deg=0.2,
        air_gap_mm=170.0,
        step_index=0,
    )

    assert step_record.docked is True
    assert step_record.velocity_mps == 0.0
    assert step_record.steering_angle_deg == 0.0
