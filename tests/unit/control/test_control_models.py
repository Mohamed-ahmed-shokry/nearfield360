"""Unit tests for closed-loop control domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.control.models import (
    ControlCommand,
    ControlPerformanceKPIs,
    ExecutionStatus,
    ManeuverExecutionReport,
    ManeuverExecutionStep,
    TrackingErrorState,
    VehicleSimState,
)
from nearfield360.planning.models import ManeuverGear, ManeuverPhase


def test_control_command_defaults_and_validation() -> None:
    cmd = ControlCommand(
        steering_angle_rad=0.25,
        target_velocity=1.0,
        acceleration_cmd=0.5,
        gear=ManeuverGear.FORWARD,
    )
    assert cmd.steering_angle_rad == 0.25
    assert cmd.steering_rate_rad_s == 0.0
    assert cmd.target_velocity == 1.0
    assert cmd.acceleration_cmd == 0.5
    assert cmd.gear == ManeuverGear.FORWARD
    assert not cmd.emergency_brake


def test_tracking_error_state() -> None:
    error = TrackingErrorState(
        cross_track_error_m=0.04,
        heading_error_rad=-0.02,
        longitudinal_error_m=0.10,
        speed_error_m_s=0.05,
        closest_waypoint_idx=5,
    )
    assert error.cross_track_error_m == 0.04
    assert error.heading_error_rad == -0.02
    assert error.longitudinal_error_m == 0.10
    assert error.speed_error_m_s == 0.05
    assert error.closest_waypoint_idx == 5


def test_vehicle_sim_state_properties() -> None:
    state = VehicleSimState(
        x=2.5,
        y=1.0,
        heading_rad=0.35,
        velocity=0.8,
        acceleration=0.2,
        steer_angle_rad=0.15,
        gear=ManeuverGear.FORWARD,
    )
    assert state.pose == (2.5, 1.0, 0.35)
    assert state.xy == (2.5, 1.0)
    assert state.velocity == 0.8
    assert state.gear == ManeuverGear.FORWARD


def test_maneuver_execution_step() -> None:
    state = VehicleSimState(
        x=0.0,
        y=0.0,
        heading_rad=0.0,
        velocity=0.5,
        acceleration=0.0,
        steer_angle_rad=0.0,
        gear=ManeuverGear.REVERSE,
    )
    cmd = ControlCommand(
        steering_angle_rad=0.1,
        target_velocity=-0.5,
        acceleration_cmd=0.0,
        gear=ManeuverGear.REVERSE,
    )
    error = TrackingErrorState(
        cross_track_error_m=0.02,
        heading_error_rad=0.01,
        longitudinal_error_m=0.05,
    )
    step = ManeuverExecutionStep(
        t=0.1,
        step_index=2,
        segment_index=1,
        phase=ManeuverPhase.DOCK,
        vehicle_state=state,
        command=cmd,
        error=error,
        nearest_obstacle_distance_m=1.85,
        status=ExecutionStatus.ACTIVE,
    )
    assert step.t == 0.1
    assert step.phase == ManeuverPhase.DOCK
    assert step.status == ExecutionStatus.ACTIVE
    assert step.nearest_obstacle_distance_m == 1.85


def test_control_kpis_and_execution_report() -> None:
    kpis = ControlPerformanceKPIs(
        max_cross_track_error_m=0.08,
        mean_cross_track_error_m=0.03,
        rmse_cross_track_error_m=0.04,
        max_heading_error_rad=0.05,
        mean_heading_error_rad=0.02,
        max_lateral_accel_m_s2=0.45,
        max_jerk_m_s3=0.80,
        docking_error_x_m=0.02,
        docking_error_y_m=0.03,
        docking_error_heading_rad=0.01,
        docking_distance_m=0.036,
        is_docked_successfully=True,
    )
    assert kpis.is_docked_successfully
    assert kpis.max_cross_track_error_m == 0.08

    report = ManeuverExecutionReport(
        plan_id="plan-001",
        slot_id="slot-left-01",
        status=ExecutionStatus.COMPLETED,
        total_steps=120,
        duration_s=6.0,
        kpis=kpis,
        message="Maneuver completed successfully",
    )
    assert report.status == ExecutionStatus.COMPLETED
    assert report.total_steps == 120
    dump = report.model_dump()
    assert dump["plan_id"] == "plan-001"
    assert dump["kpis"]["is_docked_successfully"] is True


def test_control_models_reject_inf_nan() -> None:
    with pytest.raises(ValidationError):
        ControlCommand(steering_angle_rad=float("nan"))

    with pytest.raises(ValidationError):
        VehicleSimState(x=float("inf"), y=0.0, heading_rad=0.0)
