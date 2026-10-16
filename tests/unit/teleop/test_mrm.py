"""Unit tests for MRM safe stop state machine and emergency deceleration physics."""

from __future__ import annotations

from nearfield360.config import TeleopConfig
from nearfield360.teleop.models import MRMStage
from nearfield360.teleop.mrm import MRMEngine


def test_mrm_engine_initial_state() -> None:
    engine = MRMEngine(vehicle_id="veh-test-01")
    assert not engine.is_active
    assert engine.stage == MRMStage.INACTIVE
    assert not engine.parking_brake_engaged
    assert not engine.hazard_lights_active
    assert engine.current_speed_mps == 0.0


def test_mrm_trigger_already_stopped() -> None:
    engine = MRMEngine(vehicle_id="veh-test-01")
    stage = engine.trigger_mrm(
        reason="heartbeat_loss",
        current_time_s=10.0,
        current_speed_mps=0.02,  # below standstill threshold 0.05
    )

    assert stage == MRMStage.STANDSTILL_LOCKED
    assert engine.parking_brake_engaged is True
    assert engine.hazard_lights_active is True
    assert engine.is_active is True


def test_mrm_deceleration_progression_to_standstill() -> None:
    cfg = TeleopConfig(
        mrm_deceleration_mps2=2.0,
        safe_stop_standstill_speed_mps=0.05,
    )
    engine = MRMEngine(vehicle_id="veh-test-01", config=cfg)

    # Trigger MRM at 4.0 m/s at t=10.0s
    engine.trigger_mrm(
        reason="operator_emergency_stop",
        current_time_s=10.0,
        current_speed_mps=4.0,
    )
    assert engine.stage == MRMStage.DECELERATING
    assert engine.current_speed_mps == 4.0
    assert engine.parking_brake_engaged is False

    # Step at t = 11.0s (1 second elapsed, v = 4.0 - 2.0*1.0 = 2.0 m/s)
    stage_t1 = engine.step(current_time_s=11.0)
    assert stage_t1 == MRMStage.DECELERATING
    assert abs(engine.current_speed_mps - 2.0) < 1e-3
    assert abs(engine.distance_traveled_m - 3.0) < 1e-3  # 4*1 - 0.5*2*1^2 = 3.0
    assert engine.parking_brake_engaged is False

    # Step at t = 12.0s (2 seconds elapsed, v = 4.0 - 2.0*2.0 = 0.0 m/s)
    stage_t2 = engine.step(current_time_s=12.0)
    assert stage_t2 == MRMStage.STANDSTILL_LOCKED
    assert engine.current_speed_mps == 0.0
    assert abs(engine.distance_traveled_m - 4.0) < 1e-3  # 4^2 / (2*2) = 4.0 m
    assert engine.parking_brake_engaged is True


def test_mrm_pull_to_shoulder_mode() -> None:
    engine = MRMEngine(vehicle_id="veh-test-01")
    stage = engine.trigger_mrm(
        reason="lane_clearance_override",
        current_time_s=0.0,
        current_speed_mps=3.0,
        pull_to_shoulder=True,
    )
    assert stage == MRMStage.PULLING_TO_SHOULDER


def test_mrm_generate_safe_stop_trajectory() -> None:
    cfg = TeleopConfig(mrm_deceleration_mps2=2.0)
    engine = MRMEngine(vehicle_id="veh-test-01", config=cfg)
    engine.trigger_mrm("network_severed", current_time_s=0.0, current_speed_mps=3.0)

    pose = (10.0, 5.0, 0.0)  # heading along +X
    waypoints = engine.generate_safe_stop_trajectory(current_pose=pose, step_dt_s=0.2)

    assert len(waypoints) >= 2
    # Verify monotonic speed decrease
    speeds = [wp.target_speed_mps for wp in waypoints]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] == 0.0
    # Forward progress along X
    assert waypoints[-1].x_m > pose[0]
    assert abs(waypoints[-1].y_m - pose[1]) < 1e-3


def test_mrm_diagnostic_status() -> None:
    engine = MRMEngine(vehicle_id="veh-status-01")
    engine.trigger_mrm("obstacle_hazard", current_time_s=50.0, current_speed_mps=2.5)
    status = engine.get_status()

    assert status["vehicle_id"] == "veh-status-01"
    assert status["is_active"] is True
    assert status["trigger_reason"] == "obstacle_hazard"
    assert status["initial_speed_mps"] == 2.5
