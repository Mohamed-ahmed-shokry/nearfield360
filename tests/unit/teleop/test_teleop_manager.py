"""Unit tests for teleoperation manager session coordination and safety command routing."""

from __future__ import annotations

from nearfield360.config import TeleopConfig
from nearfield360.teleop.manager import TeleopManager
from nearfield360.teleop.models import (
    MRMStage,
    TeleopCommand,
    TeleopHeartbeat,
    TeleopMode,
    TeleopWaypoint,
)


def test_teleop_manager_session_lifecycle() -> None:
    mgr = TeleopManager(facility_id="facility-alpha")
    session = mgr.start_session(
        "veh-01", "operator-1", TeleopMode.TELE_GUIDANCE, current_time_s=10.0
    )

    assert session.vehicle_id == "veh-01"
    assert session.is_active is True
    assert session.active_mode == TeleopMode.TELE_GUIDANCE
    assert "veh-01" in mgr.active_sessions

    ended = mgr.end_session("veh-01", current_time_s=50.0)
    assert ended is not None
    assert ended.is_active is False
    assert ended.ended_at_s == 50.0
    assert "veh-01" not in mgr.active_sessions
    assert len(mgr.archived_sessions) == 1


def test_teleop_manager_direct_driving_safety_check() -> None:
    cfg = TeleopConfig(max_rtt_ms=100.0, critical_rtt_ms=250.0)
    mgr = TeleopManager(facility_id="facility-alpha", config=cfg)
    mgr.start_session("veh-01", "operator-1", TeleopMode.TELE_GUIDANCE, current_time_s=0.0)

    # 1. No heartbeats yet -> SEVERED quality -> direct driving rejected
    cmd_direct = TeleopCommand(
        command_id="c1",
        vehicle_id="veh-01",
        timestamp_s=1.0,
        mode=TeleopMode.DIRECT_DRIVING,
        steering_angle_rad=0.05,
        target_velocity_mps=1.5,
    )
    ok, msg = mgr.dispatch_command(cmd_direct, current_time_s=1.0)
    assert ok is False
    assert "link quality is severed" in msg.lower()

    # 2. Provide low-latency heartbeat (40ms) -> OPTIMAL -> direct driving accepted
    hb_opt = TeleopHeartbeat(
        vehicle_id="veh-01",
        sequence_number=1,
        timestamp_s=1.5,
        rtt_ms=40.0,
    )
    mgr.process_heartbeat(hb_opt)
    ok2, msg2 = mgr.dispatch_command(cmd_direct, current_time_s=1.5)
    assert ok2 is True
    assert "dispatched successfully" in msg2

    # 3. High latency heartbeat (150ms) -> DEGRADED -> direct driving rejected
    hb_deg = TeleopHeartbeat(
        vehicle_id="veh-01",
        sequence_number=2,
        timestamp_s=2.0,
        rtt_ms=160.0,
    )
    mgr.process_heartbeat(hb_deg)
    ok3, msg3 = mgr.dispatch_command(cmd_direct, current_time_s=2.0)
    assert ok3 is False
    assert "link quality is degraded" in msg3.lower()


def test_teleop_manager_emergency_stop_command() -> None:
    mgr = TeleopManager(facility_id="facility-alpha")
    mgr.start_session("veh-01", "op-1", TeleopMode.TELE_GUIDANCE, current_time_s=0.0)

    cmd_estop = TeleopCommand(
        command_id="c-estop",
        vehicle_id="veh-01",
        timestamp_s=5.0,
        emergency_stop=True,
    )
    ok, msg = mgr.dispatch_command(cmd_estop, current_time_s=5.0, current_speed_mps=3.0)
    assert ok is True
    assert "Emergency stop executed" in msg

    session = mgr.active_sessions["veh-01"]
    assert session.active_mode == TeleopMode.MRM_SAFE_STOP
    assert session.mrm_stage == MRMStage.DECELERATING

    # Step until standstill
    mgr.step(current_time_s=10.0)
    assert mgr.active_sessions["veh-01"].mrm_stage == MRMStage.STANDSTILL_LOCKED

    # Further commands should be rejected due to MRM lockdown
    cmd_guidance = TeleopCommand(
        command_id="c-guide",
        vehicle_id="veh-01",
        timestamp_s=11.0,
        mode=TeleopMode.TELE_GUIDANCE,
        suggested_waypoints=[TeleopWaypoint(x_m=5.0, y_m=2.0)],
    )
    ok2, msg2 = mgr.dispatch_command(cmd_guidance, current_time_s=11.0)
    assert ok2 is False
    assert "locked in Minimum Risk Maneuver" in msg2


def test_teleop_manager_heartbeat_loss_watchdog() -> None:
    cfg = TeleopConfig(heartbeat_timeout_s=1.0, auto_mrm_on_heartbeat_loss=True)
    mgr = TeleopManager(facility_id="facility-alpha", config=cfg)
    mgr.start_session("veh-01", "op-1", TeleopMode.TELE_GUIDANCE, current_time_s=0.0)

    # Initial good heartbeat at t=1.0
    mgr.process_heartbeat(
        TeleopHeartbeat(vehicle_id="veh-01", sequence_number=0, timestamp_s=1.0, rtt_ms=30.0)
    )

    # Step at t=1.5 (0.5s elapsed, within 1.0s timeout)
    mgr.step(current_time_s=1.5)
    assert mgr.active_sessions["veh-01"].active_mode.value == "tele_guidance"

    # Step at t=2.2 (1.2s elapsed since last heartbeat -> exceeds 1.0s timeout)
    mgr.step(current_time_s=2.2)
    assert mgr.active_sessions["veh-01"].active_mode.value == "mrm_safe_stop"
    assert mgr.mrm_engines["veh-01"].is_active is True


def test_teleop_manager_report_generation() -> None:
    mgr = TeleopManager(facility_id="facility-alpha")
    mgr.start_session("veh-01", "op-1", TeleopMode.TELE_GUIDANCE, current_time_s=0.0)
    mgr.process_heartbeat(
        TeleopHeartbeat(vehicle_id="veh-01", sequence_number=0, timestamp_s=1.0, rtt_ms=45.0)
    )
    mgr.end_session("veh-01", current_time_s=10.0)

    report = mgr.generate_report(current_time_s=15.0)
    assert report.facility_id == "facility-alpha"
    assert report.completed_sessions_count == 1
    assert report.active_sessions_count == 0
    assert report.total_interventions_count == 1
    assert report.average_rtt_ms == 45.0
