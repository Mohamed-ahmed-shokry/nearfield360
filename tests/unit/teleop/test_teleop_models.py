"""Unit tests for teleoperation domain models and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.teleop.models import (
    MRMStage,
    NetworkQuality,
    TeleopCommand,
    TeleopHeartbeat,
    TeleopMode,
    TeleopReport,
    TeleopSession,
    TeleopWaypoint,
)


def test_teleop_enums() -> None:
    assert TeleopMode.IDLE_AUTONOMOUS.value == "idle_autonomous"
    assert TeleopMode.TELE_ASSISTANCE.value == "tele_assistance"
    assert TeleopMode.TELE_GUIDANCE.value == "tele_guidance"
    assert TeleopMode.DIRECT_DRIVING.value == "direct_driving"
    assert TeleopMode.MRM_SAFE_STOP.value == "mrm_safe_stop"

    assert MRMStage.INACTIVE.value == "inactive"
    assert MRMStage.DECELERATING.value == "decelerating"
    assert MRMStage.STANDSTILL_LOCKED.value == "standstill_locked"

    assert NetworkQuality.OPTIMAL.value == "optimal"
    assert NetworkQuality.DEGRADED.value == "degraded"
    assert NetworkQuality.CRITICAL.value == "critical"
    assert NetworkQuality.SEVERED.value == "severed"


def test_teleop_waypoint_and_command_serialization() -> None:
    wp = TeleopWaypoint(x_m=12.5, y_m=-4.0, target_speed_mps=1.5)
    cmd = TeleopCommand(
        command_id="cmd-001",
        vehicle_id="veh-teleop-01",
        timestamp_s=100.5,
        mode=TeleopMode.TELE_GUIDANCE,
        steering_angle_rad=0.15,
        target_velocity_mps=1.2,
        emergency_stop=False,
        suggested_waypoints=[wp],
    )

    data = cmd.model_dump(mode="json")
    restored = TeleopCommand.model_validate(data)

    assert restored.command_id == "cmd-001"
    assert restored.vehicle_id == "veh-teleop-01"
    assert restored.mode == TeleopMode.TELE_GUIDANCE
    assert len(restored.suggested_waypoints) == 1
    assert restored.suggested_waypoints[0].x_m == 12.5


def test_teleop_heartbeat_serialization() -> None:
    hb = TeleopHeartbeat(
        vehicle_id="veh-02",
        sequence_number=42,
        timestamp_s=205.0,
        rtt_ms=45.2,
        network_quality=NetworkQuality.OPTIMAL,
        battery_soc=0.85,
        current_speed_mps=2.1,
        active_mrm=False,
    )
    data = hb.model_dump(mode="json")
    restored = TeleopHeartbeat.model_validate(data)

    assert restored.sequence_number == 42
    assert restored.network_quality == NetworkQuality.OPTIMAL
    assert restored.rtt_ms == 45.2


def test_teleop_session_and_report() -> None:
    session = TeleopSession(
        session_id="sess-100",
        vehicle_id="veh-01",
        operator_id="op-desk-4",
        active_mode=TeleopMode.TELE_ASSISTANCE,
        started_at_s=10.0,
        ended_at_s=60.0,
        commands_processed_count=15,
        mrm_stage=MRMStage.INACTIVE,
        is_active=False,
    )

    report = TeleopReport(
        facility_id="garage-teleop-center",
        timestamp_s=70.0,
        active_sessions_count=0,
        completed_sessions_count=1,
        total_interventions_count=1,
        average_rtt_ms=52.3456,
        mrm_executions_count=0,
        sessions=[session],
    )

    data = report.model_dump(mode="json")
    restored = TeleopReport.model_validate(data)

    assert restored.facility_id == "garage-teleop-center"
    assert restored.completed_sessions_count == 1
    assert restored.average_rtt_ms == 52.35  # rounded validator


def test_teleop_models_validation_rejections() -> None:
    with pytest.raises(ValidationError):
        # empty command_id
        TeleopCommand(command_id="", vehicle_id="veh-1", timestamp_s=0.0)

    with pytest.raises(ValidationError):
        # negative sequence number
        TeleopHeartbeat(vehicle_id="veh-1", sequence_number=-1, timestamp_s=0.0)

    with pytest.raises(ValidationError):
        # invalid battery soc (> 1.0)
        TeleopHeartbeat(vehicle_id="veh-1", sequence_number=1, timestamp_s=0.0, battery_soc=1.5)
