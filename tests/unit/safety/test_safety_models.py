"""Unit tests for ISO 26262 ASIL-D functional safety domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.safety.models import (
    ActuatorSafetyOverride,
    ECUNodeId,
    HeartbeatPing,
    HeartbeatStatus,
    NodeHealthStatus,
    PlausibilityViolation,
    SafetyState,
    SafetySupervisorReport,
    SafetyTransitionRecord,
    ViolationSeverity,
)


def test_safety_enums() -> None:
    assert SafetyState.BOOT_SELF_TEST.value == "boot_self_test"
    assert SafetyState.NORMAL_OPERATION.value == "normal_operation"
    assert SafetyState.DEGRADED_PERCEPTION.value == "degraded_perception"
    assert SafetyState.EMERGENCY_SAFE_STOP.value == "emergency_safe_stop"
    assert SafetyState.FAIL_SILENT_LATCH.value == "fail_silent_latch"

    assert ECUNodeId.PERCEPTION_SURROUND.value == "perception_surround"
    assert ECUNodeId.RADAR_TRANSCEIVER.value == "radar_transceiver"
    assert ECUNodeId.CAN_GATEWAY.value == "can_gateway"

    assert HeartbeatStatus.HEALTHY.value == "healthy"
    assert HeartbeatStatus.EXPIRED.value == "expired"

    assert ViolationSeverity.CRITICAL.value == "critical"


def test_heartbeat_ping_validation() -> None:
    ping = HeartbeatPing(
        node_id=ECUNodeId.CAN_GATEWAY,
        sequence_num=42,
        timestamp_s=12.345,
    )
    assert ping.node_id == ECUNodeId.CAN_GATEWAY
    assert ping.sequence_num == 42
    assert ping.timestamp_s == 12.345

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        HeartbeatPing(node_id=ECUNodeId.CAN_GATEWAY, sequence_num=-1, timestamp_s=1.0)


def test_node_health_status_and_violation() -> None:
    health = NodeHealthStatus(
        node_id=ECUNodeId.RADAR_TRANSCEIVER,
        last_heartbeat_s=10.0,
        sequence_num=100,
        status=HeartbeatStatus.HEALTHY,
        latency_ms=22.5,
        jitter_ms=1.2,
        consecutive_misses=0,
    )
    assert health.node_id == ECUNodeId.RADAR_TRANSCEIVER
    assert health.status == HeartbeatStatus.HEALTHY

    violation = PlausibilityViolation(
        violation_type="steering_rate_exceeded",
        description="Steering velocity 0.85 rad/s exceeded limit 0.60 rad/s",
        measured_value=0.85,
        limit_value=0.60,
        severity=ViolationSeverity.CRITICAL,
        timestamp_s=10.1,
    )
    assert violation.severity == ViolationSeverity.CRITICAL
    assert violation.measured_value == 0.85


def test_safety_supervisor_report_json_roundtrip() -> None:
    override = ActuatorSafetyOverride(
        is_override_active=True,
        target_deceleration_mps2=2.5,
        max_permitted_speed_mps=0.0,
        steering_clamped=True,
        emergency_hazard_lights=True,
    )
    transition = SafetyTransitionRecord(
        from_state=SafetyState.NORMAL_OPERATION,
        to_state=SafetyState.EMERGENCY_SAFE_STOP,
        reason="Heartbeat timeout on vehicle_controller",
        timestamp_s=15.0,
    )
    node_health = NodeHealthStatus(
        node_id=ECUNodeId.VEHICLE_CONTROLLER,
        status=HeartbeatStatus.EXPIRED,
        consecutive_misses=4,
    )

    report = SafetySupervisorReport(
        timestamp_s=15.0,
        active_state=SafetyState.EMERGENCY_SAFE_STOP,
        node_healths={ECUNodeId.VEHICLE_CONTROLLER.value: node_health},
        active_violations=[],
        transition_history=[transition],
        actuator_override=override,
    )

    json_data = report.model_dump_json()
    restored = SafetySupervisorReport.model_validate_json(json_data)

    assert restored.active_state == SafetyState.EMERGENCY_SAFE_STOP
    assert restored.actuator_override.is_override_active is True
    assert restored.actuator_override.target_deceleration_mps2 == 2.5
    assert len(restored.transition_history) == 1
    assert restored.transition_history[0].from_state == SafetyState.NORMAL_OPERATION
    assert (
        restored.node_healths[ECUNodeId.VEHICLE_CONTROLLER.value].status == HeartbeatStatus.EXPIRED
    )
