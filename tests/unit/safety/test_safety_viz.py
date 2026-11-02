"""Unit tests for tactical ASIL-D safety dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.safety.models import (
    ActuatorSafetyOverride,
    ECUNodeId,
    HeartbeatStatus,
    NodeHealthStatus,
    PlausibilityViolation,
    SafetyState,
    SafetySupervisorReport,
    SafetyTransitionRecord,
    ViolationSeverity,
)
from nearfield360.safety.viz import render_safety_dashboard


def test_render_safety_dashboard_nominal(tmp_path: Path) -> None:
    nodes = {
        ECUNodeId.VEHICLE_CONTROLLER.value: NodeHealthStatus(
            node_id=ECUNodeId.VEHICLE_CONTROLLER,
            last_heartbeat_s=1.0,
            sequence_num=50,
            status=HeartbeatStatus.HEALTHY,
            latency_ms=18.5,
            jitter_ms=1.2,
        ),
        ECUNodeId.PERCEPTION_SURROUND.value: NodeHealthStatus(
            node_id=ECUNodeId.PERCEPTION_SURROUND,
            last_heartbeat_s=0.98,
            sequence_num=10,
            status=HeartbeatStatus.WARN_JITTER,
            latency_ms=95.0,
            jitter_ms=15.0,
        ),
    }

    report = SafetySupervisorReport(
        timestamp_s=1.0,
        active_state=SafetyState.NORMAL_OPERATION,
        node_healths=nodes,
        active_violations=[],
        transition_history=[
            SafetyTransitionRecord(
                from_state=SafetyState.BOOT_SELF_TEST,
                to_state=SafetyState.NORMAL_OPERATION,
                reason="Boot self-test complete",
                timestamp_s=0.5,
            )
        ],
        actuator_override=ActuatorSafetyOverride(is_override_active=False),
    )

    out_file = tmp_path / "safety_hud_nominal.png"
    img = render_safety_dashboard(report, output_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_render_safety_dashboard_emergency_with_violations(tmp_path: Path) -> None:
    nodes = {
        ECUNodeId.VEHICLE_CONTROLLER.value: NodeHealthStatus(
            node_id=ECUNodeId.VEHICLE_CONTROLLER,
            last_heartbeat_s=0.2,
            sequence_num=1,
            status=HeartbeatStatus.EXPIRED,
            latency_ms=800.0,
            jitter_ms=0.0,
            consecutive_misses=5,
        ),
    }

    violations = [
        PlausibilityViolation(
            violation_type="STEERING_RATE_EXCEEDED",
            description="Steering rack slew breach",
            measured_value=1.5,
            limit_value=0.6,
            severity=ViolationSeverity.CRITICAL,
            timestamp_s=1.0,
        )
    ]

    report = SafetySupervisorReport(
        timestamp_s=1.0,
        active_state=SafetyState.EMERGENCY_SAFE_STOP,
        node_healths=nodes,
        active_violations=violations,
        transition_history=[
            SafetyTransitionRecord(
                from_state=SafetyState.NORMAL_OPERATION,
                to_state=SafetyState.EMERGENCY_SAFE_STOP,
                reason="Critical node watchdog expired",
                timestamp_s=0.8,
            )
        ],
        actuator_override=ActuatorSafetyOverride(
            is_override_active=True,
            target_deceleration_mps2=2.5,
            max_permitted_speed_mps=0.0,
            steering_clamped=True,
            emergency_hazard_lights=True,
        ),
    )

    out_file = tmp_path / "safety_hud_emergency.png"
    img = render_safety_dashboard(report, output_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
