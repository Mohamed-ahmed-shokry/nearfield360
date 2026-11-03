"""Unit tests for cyclic hardware heartbeat watchdog and deadline monitor."""

from __future__ import annotations

from nearfield360.config import SafetyConfig
from nearfield360.safety.models import (
    ECUNodeId,
    HeartbeatPing,
    HeartbeatStatus,
)
from nearfield360.safety.watchdog import HeartbeatWatchdog


def test_watchdog_initialization() -> None:
    config = SafetyConfig(heartbeat_timeout_ms=100.0, max_consecutive_misses=3)
    wd = HeartbeatWatchdog(config)

    assert wd.timeout_s == 0.100
    assert wd.max_misses == 3
    assert len(wd.trackers) >= 7
    assert ECUNodeId.VEHICLE_CONTROLLER in wd.trackers


def test_record_heartbeat_measures_latency_and_jitter() -> None:
    wd = HeartbeatWatchdog(timeout_ms=100.0)

    # Initial ping at t=1.0s
    p1 = HeartbeatPing(node_id=ECUNodeId.CAN_GATEWAY, sequence_num=1, timestamp_s=1.0)
    h1 = wd.record_heartbeat(p1)
    assert h1.status == HeartbeatStatus.HEALTHY
    assert h1.sequence_num == 1

    # Second ping at t=1.020s (nominal period is 20ms for CAN_GATEWAY)
    p2 = HeartbeatPing(node_id=ECUNodeId.CAN_GATEWAY, sequence_num=2, timestamp_s=1.020)
    h2 = wd.record_heartbeat(p2)
    assert h2.status == HeartbeatStatus.HEALTHY
    assert abs(h2.latency_ms - 20.0) < 0.1
    assert abs(h2.jitter_ms - 0.0) < 0.1

    # Third ping with severe jitter (period 40ms instead of 20ms)
    p3 = HeartbeatPing(node_id=ECUNodeId.CAN_GATEWAY, sequence_num=3, timestamp_s=1.060)
    h3 = wd.record_heartbeat(p3)
    assert h3.status == HeartbeatStatus.WARN_JITTER
    assert h3.jitter_ms >= 15.0


def test_evaluate_triggers_missed_and_expired() -> None:
    # Timeout 100ms, max consecutive misses 2
    wd = HeartbeatWatchdog(timeout_ms=100.0, max_consecutive_misses=2)

    p1 = HeartbeatPing(node_id=ECUNodeId.VEHICLE_CONTROLLER, sequence_num=1, timestamp_s=1.0)
    wd.record_heartbeat(p1)

    # Check at t=1.05s (within 100ms deadline)
    eval1 = wd.evaluate(current_time_s=1.05)
    vc_health1 = eval1[ECUNodeId.VEHICLE_CONTROLLER.value]
    assert vc_health1.status == HeartbeatStatus.HEALTHY

    # Check at t=1.15s (150ms elapsed, 1st miss)
    eval2 = wd.evaluate(current_time_s=1.15)
    vc_health2 = eval2[ECUNodeId.VEHICLE_CONTROLLER.value]
    assert vc_health2.status == HeartbeatStatus.MISSED
    assert vc_health2.consecutive_misses == 1

    # Check at t=1.25s (250ms elapsed, 2nd miss >= max_misses) -> EXPIRED
    eval3 = wd.evaluate(current_time_s=1.25)
    vc_health3 = eval3[ECUNodeId.VEHICLE_CONTROLLER.value]
    assert vc_health3.status == HeartbeatStatus.EXPIRED
    assert vc_health3.consecutive_misses == 2


def test_check_critical_expiration() -> None:
    wd = HeartbeatWatchdog(timeout_ms=50.0, max_consecutive_misses=1)

    # Ping non-critical teleop supervisor and critical vehicle controller
    wd.record_heartbeat(
        HeartbeatPing(node_id=ECUNodeId.TELEOP_SUPERVISOR, sequence_num=1, timestamp_s=1.0)
    )
    wd.record_heartbeat(
        HeartbeatPing(node_id=ECUNodeId.VEHICLE_CONTROLLER, sequence_num=1, timestamp_s=1.0)
    )

    # Let teleop expire by advancing time past timeout
    _ = wd.evaluate(current_time_s=1.2)
    # If vehicle controller is still refreshed:
    wd.record_heartbeat(
        HeartbeatPing(node_id=ECUNodeId.VEHICLE_CONTROLLER, sequence_num=2, timestamp_s=1.2)
    )
    eval2 = wd.evaluate(current_time_s=1.2)

    # Teleop is expired, but vehicle controller is healthy
    is_crit, _ = wd.check_critical_expiration(eval2)
    assert is_crit is False

    # Now let vehicle controller expire
    eval3 = wd.evaluate(current_time_s=1.5)
    is_crit2, reason2 = wd.check_critical_expiration(eval3)
    assert is_crit2 is True
    assert "vehicle_controller" in reason2
