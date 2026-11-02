"""Unit tests for ASIL-D safety supervisor state machine and actuator overrides."""

from __future__ import annotations

from nearfield360.config import SafetyConfig
from nearfield360.safety.models import (
    ECUNodeId,
    HeartbeatStatus,
    SafetyState,
    ViolationSeverity,
)
from nearfield360.safety.supervisor import SafetySupervisor


def _seed_nominal_heartbeats(sup: SafetySupervisor, timestamp_s: float, seq: int = 1) -> None:
    """Directly seed healthy heartbeat statuses for all nodes at timestamp."""
    for tracker in sup.watchdog.trackers.values():
        tracker.last_heartbeat_s = timestamp_s
        tracker.last_sequence_num = seq
        tracker.status = HeartbeatStatus.HEALTHY
        tracker.has_received_first = True
        tracker.consecutive_misses = 0
        tracker.jitter_ms = 0.0
        tracker.latency_ms = 0.0


def test_supervisor_boot_sequence_and_nominal_transition() -> None:
    config = SafetyConfig(self_test_dwell_s=0.5)
    sup = SafetySupervisor(config=config, start_time_s=0.0)

    assert str(sup.current_state.value) == SafetyState.BOOT_SELF_TEST.value
    override = sup.get_actuator_override()
    assert override.is_override_active is False

    # Seed pings at t=0.1
    _seed_nominal_heartbeats(sup, timestamp_s=0.1, seq=1)

    # Step at t=0.2 (dwell not yet met)
    rep1 = sup.step(current_time_s=0.2)
    assert str(rep1.active_state.value) == SafetyState.BOOT_SELF_TEST.value

    # Seed pings at t=0.6 and step (dwell met + critical nodes healthy -> NORMAL_OPERATION)
    _seed_nominal_heartbeats(sup, timestamp_s=0.6, seq=2)
    rep2 = sup.step(current_time_s=0.6)
    assert str(rep2.active_state.value) == SafetyState.NORMAL_OPERATION.value
    assert str(sup.current_state.value) == SafetyState.NORMAL_OPERATION.value


def test_supervisor_degraded_perception_and_recovery() -> None:
    config = SafetyConfig(
        self_test_dwell_s=0.1,
        heartbeat_timeout_ms=100.0,
        degraded_speed_limit_mps=2.0,
    )
    sup = SafetySupervisor(config=config, start_time_s=0.0)
    _seed_nominal_heartbeats(sup, timestamp_s=0.1, seq=1)
    sup.step(current_time_s=0.15)
    assert str(sup.current_state.value) == SafetyState.NORMAL_OPERATION.value

    # Critical nodes ping at 0.25, while perception surround misses its 100ms deadline
    for node in (
        ECUNodeId.VEHICLE_CONTROLLER,
        ECUNodeId.CAN_GATEWAY,
        ECUNodeId.TRAJECTORY_PLANNER,
        ECUNodeId.TELEOP_SUPERVISOR,
        ECUNodeId.RADAR_TRANSCEIVER,
        ECUNodeId.ULTRASONIC_ARRAY,
    ):
        tracker = sup.watchdog.trackers[node]
        tracker.last_heartbeat_s = 0.25
        tracker.status = HeartbeatStatus.HEALTHY

    # Step at 0.25 -> perception missed (0.15s > 0.10s timeout) -> DEGRADED_PERCEPTION
    rep = sup.step(current_time_s=0.25)
    assert str(rep.active_state.value) == SafetyState.DEGRADED_PERCEPTION.value
    override = sup.get_actuator_override()
    assert override.is_override_active is True
    assert override.max_permitted_speed_mps == 2.0
    assert override.target_deceleration_mps2 == 0.0

    # Recover all nodes
    _seed_nominal_heartbeats(sup, timestamp_s=0.35, seq=3)
    rep_recovered = sup.step(current_time_s=0.35)
    assert str(rep_recovered.active_state.value) == SafetyState.NORMAL_OPERATION.value


def test_supervisor_critical_watchdog_triggers_safe_stop_and_latch() -> None:
    config = SafetyConfig(self_test_dwell_s=0.1, heartbeat_timeout_ms=100.0)
    sup = SafetySupervisor(config=config, start_time_s=0.0)
    _seed_nominal_heartbeats(sup, timestamp_s=0.1, seq=1)
    sup.step(current_time_s=0.15)
    assert str(sup.current_state.value) == SafetyState.NORMAL_OPERATION.value

    # Time advances to 0.7 without any pings -> vehicle controller expires (0.6s elapsed > 3 * 0.1s)
    rep_stop = sup.step(current_time_s=0.7, kinematics={"wheel_speed_mps": 2.5})
    assert str(rep_stop.active_state.value) == SafetyState.EMERGENCY_SAFE_STOP.value
    override = rep_stop.actuator_override
    assert override.is_override_active is True
    assert override.target_deceleration_mps2 > 0.0
    assert override.steering_clamped is True
    assert override.emergency_hazard_lights is True

    # Vehicle achieves standstill (speed = 0.0) -> transitions to FAIL_SILENT_LATCH
    rep_latch = sup.step(current_time_s=1.2, kinematics={"wheel_speed_mps": 0.0})
    assert str(rep_latch.active_state.value) == SafetyState.FAIL_SILENT_LATCH.value

    # Attempting normal transition while latched must be blocked
    sup.transition_to(SafetyState.NORMAL_OPERATION, "Illegal attempt", timestamp_s=1.3)
    assert str(sup.current_state.value) == SafetyState.FAIL_SILENT_LATCH.value

    # Invalid reset key fails
    assert sup.reset_latch("wrong_key", timestamp_s=1.4) is False
    assert str(sup.current_state.value) == SafetyState.FAIL_SILENT_LATCH.value

    # Valid key resets to BOOT_SELF_TEST
    assert sup.reset_latch("ASIL-D-RESET-KEY-2026", timestamp_s=1.5) is True
    assert str(sup.current_state.value) == SafetyState.BOOT_SELF_TEST.value


def test_supervisor_kinematic_violation_triggers_emergency_stop() -> None:
    config = SafetyConfig(self_test_dwell_s=0.1, max_steering_rate_rads=0.5)
    sup = SafetySupervisor(config=config, start_time_s=0.0)
    _seed_nominal_heartbeats(sup, timestamp_s=0.1, seq=1)
    sup.step(
        current_time_s=0.15,
        kinematics={"steering_angle_rad": 0.0, "wheel_speed_mps": 1.0},
    )
    assert str(sup.current_state.value) == SafetyState.NORMAL_OPERATION.value

    _seed_nominal_heartbeats(sup, timestamp_s=0.20, seq=2)
    # Violent steering rate breach: 0.8 rad in 0.05s = 16 rad/s >> 0.5
    rep = sup.step(
        current_time_s=0.20,
        kinematics={"steering_angle_rad": 0.8, "wheel_speed_mps": 1.0},
    )
    assert str(rep.active_state.value) == SafetyState.EMERGENCY_SAFE_STOP.value
    assert any(v.severity == ViolationSeverity.CRITICAL for v in rep.active_violations)
