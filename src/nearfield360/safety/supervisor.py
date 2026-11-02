"""ISO 26262 ASIL-D functional safety supervisor and state machine.

Orchestrates vehicle safety states (BOOT_SELF_TEST, NORMAL_OPERATION,
DEGRADED_PERCEPTION, EMERGENCY_SAFE_STOP, FAIL_SILENT_LATCH), oversees cyclic
watchdog heartbeats and plausibility validators, and enforces fail-safe actuator
overrides (emergency deceleration, speed capping, steering clamping).
"""

from __future__ import annotations

import logging
from typing import Any

from nearfield360.config import SafetyConfig
from nearfield360.safety.models import (
    ActuatorSafetyOverride,
    ECUNodeId,
    HeartbeatStatus,
    PlausibilityViolation,
    SafetyState,
    SafetySupervisorReport,
    SafetyTransitionRecord,
    ViolationSeverity,
)
from nearfield360.safety.plausibility import PlausibilityChecker
from nearfield360.safety.watchdog import HeartbeatWatchdog

logger = logging.getLogger(__name__)


class SafetySupervisor:
    """ASIL-D functional safety supervisor state machine and actuator guard."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        watchdog: HeartbeatWatchdog | None = None,
        plausibility: PlausibilityChecker | None = None,
        start_time_s: float = 0.0,
    ) -> None:
        self.config = config or SafetyConfig()
        self.watchdog = watchdog or HeartbeatWatchdog(self.config)
        self.plausibility = plausibility or PlausibilityChecker(self.config)

        self._state: SafetyState = SafetyState.BOOT_SELF_TEST
        self._boot_start_time_s: float = start_time_s
        self._last_time_s: float = start_time_s
        self._transition_history: list[SafetyTransitionRecord] = [
            SafetyTransitionRecord(
                from_state=SafetyState.BOOT_SELF_TEST,
                to_state=SafetyState.BOOT_SELF_TEST,
                reason="System initialized into boot self-test",
                timestamp_s=start_time_s,
            )
        ]
        self._active_violations: list[PlausibilityViolation] = []
        self._standstill_confirmed: bool = False

    @property
    def current_state(self) -> SafetyState:
        """Active operational safety state."""
        return self._state

    @property
    def transition_history(self) -> list[SafetyTransitionRecord]:
        """Immutable copy of state transition history."""
        return list(self._transition_history)

    def transition_to(
        self,
        target_state: SafetyState,
        reason: str,
        timestamp_s: float,
    ) -> None:
        """Perform a state transition and append to the audit record."""
        if self._state == target_state:
            return

        # Irreversible latch constraint: FAIL_SILENT_LATCH cannot transition out automatically
        if self._state == SafetyState.FAIL_SILENT_LATCH:
            logger.warning(
                "Attempted transition from FAIL_SILENT_LATCH to %s blocked (latch active)",
                target_state.value,
            )
            return

        record = SafetyTransitionRecord(
            from_state=self._state,
            to_state=target_state,
            reason=reason,
            timestamp_s=timestamp_s,
        )
        self._transition_history.append(record)
        logger.info(
            "Safety supervisor state transition: %s -> %s (Reason: %s)",
            self._state.value,
            target_state.value,
            reason,
        )
        self._state = target_state

    def trigger_emergency_stop(self, reason: str, timestamp_s: float) -> None:
        """Manually or externally invoke emergency safe stop."""
        self.transition_to(
            target_state=SafetyState.EMERGENCY_SAFE_STOP,
            reason=f"Emergency safe stop triggered: {reason}",
            timestamp_s=timestamp_s,
        )

    def trigger_latch(self, reason: str, timestamp_s: float) -> None:
        """Transition into irrevocable fail-silent hardware latch."""
        self.transition_to(
            target_state=SafetyState.FAIL_SILENT_LATCH,
            reason=f"Fail-silent latch engaged: {reason}",
            timestamp_s=timestamp_s,
        )

    def reset_latch(self, authorization_key: str, timestamp_s: float) -> bool:
        """Authorized diagnostic override to clear fail-silent latch."""
        if authorization_key != "ASIL-D-RESET-KEY-2026":
            logger.warning("Invalid authorization key provided to reset safety latch")
            return False

        self._state = SafetyState.BOOT_SELF_TEST
        self._boot_start_time_s = timestamp_s
        self._standstill_confirmed = False
        self._transition_history.append(
            SafetyTransitionRecord(
                from_state=SafetyState.FAIL_SILENT_LATCH,
                to_state=SafetyState.BOOT_SELF_TEST,
                reason="Authorized hardware reset of fail-silent latch",
                timestamp_s=timestamp_s,
            )
        )
        self.watchdog.reset()
        self.plausibility.reset_history()
        return True

    def get_actuator_override(self) -> ActuatorSafetyOverride:
        """Calculate actuation limits and overrides based on current safety state."""
        if self._state in (SafetyState.FAIL_SILENT_LATCH, SafetyState.EMERGENCY_SAFE_STOP):
            return ActuatorSafetyOverride(
                is_override_active=True,
                target_deceleration_mps2=self.config.safe_stop_deceleration_mps2,
                max_permitted_speed_mps=0.0,
                steering_clamped=True,
                emergency_hazard_lights=True,
            )
        if self._state == SafetyState.DEGRADED_PERCEPTION:
            return ActuatorSafetyOverride(
                is_override_active=True,
                target_deceleration_mps2=0.0,
                max_permitted_speed_mps=self.config.degraded_speed_limit_mps,
                steering_clamped=False,
                emergency_hazard_lights=False,
            )
        # NORMAL_OPERATION or BOOT_SELF_TEST
        return ActuatorSafetyOverride(
            is_override_active=False,
            target_deceleration_mps2=0.0,
            max_permitted_speed_mps=10.0,
            steering_clamped=False,
            emergency_hazard_lights=False,
        )

    def step(
        self,
        current_time_s: float,
        kinematics: dict[str, Any] | None = None,
    ) -> SafetySupervisorReport:
        """Evaluate one cyclic safety control tick across watchdog and plausibility inputs."""
        self._last_time_s = current_time_s

        # 1. Evaluate Heartbeat Watchdogs
        node_healths = self.watchdog.evaluate(current_time_s)
        is_crit_watchdog, crit_node_reason = self.watchdog.check_critical_expiration(node_healths)

        # 2. Evaluate Plausibility checks if kinematics provided
        new_violations: list[PlausibilityViolation] = []
        if kinematics is not None:
            steer = float(kinematics.get("steering_angle_rad", 0.0))
            wheel_speed = float(kinematics.get("wheel_speed_mps", 0.0))
            sec_spd = kinematics.get("secondary_speed_mps")
            secondary_speed = float(sec_spd) if sec_spd is not None else None

            new_violations.extend(
                self.plausibility.evaluate_kinematics(
                    timestamp_s=current_time_s,
                    steering_angle_rad=steer,
                    wheel_speed_mps=wheel_speed,
                    secondary_speed_mps=secondary_speed,
                )
            )

        self._active_violations = new_violations

        # 3. State Evaluation & Transitions
        if self._state == SafetyState.BOOT_SELF_TEST:
            dwell_elapsed = (
                current_time_s - self._boot_start_time_s
            ) >= self.config.self_test_dwell_s
            all_critical_present = True
            for node in (
                ECUNodeId.VEHICLE_CONTROLLER,
                ECUNodeId.CAN_GATEWAY,
                ECUNodeId.TRAJECTORY_PLANNER,
            ):
                health = node_healths.get(node.value)
                if health is None or health.status != HeartbeatStatus.HEALTHY:
                    all_critical_present = False
                    break

            if dwell_elapsed and all_critical_present:
                elapsed_dwell = current_time_s - self._boot_start_time_s
                self.transition_to(
                    target_state=SafetyState.NORMAL_OPERATION,
                    reason=f"Boot self-test passed after {elapsed_dwell:.2f}s",
                    timestamp_s=current_time_s,
                )
            elif (current_time_s - self._boot_start_time_s) > (self.config.self_test_dwell_s * 4):
                self.transition_to(
                    target_state=SafetyState.EMERGENCY_SAFE_STOP,
                    reason="Boot self-test timed out: critical ECUs not reporting",
                    timestamp_s=current_time_s,
                )

        elif self._state in (SafetyState.NORMAL_OPERATION, SafetyState.DEGRADED_PERCEPTION):
            if is_crit_watchdog:
                self.transition_to(
                    target_state=SafetyState.EMERGENCY_SAFE_STOP,
                    reason=f"Critical ECU watchdog failure: {crit_node_reason}",
                    timestamp_s=current_time_s,
                )
            else:
                has_fatal = any(
                    v.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.FATAL)
                    for v in self._active_violations
                )
                if has_fatal:
                    first_v = next(
                        v
                        for v in self._active_violations
                        if v.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.FATAL)
                    )
                    self.transition_to(
                        target_state=SafetyState.EMERGENCY_SAFE_STOP,
                        reason=f"Violation: {first_v.violation_type} ({first_v.description})",
                        timestamp_s=current_time_s,
                    )
                else:
                    has_warning = any(
                        v.severity == ViolationSeverity.WARNING for v in self._active_violations
                    )
                    teleop_h = node_healths.get(ECUNodeId.TELEOP_SUPERVISOR.value)
                    teleop_deg = teleop_h is not None and teleop_h.status in (
                        HeartbeatStatus.MISSED,
                        HeartbeatStatus.EXPIRED,
                        HeartbeatStatus.WARN_JITTER,
                    )
                    percept_h = node_healths.get(ECUNodeId.PERCEPTION_SURROUND.value)
                    percept_deg = percept_h is not None and percept_h.status in (
                        HeartbeatStatus.MISSED,
                        HeartbeatStatus.WARN_JITTER,
                    )

                    if has_warning or teleop_deg or percept_deg:
                        if self._state == SafetyState.NORMAL_OPERATION:
                            self.transition_to(
                                target_state=SafetyState.DEGRADED_PERCEPTION,
                                reason="Sensor jitter or non-critical node degradation detected",
                                timestamp_s=current_time_s,
                            )
                    elif self._state == SafetyState.DEGRADED_PERCEPTION:
                        self.transition_to(
                            target_state=SafetyState.NORMAL_OPERATION,
                            reason="Non-critical node telemetry recovered to nominal bounds",
                            timestamp_s=current_time_s,
                        )

        elif self._state == SafetyState.EMERGENCY_SAFE_STOP:
            speed = float(kinematics.get("wheel_speed_mps", 0.0)) if kinematics else 0.0
            if abs(speed) < 0.05:
                self._standstill_confirmed = True
                self.transition_to(
                    target_state=SafetyState.FAIL_SILENT_LATCH,
                    reason="Standstill verified following safe stop; engaging park pawl latch",
                    timestamp_s=current_time_s,
                )

        return self.generate_report(current_time_s)

    def generate_report(self, current_time_s: float) -> SafetySupervisorReport:
        """Produce consolidated safety audit report."""
        return SafetySupervisorReport(
            timestamp_s=current_time_s,
            active_state=self._state,
            node_healths=dict(self.watchdog.node_records),
            active_violations=list(self._active_violations),
            transition_history=list(self._transition_history),
            actuator_override=self.get_actuator_override(),
        )
