"""Sensor plausibility cross-checker and kinematic invariant validator.

Validates sensor telemetry freshness, non-frozen signals, steering slew rates,
longitudinal acceleration thresholds, and cross-sensor velocity consistency according
to ISO 26262 ASIL-D functional safety specifications.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque

from nearfield360.config import SafetyConfig
from nearfield360.safety.models import PlausibilityViolation, ViolationSeverity

logger = logging.getLogger(__name__)


class PlausibilityChecker:
    """ASIL-D sensor telemetry and kinematic plausibility monitoring engine."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._signal_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self._last_kinematics: dict[str, float] | None = None

    def reset_history(self) -> None:
        """Clear all historical sensor signal caches and previous kinematic states."""
        self._signal_history.clear()
        self._last_kinematics = None

    def check_telemetry_freshness(
        self,
        sensor_id: str,
        timestamp_s: float,
        current_time_s: float,
        max_stale_s: float = 0.25,
    ) -> PlausibilityViolation | None:
        """Verify that sensor telemetry is within acceptable latency limits."""
        delta_s = current_time_s - timestamp_s
        if delta_s < -0.05:
            # Future timestamp invariant breach
            return PlausibilityViolation(
                violation_type="FUTURE_TIMESTAMP",
                description=(
                    f"Sensor {sensor_id} reported telemetry timestamp ahead of current clock"
                ),
                measured_value=delta_s,
                limit_value=0.0,
                severity=ViolationSeverity.CRITICAL,
                timestamp_s=current_time_s,
            )
        if delta_s > max_stale_s:
            return PlausibilityViolation(
                violation_type="STALE_TELEMETRY",
                description=f"Sensor {sensor_id} telemetry is stale by {delta_s * 1e3:.1f}ms",
                measured_value=delta_s,
                limit_value=max_stale_s,
                severity=ViolationSeverity.WARNING
                if delta_s < max_stale_s * 2
                else ViolationSeverity.CRITICAL,
                timestamp_s=current_time_s,
            )
        return None

    def check_frozen_signal(
        self,
        signal_name: str,
        value: float,
        timestamp_s: float,
        window_size: int = 5,
        variance_tol: float = 1e-5,
    ) -> PlausibilityViolation | None:
        """Detect stuck-at or frozen sensor values over consecutive sample windows."""
        history = self._signal_history[signal_name]
        history.append(value)

        if len(history) < window_size:
            return None

        recent = list(history)[-window_size:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)

        # Non-zero readings that remain completely identical indicate frozen sensor bus
        if abs(mean_val) > 0.05 and variance < variance_tol:
            return PlausibilityViolation(
                violation_type="FROZEN_SENSOR_SIGNAL",
                description=(
                    f"Signal {signal_name} frozen at {mean_val:.4f} across {window_size} frames"
                ),
                measured_value=variance,
                limit_value=variance_tol,
                severity=ViolationSeverity.CRITICAL,
                timestamp_s=timestamp_s,
            )
        return None

    def check_steering_slew(
        self,
        current_angle_rad: float,
        previous_angle_rad: float,
        dt_s: float,
        max_rate_rads: float | None = None,
        timestamp_s: float = 0.0,
    ) -> PlausibilityViolation | None:
        """Check if steering rate exceeds physical steering rack slew limit."""
        if dt_s <= 0.0:
            return None
        limit = max_rate_rads if max_rate_rads is not None else self.config.max_steering_rate_rads
        rate = abs(current_angle_rad - previous_angle_rad) / dt_s
        if rate > limit:
            return PlausibilityViolation(
                violation_type="STEERING_RATE_EXCEEDED",
                description=f"Steering slew rate {rate:.3f} rad/s exceeds limit {limit:.3f} rad/s",
                measured_value=rate,
                limit_value=limit,
                severity=ViolationSeverity.CRITICAL,
                timestamp_s=timestamp_s,
            )
        return None

    def check_acceleration(
        self,
        current_speed_mps: float,
        previous_speed_mps: float,
        dt_s: float,
        max_accel_mps2: float | None = None,
        timestamp_s: float = 0.0,
    ) -> PlausibilityViolation | None:
        """Validate that longitudinal acceleration does not violate kinematic bounds."""
        if dt_s <= 0.0:
            return None
        limit = max_accel_mps2 if max_accel_mps2 is not None else self.config.max_acceleration_mps2
        accel = abs(current_speed_mps - previous_speed_mps) / dt_s
        if accel > limit:
            return PlausibilityViolation(
                violation_type="ACCELERATION_ENVELOPE_EXCEEDED",
                description=f"Acceleration {accel:.2f} m/s² exceeds limit {limit:.2f} m/s²",
                measured_value=accel,
                limit_value=limit,
                severity=ViolationSeverity.CRITICAL,
                timestamp_s=timestamp_s,
            )
        return None

    def check_speed_discrepancy(
        self,
        wheel_speed_mps: float,
        secondary_speed_mps: float,
        tolerance_mps: float | None = None,
        timestamp_s: float = 0.0,
    ) -> PlausibilityViolation | None:
        """Cross-check wheel encoder speed against secondary sensor (radar / odometry)."""
        limit = (
            tolerance_mps
            if tolerance_mps is not None
            else self.config.speed_discrepancy_tolerance_mps
        )
        diff = abs(wheel_speed_mps - secondary_speed_mps)
        if diff > limit:
            return PlausibilityViolation(
                violation_type="VELOCITY_DISCREPANCY",
                description=(
                    f"Speed mismatch: wheel={wheel_speed_mps:.2f} m/s vs "
                    f"secondary={secondary_speed_mps:.2f} m/s (diff {diff:.2f} > {limit:.2f})"
                ),
                measured_value=diff,
                limit_value=limit,
                severity=ViolationSeverity.CRITICAL,
                timestamp_s=timestamp_s,
            )
        return None

    def check_range_bounds(
        self,
        sensor_name: str,
        range_m: float,
        min_range_m: float = 0.05,
        max_range_m: float = 120.0,
        timestamp_s: float = 0.0,
    ) -> PlausibilityViolation | None:
        """Verify that sensor distance measurements lie within realistic boundaries."""
        if math.isnan(range_m) or math.isinf(range_m):
            return PlausibilityViolation(
                violation_type="INVALID_NUMERIC_RANGE",
                description=f"Sensor {sensor_name} yielded non-finite range {range_m}",
                measured_value=0.0,
                limit_value=0.0,
                severity=ViolationSeverity.FATAL,
                timestamp_s=timestamp_s,
            )
        if range_m < min_range_m or range_m > max_range_m:
            return PlausibilityViolation(
                violation_type="OUT_OF_BOUNDS_RANGE",
                description=(
                    f"Sensor {sensor_name} range {range_m:.2f}m is outside valid span "
                    f"[{min_range_m:.2f}, {max_range_m:.2f}]m"
                ),
                measured_value=range_m,
                limit_value=min_range_m if range_m < min_range_m else max_range_m,
                severity=ViolationSeverity.WARNING,
                timestamp_s=timestamp_s,
            )
        return None

    def evaluate_kinematics(
        self,
        timestamp_s: float,
        steering_angle_rad: float,
        wheel_speed_mps: float,
        secondary_speed_mps: float | None = None,
    ) -> list[PlausibilityViolation]:
        """Perform end-to-end kinematic cross-validation using historical tracking."""
        violations: list[PlausibilityViolation] = []

        if secondary_speed_mps is not None:
            v_disc = self.check_speed_discrepancy(
                wheel_speed_mps=wheel_speed_mps,
                secondary_speed_mps=secondary_speed_mps,
                timestamp_s=timestamp_s,
            )
            if v_disc:
                violations.append(v_disc)

        if self._last_kinematics is not None:
            prev_t = self._last_kinematics["timestamp_s"]
            dt_s = timestamp_s - prev_t
            if dt_s > 0.0:
                prev_steer = self._last_kinematics["steering_angle_rad"]
                v_steer = self.check_steering_slew(
                    current_angle_rad=steering_angle_rad,
                    previous_angle_rad=prev_steer,
                    dt_s=dt_s,
                    timestamp_s=timestamp_s,
                )
                if v_steer:
                    violations.append(v_steer)

                prev_speed = self._last_kinematics["wheel_speed_mps"]
                v_accel = self.check_acceleration(
                    current_speed_mps=wheel_speed_mps,
                    previous_speed_mps=prev_speed,
                    dt_s=dt_s,
                    timestamp_s=timestamp_s,
                )
                if v_accel:
                    violations.append(v_accel)

        self._last_kinematics = {
            "timestamp_s": timestamp_s,
            "steering_angle_rad": steering_angle_rad,
            "wheel_speed_mps": wheel_speed_mps,
        }
        return violations
