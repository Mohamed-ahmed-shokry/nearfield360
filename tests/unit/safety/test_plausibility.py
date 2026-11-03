"""Unit tests for sensor plausibility cross-checker and kinematic invariant validator."""

from __future__ import annotations

from nearfield360.config import SafetyConfig
from nearfield360.safety.models import ViolationSeverity
from nearfield360.safety.plausibility import PlausibilityChecker


def test_plausibility_telemetry_freshness() -> None:
    checker = PlausibilityChecker()

    # Normal latency (50ms) -> passes
    assert checker.check_telemetry_freshness("radar_1", 10.0, 10.05) is None

    # Future timestamp breach (-100ms into future)
    v_future = checker.check_telemetry_freshness("radar_1", 10.10, 10.0)
    assert v_future is not None
    assert v_future.violation_type == "FUTURE_TIMESTAMP"
    assert v_future.severity == ViolationSeverity.CRITICAL

    # Stale telemetry (300ms vs default 250ms limit)
    v_stale = checker.check_telemetry_freshness("radar_1", 10.0, 10.30)
    assert v_stale is not None
    assert v_stale.violation_type == "STALE_TELEMETRY"
    assert v_stale.severity == ViolationSeverity.WARNING

    # Highly stale telemetry (> 2x limit = 600ms)
    v_crit_stale = checker.check_telemetry_freshness("radar_1", 10.0, 10.60)
    assert v_crit_stale is not None
    assert v_crit_stale.severity == ViolationSeverity.CRITICAL


def test_plausibility_frozen_signal() -> None:
    checker = PlausibilityChecker()

    # Dynamic varying signal -> passes
    for i in range(10):
        v = checker.check_frozen_signal("wheel_speed", 5.0 + (i % 3) * 0.1, timestamp_s=float(i))
        assert v is None

    checker.reset_history()

    # Constant non-zero reading for 5 cycles -> frozen detection
    violation = None
    for i in range(5):
        violation = checker.check_frozen_signal("radar_range", 12.345, timestamp_s=float(i))
    assert violation is not None
    assert violation.violation_type == "FROZEN_SENSOR_SIGNAL"
    assert violation.severity == ViolationSeverity.CRITICAL

    # Constant zero reading (e.g. stopped vehicle) -> permitted
    checker.reset_history()
    for i in range(10):
        v_zero = checker.check_frozen_signal("speed", 0.0, timestamp_s=float(i))
        assert v_zero is None


def test_plausibility_steering_and_acceleration() -> None:
    config = SafetyConfig(max_steering_rate_rads=0.5, max_acceleration_mps2=3.0)
    checker = PlausibilityChecker(config=config)

    # Valid steering rate (0.02 rad in 0.1s = 0.2 rad/s < 0.5)
    assert checker.check_steering_slew(0.02, 0.0, dt_s=0.1) is None

    # Excessive steering rate (0.1 rad in 0.1s = 1.0 rad/s > 0.5)
    v_steer = checker.check_steering_slew(0.1, 0.0, dt_s=0.1)
    assert v_steer is not None
    assert v_steer.violation_type == "STEERING_RATE_EXCEEDED"
    assert v_steer.measured_value == 1.0

    # Valid acceleration (0.2 m/s in 0.1s = 2.0 m/s² < 3.0)
    assert checker.check_acceleration(5.2, 5.0, dt_s=0.1) is None

    # Excessive acceleration (0.5 m/s in 0.1s = 5.0 m/s² > 3.0)
    v_accel = checker.check_acceleration(5.5, 5.0, dt_s=0.1)
    assert v_accel is not None
    assert v_accel.violation_type == "ACCELERATION_ENVELOPE_EXCEEDED"
    assert v_accel.measured_value == 5.0


def test_plausibility_speed_discrepancy_and_range_bounds() -> None:
    config = SafetyConfig(speed_discrepancy_tolerance_mps=1.0)
    checker = PlausibilityChecker(config=config)

    # Consistent velocities
    assert checker.check_speed_discrepancy(5.0, 5.5) is None

    # Inconsistent velocities (diff 1.5 > 1.0)
    v_disc = checker.check_speed_discrepancy(5.0, 6.5)
    assert v_disc is not None
    assert v_disc.violation_type == "VELOCITY_DISCREPANCY"
    assert v_disc.measured_value == 1.5

    # Valid distance range
    assert checker.check_range_bounds("ultrasonic_front", 2.5) is None

    # Negative distance reading
    v_neg = checker.check_range_bounds("ultrasonic_front", -0.5)
    assert v_neg is not None
    assert v_neg.violation_type == "OUT_OF_BOUNDS_RANGE"
    assert v_neg.severity == ViolationSeverity.WARNING

    # Non-finite distance reading
    v_nan = checker.check_range_bounds("radar_front", float("nan"))
    assert v_nan is not None
    assert v_nan.violation_type == "INVALID_NUMERIC_RANGE"
    assert v_nan.severity == ViolationSeverity.FATAL


def test_plausibility_kinematic_evaluation_pipeline() -> None:
    checker = PlausibilityChecker()

    # First frame initializes state
    v1 = checker.evaluate_kinematics(timestamp_s=0.0, steering_angle_rad=0.0, wheel_speed_mps=2.0)
    assert len(v1) == 0

    # Smooth second frame
    v2 = checker.evaluate_kinematics(
        timestamp_s=0.1,
        steering_angle_rad=0.02,
        wheel_speed_mps=2.1,
        secondary_speed_mps=2.15,
    )
    assert len(v2) == 0

    # Third frame with violent steering change (1.0 rad in 0.1s = 10 rad/s)
    v3 = checker.evaluate_kinematics(
        timestamp_s=0.2,
        steering_angle_rad=1.02,
        wheel_speed_mps=2.1,
    )
    assert len(v3) == 1
    assert v3[0].violation_type == "STEERING_RATE_EXCEEDED"
