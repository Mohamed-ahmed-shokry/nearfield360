"""Tests for 4-wheel tire tread depth profiling and TPMS pressure analyzer."""

from __future__ import annotations

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import TirePosition, TireWearStatus
from nearfield360.inspection.tires import TireProfiler


def test_evaluate_wheel_good_condition() -> None:
    profiler = TireProfiler()
    telemetry = profiler.evaluate_wheel(
        position=TirePosition.FRONT_LEFT,
        inner_depth_mm=6.2,
        center_depth_mm=6.3,
        outer_depth_mm=6.1,
        pressure_bar=2.40,
        temperature_c=21.0,
    )

    assert telemetry.position == TirePosition.FRONT_LEFT
    assert telemetry.mean_tread_depth_mm == 6.2
    assert telemetry.status == TireWearStatus.GOOD
    assert telemetry.uneven_wear_detected is False


def test_evaluate_wheel_critical_replace_legal_limit() -> None:
    config = InspectionConfig(min_legal_tread_depth_mm=1.6)
    profiler = TireProfiler(config=config)

    telemetry = profiler.evaluate_wheel(
        position=TirePosition.REAR_RIGHT,
        inner_depth_mm=1.4,
        center_depth_mm=1.5,
        outer_depth_mm=1.3,
        pressure_bar=2.38,
    )

    assert telemetry.mean_tread_depth_mm == 1.4
    assert telemetry.status == TireWearStatus.CRITICAL_REPLACE


def test_evaluate_wheel_asymmetric_shoulder_wear() -> None:
    config = InspectionConfig(max_tread_wear_variance_mm=1.5)
    profiler = TireProfiler(config=config)

    # 2.2 mm difference between inner and outer shoulder (camber/toe wear)
    telemetry = profiler.evaluate_wheel(
        position=TirePosition.FRONT_RIGHT,
        inner_depth_mm=5.8,
        center_depth_mm=5.0,
        outer_depth_mm=3.6,
        pressure_bar=2.40,
    )

    assert telemetry.uneven_wear_detected is True
    assert telemetry.status == TireWearStatus.ACCEPTABLE


def test_evaluate_wheel_pressure_fault() -> None:
    config = InspectionConfig(
        nominal_cold_pressure_bar=2.4,
        pressure_deviation_tolerance_bar=0.25,
    )
    profiler = TireProfiler(config=config)

    # Under-inflated tire (1.95 bar < 2.15 bar limit)
    telemetry = profiler.evaluate_wheel(
        position=TirePosition.REAR_LEFT,
        inner_depth_mm=5.5,
        center_depth_mm=5.5,
        outer_depth_mm=5.5,
        pressure_bar=1.95,
    )

    assert telemetry.status == TireWearStatus.PRESSURE_FAULT


def test_evaluate_four_wheels_and_axle_variance() -> None:
    profiler = TireProfiler()
    wheel_data = {
        TirePosition.FRONT_LEFT: {"inner": 6.0, "center": 6.0, "outer": 6.0, "pressure": 2.4},
        TirePosition.FRONT_RIGHT: {"inner": 5.8, "center": 5.8, "outer": 5.8, "pressure": 2.4},
        TirePosition.REAR_LEFT: {"inner": 4.5, "center": 4.5, "outer": 4.5, "pressure": 2.4},
        TirePosition.REAR_RIGHT: {"inner": 4.5, "center": 4.5, "outer": 4.5, "pressure": 2.4},
    }

    tires = profiler.evaluate_four_wheels(wheel_data)
    assert len(tires) == 4

    variance = profiler.compute_axle_variance(tires)
    assert variance["front_cross_variance_mm"] == 0.2
    assert variance["rear_cross_variance_mm"] == 0.0
    assert variance["axle_longitudinal_variance_mm"] == 1.4
