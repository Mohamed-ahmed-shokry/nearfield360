"""Unit tests for CANGateway discrete-event scheduler and bus metrics."""

from __future__ import annotations

from nearfield360.can.faults import FaultInjector
from nearfield360.can.gateway import CANGateway
from nearfield360.can.models import VehicleControlSignal, VehicleFeedbackSignal
from nearfield360.config import CANConfig


def test_gateway_nominal_simulation() -> None:
    config = CANConfig(bus_bitrate_kbps=500, fault_injection_rate=0.0)
    gateway = CANGateway(config)

    report = gateway.run_simulation(duration_s=1.0)
    assert report.duration_s == 1.0
    assert len(report.target_ecus) == 3

    # Rates: ECM (50 Hz), Chassis (100 Hz), Sonar (20 Hz) -> total 170 frames/sec
    assert report.telemetry.frames_sent == 170
    assert report.telemetry.frames_received == 170
    assert report.telemetry.dropped_frames == 0
    assert report.telemetry.corrupted_frames == 0
    assert 0.0 < report.telemetry.bus_load_percent < 80.0
    assert report.telemetry.is_overloaded is False
    assert all(status == "nominal" for status in report.ecu_health_status.values())


def test_gateway_with_fault_injection() -> None:
    config = CANConfig(bus_bitrate_kbps=500, fault_injection_rate=0.20)
    injector = FaultInjector(fault_rate=0.20, seed=77)
    gateway = CANGateway(config, fault_injector=injector)

    report = gateway.run_simulation(duration_s=1.0)
    assert report.telemetry.dropped_frames > 0 or report.telemetry.corrupted_frames > 0
    assert len(report.fault_injection_log) > 0
    assert any(
        status in ("degraded", "fault_present") for status in report.ecu_health_status.values()
    )


def test_gateway_overload_detection() -> None:
    # Set low bitrate and max bus load threshold to trigger overload
    config = CANConfig(bus_bitrate_kbps=125, max_bus_load_percent=10.0)
    gateway = CANGateway(config)

    report = gateway.run_simulation(duration_s=0.5)
    assert report.telemetry.bus_load_percent > 10.0
    assert report.telemetry.is_overloaded is True


def test_gateway_custom_signals() -> None:
    config = CANConfig()
    gateway = CANGateway(config)

    ctrl = VehicleControlSignal(steering_angle_rad=0.5, speed_mps=12.0)
    fb = VehicleFeedbackSignal(wheel_speed_fl_mps=12.0)

    report = gateway.run_simulation(duration_s=0.1, control_signal=ctrl, feedback_signal=fb)
    assert report.telemetry.frames_sent > 0
