"""Unit tests for CAN domain models, validations, and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.can.models import (
    CANBusTelemetry,
    CANFaultType,
    CANFrame,
    CANFrameType,
    HILSimulationReport,
    VehicleControlSignal,
    VehicleFeedbackSignal,
)


def test_can_enums() -> None:
    assert CANFrameType.STANDARD == "standard"
    assert CANFrameType.EXTENDED == "extended"
    assert CANFrameType.CAN_FD_STANDARD == "can_fd_standard"
    assert CANFrameType.CAN_FD_EXTENDED == "can_fd_extended"

    assert CANFaultType.NONE == "none"
    assert CANFaultType.CORRUPTED_CRC == "corrupted_crc"
    assert CANFaultType.FRAME_DROP == "frame_drop"
    assert CANFaultType.TIMING_JITTER == "timing_jitter"
    assert CANFaultType.BIT_FLIP == "bit_flip"
    assert CANFaultType.DLC_MISMATCH == "dlc_mismatch"


def test_can_frame_validation_and_frozen() -> None:
    frame = CANFrame(
        arbitration_id=0x120,
        dlc=8,
        payload=b"\x00\x01\x02\x03\x04\x05\x06\x07",
        crc=0xA5B6,
    )
    assert frame.arbitration_id == 0x120
    assert frame.dlc == 8
    assert frame.is_extended_id is False

    with pytest.raises(ValidationError):
        frame.dlc = 4  # type: ignore[misc]

    # Reject invalid arbitration ID
    with pytest.raises(ValidationError):
        CANFrame(arbitration_id=-1, dlc=0)

    # Reject oversized payload
    with pytest.raises(ValidationError):
        CANFrame(arbitration_id=0x100, dlc=8, payload=b"x" * 65)


def test_vehicle_control_signal_validation() -> None:
    sig = VehicleControlSignal(
        steering_angle_rad=0.25,
        throttle_pct=30.0,
        brake_pct=0.0,
        gear="D",
        speed_mps=5.0,
        heartbeat_counter=3,
    )
    assert sig.gear == "D"
    assert sig.heartbeat_counter == 3

    # Invalid steering angle (out of [-1.0, 1.0])
    with pytest.raises(ValidationError):
        VehicleControlSignal(steering_angle_rad=2.5)

    # Invalid gear
    with pytest.raises(ValidationError):
        VehicleControlSignal(gear="X")  # type: ignore[arg-type]

    # Invalid heartbeat counter (out of [0, 15])
    with pytest.raises(ValidationError):
        VehicleControlSignal(heartbeat_counter=16)


def test_vehicle_feedback_signal_validation() -> None:
    fb = VehicleFeedbackSignal(
        wheel_speed_fl_mps=10.0,
        wheel_speed_fr_mps=10.1,
        wheel_speed_rl_mps=10.0,
        wheel_speed_rr_mps=10.1,
        yaw_rate_rads=0.02,
        longitudinal_accel_mps2=0.5,
        ultrasound_distances_m=[1.2, 2.5, 0.8, 1.5],
        heartbeat_counter=7,
    )
    assert len(fb.ultrasound_distances_m) == 4
    assert fb.heartbeat_counter == 7

    # Reject negative wheel speed
    with pytest.raises(ValidationError):
        VehicleFeedbackSignal(wheel_speed_fl_mps=-1.0)


def test_hil_report_round_trip() -> None:
    telemetry = CANBusTelemetry(
        frames_sent=100,
        frames_received=98,
        dropped_frames=2,
        corrupted_frames=1,
        bus_load_percent=42.5,
        mean_jitter_us=120.0,
        peak_jitter_us=350.0,
        is_overloaded=False,
    )
    report = HILSimulationReport(
        duration_s=5.0,
        target_ecus=["ECM", "TCM", "EPS", "BCM"],
        telemetry=telemetry,
        fault_injection_log=[{"time_s": 1.2, "fault": "frame_drop", "id": 0x120}],
        ecu_health_status={"ECM": "healthy", "EPS": "healthy"},
    )
    serialized = report.model_dump_json()
    restored = HILSimulationReport.model_validate_json(serialized)
    assert restored == report
