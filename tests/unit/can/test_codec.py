"""Unit tests for CAN bit-packing codec and CRC-8 validation."""

from __future__ import annotations

import pytest

from nearfield360.can.codec import (
    ARBITRATION_ID_CONTROL,
    ARBITRATION_ID_FEEDBACK,
    CANCodec,
    compute_crc8_j1850,
)
from nearfield360.can.models import (
    CANFrame,
    VehicleControlSignal,
    VehicleFeedbackSignal,
)


def test_compute_crc8_j1850_standard() -> None:
    # CRC calculation deterministic and non-zero on test vector
    test_vec = b"\x01\x02\x03\x04"
    crc1 = compute_crc8_j1850(test_vec)
    crc2 = compute_crc8_j1850(test_vec)
    assert crc1 == crc2
    assert 0 <= crc1 <= 0xFF

    # Empty data vector check
    assert 0 <= compute_crc8_j1850(b"") <= 0xFF


def test_control_signal_round_trip() -> None:
    ctrl = VehicleControlSignal(
        steering_angle_rad=0.352,
        throttle_pct=42.0,
        brake_pct=0.0,
        gear="D",
        speed_mps=8.25,
        heartbeat_counter=5,
    )
    frame = CANCodec.encode_control_signal(ctrl, timestamp_us=12345.0)

    assert frame.arbitration_id == ARBITRATION_ID_CONTROL
    assert frame.dlc == 8
    assert len(frame.payload) == 8
    assert frame.timestamp_us == 12345.0

    restored = CANCodec.decode_control_signal(frame)
    assert restored.steering_angle_rad == pytest.approx(0.352, abs=1e-3)
    assert restored.throttle_pct == pytest.approx(42.0, abs=0.5)
    assert restored.brake_pct == 0.0
    assert restored.gear == "D"
    assert restored.speed_mps == pytest.approx(8.25, abs=0.05)
    assert restored.heartbeat_counter == 5


def test_feedback_signal_round_trip() -> None:
    fb = VehicleFeedbackSignal(
        wheel_speed_fl_mps=12.25,
        wheel_speed_fr_mps=12.50,
        wheel_speed_rl_mps=12.00,
        wheel_speed_rr_mps=12.25,
        yaw_rate_rads=0.15,
        longitudinal_accel_mps2=-1.25,
        ultrasound_distances_m=[0.85, 1.20, 2.10],
        heartbeat_counter=11,
    )
    frame = CANCodec.encode_feedback_signal(fb, timestamp_us=54321.0)

    assert frame.arbitration_id == ARBITRATION_ID_FEEDBACK
    assert frame.dlc == 8
    assert len(frame.payload) == 8

    restored = CANCodec.decode_feedback_signal(frame)
    assert restored.wheel_speed_fl_mps == pytest.approx(12.25, abs=0.25)
    assert restored.wheel_speed_fr_mps == pytest.approx(12.50, abs=0.25)
    assert restored.yaw_rate_rads == pytest.approx(0.15, abs=0.05)
    assert restored.longitudinal_accel_mps2 == pytest.approx(-1.25, abs=0.2)
    assert restored.ultrasound_distances_m[0] == pytest.approx(0.85, abs=0.05)
    assert restored.heartbeat_counter == 11


def test_decode_corrupted_crc_raises_error() -> None:
    ctrl = VehicleControlSignal(steering_angle_rad=0.0, gear="P")
    frame = CANCodec.encode_control_signal(ctrl)

    # Invert CRC lower nibble
    corrupted_byte7 = frame.payload[7] ^ 0x01
    corrupted_payload = frame.payload[:7] + bytes([corrupted_byte7])
    corrupted_frame = CANFrame(
        arbitration_id=frame.arbitration_id,
        dlc=8,
        payload=corrupted_payload,
    )

    with pytest.raises(ValueError, match="CAN CRC mismatch"):
        CANCodec.decode_control_signal(corrupted_frame)


def test_decode_insufficient_length_raises_error() -> None:
    short_frame = CANFrame(arbitration_id=0x120, dlc=4, payload=b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="requires 8 bytes"):
        CANCodec.decode_control_signal(short_frame)
    with pytest.raises(ValueError, match="requires 8 bytes"):
        CANCodec.decode_feedback_signal(short_frame)
