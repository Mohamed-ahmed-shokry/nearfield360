"""CAN frame serialization, signal packing/unpacking, and CRC verification (ISO 11898)."""

from __future__ import annotations

import struct

from nearfield360.can.models import (
    CANFrame,
    VehicleControlSignal,
    VehicleFeedbackSignal,
)

# Standard arbitration IDs for drive-by-wire AVP control and feedback
ARBITRATION_ID_CONTROL = 0x120  # Actuator commands (EPS, ECM, EBCM)
ARBITRATION_ID_FEEDBACK = 0x220  # Chassis wheel speeds, IMU, ultrasonic

# CRC-8-SAE J1850 polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D), init 0xFF, xorout 0xFF
CRC8_POLYNOMIAL = 0x1D
CRC8_INIT = 0xFF
CRC8_XOROUT = 0xFF


def compute_crc8_j1850(data: bytes) -> int:
    """Calculate CRC-8 conforming to SAE J1850 standard."""
    crc = CRC8_INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1 ^ CRC8_POLYNOMIAL) & 255 if crc & 128 else crc << 1 & 255
    return crc ^ CRC8_XOROUT


class CANCodec:
    """Encodes and decodes automotive control and feedback signals to/from CAN frames."""

    @staticmethod
    def encode_control_signal(
        signal: VehicleControlSignal,
        timestamp_us: float = 0.0,
    ) -> CANFrame:
        """Pack VehicleControlSignal into standard 8-byte CAN frame.

        Layout (8 bytes):
        - Bytes 0-1: steering_angle (int16, -1.0 to 1.0 rad scaled by 10000 -> -10000 to +10000)
        - Byte 2: throttle_pct (uint8, 0 to 100 scaled by 2.5 -> 0 to 250)
        - Byte 3: brake_pct (uint8, 0 to 100 scaled by 2.5 -> 0 to 250)
        - Byte 4: gear (ascii ord: 'P', 'R', 'N', 'D')
        - Bytes 5-6: speed (uint16, m/s * 100, e.g. 5.5 m/s -> 550)
        - Byte 7: [bits 7-4: heartbeat_counter (0-15)], [bits 3-0: CRC-4 / lower nibble of CRC-8]
        """
        steer_scaled = round(signal.steering_angle_rad * 10000.0)
        steer_scaled = max(-10000, min(10000, steer_scaled))

        throttle_scaled = round(signal.throttle_pct * 2.5)
        throttle_scaled = max(0, min(250, throttle_scaled))

        brake_scaled = round(signal.brake_pct * 2.5)
        brake_scaled = max(0, min(250, brake_scaled))

        gear_byte = ord(signal.gear)
        speed_scaled = round(max(0.0, signal.speed_mps) * 100.0)
        speed_scaled = min(65535, speed_scaled)

        # Pack first 7 bytes
        data7 = struct.pack(
            ">hBBBH",
            steer_scaled,
            throttle_scaled,
            brake_scaled,
            gear_byte,
            speed_scaled,
        )

        crc8 = compute_crc8_j1850(data7)
        # Byte 7 combines 4-bit rolling heartbeat and lower 4 bits of CRC
        byte7 = ((signal.heartbeat_counter & 0x0F) << 4) | (crc8 & 0x0F)
        payload = data7 + bytes([byte7])

        return CANFrame(
            arbitration_id=ARBITRATION_ID_CONTROL,
            is_extended_id=False,
            is_fd=False,
            payload=payload,
            dlc=8,
            timestamp_us=timestamp_us,
            crc=crc8,
        )

    @staticmethod
    def decode_control_signal(frame: CANFrame) -> VehicleControlSignal:
        """Unpack CAN frame into VehicleControlSignal with CRC validation."""
        if len(frame.payload) < 8 or frame.dlc < 8:
            raise ValueError(f"CANControl frame requires 8 bytes, got {len(frame.payload)}")

        data7 = frame.payload[:7]
        byte7 = frame.payload[7]

        expected_crc = compute_crc8_j1850(data7)
        received_crc_low = byte7 & 0x0F
        if (expected_crc & 0x0F) != received_crc_low:
            raise ValueError(
                f"CAN CRC mismatch: expected 0x{expected_crc:02X}, "
                f"payload low nibble was 0x{received_crc_low:01X}"
            )

        heartbeat = (byte7 >> 4) & 0x0F
        steer_scaled, throttle_scaled, brake_scaled, gear_byte, speed_scaled = struct.unpack(
            ">hBBBH",
            data7,
        )

        gear_char = chr(gear_byte)
        if gear_char not in ("P", "R", "N", "D"):
            gear_char = "P"

        return VehicleControlSignal(
            steering_angle_rad=round(steer_scaled / 10000.0, 4),
            throttle_pct=round(throttle_scaled / 2.5, 1),
            brake_pct=round(brake_scaled / 2.5, 1),
            gear=gear_char,
            speed_mps=round(speed_scaled / 100.0, 2),
            heartbeat_counter=heartbeat,
        )

    @staticmethod
    def encode_feedback_signal(
        signal: VehicleFeedbackSignal,
        timestamp_us: float = 0.0,
    ) -> CANFrame:
        """Pack VehicleFeedbackSignal into standard 8-byte CAN frame.

        Layout (8 bytes):
        - Byte 0: wheel_speed_fl (uint8, m/s * 4 -> max 63.75 m/s)
        - Byte 1: wheel_speed_fr (uint8, m/s * 4)
        - Byte 2: wheel_speed_rl (uint8, m/s * 4)
        - Byte 3: wheel_speed_rr (uint8, m/s * 4)
        - Byte 4: yaw_rate (int8, rad/s * 64 -> range -2.0 to +1.98 rad/s)
        - Byte 5: longitudinal_accel (int8, m/s^2 * 8 -> range -15.0 to +15.0 m/s^2)
        - Byte 6: ultrasound_min_dist (uint8, cm -> 0 to 255 cm / 2.55 m)
        - Byte 7: [bits 7-4: heartbeat_counter], [bits 3-0: CRC-4]
        """
        w_fl = max(0, min(255, round(signal.wheel_speed_fl_mps * 4.0)))
        w_fr = max(0, min(255, round(signal.wheel_speed_fr_mps * 4.0)))
        w_rl = max(0, min(255, round(signal.wheel_speed_rl_mps * 4.0)))
        w_rr = max(0, min(255, round(signal.wheel_speed_rr_mps * 4.0)))

        yaw = max(-128, min(127, round(signal.yaw_rate_rads * 64.0)))
        accel = max(-128, min(127, round(signal.longitudinal_accel_mps2 * 8.0)))

        min_sonar_cm = 255
        if signal.ultrasound_distances_m:
            min_sonar_cm = max(0, min(255, round(min(signal.ultrasound_distances_m) * 100.0)))

        data7 = struct.pack(
            ">BBBBbbB",
            w_fl,
            w_fr,
            w_rl,
            w_rr,
            yaw,
            accel,
            min_sonar_cm,
        )
        crc8 = compute_crc8_j1850(data7)
        byte7 = ((signal.heartbeat_counter & 0x0F) << 4) | (crc8 & 0x0F)
        payload = data7 + bytes([byte7])

        return CANFrame(
            arbitration_id=ARBITRATION_ID_FEEDBACK,
            is_extended_id=False,
            is_fd=False,
            payload=payload,
            dlc=8,
            timestamp_us=timestamp_us,
            crc=crc8,
        )

    @staticmethod
    def decode_feedback_signal(frame: CANFrame) -> VehicleFeedbackSignal:
        """Unpack CAN frame into VehicleFeedbackSignal with CRC validation."""
        if len(frame.payload) < 8 or frame.dlc < 8:
            raise ValueError(f"CANFeedback frame requires 8 bytes, got {len(frame.payload)}")

        data7 = frame.payload[:7]
        byte7 = frame.payload[7]

        expected_crc = compute_crc8_j1850(data7)
        received_crc_low = byte7 & 0x0F
        if (expected_crc & 0x0F) != received_crc_low:
            raise ValueError(
                f"CAN CRC mismatch: expected 0x{expected_crc:02X}, "
                f"payload low nibble was 0x{received_crc_low:01X}"
            )

        heartbeat = (byte7 >> 4) & 0x0F
        w_fl, w_fr, w_rl, w_rr, yaw, accel, min_sonar_cm = struct.unpack(
            ">BBBBbbB",
            data7,
        )

        sonar_dist = [round(min_sonar_cm / 100.0, 2)]

        return VehicleFeedbackSignal(
            wheel_speed_fl_mps=round(w_fl / 4.0, 2),
            wheel_speed_fr_mps=round(w_fr / 4.0, 2),
            wheel_speed_rl_mps=round(w_rl / 4.0, 2),
            wheel_speed_rr_mps=round(w_rr / 4.0, 2),
            yaw_rate_rads=round(yaw / 64.0, 3),
            longitudinal_accel_mps2=round(accel / 8.0, 2),
            ultrasound_distances_m=sonar_dist,
            heartbeat_counter=heartbeat,
        )
