"""Automated hardware fault injection and clock timing jitter engine for HIL testing."""

from __future__ import annotations

import random

from nearfield360.can.models import CANFaultType, CANFrame


class FaultInjector:
    """Injects synthetic physical and protocol-level faults into CAN frame streams."""

    def __init__(
        self,
        fault_rate: float = 0.0,
        max_jitter_us: float = 500.0,
        seed: int | None = 42,
    ) -> None:
        self.fault_rate = max(0.0, min(1.0, fault_rate))
        self.max_jitter_us = max(0.0, max_jitter_us)
        self.rng = random.Random(seed)  # noqa: S311

    def process_frame(self, frame: CANFrame) -> CANFrame | None:
        """Evaluate and inject a fault into the candidate frame.

        Returns None if the frame is dropped, or a mutated CANFrame with fault recorded.
        """
        # Determine if a fault should be injected
        if self.fault_rate <= 0.0 or self.rng.random() >= self.fault_rate:
            # Nominal jitter application only
            jitter = self.rng.uniform(0.0, min(20.0, self.max_jitter_us))
            return CANFrame(
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_fd=frame.is_fd,
                payload=frame.payload,
                dlc=frame.dlc,
                timestamp_us=frame.timestamp_us + jitter,
                crc=frame.crc,
                fault_type=CANFaultType.NONE,
            )

        # Select fault category
        fault_choice = self.rng.choice(
            [
                CANFaultType.FRAME_DROP,
                CANFaultType.CORRUPTED_CRC,
                CANFaultType.BIT_FLIP,
                CANFaultType.TIMING_JITTER,
                CANFaultType.DLC_MISMATCH,
            ]
        )

        if fault_choice == CANFaultType.FRAME_DROP:
            return None

        if fault_choice == CANFaultType.CORRUPTED_CRC:
            # Invert lower byte of CRC
            corrupted_crc = frame.crc ^ 0xFF
            payload = frame.payload
            if len(payload) >= 8:
                # Corrupt the CRC nibble in byte 7
                byte7 = payload[7] ^ 0x05
                payload = payload[:7] + bytes([byte7])
            return CANFrame(
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_fd=frame.is_fd,
                payload=payload,
                dlc=frame.dlc,
                timestamp_us=frame.timestamp_us,
                crc=corrupted_crc,
                fault_type=CANFaultType.CORRUPTED_CRC,
            )

        if fault_choice == CANFaultType.BIT_FLIP:
            mutated_bytes = bytearray(frame.payload)
            if mutated_bytes:
                # Flip random bit in byte 0 or 1
                byte_idx = self.rng.randint(0, min(1, len(mutated_bytes) - 1))
                mutated_bytes[byte_idx] ^= 0x01
            return CANFrame(
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_fd=frame.is_fd,
                payload=bytes(mutated_bytes),
                dlc=frame.dlc,
                timestamp_us=frame.timestamp_us,
                crc=frame.crc,
                fault_type=CANFaultType.BIT_FLIP,
            )

        if fault_choice == CANFaultType.TIMING_JITTER:
            # Inject substantial timing jitter exceeding nominal threshold
            jitter = self.rng.uniform(self.max_jitter_us * 0.8, self.max_jitter_us * 1.5)
            return CANFrame(
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_fd=frame.is_fd,
                payload=frame.payload,
                dlc=frame.dlc,
                timestamp_us=frame.timestamp_us + jitter,
                crc=frame.crc,
                fault_type=CANFaultType.TIMING_JITTER,
            )

        if fault_choice == CANFaultType.DLC_MISMATCH:
            # Alter reported DLC
            new_dlc = max(0, frame.dlc - 2)
            return CANFrame(
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_fd=frame.is_fd,
                payload=frame.payload[:new_dlc],
                dlc=new_dlc,
                timestamp_us=frame.timestamp_us,
                crc=frame.crc,
                fault_type=CANFaultType.DLC_MISMATCH,
            )

        return frame
