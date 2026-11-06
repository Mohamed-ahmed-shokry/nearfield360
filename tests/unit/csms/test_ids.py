"""Unit tests for In-Vehicle Intrusion Detection System (IDS)."""

from __future__ import annotations

import pytest

from nearfield360.can.models import CANFaultType, CANFrame
from nearfield360.config import CSMSConfig
from nearfield360.csms.ids import IntrusionDetectionSystem, compute_shannon_entropy
from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
)


def test_shannon_entropy_calculation() -> None:
    assert compute_shannon_entropy(b"") == 0.0
    assert compute_shannon_entropy(b"\x00" * 8) == 0.0

    # Alternating 2 symbols has entropy = 1.0 bit
    entropy_2 = compute_shannon_entropy(b"\x00\xff" * 4)
    assert pytest.approx(entropy_2, abs=1e-3) == 1.0

    # 8 distinct bytes has entropy = log2(8) = 3.0 bits
    entropy_8 = compute_shannon_entropy(bytes(range(8)))
    assert pytest.approx(entropy_8, abs=1e-3) == 3.0


def test_ids_nominal_traffic() -> None:
    ids = IntrusionDetectionSystem()
    t = 10.0
    for i in range(20):
        frame = CANFrame(
            arbitration_id=0x120,
            payload=bytes([0x01, i, 0x03, 0x04]),
            dlc=4,
            timestamp_us=t * 1_000_000,
            crc=0,
            fault_type=CANFaultType.NONE,
        )
        evt = ids.inspect_frame(frame, current_time_s=t, vehicle_speed_mps=0.0)
        assert evt is None
        t += 0.020  # 50 Hz


def test_ids_can_dos_flood_detection() -> None:
    config = CSMSConfig(max_can_frame_rate_per_sec=100.0)
    ids = IntrusionDetectionSystem(config=config)

    t = 100.0
    detected_event = None
    for i in range(120):
        frame = CANFrame(
            arbitration_id=0x120,
            payload=bytes([0x10, (i % 256), 0x20, 0x30]),
            dlc=4,
            timestamp_us=t * 1_000_000,
            crc=0,
            fault_type=CANFaultType.NONE,
        )
        evt = ids.inspect_frame(frame, current_time_s=t, vehicle_speed_mps=0.0)
        if evt is not None:
            detected_event = evt
            break
        t += 0.005  # 200 Hz > 100 Hz limit

    assert detected_event is not None
    assert detected_event.category == CyberThreatCategory.CAN_DOS_FLOOD
    assert detected_event.severity == IncidentSeverity.CRITICAL  # 0x120 is powertrain/control
    assert detected_event.defense_action == DefenseAction.REQUEST_SAFE_STOP
    assert detected_event.arbitration_id == 0x120


def test_ids_replay_attack_detection() -> None:
    ids = IntrusionDetectionSystem()
    fixed_payload = b"\xaa\xbb\xcc\xdd\xee\xff"

    frame1 = CANFrame(
        arbitration_id=0x150,
        payload=fixed_payload,
        dlc=6,
        timestamp_us=50.0 * 1_000_000,
        crc=0,
        fault_type=CANFaultType.NONE,
    )
    evt1 = ids.inspect_frame(frame1, current_time_s=50.0)
    assert evt1 is None

    # Immediate identical payload retransmitted 5ms later (< 50ms tolerance)
    frame2 = CANFrame(
        arbitration_id=0x150,
        payload=fixed_payload,
        dlc=6,
        timestamp_us=50.005 * 1_000_000,
        crc=0,
        fault_type=CANFaultType.NONE,
    )
    evt2 = ids.inspect_frame(frame2, current_time_s=50.005)
    assert evt2 is not None
    assert evt2.category == CyberThreatCategory.REPLAY_ATTACK
    assert evt2.severity == IncidentSeverity.MEDIUM
    assert evt2.defense_action == DefenseAction.DROP_FRAME


def test_ids_unauthorized_uds_diagnostic() -> None:
    ids = IntrusionDetectionSystem()
    # 0x7DF diagnostic broadcast with Service 0x27 (SecurityAccess)
    diag_payload = bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
    frame = CANFrame(
        arbitration_id=0x7DF,
        payload=diag_payload,
        dlc=8,
        timestamp_us=25.0 * 1_000_000,
        crc=0,
        fault_type=CANFaultType.NONE,
    )

    # Allowed if standstill
    evt_standstill = ids.inspect_frame(frame, current_time_s=25.0, vehicle_speed_mps=0.0)
    assert evt_standstill is None

    # Prohibited if vehicle in motion (> 0.05 m/s)
    evt_motion = ids.inspect_frame(frame, current_time_s=25.1, vehicle_speed_mps=3.5)
    assert evt_motion is not None
    assert evt_motion.category == CyberThreatCategory.UDS_UNAUTHORIZED_DIAG
    assert evt_motion.severity == IncidentSeverity.CRITICAL
    assert evt_motion.defense_action == DefenseAction.LOCKOUT_DIAGNOSTICS


def test_ids_payload_fuzzing_detection() -> None:
    config = CSMSConfig(payload_entropy_max=4.5)  # Restrict entropy to 4.5
    ids = IntrusionDetectionSystem(config=config)

    # 64 distinct bytes has entropy = log2(64) = 6.0 bits > 4.5
    fuzzed_payload = bytes(range(64))
    frame = CANFrame(
        arbitration_id=0x200,
        is_fd=True,
        payload=fuzzed_payload,
        dlc=64,
        timestamp_us=30.0 * 1_000_000,
        crc=0,
        fault_type=CANFaultType.NONE,
    )
    evt = ids.inspect_frame(frame, current_time_s=30.0)
    assert evt is not None
    assert evt.category == CyberThreatCategory.ANOMALOUS_PAYLOAD_FUZZ
    assert evt.defense_action == DefenseAction.DROP_FRAME
