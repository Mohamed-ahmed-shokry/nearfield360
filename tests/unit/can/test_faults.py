"""Unit tests for synthetic CAN fault injection and timing jitter."""

from __future__ import annotations

from nearfield360.can.codec import CANCodec
from nearfield360.can.faults import FaultInjector
from nearfield360.can.models import CANFaultType, VehicleControlSignal


def test_fault_injector_zero_rate_applies_only_nominal_jitter() -> None:
    injector = FaultInjector(fault_rate=0.0, max_jitter_us=500.0, seed=123)
    ctrl = VehicleControlSignal(gear="D")
    frame = CANCodec.encode_control_signal(ctrl, timestamp_us=1000.0)

    for _ in range(50):
        out = injector.process_frame(frame)
        assert out is not None
        assert out.fault_type == CANFaultType.NONE
        assert out.arbitration_id == frame.arbitration_id
        assert out.payload == frame.payload
        assert 1000.0 <= out.timestamp_us <= 1025.0


def test_fault_injector_high_rate_produces_faults() -> None:
    injector = FaultInjector(fault_rate=1.0, max_jitter_us=500.0, seed=42)
    ctrl = VehicleControlSignal(gear="N")
    frame = CANCodec.encode_control_signal(ctrl, timestamp_us=1000.0)

    fault_counts: dict[str, int] = {}
    for _ in range(100):
        out = injector.process_frame(frame)
        if out is None:
            fault_counts["dropped"] = fault_counts.get("dropped", 0) + 1
        else:
            fault_counts[out.fault_type.value] = fault_counts.get(out.fault_type.value, 0) + 1

    assert fault_counts.get("dropped", 0) > 0
    assert fault_counts.get(CANFaultType.CORRUPTED_CRC.value, 0) > 0
    assert fault_counts.get(CANFaultType.BIT_FLIP.value, 0) > 0
    assert fault_counts.get(CANFaultType.TIMING_JITTER.value, 0) > 0
    assert fault_counts.get(CANFaultType.DLC_MISMATCH.value, 0) > 0


def test_fault_injector_determinism_per_seed() -> None:
    ctrl = VehicleControlSignal(gear="P")
    frame = CANCodec.encode_control_signal(ctrl, timestamp_us=0.0)

    inj1 = FaultInjector(fault_rate=0.5, seed=99)
    inj2 = FaultInjector(fault_rate=0.5, seed=99)

    seq1 = [inj1.process_frame(frame) for _ in range(20)]
    seq2 = [inj2.process_frame(frame) for _ in range(20)]

    assert seq1 == seq2
