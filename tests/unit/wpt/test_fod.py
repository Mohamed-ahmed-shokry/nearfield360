"""Unit tests for FOD and LOD safety interlock engine."""

from nearfield360.config import WPTConfig
from nearfield360.wpt.fod import FODDetector
from nearfield360.wpt.models import WPTInterlockState


def test_nominal_resonance_no_faults() -> None:
    detector = FODDetector()
    status = detector.evaluate_resonance(
        measured_frequency_khz=85.0,
        measured_q_factor=280.0,
        living_object_present=False,
    )
    assert status.metal_object_detected is False
    assert status.living_object_detected is False
    assert status.power_inhibited is False
    assert status.interlock_state == WPTInterlockState.NORMAL
    assert status.frequency_shift_khz == 0.0
    assert status.q_drop_percent == 0.0


def test_fod_metal_detection_via_q_drop() -> None:
    cfg = WPTConfig(fod_q_drop_threshold_percent=15.0)
    detector = FODDetector(config=cfg, nominal_q_factor=300.0)

    # 300 -> 240 is 20% drop (exceeds 15% threshold)
    status = detector.evaluate_resonance(
        measured_frequency_khz=85.1,
        measured_q_factor=240.0,
    )
    assert status.metal_object_detected is True
    assert status.power_inhibited is True
    assert status.interlock_state == WPTInterlockState.FOD_FAULT
    assert "FOD FAULT" in status.reason
    assert status.q_drop_percent == 20.0


def test_fod_metal_detection_via_freq_shift() -> None:
    cfg = WPTConfig(fod_freq_shift_threshold_khz=1.5)
    detector = FODDetector(config=cfg)

    # 85.0 -> 87.2 is 2.2 kHz shift (exceeds 1.5 kHz threshold)
    status = detector.evaluate_resonance(
        measured_frequency_khz=87.2,
        measured_q_factor=275.0,
    )
    assert status.metal_object_detected is True
    assert status.power_inhibited is True
    assert status.interlock_state == WPTInterlockState.FOD_FAULT
    assert status.frequency_shift_khz == 2.2


def test_lod_living_object_fault() -> None:
    detector = FODDetector()
    status = detector.evaluate_resonance(
        measured_frequency_khz=85.0,
        measured_q_factor=280.0,
        living_object_present=True,
    )
    assert status.living_object_detected is True
    assert status.power_inhibited is True
    assert status.interlock_state == WPTInterlockState.LOD_FAULT
    assert "LOD FAULT" in status.reason


def test_manual_emergency_shutdown() -> None:
    detector = FODDetector()
    status = detector.evaluate_resonance(
        measured_frequency_khz=85.0,
        measured_q_factor=280.0,
        manual_emergency_trip=True,
    )
    assert status.power_inhibited is True
    assert status.interlock_state == WPTInterlockState.EMERGENCY_SHUTDOWN
