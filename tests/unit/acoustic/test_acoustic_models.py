"""Unit tests for acoustic domain models, classifications, and sector mapping."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nearfield360.acoustic.models import (
    AcousticDoAResult,
    AcousticReport,
    AudioVisualSectorMatch,
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    MicrophoneSpec,
    SirenDetectionResult,
    SirenType,
    azimuth_to_camera_sector,
)


def test_azimuth_to_camera_sector() -> None:
    # Front Sector: [-45, 45)
    assert azimuth_to_camera_sector(0.0) == CameraSector.FRONT
    assert azimuth_to_camera_sector(30.0) == CameraSector.FRONT
    assert azimuth_to_camera_sector(-30.0) == CameraSector.FRONT
    assert azimuth_to_camera_sector(-45.0) == CameraSector.FRONT

    # Right Sector: [45, 135)
    assert azimuth_to_camera_sector(45.0) == CameraSector.RIGHT
    assert azimuth_to_camera_sector(90.0) == CameraSector.RIGHT
    assert azimuth_to_camera_sector(134.9) == CameraSector.RIGHT

    # Left Sector: [-135, -45)
    assert azimuth_to_camera_sector(-45.1) == CameraSector.LEFT
    assert azimuth_to_camera_sector(-90.0) == CameraSector.LEFT
    assert azimuth_to_camera_sector(-135.0) == CameraSector.LEFT

    # Rear Sector: [135, 180] and [-180, -135)
    assert azimuth_to_camera_sector(135.0) == CameraSector.REAR
    assert azimuth_to_camera_sector(180.0) == CameraSector.REAR
    assert azimuth_to_camera_sector(-180.0) == CameraSector.REAR
    assert azimuth_to_camera_sector(-150.0) == CameraSector.REAR

    # Wrap-around angles (> 180 or < -180)
    assert azimuth_to_camera_sector(360.0) == CameraSector.FRONT
    assert azimuth_to_camera_sector(450.0) == CameraSector.RIGHT
    assert azimuth_to_camera_sector(-270.0) == CameraSector.RIGHT


def test_microphone_spec() -> None:
    mic = MicrophoneSpec(mic_id="mic_front_01", x_m=1.2, y_m=0.0, z_m=1.5)
    assert mic.mic_id == "mic_front_01"
    assert mic.x_m == 1.2
    assert mic.y_m == 0.0
    assert mic.z_m == 1.5

    data = mic.model_dump(mode="json")
    assert data["mic_id"] == "mic_front_01"
    assert data["x_m"] == 1.2

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        MicrophoneSpec.model_validate({"mic_id": "m1", "x_m": 0.0, "y_m": 0.0, "extra": 123})


def test_siren_detection_result() -> None:
    res = SirenDetectionResult(
        detected=True,
        siren_type=SirenType.WAIL,
        fundamental_freq_hz=950.0,
        snr_db=14.5,
        confidence=0.92,
        harmonic_energy_ratio=0.78,
        cycle_rate_hz=0.25,
    )
    assert res.detected is True
    assert res.siren_type == SirenType.WAIL
    assert res.fundamental_freq_hz == 950.0

    # Bounds validation
    with pytest.raises(ValidationError):
        SirenDetectionResult(confidence=1.5)
    with pytest.raises(ValidationError):
        SirenDetectionResult(fundamental_freq_hz=-10.0)


def test_acoustic_doa_result() -> None:
    doa = AcousticDoAResult(
        azimuth_deg=45.0,
        elevation_deg=0.0,
        confidence=0.88,
        sound_pressure_level_db=85.0,
        primary_pair=("mic_FL", "mic_FR"),
        time_diff_s=0.00045,
    )
    assert doa.azimuth_deg == 45.0
    assert doa.primary_pair == ("mic_FL", "mic_FR")

    # Invalid azimuth bounds
    with pytest.raises(ValidationError):
        AcousticDoAResult(
            azimuth_deg=250.0,
            primary_pair=("m1", "m2"),
        )


def test_acoustic_report_and_emergency_active() -> None:
    siren = SirenDetectionResult(
        detected=True,
        siren_type=SirenType.YELP,
        fundamental_freq_hz=1100.0,
        snr_db=18.0,
        confidence=0.95,
        harmonic_energy_ratio=0.82,
        cycle_rate_hz=3.2,
    )
    doa = AcousticDoAResult(
        azimuth_deg=-60.0,
        confidence=0.89,
        sound_pressure_level_db=92.0,
        primary_pair=("mic_FL", "mic_RL"),
        time_diff_s=-0.0008,
    )
    sector = AudioVisualSectorMatch(
        azimuth_deg=-60.0,
        camera_sector=CameraSector.LEFT,
        visual_confirmation=True,
        fused_confidence=0.98,
        yield_action_recommended=True,
    )

    report = AcousticReport(
        timestamp_s=12.5,
        siren_detection=siren,
        doa=doa,
        doppler=DopplerState.APPROACHING,
        estimated_radial_speed_mps=15.2,
        yield_state=EmergencyYieldState.EMERGENCY_STOP_ACTIVE,
        sector_match=sector,
        summary_message="Approaching police siren on left flank.",
    )

    assert report.is_emergency_active() is True

    # Receding siren should not command active emergency
    report_receding = report.model_copy(update={"doppler": DopplerState.RECEDING})
    assert report_receding.is_emergency_active() is False

    # Clear yield state should not trigger active emergency
    report_clear = report.model_copy(update={"yield_state": EmergencyYieldState.CLEAR})
    assert report_clear.is_emergency_active() is False

    # JSON serialization
    serialized = json.loads(report.model_dump_json())
    assert serialized["siren_detection"]["siren_type"] == "yelp"
    assert serialized["sector_match"]["camera_sector"] == "MVL"
    assert serialized["doppler"] == "approaching"
