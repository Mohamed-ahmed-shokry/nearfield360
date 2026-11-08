"""Unit tests for tactical acoustic HUD dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2

from nearfield360.acoustic.models import (
    AcousticDoAResult,
    AcousticReport,
    AudioVisualSectorMatch,
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    SirenDetectionResult,
    SirenType,
)
from nearfield360.acoustic.viz import render_acoustic_dashboard


def test_render_acoustic_dashboard_nominal(tmp_path: Path) -> None:
    report = AcousticReport(
        timestamp_s=10.0,
        siren_detection=SirenDetectionResult(
            detected=False,
            siren_type=SirenType.NONE,
            fundamental_freq_hz=0.0,
            snr_db=2.5,
            confidence=0.0,
        ),
        doa=None,
        doppler=DopplerState.UNKNOWN,
        estimated_radial_speed_mps=0.0,
        yield_state=EmergencyYieldState.CLEAR,
        sector_match=None,
        summary_message="Acoustic field clear.",
    )

    out_file = tmp_path / "nominal_hud.png"
    rendered_path = render_acoustic_dashboard(report, output_path=out_file)

    assert rendered_path == out_file
    assert out_file.is_file()
    assert out_file.stat().st_size > 10000

    img = cv2.imread(str(out_file))
    assert img is not None
    assert img.shape == (720, 1280, 3)


def test_render_acoustic_dashboard_emergency_active(tmp_path: Path) -> None:
    siren = SirenDetectionResult(
        detected=True,
        siren_type=SirenType.WAIL,
        fundamental_freq_hz=1150.0,
        snr_db=22.5,
        confidence=0.96,
        harmonic_energy_ratio=0.85,
        cycle_rate_hz=0.25,
    )
    doa = AcousticDoAResult(
        azimuth_deg=45.0,
        confidence=0.92,
        sound_pressure_level_db=94.5,
        primary_pair=("mic_FL", "mic_FR"),
        time_diff_s=0.00042,
    )
    sector = AudioVisualSectorMatch(
        azimuth_deg=45.0,
        camera_sector=CameraSector.RIGHT,
        visual_confirmation=True,
        fused_confidence=0.97,
        yield_action_recommended=True,
    )
    report = AcousticReport(
        timestamp_s=15.5,
        siren_detection=siren,
        doa=doa,
        doppler=DopplerState.APPROACHING,
        estimated_radial_speed_mps=14.2,
        yield_state=EmergencyYieldState.EMERGENCY_STOP_ACTIVE,
        sector_match=sector,
        summary_message="EMERGENCY VEHICLE INTERLOCK ACTIVE: Approaching WAIL siren on sector MVR.",
    )

    out_file = tmp_path / "emergency_hud.png"
    rendered_path = render_acoustic_dashboard(report, output_path=out_file)

    assert rendered_path.is_file()
    assert rendered_path.stat().st_size > 10000

    img = cv2.imread(str(rendered_path))
    assert img is not None
    assert img.shape == (720, 1280, 3)
