"""Unit tests for acoustic sensing executive, Doppler tracking, and yield interlocks."""

from __future__ import annotations

from unittest.mock import MagicMock

from nearfield360.acoustic.detector import generate_synthetic_siren
from nearfield360.acoustic.doa import build_default_microphone_array, simulate_multichannel_siren
from nearfield360.acoustic.executive import AcousticExecutive
from nearfield360.acoustic.models import (
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    SirenType,
)
from nearfield360.config import AcousticConfig


def test_executive_clear_field() -> None:
    config = AcousticConfig()
    exec_engine = AcousticExecutive(config)

    # Ambient noise only
    mics = build_default_microphone_array()
    noise = generate_synthetic_siren(SirenType.NONE, duration_s=1.0)
    audio = simulate_multichannel_siren(noise, azimuth_deg=0.0, mic_specs=mics)

    report = exec_engine.process_audio(audio, timestamp_s=1.0)

    assert report.siren_detection.detected is False
    assert report.yield_state == EmergencyYieldState.CLEAR
    assert report.is_emergency_active() is False
    assert "Acoustic field clear" in report.summary_message


def test_executive_approaching_siren_triggers_stop() -> None:
    config = AcousticConfig(mic_array_radius_m=0.75)
    mics = build_default_microphone_array(0.75)
    exec_engine = AcousticExecutive(config)

    # Frame 1: nominal wail at t=1.0s (anchor frequency)
    sig_1 = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=0.0)
    audio_1 = simulate_multichannel_siren(sig_1, azimuth_deg=10.0, mic_specs=mics)
    report_1 = exec_engine.process_audio(audio_1, timestamp_s=1.0, estimated_distance_m=40.0)

    assert report_1.siren_detection.detected is True
    assert report_1.yield_state in (
        EmergencyYieldState.MONITORING,
        EmergencyYieldState.YIELD_PREPARED,
    )

    # Frame 2 & 3: positive Doppler shift (+60 Hz approaching) at t=2.0s & t=3.0s, distance = 25m
    sig_2 = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=60.0)
    audio_2 = simulate_multichannel_siren(sig_2, azimuth_deg=10.0, mic_specs=mics)
    exec_engine.process_audio(audio_2, timestamp_s=2.0, estimated_distance_m=28.0)

    report_3 = exec_engine.process_audio(audio_2, timestamp_s=3.0, estimated_distance_m=22.0)

    assert report_3.siren_detection.detected is True
    assert report_3.doppler == DopplerState.APPROACHING
    assert report_3.estimated_radial_speed_mps > 2.0
    assert report_3.yield_state == EmergencyYieldState.EMERGENCY_STOP_ACTIVE
    assert report_3.is_emergency_active() is True
    assert report_3.sector_match is not None
    assert report_3.sector_match.camera_sector == CameraSector.FRONT
    assert report_3.sector_match.yield_action_recommended is True


def test_executive_safety_supervisor_interlock() -> None:
    config = AcousticConfig(enable_yield_interlock=True)
    mics = build_default_microphone_array()
    mock_supervisor = MagicMock()
    exec_engine = AcousticExecutive(config, safety_supervisor=mock_supervisor)

    # Prime anchor
    sig = generate_synthetic_siren(SirenType.YELP, duration_s=1.0, doppler_shift_hz=0.0)
    audio = simulate_multichannel_siren(sig, azimuth_deg=90.0, mic_specs=mics)
    exec_engine.process_audio(audio, timestamp_s=1.0)

    # Approaching frame with distance <= 35m
    sig_app = generate_synthetic_siren(SirenType.YELP, duration_s=1.0, doppler_shift_hz=50.0)
    audio_app = simulate_multichannel_siren(sig_app, azimuth_deg=90.0, mic_specs=mics)
    exec_engine.process_audio(audio_app, timestamp_s=2.0, estimated_distance_m=20.0)
    exec_engine.process_audio(audio_app, timestamp_s=3.0, estimated_distance_m=18.0)

    mock_supervisor.trigger_emergency_stop.assert_called()


def test_executive_visual_confirmation_escalation() -> None:
    config = AcousticConfig(yield_distance_threshold_m=35.0)
    mics = build_default_microphone_array()
    exec_engine = AcousticExecutive(config)

    # Setup approaching siren at 60m (> threshold 35m)
    sig_anchor = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=0.0)
    audio_anchor = simulate_multichannel_siren(sig_anchor, azimuth_deg=0.0, mic_specs=mics)
    exec_engine.process_audio(audio_anchor, timestamp_s=1.0)

    sig_app = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=50.0)
    audio_app = simulate_multichannel_siren(sig_app, azimuth_deg=0.0, mic_specs=mics)
    exec_engine.process_audio(audio_app, timestamp_s=2.0)

    # Without visual confirmation at 60m: YIELD_PREPARED
    rep_no_vis = exec_engine.process_audio(audio_app, timestamp_s=3.0, estimated_distance_m=60.0)
    assert rep_no_vis.yield_state == EmergencyYieldState.YIELD_PREPARED

    # With visual confirmation on FRONT camera: escalates to EMERGENCY_STOP_ACTIVE
    rep_vis = exec_engine.process_audio(
        audio_app,
        timestamp_s=4.0,
        estimated_distance_m=60.0,
        visual_camera_sectors=[CameraSector.FRONT],
    )
    assert rep_vis.yield_state == EmergencyYieldState.EMERGENCY_STOP_ACTIVE
    assert rep_vis.sector_match is not None
    assert rep_vis.sector_match.visual_confirmation is True
    assert rep_vis.sector_match.fused_confidence > 0.80


def test_executive_receding_siren() -> None:
    config = AcousticConfig()
    mics = build_default_microphone_array()
    exec_engine = AcousticExecutive(config)

    sig_anchor = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=0.0)
    audio_anchor = simulate_multichannel_siren(sig_anchor, azimuth_deg=180.0, mic_specs=mics)
    exec_engine.process_audio(audio_anchor, timestamp_s=1.0)

    # Negative Doppler shift (-60 Hz receding)
    sig_rec = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, doppler_shift_hz=-60.0)
    audio_rec = simulate_multichannel_siren(sig_rec, azimuth_deg=180.0, mic_specs=mics)
    exec_engine.process_audio(audio_rec, timestamp_s=2.0)
    rep_rec = exec_engine.process_audio(audio_rec, timestamp_s=3.0, estimated_distance_m=50.0)

    assert rep_rec.doppler == DopplerState.RECEDING
    assert rep_rec.estimated_radial_speed_mps < -2.0
    assert rep_rec.yield_state == EmergencyYieldState.MONITORING
    assert rep_rec.is_emergency_active() is False
