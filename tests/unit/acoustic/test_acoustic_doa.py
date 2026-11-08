"""Unit tests for GCC-PHAT direction-of-arrival (DoA) triangulation engine."""

from __future__ import annotations

import math

import numpy as np

from nearfield360.acoustic.detector import generate_synthetic_siren
from nearfield360.acoustic.doa import (
    AcousticDoAEstimator,
    build_default_microphone_array,
    gcc_phat,
    simulate_multichannel_siren,
)
from nearfield360.acoustic.models import SirenType
from nearfield360.config import AcousticConfig


def test_default_microphone_array() -> None:
    mics = build_default_microphone_array(radius_m=0.8)
    assert len(mics) == 4
    assert mics[0].mic_id == "mic_FL"
    assert mics[0].x_m == 0.8
    assert mics[0].y_m == -0.8
    assert mics[1].mic_id == "mic_FR"
    assert mics[1].x_m == 0.8
    assert mics[1].y_m == 0.8


def test_gcc_phat_time_delay() -> None:
    fs = 16000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    # Reference broadband chirping signal
    sig1 = np.sin(2 * np.pi * 1000.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)

    # Shift sig2 by exactly 5 samples (5 / 16000 = 0.0003125 s)
    delay_samples = 5
    sig2 = np.zeros_like(sig1)
    sig2[delay_samples:] = sig1[:-delay_samples]

    max_tau = 0.005
    tau_est, peak_val = gcc_phat(sig1, sig2, fs=fs, max_tau_s=max_tau)

    expected_tau = -delay_samples / float(fs)  # sig1 leads sig2
    assert math.isclose(tau_est, expected_tau, abs_tol=1.0 / fs)
    assert peak_val > 0.5


def test_doa_estimation_cardinal_angles() -> None:
    config = AcousticConfig(sampling_rate_hz=16000, mic_array_radius_m=0.75)
    mics = build_default_microphone_array(radius_m=0.75)
    estimator = AcousticDoAEstimator(config, mic_specs=mics)

    # 1. Front (0 deg)
    source_sig = generate_synthetic_siren(SirenType.WAIL, duration_s=1.0, sampling_rate_hz=16000)
    audio_front = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=0.0,
        mic_specs=mics,
        fs=16000,
        snr_db=30.0,
    )
    res_front = estimator.estimate_doa(audio_front)
    assert math.isclose(res_front.azimuth_deg, 0.0, abs_tol=4.0)
    assert res_front.confidence > 0.4
    assert res_front.sound_pressure_level_db > 50.0

    # 2. Right (90 deg)
    audio_right = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=90.0,
        mic_specs=mics,
        fs=16000,
        snr_db=30.0,
    )
    res_right = estimator.estimate_doa(audio_right)
    assert math.isclose(res_right.azimuth_deg, 90.0, abs_tol=4.0)

    # 3. Left (-90 deg)
    audio_left = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=-90.0,
        mic_specs=mics,
        fs=16000,
        snr_db=30.0,
    )
    res_left = estimator.estimate_doa(audio_left)
    assert math.isclose(res_left.azimuth_deg, -90.0, abs_tol=4.0)


def test_doa_estimation_oblique_angles() -> None:
    config = AcousticConfig(sampling_rate_hz=16000, mic_array_radius_m=0.75)
    mics = build_default_microphone_array(radius_m=0.75)
    estimator = AcousticDoAEstimator(config, mic_specs=mics)

    source_sig = generate_synthetic_siren(SirenType.YELP, duration_s=1.0, sampling_rate_hz=16000)

    # Front-Right: 45.0 deg
    audio_fr = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=45.0,
        mic_specs=mics,
        fs=16000,
        snr_db=30.0,
    )
    res_fr = estimator.estimate_doa(audio_fr)
    assert math.isclose(res_fr.azimuth_deg, 45.0, abs_tol=5.0)

    # Rear-Left: -135.0 deg
    audio_rl = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=-135.0,
        mic_specs=mics,
        fs=16000,
        snr_db=30.0,
    )
    res_rl = estimator.estimate_doa(audio_rl)
    assert math.isclose(res_rl.azimuth_deg, -135.0, abs_tol=5.0)


def test_doa_edge_cases() -> None:
    config = AcousticConfig()
    estimator = AcousticDoAEstimator(config)

    # Single channel buffer
    single_ch = np.zeros((1, 1000))
    res_single = estimator.estimate_doa(single_ch)
    assert res_single.confidence == 0.0

    # Very short buffer
    tiny = np.zeros((4, 50))
    res_tiny = estimator.estimate_doa(tiny)
    assert res_tiny.confidence == 0.0
