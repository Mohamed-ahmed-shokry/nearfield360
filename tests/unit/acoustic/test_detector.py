"""Unit tests for STFT siren detector and modulation pattern classifier."""

from __future__ import annotations

import numpy as np

from nearfield360.acoustic.detector import SirenDetector, compute_stft, generate_synthetic_siren
from nearfield360.acoustic.models import SirenType
from nearfield360.config import AcousticConfig


def test_compute_stft() -> None:
    fs = 16000
    t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
    # 1000 Hz pure tone
    sig = np.sin(2 * np.pi * 1000.0 * t)

    stft_mat = compute_stft(sig, window_size=512, hop_size=128)
    assert stft_mat.ndim == 2
    assert stft_mat.shape[1] == 257  # 512 // 2 + 1 freq bins

    # Short signal padding
    short_sig = np.sin(2 * np.pi * 1000.0 * t[:100])
    stft_short = compute_stft(short_sig, window_size=512, hop_size=128)
    assert stft_short.shape[0] >= 1


def test_detect_wail_siren() -> None:
    config = AcousticConfig()
    detector = SirenDetector(config)

    audio = generate_synthetic_siren(
        siren_type=SirenType.WAIL,
        duration_s=2.5,
        sampling_rate_hz=config.sampling_rate_hz,
        snr_db=25.0,
    )
    result = detector.analyze_signal(audio)

    assert result.detected is True
    assert result.siren_type == SirenType.WAIL
    assert result.confidence >= config.min_confidence_threshold
    assert 600.0 <= result.fundamental_freq_hz <= 1500.0
    assert result.snr_db >= config.min_siren_snr_db


def test_detect_yelp_siren() -> None:
    config = AcousticConfig()
    detector = SirenDetector(config)

    audio = generate_synthetic_siren(
        siren_type=SirenType.YELP,
        duration_s=2.0,
        sampling_rate_hz=config.sampling_rate_hz,
        snr_db=25.0,
    )
    result = detector.analyze_signal(audio)

    assert result.detected is True
    assert result.siren_type == SirenType.YELP
    assert result.cycle_rate_hz >= 1.2
    assert result.confidence >= config.min_confidence_threshold


def test_detect_hi_lo_siren() -> None:
    config = AcousticConfig()
    detector = SirenDetector(config)

    audio = generate_synthetic_siren(
        siren_type=SirenType.HI_LO,
        duration_s=2.5,
        sampling_rate_hz=config.sampling_rate_hz,
        snr_db=25.0,
    )
    result = detector.analyze_signal(audio)

    assert result.detected is True
    assert result.siren_type == SirenType.HI_LO
    assert 400.0 <= result.fundamental_freq_hz <= 700.0


def test_reject_ambient_noise() -> None:
    config = AcousticConfig()
    detector = SirenDetector(config)

    # Pure ambient noise with no siren
    noise = generate_synthetic_siren(
        siren_type=SirenType.NONE,
        duration_s=1.5,
        sampling_rate_hz=config.sampling_rate_hz,
        snr_db=-5.0,
    )
    result = detector.analyze_signal(noise)

    assert result.detected is False
    assert result.siren_type == SirenType.NONE
    assert result.confidence == 0.0


def test_short_audio_rejection() -> None:
    config = AcousticConfig(stft_window_samples=512)
    detector = SirenDetector(config)

    tiny_audio = np.zeros(200, dtype=np.float32)
    result = detector.analyze_signal(tiny_audio)

    assert result.detected is False
    assert result.siren_type == SirenType.NONE
