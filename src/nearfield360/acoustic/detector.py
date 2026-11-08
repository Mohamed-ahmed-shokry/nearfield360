"""Short-Time Fourier Transform (STFT) siren detection and modulation classification."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.acoustic.models import SirenDetectionResult, SirenType

if TYPE_CHECKING:
    from nearfield360.config import AcousticConfig


def compute_stft(
    signal: np.ndarray,
    window_size: int = 512,
    hop_size: int = 128,
) -> np.ndarray:
    """Compute Short-Time Fourier Transform (STFT) using a Hann window.

    Returns:
        stft_matrix with shape (num_frames, num_freq_bins).
    """
    n_samples = len(signal)
    if n_samples < window_size:
        # Pad with zeros if signal shorter than window
        signal = np.pad(signal, (0, window_size - n_samples))
        n_samples = window_size

    window = np.hanning(window_size)
    num_frames = max(1, 1 + (n_samples - window_size) // hop_size)

    frames = np.lib.stride_tricks.sliding_window_view(
        signal[: (num_frames - 1) * hop_size + window_size],
        window_shape=window_size,
    )[::hop_size]

    windowed = frames * window
    stft_matrix = np.fft.rfft(windowed, n=window_size)
    return stft_matrix


class SirenDetector:
    """Spectral analyzer and emergency acoustic siren modulation classifier."""

    def __init__(self, config: AcousticConfig) -> None:
        self.config = config
        self.fs = config.sampling_rate_hz
        self.win_size = config.stft_window_samples
        self.hop_size = config.stft_hop_samples
        self.freq_bins = np.fft.rfftfreq(self.win_size, d=1.0 / self.fs)

    def analyze_signal(self, audio: np.ndarray) -> SirenDetectionResult:
        """Analyze a monophonic or primary channel audio buffer for siren signatures."""
        if audio.ndim > 1:
            # Use first channel if multichannel
            audio = audio[0]

        if len(audio) < self.win_size:
            return SirenDetectionResult(detected=False, siren_type=SirenType.NONE)

        stft_matrix = compute_stft(audio, window_size=self.win_size, hop_size=self.hop_size)
        mag_spec = np.abs(stft_matrix)
        power_spec = mag_spec**2

        # 1. Frequency masking for siren band
        f_min = self.config.siren_freq_min_hz
        f_max = self.config.siren_freq_max_hz
        in_band_mask = (self.freq_bins >= f_min) & (self.freq_bins <= f_max)
        out_band_mask = ~in_band_mask

        in_band_power = np.sum(power_spec[:, in_band_mask], axis=1)
        out_band_power = np.sum(power_spec[:, out_band_mask], axis=1)

        mean_in_band = float(np.mean(in_band_power))
        mean_out_band = float(np.mean(out_band_power)) + 1e-9
        snr_db = 10.0 * math.log10(max(1e-9, mean_in_band / mean_out_band))

        if snr_db < self.config.min_siren_snr_db:
            return SirenDetectionResult(
                detected=False,
                siren_type=SirenType.NONE,
                fundamental_freq_hz=0.0,
                snr_db=round(snr_db, 2),
                confidence=0.0,
            )

        # 2. Track fundamental peak frequency per frame in siren band
        in_band_indices = np.where(in_band_mask)[0]
        peak_freqs = []
        harmonic_ratios = []

        for frame_idx in range(len(mag_spec)):
            frame_mag = mag_spec[frame_idx]
            in_band_mag = frame_mag[in_band_mask]
            if len(in_band_mag) == 0:
                continue

            local_peak_idx = int(np.argmax(in_band_mag))
            global_peak_idx = in_band_indices[local_peak_idx]
            peak_freq = float(self.freq_bins[global_peak_idx])
            peak_freqs.append(peak_freq)

            # Harmonic energy check: fundamental + 2nd harmonic
            harm2_idx = round(global_peak_idx * 2)
            harm_power = frame_mag[global_peak_idx] ** 2
            if harm2_idx < len(frame_mag):
                harm_power += frame_mag[harm2_idx] ** 2
            total_frame_power = float(np.sum(frame_mag[in_band_mask] ** 2)) + 1e-9
            harmonic_ratios.append(min(1.0, harm_power / total_frame_power))

        if not peak_freqs:
            return SirenDetectionResult(detected=False, siren_type=SirenType.NONE)

        peak_freq_arr = np.array(peak_freqs)
        mean_fund_freq = float(np.mean(peak_freq_arr))
        mean_harm_ratio = float(np.mean(harmonic_ratios)) if harmonic_ratios else 0.5
        freq_sweep_range = float(np.ptp(peak_freq_arr))

        # 3. Frequency modulation cycle rate estimation (zero-crossing rate of pitch curve)
        detrended_freq = peak_freq_arr - mean_fund_freq
        zero_crossings = np.where(np.diff(np.signbit(detrended_freq)))[0]
        time_per_frame = self.hop_size / self.fs
        total_time_s = len(peak_freq_arr) * time_per_frame

        if total_time_s > 0 and len(zero_crossings) >= 2:
            cycle_rate_hz = (len(zero_crossings) / 2.0) / total_time_s
        else:
            cycle_rate_hz = 0.0

        # 4. Modulation pattern classification
        siren_type, match_confidence = self._classify_modulation(
            peak_freq_arr, freq_sweep_range, cycle_rate_hz, snr_db
        )

        snr_factor = min(1.0, max(0.0, (snr_db - self.config.min_siren_snr_db) / 10.0))
        final_conf = min(
            1.0,
            round(match_confidence * (0.80 + 0.20 * snr_factor), 4),
        )

        detected = (
            siren_type != SirenType.NONE and final_conf >= self.config.min_confidence_threshold
        )

        return SirenDetectionResult(
            detected=detected,
            siren_type=siren_type if detected else SirenType.NONE,
            fundamental_freq_hz=round(mean_fund_freq, 1),
            snr_db=round(snr_db, 2),
            confidence=final_conf if detected else 0.0,
            harmonic_energy_ratio=round(mean_harm_ratio, 4),
            cycle_rate_hz=round(cycle_rate_hz, 2),
        )

    def _classify_modulation(
        self,
        freq_trajectory: np.ndarray,
        sweep_range_hz: float,
        cycle_rate_hz: float,
        snr_db: float,
    ) -> tuple[SirenType, float]:
        """Classify siren frequency modulation pattern into Wail, Yelp, Hi-Lo, or Horn."""
        # Check Hi-Lo (two alternating frequencies with bimodal distribution)
        hist, _ = np.histogram(freq_trajectory, bins=8)
        top_bins = np.sort(hist)[-2:]
        if (
            len(freq_trajectory) >= 10
            and top_bins[0] > len(freq_trajectory) * 0.25
            and top_bins[1] > len(freq_trajectory) * 0.25
            and 150.0 <= sweep_range_hz <= 400.0
        ):
            return SirenType.HI_LO, 0.90

        # Check Yelp: rapid pitch oscillation (1.5 Hz to 5.0 Hz) with substantial sweep
        if sweep_range_hz >= 300.0 and 1.2 <= cycle_rate_hz <= 6.0:
            return SirenType.YELP, 0.95

        # Check Wail: slow continuous pitch sweep (0.1 Hz to 0.8 Hz) with substantial sweep
        if sweep_range_hz >= 350.0 and 0.05 <= cycle_rate_hz <= 0.9:
            return SirenType.WAIL, 0.93

        # Check Horn: steady piercing tone with high SNR and narrow frequency spread
        if sweep_range_hz < 150.0 and snr_db >= 10.0:
            return SirenType.HORN, 0.80

        # Fallback if wide sweep but irregular
        if sweep_range_hz >= 400.0:
            return SirenType.WAIL, 0.70

        return SirenType.NONE, 0.0


def generate_synthetic_siren(
    siren_type: SirenType,
    duration_s: float = 2.0,
    sampling_rate_hz: int = 16000,
    snr_db: float = 20.0,
    doppler_shift_hz: float = 0.0,
) -> np.ndarray:
    """Generate synthetic siren audio waveform for testing and simulation."""
    t = np.linspace(0, duration_s, int(sampling_rate_hz * duration_s), endpoint=False)

    if siren_type == SirenType.WAIL:
        # Swept cosine between 700 Hz and 1400 Hz at 0.25 Hz cycle
        f_center = 1050.0 + doppler_shift_hz
        f_amp = 350.0
        mod_rate = 0.25
        phase = 2 * np.pi * f_center * t - (f_amp / mod_rate) * np.cos(2 * np.pi * mod_rate * t)
        signal = np.sin(phase) + 0.3 * np.sin(2 * phase)
    elif siren_type == SirenType.YELP:
        # Fast swept cosine between 700 Hz and 1400 Hz at 2.5 Hz cycle
        f_center = 1050.0 + doppler_shift_hz
        f_amp = 350.0
        mod_rate = 2.5
        phase = 2 * np.pi * f_center * t - (f_amp / mod_rate) * np.cos(2 * np.pi * mod_rate * t)
        signal = np.sin(phase) + 0.3 * np.sin(2 * phase)
    elif siren_type == SirenType.HI_LO:
        # Alternates between 450 Hz and 650 Hz every 0.5s
        f_hi = 650.0 + doppler_shift_hz
        f_lo = 450.0 + doppler_shift_hz
        freqs = np.where((t % 1.0) < 0.5, f_hi, f_lo)
        phase = 2 * np.pi * np.cumsum(freqs) / sampling_rate_hz
        signal = np.sin(phase)
    elif siren_type == SirenType.HORN:
        # Steady 500 Hz tone + rich harmonics
        f_horn = 500.0 + doppler_shift_hz
        signal = (
            np.sin(2 * np.pi * f_horn * t)
            + 0.5 * np.sin(2 * np.pi * 2 * f_horn * t)
            + 0.25 * np.sin(2 * np.pi * 3 * f_horn * t)
        )
    else:
        # Pure ambient noise (no siren)
        signal = np.zeros_like(t)

    # Normalize signal power to 1.0
    sig_power = float(np.mean(signal**2))
    if sig_power > 0:
        signal = signal / math.sqrt(sig_power)

    # Add Gaussian background noise scaled to requested SNR
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise = np.random.default_rng(42).normal(0.0, math.sqrt(noise_power), len(t))

    composite = signal + noise
    max_amp = float(np.max(np.abs(composite))) + 1e-9
    return composite / max_amp
