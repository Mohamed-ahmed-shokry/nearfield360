"""Direction-of-Arrival (DoA) triangulation engine using GCC-PHAT."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.acoustic.models import AcousticDoAResult, MicrophoneSpec

if TYPE_CHECKING:
    from nearfield360.config import AcousticConfig


def build_default_microphone_array(radius_m: float = 0.75) -> list[MicrophoneSpec]:
    """Create a default 4-microphone perimeter planar array on vehicle roof."""
    return [
        MicrophoneSpec(mic_id="mic_FL", x_m=radius_m, y_m=-radius_m, z_m=1.5),
        MicrophoneSpec(mic_id="mic_FR", x_m=radius_m, y_m=radius_m, z_m=1.5),
        MicrophoneSpec(mic_id="mic_RL", x_m=-radius_m, y_m=-radius_m, z_m=1.5),
        MicrophoneSpec(mic_id="mic_RR", x_m=-radius_m, y_m=radius_m, z_m=1.5),
    ]


def gcc_phat(
    sig_i: np.ndarray,
    sig_j: np.ndarray,
    fs: int,
    max_tau_s: float,
) -> tuple[float, float]:
    """Compute Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

    Returns:
        (tau_seconds, peak_correlation_value)
    """
    n = len(sig_i) + len(sig_j)
    # Next power of 2 for fast FFT
    n_fft = 2 ** math.ceil(math.log2(max(512, n)))

    fft_i = np.fft.rfft(sig_i, n=n_fft)
    fft_j = np.fft.rfft(sig_j, n=n_fft)

    cross_spec = fft_i * np.conj(fft_j)
    denom = np.abs(cross_spec) + 1e-12
    norm_cross_spec = cross_spec / denom

    cc = np.fft.irfft(norm_cross_spec, n=n_fft)
    # Zero delay is at index 0; shift so zero delay is centered
    half = n_fft // 2
    cc = np.concatenate((cc[half:], cc[:half]))

    max_samples = math.ceil(max_tau_s * fs)
    center = half
    search_start = max(0, center - max_samples)
    search_end = min(n_fft, center + max_samples + 1)

    sub_cc = cc[search_start:search_end]
    if len(sub_cc) < 3:
        return 0.0, 0.0

    peak_local_idx = int(np.argmax(sub_cc))
    peak_global_idx = search_start + peak_local_idx

    # Parabolic sub-sample peak refinement
    delta_sample = 0.0
    if 0 < peak_local_idx < len(sub_cc) - 1:
        alpha = float(sub_cc[peak_local_idx - 1])
        beta = float(sub_cc[peak_local_idx])
        gamma = float(sub_cc[peak_local_idx + 1])
        denom_p = alpha - 2.0 * beta + gamma
        if abs(denom_p) > 1e-12:
            delta_sample = 0.5 * (alpha - gamma) / denom_p

    refined_sample = (peak_global_idx - center) + delta_sample
    tau_s = refined_sample / float(fs)
    peak_val = float(sub_cc[peak_local_idx])

    return tau_s, peak_val


class AcousticDoAEstimator:
    """Direction-of-Arrival (DoA) triangulation engine using multi-microphone GCC-PHAT."""

    def __init__(
        self,
        config: AcousticConfig,
        mic_specs: list[MicrophoneSpec] | None = None,
    ) -> None:
        self.config = config
        self.fs = config.sampling_rate_hz
        self.c = config.sound_speed_mps
        self.mics = mic_specs or build_default_microphone_array(config.mic_array_radius_m)

        # Build unique microphone pairs
        self.pairs: list[tuple[int, int]] = []
        for i in range(len(self.mics)):
            for j in range(i + 1, len(self.mics)):
                self.pairs.append((i, j))

    def estimate_doa(self, multichannel_audio: np.ndarray) -> AcousticDoAResult:
        """Estimate 2D planar acoustic azimuth from multichannel audio buffer.

        multichannel_audio shape: (C, N) where C is number of microphones.
        """
        num_channels, num_samples = multichannel_audio.shape
        if num_channels < 2 or num_samples < self.config.stft_window_samples:
            return AcousticDoAResult(
                azimuth_deg=0.0,
                elevation_deg=0.0,
                confidence=0.0,
                sound_pressure_level_db=40.0,
                primary_pair=(self.mics[0].mic_id, self.mics[1].mic_id),
                time_diff_s=0.0,
            )

        # 1. Compute pairwise TDoAs using GCC-PHAT
        pair_delays: dict[tuple[int, int], tuple[float, float]] = {}
        best_pair = self.pairs[0]
        best_peak = -1.0

        for i, j in self.pairs:
            if i >= num_channels or j >= num_channels:
                continue

            mic_i = self.mics[i]
            mic_j = self.mics[j]
            dist_ij = math.dist([mic_i.x_m, mic_i.y_m], [mic_j.x_m, mic_j.y_m])
            max_tau = (dist_ij / self.c) * 1.05  # 5% safety margin

            tau_ij, peak_val = gcc_phat(
                multichannel_audio[i],
                multichannel_audio[j],
                self.fs,
                max_tau,
            )
            pair_delays[(i, j)] = (tau_ij, peak_val)
            if peak_val > best_peak:
                best_peak = peak_val
                best_pair = (i, j)

        # 2. Steered Response Power (SRP) grid search over azimuth theta in [-180, 180)
        theta_angles = np.radians(np.linspace(-180.0, 179.0, 360))
        scores = np.zeros(len(theta_angles))

        for (i, j), (tau_measured, peak_weight) in pair_delays.items():
            dx = self.mics[i].x_m - self.mics[j].x_m
            dy = self.mics[i].y_m - self.mics[j].y_m

            # Theoretical delay: tau = -(dx*cos(theta) + dy*sin(theta)) / c
            tau_model = -(dx * np.cos(theta_angles) + dy * np.sin(theta_angles)) / self.c

            # Error penalty between model and measured TDoA
            error = np.abs(tau_model - tau_measured)
            # Gaussian similarity kernel (sigma = 3 samples)
            sigma = 3.0 / self.fs
            scores += peak_weight * np.exp(-0.5 * (error / sigma) ** 2)

        best_theta_idx = int(np.argmax(scores))
        best_azimuth_deg = float(np.degrees(theta_angles[best_theta_idx]))

        # Sub-degree parabolic peak refinement
        if 0 < best_theta_idx < len(scores) - 1:
            s_left = scores[best_theta_idx - 1]
            s_center = scores[best_theta_idx]
            s_right = scores[best_theta_idx + 1]
            denom_s = s_left - 2.0 * s_center + s_right
            if abs(denom_s) > 1e-9:
                delta_idx = 0.5 * (s_left - s_right) / denom_s
                best_azimuth_deg += float(delta_idx * 1.0)

        # Normalize to [-180, 180]
        norm_azimuth = (best_azimuth_deg + 180.0) % 360.0 - 180.0

        # Confidence: peak score relative to mean score
        mean_score = float(np.mean(scores))
        peak_score = float(scores[best_theta_idx])
        conf = 0.0
        if peak_score > 0:
            contrast = (peak_score - mean_score) / peak_score
            conf = min(1.0, max(0.1, round(contrast * min(1.0, best_peak * 2.0), 4)))

        # 3. Sound Pressure Level (SPL) estimation in dB
        rms_amp = float(np.sqrt(np.mean(multichannel_audio**2))) + 1e-9
        spl_db = round(20.0 * math.log10(rms_amp / 2e-5), 1)
        spl_db = max(30.0, min(140.0, spl_db))

        primary_mic_i = self.mics[best_pair[0]].mic_id
        primary_mic_j = self.mics[best_pair[1]].mic_id
        measured_tau = pair_delays[best_pair][0]

        return AcousticDoAResult(
            azimuth_deg=round(norm_azimuth, 2),
            elevation_deg=0.0,
            confidence=conf,
            sound_pressure_level_db=spl_db,
            primary_pair=(primary_mic_i, primary_mic_j),
            time_diff_s=round(measured_tau, 6),
        )


def simulate_multichannel_siren(
    signal: np.ndarray,
    azimuth_deg: float,
    mic_specs: list[MicrophoneSpec],
    fs: int = 16000,
    sound_speed_mps: float = 343.0,
    snr_db: float = 25.0,
) -> np.ndarray:
    """Simulate realistic multichannel microphone audio from a planar acoustic source."""
    num_mics = len(mic_specs)
    n_samples = len(signal)
    multichannel = np.zeros((num_mics, n_samples), dtype=np.float64)

    theta_rad = math.radians(azimuth_deg)
    u_x = math.cos(theta_rad)
    u_y = math.sin(theta_rad)

    # Compute arrival delay for each mic relative to origin
    delays_s = [-(mic.x_m * u_x + mic.y_m * u_y) / sound_speed_mps for mic in mic_specs]
    # Re-reference so min delay is 0
    min_delay = min(delays_s)
    relative_delays_s = [d - min_delay for d in delays_s]

    rng = np.random.default_rng(1234)

    for i, delay_s in enumerate(relative_delays_s):
        delay_samples = delay_s * fs
        int_delay = math.floor(delay_samples)
        frac_delay = delay_samples - int_delay

        # Linear fractional sample delay interpolation
        delayed_sig = np.zeros(n_samples)
        if int_delay < n_samples:
            delayed_sig[int_delay:] = (1.0 - frac_delay) * signal[: n_samples - int_delay]
        if int_delay + 1 < n_samples:
            delayed_sig[int_delay + 1 :] += frac_delay * signal[: n_samples - (int_delay + 1)]

        # Add uncorrelated channel sensor noise
        noise_power = 10.0 ** (-snr_db / 10.0)
        noise = rng.normal(0.0, math.sqrt(noise_power), n_samples)
        multichannel[i] = delayed_sig + noise

    # Normalize max amplitude to ~0.9
    max_amp = float(np.max(np.abs(multichannel))) + 1e-9
    return (multichannel / max_amp) * 0.9
