"""Synthetic 77 GHz FMCW automotive radar simulation and Range-Doppler matrix processing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.sensor.models import RadarTarget, RangeDopplerMap

if TYPE_CHECKING:
    from nearfield360.config import SensorConfig

SPEED_OF_LIGHT: float = 299_792_458.0  # m/s


class FMCWRadarSimulator:
    """Frequency-Modulated Continuous-Wave (FMCW) automotive radar transceiver simulator."""

    def __init__(self, config: SensorConfig | None = None) -> None:
        if config is None:
            from nearfield360.config import SensorConfig

            config = SensorConfig()

        self.config = config
        self.fc: float = config.carrier_frequency_ghz * 1e9  # Hz
        self.c: float = SPEED_OF_LIGHT
        self.wavelength: float = self.c / self.fc  # meters (~3.9 mm at 77 GHz)
        self.bandwidth: float = config.sweep_bandwidth_mhz * 1e6  # Hz
        self.tc: float = config.chirp_duration_us * 1e-6  # seconds
        self.slope: float = self.bandwidth / self.tc  # Hz / s
        self.num_chirps: int = config.num_chirps
        self.samples_per_chirp: int = config.samples_per_chirp
        self.sampling_period: float = self.tc / self.samples_per_chirp
        self.fs: float = 1.0 / self.sampling_period
        self.max_range: float = config.radar_max_range_m

    def simulate_baseband(
        self,
        targets: list[RadarTarget],
        noise_level_db: float = -30.0,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Synthesize 2D complex baseband de-chirped beat matrix [num_chirps, samples_per_chirp]."""
        rng = np.random.default_rng(seed)

        # Time vectors
        # fast-time within chirp: shape (1, samples)
        t_fast = np.linspace(
            0.0,
            self.tc,
            self.samples_per_chirp,
            endpoint=False,
            dtype=np.float64,
        ).reshape(1, -1)
        # slow-time across chirps: shape (chirps, 1)
        chirp_indices = np.arange(self.num_chirps, dtype=np.float64).reshape(-1, 1)
        t_slow = chirp_indices * self.tc

        # Baseband complex signal initialized to zero
        signal = np.zeros((self.num_chirps, self.samples_per_chirp), dtype=np.complex128)

        for target in targets:
            r0 = target.range_m
            vr = target.radial_velocity_mps  # positive = receding, negative = approaching
            rcs = max(0.1, 10.0 ** (target.rcs_dbm2 / 10.0))

            # Amplitude with free-space 1/R^2 attenuation
            amp = math.sqrt(rcs) / max(1.0, r0**2) * 50.0

            # Beat frequency due to range
            fb = self.slope * (2.0 * r0 / self.c)
            # Doppler frequency shift
            fd = 2.0 * vr / self.wavelength

            # 2D phase evolution: fast time beat + slow time doppler
            phase = 2.0 * np.pi * (fb * t_fast + fd * t_slow)
            signal += amp * np.exp(1j * phase)

        # Add complex Gaussian thermal receiver noise
        noise_sigma = 10.0 ** (noise_level_db / 20.0)
        noise = (
            rng.normal(0.0, noise_sigma, signal.shape)
            + 1j * rng.normal(0.0, noise_sigma, signal.shape)
        ).astype(np.complex128)

        return signal + noise

    def compute_range_doppler(
        self,
        baseband: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute 2D FFT to convert baseband signal to Range-Doppler power spectrum in dB.

        Returns:
            (rd_matrix_db, range_axis_m, doppler_axis_mps)
        """
        num_chirps, num_samples = baseband.shape

        # 1. Fast-time FFT (Range processing with Hanning window)
        window_fast = np.hanning(num_samples).reshape(1, -1)
        windowed_samples = baseband * window_fast
        range_fft = np.fft.fft(windowed_samples, axis=1)

        # Retain only positive range bins (first half of FFT)
        half_samples = num_samples // 2
        range_half = range_fft[:, :half_samples]

        # 2. Slow-time FFT (Doppler processing with Hanning window)
        window_slow = np.hanning(num_chirps).reshape(-1, 1)
        windowed_chirps = range_half * window_slow
        doppler_fft = np.fft.fft(windowed_chirps, axis=0)
        doppler_shift = np.fft.fftshift(doppler_fft, axes=0)

        # 3. Logarithmic Power Spectrum (dB)
        power = np.abs(doppler_shift) ** 2
        power_db = 10.0 * np.log10(np.maximum(power, 1e-12))

        # Calculate physical metric axes
        # Range resolution: delta_r = c / (2 * B)
        range_resolution = self.c / (2.0 * self.bandwidth)
        range_axis_m = np.arange(half_samples, dtype=np.float64) * range_resolution

        # Doppler velocity resolution: v_max = lambda / (4 * Tc), delta_v = lambda / (2 * Nc * Tc)
        v_max = self.wavelength / (4.0 * self.tc)
        doppler_axis_mps = np.linspace(-v_max, v_max, num_chirps, endpoint=False, dtype=np.float64)

        return power_db, range_axis_m, doppler_axis_mps

    def generate_map(
        self,
        targets: list[RadarTarget],
        seed: int | None = 42,
    ) -> RangeDopplerMap:
        """Run full end-to-end radar simulation and return serializable RangeDopplerMap."""
        baseband = self.simulate_baseband(targets, seed=seed)
        power_db, range_axis_m, doppler_axis_mps = self.compute_range_doppler(baseband)

        min_power = float(np.min(power_db))
        max_power = float(np.max(power_db))

        return RangeDopplerMap(
            matrix_db=power_db.tolist(),
            range_bins_m=range_axis_m.tolist(),
            doppler_bins_mps=doppler_axis_mps.tolist(),
            min_power_db=min_power,
            max_power_db=max_power,
        )
