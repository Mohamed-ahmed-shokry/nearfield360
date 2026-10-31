"""Unit tests for 77 GHz FMCW automotive radar simulation and Range-Doppler matrix."""

from __future__ import annotations

import numpy as np

from nearfield360.config import SensorConfig
from nearfield360.sensor.models import RadarTarget, RangeDopplerMap
from nearfield360.sensor.radar import SPEED_OF_LIGHT, FMCWRadarSimulator


def test_radar_simulator_init_constants() -> None:
    config = SensorConfig(carrier_frequency_ghz=77.0, sweep_bandwidth_mhz=250.0)
    sim = FMCWRadarSimulator(config)

    assert sim.fc == 77.0e9
    assert abs(sim.wavelength - (SPEED_OF_LIGHT / 77.0e9)) < 1e-9
    assert sim.bandwidth == 250.0e6
    assert sim.num_chirps == 128
    assert sim.samples_per_chirp == 256


def test_simulate_baseband_shape_and_type() -> None:
    sim = FMCWRadarSimulator()
    targets = [RadarTarget(range_m=15.0, radial_velocity_mps=-2.0, rcs_dbm2=10.0)]
    baseband = sim.simulate_baseband(targets, seed=123)

    assert baseband.shape == (sim.num_chirps, sim.samples_per_chirp)
    assert baseband.dtype == np.complex128
    assert not np.isnan(baseband).any()


def test_compute_range_doppler_detects_peak_at_target() -> None:
    config = SensorConfig(
        carrier_frequency_ghz=77.0,
        sweep_bandwidth_mhz=250.0,
        num_chirps=128,
        samples_per_chirp=256,
    )
    sim = FMCWRadarSimulator(config)

    target_range = 10.0  # meters
    target_velocity = 4.0  # m/s
    targets = [
        RadarTarget(
            range_m=target_range,
            radial_velocity_mps=target_velocity,
            rcs_dbm2=15.0,
        )
    ]

    baseband = sim.simulate_baseband(targets, noise_level_db=-50.0, seed=42)
    power_db, range_axis_m, doppler_axis_mps = sim.compute_range_doppler(baseband)

    assert power_db.shape == (128, 128)
    assert len(range_axis_m) == 128
    assert len(doppler_axis_mps) == 128

    # Find the peak coordinate in (doppler_idx, range_idx)
    peak_doppler_idx, peak_range_idx = np.unravel_index(np.argmax(power_db), power_db.shape)
    detected_range = range_axis_m[peak_range_idx]
    detected_velocity = doppler_axis_mps[peak_doppler_idx]

    # Verification: peak is close to true values (within resolution bin delta_r = 0.6m)
    assert abs(detected_range - target_range) < 0.7
    assert abs(detected_velocity - target_velocity) < 0.5


def test_generate_map_serializable() -> None:
    sim = FMCWRadarSimulator()
    targets = [RadarTarget(range_m=8.0, radial_velocity_mps=1.5)]
    rd_map = sim.generate_map(targets, seed=42)

    assert isinstance(rd_map, RangeDopplerMap)
    assert len(rd_map.matrix_db) == sim.num_chirps
    assert len(rd_map.range_bins_m) == sim.samples_per_chirp // 2
    assert rd_map.max_power_db > rd_map.min_power_db

    # JSON serializability check
    json_data = rd_map.model_dump_json()
    assert len(json_data) > 100
