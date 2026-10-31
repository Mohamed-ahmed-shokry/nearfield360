"""Unit tests for 2D CA-CFAR radar peak detection and point cloud extraction."""

from __future__ import annotations

import numpy as np

from nearfield360.config import SensorConfig
from nearfield360.sensor.cfar import CACFARDetector
from nearfield360.sensor.models import RadarPointCloud, RadarTarget
from nearfield360.sensor.radar import FMCWRadarSimulator


def test_cfar_detector_from_config() -> None:
    config = SensorConfig(
        cfar_guard_cells=3,
        cfar_train_cells=5,
        cfar_threshold_factor=1.35,
    )
    detector = CACFARDetector.from_config(config)

    assert detector.guard == 3
    assert detector.train == 5
    assert detector.threshold_factor == 1.35


def test_cfar_noise_only_suppresses_detections() -> None:
    detector = CACFARDetector(guard_cells=2, train_cells=4, threshold_factor=1.5, min_snr_db=12.0)
    rng = np.random.default_rng(123)
    power_db = rng.normal(-40.0, 1.0, (128, 128)).astype(np.float32)
    range_bins = np.linspace(0.0, 50.0, 128)
    doppler_bins = np.linspace(-20.0, 20.0, 128)

    cloud = detector.detect(power_db, range_bins, doppler_bins)

    assert isinstance(cloud, RadarPointCloud)
    # Pure uniform noise should not produce spurious strong detections
    assert len(cloud.points) == 0


def test_cfar_detects_simulated_targets() -> None:
    config = SensorConfig(
        num_chirps=128,
        samples_per_chirp=256,
        cfar_guard_cells=2,
        cfar_train_cells=4,
        cfar_threshold_factor=1.2,
    )
    sim = FMCWRadarSimulator(config)
    detector = CACFARDetector.from_config(config)

    t1 = RadarTarget(range_m=12.0, radial_velocity_mps=-3.0, rcs_dbm2=15.0)
    t2 = RadarTarget(range_m=28.0, radial_velocity_mps=4.5, rcs_dbm2=20.0)

    baseband = sim.simulate_baseband([t1, t2], noise_level_db=-45.0, seed=42)
    power_db, range_axis_m, doppler_axis_mps = sim.compute_range_doppler(baseband)

    cloud = detector.detect(
        power_db,
        range_axis_m,
        doppler_axis_mps,
        timestamp_s=0.5,
        sensor_id="test_radar",
    )

    assert cloud.sensor_id == "test_radar"
    assert cloud.timestamp_s == 0.5
    assert len(cloud.points) >= 2

    # Check that targets are detected with accurate parameters
    detected_ranges = [p.range_m for p in cloud.points]
    detected_velocities = [p.radial_velocity_mps for p in cloud.points]

    found_t1 = any(
        abs(r - 12.0) < 0.7 and abs(v - (-3.0)) < 0.6
        for r, v in zip(detected_ranges, detected_velocities, strict=False)
    )
    found_t2 = any(
        abs(r - 28.0) < 0.7 and abs(v - 4.5) < 0.6
        for r, v in zip(detected_ranges, detected_velocities, strict=False)
    )

    assert found_t1, f"Target 1 not detected in {cloud.points}"
    assert found_t2, f"Target 2 not detected in {cloud.points}"
