"""Unit tests for tactical FMCW radar and ultrasonic sensor dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.sensor.models import (
    FusedSensorObstacle,
    RadarPointCloud,
    RadarTarget,
    RangeDopplerMap,
    SensorPose,
    SensorSimulationReport,
    UltrasonicArrayFrame,
    UltrasonicEcho,
)
from nearfield360.sensor.viz import render_sensor_dashboard


def test_render_sensor_dashboard_returns_valid_canvas(tmp_path: Path) -> None:
    t1 = RadarTarget(range_m=12.5, radial_velocity_mps=-2.0, rcs_dbm2=15.0)
    cloud = RadarPointCloud(points=[t1], timestamp_s=1.0)

    pose = SensorPose(x_m=3.8, y_m=0.3, yaw_rad=0.0)
    echo = UltrasonicEcho(
        sensor_id="uss_front_center_left",
        sensor_pose=pose,
        range_m=0.75,
        in_detection_cone=True,
    )
    frame = UltrasonicArrayFrame(echoes=[echo], timestamp_s=1.0)

    obs = FusedSensorObstacle(
        obstacle_id="obs-001",
        x_m=4.55,
        y_m=0.3,
        radial_velocity_mps=-2.0,
        min_range_m=0.75,
        source_modalities=["radar", "ultrasonic"],
        confidence=0.95,
    )
    report = SensorSimulationReport(
        timestamp_s=1.0,
        radar_points_count=1,
        radar_point_cloud=cloud,
        ultrasonic_echoes_count=1,
        ultrasonic_frame=frame,
        fused_obstacles_count=1,
        fused_obstacles=[obs],
    )

    rd_map = RangeDopplerMap(
        matrix_db=[[-60.0 + (i + j) % 30 for j in range(128)] for i in range(128)],
        range_bins_m=list(np.linspace(0.0, 50.0, 128)),
        doppler_bins_mps=list(np.linspace(-15.0, 15.0, 128)),
        min_power_db=-60.0,
        max_power_db=-10.0,
    )

    out_file = tmp_path / "sensor_dashboard.png"
    canvas = render_sensor_dashboard(report, rd_map=rd_map, output_path=out_file)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000

    # Verify PNG magic header
    with out_file.open("rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"
