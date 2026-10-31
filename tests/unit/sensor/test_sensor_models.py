"""Unit tests for synthetic radar and ultrasonic sensor domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def test_radar_target_valid_and_bounds() -> None:
    target = RadarTarget(
        range_m=12.5,
        radial_velocity_mps=-3.2,
        azimuth_rad=0.15,
        rcs_dbm2=12.0,
        snr_db=18.5,
    )
    assert target.range_m == 12.5
    assert target.radial_velocity_mps == -3.2
    assert target.azimuth_rad == 0.15
    assert target.rcs_dbm2 == 12.0
    assert target.snr_db == 18.5

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RadarTarget(range_m=-1.0)


def test_radar_point_cloud_json_roundtrip() -> None:
    t1 = RadarTarget(range_m=5.0, radial_velocity_mps=1.0)
    t2 = RadarTarget(range_m=10.0, radial_velocity_mps=-2.5)
    cloud = RadarPointCloud(
        points=[t1, t2],
        timestamp_s=1.23,
        sensor_id="front_radar_77ghz",
        noise_floor_db=-92.0,
    )

    json_str = cloud.model_dump_json()
    restored = RadarPointCloud.model_validate_json(json_str)

    assert len(restored.points) == 2
    assert restored.points[0].range_m == 5.0
    assert restored.points[1].radial_velocity_mps == -2.5
    assert restored.sensor_id == "front_radar_77ghz"
    assert restored.noise_floor_db == -92.0


def test_range_doppler_map_schema() -> None:
    rd_map = RangeDopplerMap(
        matrix_db=[[-50.0, -40.0], [-60.0, -20.0]],
        range_bins_m=[0.0, 5.0],
        doppler_bins_mps=[-10.0, 10.0],
        min_power_db=-60.0,
        max_power_db=-20.0,
    )
    assert len(rd_map.matrix_db) == 2
    assert rd_map.max_power_db == -20.0
    assert rd_map.min_power_db == -60.0


def test_ultrasonic_echo_and_frame() -> None:
    pose = SensorPose(x_m=2.0, y_m=0.5, yaw_rad=0.0)
    echo = UltrasonicEcho(
        sensor_id="uss_front_center_left",
        sensor_pose=pose,
        range_m=0.85,
        echo_amplitude=0.92,
        in_detection_cone=True,
    )
    frame = UltrasonicArrayFrame(
        echoes=[echo],
        ambient_temp_c=25.0,
        timestamp_s=1.0,
    )

    assert len(frame.echoes) == 1
    assert frame.echoes[0].range_m == 0.85
    assert frame.ambient_temp_c == 25.0

    with pytest.raises(ValidationError, match="less than or equal to 60"):
        UltrasonicArrayFrame(ambient_temp_c=100.0)


def test_fused_sensor_obstacle_and_report() -> None:
    obstacle = FusedSensorObstacle(
        obstacle_id="obs-001",
        x_m=4.5,
        y_m=-1.2,
        radial_velocity_mps=-1.5,
        min_range_m=4.65,
        source_modalities=["radar", "ultrasonic"],
        confidence=0.95,
    )
    pose = SensorPose(x_m=2.0, y_m=0.0)
    echo = UltrasonicEcho(sensor_id="uss_front_center", sensor_pose=pose, range_m=4.65)
    frame = UltrasonicArrayFrame(echoes=[echo])
    cloud = RadarPointCloud(points=[RadarTarget(range_m=4.65, radial_velocity_mps=-1.5)])

    report = SensorSimulationReport(
        timestamp_s=2.0,
        radar_points_count=1,
        radar_point_cloud=cloud,
        ultrasonic_echoes_count=1,
        ultrasonic_frame=frame,
        fused_obstacles_count=1,
        fused_obstacles=[obstacle],
    )

    json_str = report.model_dump_json()
    restored = SensorSimulationReport.model_validate_json(json_str)

    assert restored.fused_obstacles_count == 1
    assert restored.fused_obstacles[0].obstacle_id == "obs-001"
    assert restored.fused_obstacles[0].confidence == 0.95
    assert "radar" in restored.fused_obstacles[0].source_modalities
