"""Unit tests for cross-modal radar-ultrasonic-vision sensor fusion."""

from __future__ import annotations

from nearfield360.sensor.fusion import CrossModalSensorFusion
from nearfield360.sensor.models import (
    RadarPointCloud,
    RadarTarget,
    SensorPose,
    SensorSimulationReport,
    UltrasonicArrayFrame,
    UltrasonicEcho,
)


def test_fusion_associates_colocated_radar_and_ultrasonic() -> None:
    fusion = CrossModalSensorFusion(front_radar_x_m=3.80, front_radar_y_m=0.0)

    # Radar target at range 1.05m straight ahead: x = 3.80 + 1.05 = 4.85m, y = 0.0m
    radar_target = RadarTarget(range_m=1.05, radial_velocity_mps=-1.2, azimuth_rad=0.0)
    radar_cloud = RadarPointCloud(points=[radar_target], timestamp_s=1.0)

    # Front center ultrasonic echo at range 1.0m from sensor at (3.85, 0.0, yaw=0.0): x = 4.85m
    pose = SensorPose(x_m=3.85, y_m=0.0, yaw_rad=0.0)
    uss_echo = UltrasonicEcho(
        sensor_id="uss_front_center",
        sensor_pose=pose,
        range_m=1.0,
        in_detection_cone=True,
    )
    uss_frame = UltrasonicArrayFrame(echoes=[uss_echo], timestamp_s=1.0)

    report = fusion.fuse(radar_cloud, uss_frame, timestamp_s=1.0)

    assert isinstance(report, SensorSimulationReport)
    assert report.fused_obstacles_count == 1
    fused_obs = report.fused_obstacles[0]

    # Should be associated into multi-modal obstacle
    assert "radar" in fused_obs.source_modalities
    assert "ultrasonic" in fused_obs.source_modalities
    assert fused_obs.radial_velocity_mps == -1.2
    assert fused_obs.confidence >= 0.90
    assert abs(fused_obs.x_m - 4.85) < 0.1


def test_fusion_handles_independent_radar_and_ultrasonic() -> None:
    fusion = CrossModalSensorFusion()

    # Long-range radar target at 25.0m (outside ultrasonic range)
    radar_target = RadarTarget(range_m=25.0, radial_velocity_mps=3.5, azimuth_rad=0.0)
    radar_cloud = RadarPointCloud(points=[radar_target])

    # Rear ultrasonic echo at 0.5m behind vehicle (outside front radar FOV)
    pose = SensorPose(x_m=-1.0, y_m=0.0, yaw_rad=3.14159)
    uss_echo = UltrasonicEcho(
        sensor_id="uss_rear_center",
        sensor_pose=pose,
        range_m=0.5,
        in_detection_cone=True,
    )
    uss_frame = UltrasonicArrayFrame(echoes=[uss_echo])

    report = fusion.fuse(radar_cloud, uss_frame)

    assert report.fused_obstacles_count == 2
    radar_obs = next(o for o in report.fused_obstacles if "radar" in o.source_modalities)
    uss_obs = next(o for o in report.fused_obstacles if "ultrasonic" in o.source_modalities)

    assert radar_obs.radial_velocity_mps == 3.5
    assert uss_obs.radial_velocity_mps == 0.0
    assert uss_obs.x_m < 0.0  # located behind vehicle


def test_fusion_camera_modality_association() -> None:
    fusion = CrossModalSensorFusion(front_radar_x_m=3.80, front_radar_y_m=0.0)

    radar_target = RadarTarget(range_m=5.0, radial_velocity_mps=-0.5, azimuth_rad=0.0)
    radar_cloud = RadarPointCloud(points=[radar_target])
    uss_frame = UltrasonicArrayFrame(echoes=[])

    # Camera obstacle at (8.8, 0.0) matches radar target at 3.8 + 5.0 = 8.8m
    camera_obstacles = [{"x": 8.8, "y": 0.0, "width": 1.8, "length": 4.5}]

    report = fusion.fuse(radar_cloud, uss_frame, camera_obstacles=camera_obstacles)

    assert report.fused_obstacles_count == 1
    fused_obs = report.fused_obstacles[0]
    assert "radar" in fused_obs.source_modalities
    assert "camera" in fused_obs.source_modalities
    assert fused_obs.confidence >= 0.90
