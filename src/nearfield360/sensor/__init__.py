"""Multi-modal FMCW radar and ultrasonic sensor simulation package."""

from __future__ import annotations

from nearfield360.sensor.cfar import CACFARDetector
from nearfield360.sensor.fusion import CrossModalSensorFusion
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
from nearfield360.sensor.radar import FMCWRadarSimulator
from nearfield360.sensor.ultrasonic import (
    UltrasonicArraySimulator,
    build_default_sensor_layout,
    calculate_speed_of_sound,
)
from nearfield360.sensor.viz import render_sensor_dashboard

__all__ = [
    "CACFARDetector",
    "CrossModalSensorFusion",
    "FMCWRadarSimulator",
    "FusedSensorObstacle",
    "RadarPointCloud",
    "RadarTarget",
    "RangeDopplerMap",
    "SensorPose",
    "SensorSimulationReport",
    "UltrasonicArrayFrame",
    "UltrasonicArraySimulator",
    "UltrasonicEcho",
    "build_default_sensor_layout",
    "calculate_speed_of_sound",
    "render_sensor_dashboard",
]
