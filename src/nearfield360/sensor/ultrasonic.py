"""Acoustic wave propagation, time-of-flight (ToF), and 12-sensor ultrasonic array simulator."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from nearfield360.sensor.models import (
    SensorPose,
    UltrasonicArrayFrame,
    UltrasonicEcho,
)

if TYPE_CHECKING:
    from nearfield360.config import SensorConfig


def calculate_speed_of_sound(temp_c: float = 20.0) -> float:
    """Calculate temperature-compensated speed of sound in air (m/s)."""
    kelvin_ratio = 1.0 + (temp_c / 273.15)
    return 331.3 * math.sqrt(max(0.1, kelvin_ratio))


def normalize_angle_rad(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def build_default_sensor_layout(num_sensors: int = 12) -> dict[str, SensorPose]:
    """Generate automotive perimeter ultrasonic sensor poses in vehicle coordinates.

    Vehicle coordinate convention:
    - X: forward (+X), rear (-X)
    - Y: left (+Y), right (-Y)
    - Yaw: counter-clockwise heading angle from +X
    """
    poses: dict[str, SensorPose] = {}

    # Front bumper (X = 3.8m, curvature facing forward)
    poses["uss_front_left"] = SensorPose(x_m=3.75, y_m=0.85, yaw_rad=math.radians(35.0))
    poses["uss_front_center_left"] = SensorPose(x_m=3.85, y_m=0.30, yaw_rad=math.radians(5.0))
    poses["uss_front_center_right"] = SensorPose(x_m=3.85, y_m=-0.30, yaw_rad=math.radians(-5.0))
    poses["uss_front_right"] = SensorPose(x_m=3.75, y_m=-0.85, yaw_rad=math.radians(-35.0))

    # Rear bumper (X = -1.0m, curvature facing backward)
    poses["uss_rear_left"] = SensorPose(x_m=-0.95, y_m=0.85, yaw_rad=math.radians(145.0))
    poses["uss_rear_center_left"] = SensorPose(x_m=-1.05, y_m=0.30, yaw_rad=math.radians(175.0))
    poses["uss_rear_center_right"] = SensorPose(x_m=-1.05, y_m=-0.30, yaw_rad=math.radians(-175.0))
    poses["uss_rear_right"] = SensorPose(x_m=-0.95, y_m=-0.85, yaw_rad=math.radians(-145.0))

    # Flank side sensors for parking slot & curb detection
    poses["uss_side_front_left"] = SensorPose(x_m=3.20, y_m=0.95, yaw_rad=math.radians(90.0))
    poses["uss_side_front_right"] = SensorPose(x_m=3.20, y_m=-0.95, yaw_rad=math.radians(-90.0))
    poses["uss_side_rear_left"] = SensorPose(x_m=-0.40, y_m=0.95, yaw_rad=math.radians(90.0))
    poses["uss_side_rear_right"] = SensorPose(x_m=-0.40, y_m=-0.95, yaw_rad=math.radians(-90.0))

    if num_sensors < 12:
        # If configured for fewer sensors (e.g. 8 bumper sensors), take first num_sensors
        selected_keys = list(poses.keys())[:num_sensors]
        return {k: poses[k] for k in selected_keys}

    return poses


class UltrasonicArraySimulator:
    """Perimeter ultrasonic transducer array time-of-flight acoustic simulator."""

    def __init__(self, config: SensorConfig | None = None) -> None:
        if config is None:
            from nearfield360.config import SensorConfig

            config = SensorConfig()

        self.config = config
        self.beam_angle_rad: float = math.radians(config.ultrasonic_beam_angle_deg)
        self.half_beam_rad: float = self.beam_angle_rad / 2.0
        self.min_range: float = config.ultrasonic_min_range_m
        self.max_range: float = config.ultrasonic_max_range_m
        self.sensors: dict[str, SensorPose] = build_default_sensor_layout(
            config.num_ultrasonic_sensors
        )

    def simulate_echoes(
        self,
        obstacles: list[tuple[float, float]],
        ambient_temp_c: float = 20.0,
        timestamp_s: float = 0.0,
    ) -> UltrasonicArrayFrame:
        """Simulate acoustic wave propagation and first-echo returns for all transducers.

        Args:
            obstacles: List of (x, y) 2D obstacle positions in vehicle coordinates.
            ambient_temp_c: Ambient air temperature in Celsius.
            timestamp_s: Current simulation timestamp.

        Returns:
            UltrasonicArrayFrame with detected distance echoes.
        """
        echoes: list[UltrasonicEcho] = []

        for sensor_id, pose in self.sensors.items():
            closest_dist: float = self.max_range
            best_amp: float = 0.0
            detected = False

            for ox, oy in obstacles:
                dx = ox - pose.x_m
                dy = oy - pose.y_m
                dist = math.hypot(dx, dy)

                if dist < self.min_range or dist > self.max_range:
                    continue

                bearing_global = math.atan2(dy, dx)
                relative_bearing = normalize_angle_rad(bearing_global - pose.yaw_rad)

                # Check if obstacle lies within the directivity beam cone
                if abs(relative_bearing) <= self.half_beam_rad:
                    # Directivity attenuation: cosine of relative bearing angle
                    cone_ratio = relative_bearing / self.half_beam_rad
                    directivity = max(0.0, math.cos(cone_ratio * (math.pi / 2.0)))

                    # Spherical spreading attenuation (1 / R)
                    dist_att = self.min_range / dist
                    amplitude = min(1.0, max(0.05, directivity * dist_att * 2.0))

                    if dist < closest_dist:
                        closest_dist = dist
                        best_amp = amplitude
                        detected = True

            echoes.append(
                UltrasonicEcho(
                    sensor_id=sensor_id,
                    sensor_pose=pose,
                    range_m=round(closest_dist, 3),
                    echo_amplitude=round(best_amp, 3),
                    in_detection_cone=detected,
                )
            )

        return UltrasonicArrayFrame(
            echoes=echoes,
            ambient_temp_c=ambient_temp_c,
            timestamp_s=timestamp_s,
        )
