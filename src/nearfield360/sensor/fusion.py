"""Cross-modal radar, ultrasonic, and camera BEV sensor fusion engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from nearfield360.sensor.models import (
    FusedSensorObstacle,
    RadarPointCloud,
    SensorSimulationReport,
    UltrasonicArrayFrame,
)

if TYPE_CHECKING:
    from nearfield360.config import SensorConfig


class CrossModalSensorFusion:
    """Associates and fuses radar point clouds, ultrasonic ToF echoes, and vision tracks."""

    def __init__(
        self,
        config: SensorConfig | None = None,
        association_gate_m: float = 1.5,
        front_radar_x_m: float = 3.80,
        front_radar_y_m: float = 0.0,
    ) -> None:
        if config is None:
            from nearfield360.config import SensorConfig

            config = SensorConfig()

        self.config = config
        self.association_gate_m = association_gate_m
        self.front_radar_x = front_radar_x_m
        self.front_radar_y = front_radar_y_m

    def fuse(
        self,
        radar_cloud: RadarPointCloud,
        ultrasonic_frame: UltrasonicArrayFrame,
        camera_obstacles: list[dict[str, float]] | None = None,
        timestamp_s: float = 0.0,
    ) -> SensorSimulationReport:
        """Fuse multi-modal sensor inputs into unified vehicle-frame obstacle estimates.

        Args:
            radar_cloud: Detected radar points with radial velocities and RCS.
            ultrasonic_frame: Active ultrasonic echoes with transducer poses.
            camera_obstacles: Optional camera BEV obstacle centroids [{'x': ..., 'y': ...}].
            timestamp_s: Current sensor capture timestamp.

        Returns:
            SensorSimulationReport containing fused obstacles and source telemetry.
        """
        fused_obstacles: list[FusedSensorObstacle] = []
        obs_counter = 1

        # 1. Project ultrasonic echoes into vehicle coordinates (x, y)
        active_uss = [e for e in ultrasonic_frame.echoes if e.in_detection_cone]
        uss_candidates: list[dict[str, float | str]] = []
        for echo in active_uss:
            px = echo.sensor_pose.x_m + echo.range_m * math.cos(echo.sensor_pose.yaw_rad)
            py = echo.sensor_pose.y_m + echo.range_m * math.sin(echo.sensor_pose.yaw_rad)
            uss_candidates.append(
                {
                    "x": px,
                    "y": py,
                    "range_m": echo.range_m,
                    "sensor_id": echo.sensor_id,
                    "matched": 0.0,
                }
            )

        # 2. Process radar targets and associate with nearby ultrasonic echoes
        for r_target in radar_cloud.points:
            rx = self.front_radar_x + r_target.range_m * math.cos(r_target.azimuth_rad)
            ry = self.front_radar_y + r_target.range_m * math.sin(r_target.azimuth_rad)

            # Search for closest active ultrasonic echo within gating distance
            matched_uss: dict[str, float | str] | None = None
            min_dist = self.association_gate_m

            for uss in uss_candidates:
                d = math.hypot(rx - float(uss["x"]), ry - float(uss["y"]))
                if d < min_dist:
                    min_dist = d
                    matched_uss = uss

            modalities = ["radar"]
            confidence = 0.85
            refined_x = rx
            refined_y = ry
            r_val = r_target.range_m

            if matched_uss is not None:
                modalities.append("ultrasonic")
                matched_uss["matched"] = 1.0
                confidence = 0.95
                # Ultrasonic provides millimeter-level near-field radial accuracy
                # Combine positions weighted by sensor precision
                refined_x = 0.4 * rx + 0.6 * float(matched_uss["x"])
                refined_y = 0.4 * ry + 0.6 * float(matched_uss["y"])
                r_val = min(r_target.range_m, float(matched_uss["range_m"]))

            fused_obstacles.append(
                FusedSensorObstacle(
                    obstacle_id=f"fused-obs-{obs_counter:03d}",
                    x_m=round(refined_x, 2),
                    y_m=round(refined_y, 2),
                    radial_velocity_mps=r_target.radial_velocity_mps,
                    min_range_m=round(r_val, 2),
                    source_modalities=modalities,
                    confidence=confidence,
                )
            )
            obs_counter += 1

        # 3. Add remaining unmatched ultrasonic echoes as standalone near-field obstacles
        for uss in uss_candidates:
            if float(uss["matched"]) == 0.0:
                ux = float(uss["x"])
                uy = float(uss["y"])
                fused_obstacles.append(
                    FusedSensorObstacle(
                        obstacle_id=f"fused-obs-{obs_counter:03d}",
                        x_m=round(ux, 2),
                        y_m=round(uy, 2),
                        radial_velocity_mps=0.0,
                        min_range_m=round(float(uss["range_m"]), 2),
                        source_modalities=["ultrasonic"],
                        confidence=0.80,
                    )
                )
                obs_counter += 1

        # 4. Associate camera obstacles if provided
        if camera_obstacles:
            for cam_obs in camera_obstacles:
                cx = cam_obs.get("x", 0.0)
                cy = cam_obs.get("y", 0.0)
                # Associate with closest fused obstacle
                matched_fused: FusedSensorObstacle | None = None
                best_gate = self.association_gate_m

                for fo in fused_obstacles:
                    d = math.hypot(fo.x_m - cx, fo.y_m - cy)
                    if d < best_gate:
                        best_gate = d
                        matched_fused = fo

                if matched_fused is not None:
                    if "camera" not in matched_fused.source_modalities:
                        matched_fused.source_modalities.append("camera")
                    # Boost confidence when confirmed by camera vision
                    matched_fused.confidence = min(0.99, matched_fused.confidence + 0.05)

        return SensorSimulationReport(
            timestamp_s=timestamp_s,
            radar_points_count=len(radar_cloud.points),
            radar_point_cloud=radar_cloud,
            ultrasonic_echoes_count=len(active_uss),
            ultrasonic_frame=ultrasonic_frame,
            fused_obstacles_count=len(fused_obstacles),
            fused_obstacles=fused_obstacles,
        )
