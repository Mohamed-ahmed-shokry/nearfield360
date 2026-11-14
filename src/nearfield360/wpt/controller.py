"""Closed-loop autonomous micro-docking inching controller for SAE J2954 WPT."""

from __future__ import annotations

import math

from nearfield360.config import WPTConfig
from nearfield360.wpt.beacon import MagneticBeaconEstimator
from nearfield360.wpt.models import (
    CoilOffset3D,
    WPTAlignmentStatus,
    WPTDockingStep,
)


class WPTInchingController:
    """Closed-loop creeping controller for millimeter-precision coil centering."""

    def __init__(
        self,
        config: WPTConfig | None = None,
        kp_lat: float = 2.0,
        kd_heading: float = 0.5,
        dt_s: float = 0.05,
        wheelbase_m: float = 2.8,
    ) -> None:
        self.config = config or WPTConfig()
        self.kp_lat = kp_lat
        self.kd_heading = kd_heading
        self.dt_s = dt_s
        self.wheelbase_m = wheelbase_m
        self.estimator = MagneticBeaconEstimator(config=self.config)

    def compute_steering_command(self, delta_y_m: float, heading_err_deg: float) -> float:
        """Calculate Ackermann steering angle trim to cancel lateral and heading error."""
        # Positive lateral error (right of center) requires left turn (negative steer)
        steer_p = -self.kp_lat * delta_y_m * (180.0 / math.pi)
        steer_d = -self.kd_heading * heading_err_deg
        raw_steer = steer_p + steer_d
        # Limit steering angle to practical inching rack limits (+/- 25 deg)
        return float(max(-25.0, min(25.0, raw_steer)))

    def compute_velocity_command(self, delta_x_m: float, delta_r_mm: float) -> float:
        """Calculate creeping velocity, decelerating smoothly to 0 as alignment is reached."""
        if delta_r_mm <= self.config.docked_radius_threshold_mm and abs(delta_x_m) <= 0.02:
            return 0.0

        max_speed = self.config.inching_speed_mps
        speed_factor = max(0.08, min(1.0, abs(delta_x_m) / 0.12))
        if abs(delta_x_m) <= 0.025:
            speed_factor = 0.08
        return float(round(max_speed * speed_factor, 3))

    def step(
        self,
        current_x_m: float,
        current_y_m: float,
        current_heading_deg: float,
        air_gap_mm: float = 170.0,
        step_index: int = 0,
    ) -> tuple[WPTDockingStep, float, float, float]:
        """Execute one control cycle, returning step report and next state (x, y, heading)."""
        offset_x_mm = current_x_m * 1000.0
        offset_y_mm = current_y_m * 1000.0
        offset = self.estimator.compute_status(
            delta_x_mm=offset_x_mm,
            delta_y_mm=offset_y_mm,
            delta_z_mm=air_gap_mm,
            delta_theta_deg=current_heading_deg,
        )

        is_docked = (
            offset.radial_offset_mm <= self.config.docked_radius_threshold_mm
            and abs(offset.delta_theta_deg) <= self.config.max_heading_error_deg
        )

        steering_cmd_deg = self.compute_steering_command(
            delta_y_m=current_y_m,
            heading_err_deg=current_heading_deg,
        )
        velocity_cmd = self.compute_velocity_command(
            delta_x_m=current_x_m,
            delta_r_mm=offset.radial_offset_mm,
        )

        # If already docked, lock brakes
        if is_docked:
            velocity_cmd = 0.0
            steering_cmd_deg = 0.0
            offset = CoilOffset3D(
                delta_x_mm=offset.delta_x_mm,
                delta_y_mm=offset.delta_y_mm,
                delta_z_mm=offset.delta_z_mm,
                radial_offset_mm=offset.radial_offset_mm,
                delta_theta_deg=offset.delta_theta_deg,
                is_within_tolerance=True,
                alignment_status=WPTAlignmentStatus.DOCKED,
            )

        step_record = WPTDockingStep(
            step_index=step_index,
            vehicle_x_m=round(current_x_m, 4),
            vehicle_y_m=round(current_y_m, 4),
            heading_deg=round(current_heading_deg, 2),
            velocity_mps=round(velocity_cmd, 3),
            steering_angle_deg=round(steering_cmd_deg, 2),
            coil_offset=offset,
            docked=is_docked,
        )

        if is_docked or velocity_cmd == 0.0:
            return step_record, current_x_m, current_y_m, current_heading_deg

        # Kinematic bicycle model update: secondary coil situated on steered front axle
        heading_rad = math.radians(current_heading_deg)
        steer_rad = math.radians(steering_cmd_deg)
        course_rad = heading_rad + steer_rad

        # Vehicle moves forward (decreasing remaining positive x)
        next_x = current_x_m - velocity_cmd * math.cos(course_rad) * self.dt_s
        next_y = current_y_m + velocity_cmd * math.sin(course_rad) * self.dt_s
        angular_rate = (velocity_cmd / self.wheelbase_m) * math.tan(steer_rad)
        next_heading = current_heading_deg + math.degrees(angular_rate) * self.dt_s

        return step_record, float(next_x), float(next_y), float(next_heading)

    def simulate_docking(
        self,
        initial_x_m: float = 0.25,
        initial_y_m: float = 0.08,
        initial_heading_deg: float = 2.0,
        air_gap_mm: float = 170.0,
        max_steps: int = 150,
    ) -> list[WPTDockingStep]:
        """Simulate autonomous creeping docking maneuver until coils align or max steps."""
        x, y, heading = initial_x_m, initial_y_m, initial_heading_deg
        steps: list[WPTDockingStep] = []

        for step_idx in range(max_steps):
            step_record, x, y, heading = self.step(
                current_x_m=x,
                current_y_m=y,
                current_heading_deg=heading,
                air_gap_mm=air_gap_mm,
                step_index=step_idx,
            )
            steps.append(step_record)
            if step_record.docked:
                break

        return steps
