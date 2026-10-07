"""Path tracking controller with forward and reverse Stanley steering and longitudinal PI."""

from __future__ import annotations

import math

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.models import ControlCommand, TrackingErrorState
from nearfield360.planning.kinematics import normalize_angle
from nearfield360.planning.models import ManeuverGear, TrajectoryWaypoint


class PathTrackingController:
    """Closed-loop Stanley steering and longitudinal speed controller for parking maneuvers."""

    def __init__(
        self,
        control_config: ParkingControlConfig | None = None,
        planner_config: ParkingPlannerConfig | None = None,
    ) -> None:
        self.config = control_config or ParkingControlConfig()
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.wheelbase = self.planner_config.wheelbase
        self.max_steer_angle_rad = self.planner_config.max_steer_angle_rad
        self.max_acceleration = self.planner_config.max_acceleration

        # Internal integrator state for longitudinal PI control
        self._speed_error_integral = 0.0

    def reset(self) -> None:
        """Reset internal controller states (integrators)."""
        self._speed_error_integral = 0.0

    def find_target_waypoint_index(
        self,
        x: float,
        y: float,
        waypoints: list[TrajectoryWaypoint],
        start_idx: int = 0,
        search_window: int = 25,
    ) -> int:
        """Find the index of the closest reference waypoint within a forward search window."""
        if not waypoints:
            return 0
        end_idx = min(len(waypoints), start_idx + search_window)
        best_idx = start_idx
        best_dist_sq = float("inf")

        for idx in range(start_idx, end_idx):
            wp = waypoints[idx]
            dist_sq = (wp.x - x) ** 2 + (wp.y - y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx

        return best_idx

    def compute_tracking_error(
        self,
        x: float,
        y: float,
        heading_rad: float,
        velocity: float,
        waypoints: list[TrajectoryWaypoint],
        active_idx: int,
        gear: ManeuverGear,
    ) -> tuple[TrackingErrorState, TrajectoryWaypoint]:
        """Compute lateral cross-track error, heading error, and longitudinal speed error."""
        if not waypoints:
            empty_wp = TrajectoryWaypoint(x=x, y=y, heading_rad=heading_rad)
            return (
                TrackingErrorState(
                    cross_track_error_m=0.0,
                    heading_error_rad=0.0,
                    longitudinal_error_m=0.0,
                    speed_error_m_s=0.0,
                    closest_waypoint_idx=0,
                ),
                empty_wp,
            )

        clamped_idx = max(0, min(len(waypoints) - 1, active_idx))
        target_wp = waypoints[clamped_idx]

        # In forward gear, track front axle; in reverse gear, track rear axle
        if gear == ManeuverGear.FORWARD:
            axle_x = x + self.wheelbase * math.cos(heading_rad)
            axle_y = y + self.wheelbase * math.sin(heading_rad)
        else:
            axle_x = x
            axle_y = y

        dx = axle_x - target_wp.x
        dy = axle_y - target_wp.y

        # Cross track error perpendicular to path tangent
        # Path tangent: (cos(theta_ref), sin(theta_ref))
        # Path normal (pointing left): (-sin(theta_ref), cos(theta_ref))
        cos_ref = math.cos(target_wp.heading_rad)
        sin_ref = math.sin(target_wp.heading_rad)

        cross_track_error = -sin_ref * dx + cos_ref * dy
        longitudinal_error = cos_ref * dx + sin_ref * dy

        heading_error = normalize_angle(target_wp.heading_rad - heading_rad)
        speed_error = target_wp.velocity - abs(velocity)

        error_state = TrackingErrorState(
            cross_track_error_m=round(cross_track_error, 4),
            heading_error_rad=round(heading_error, 4),
            longitudinal_error_m=round(longitudinal_error, 4),
            speed_error_m_s=round(speed_error, 4),
            closest_waypoint_idx=clamped_idx,
        )
        return error_state, target_wp

    def compute_control_command(
        self,
        current_steer_rad: float,
        current_velocity: float,
        error_state: TrackingErrorState,
        target_wp: TrajectoryWaypoint,
        gear: ManeuverGear,
        emergency_brake: bool = False,
    ) -> ControlCommand:
        """Compute bounded steering and longitudinal control commands."""
        if emergency_brake:
            return ControlCommand(
                steering_angle_rad=round(current_steer_rad, 4),
                steering_rate_rad_s=0.0,
                target_velocity=0.0,
                acceleration_cmd=-self.config.emergency_brake_decel,
                gear=gear,
                emergency_brake=True,
            )

        speed_mag = abs(current_velocity)
        k_cross = self.config.stanley_k
        v_soft = self.config.stanley_soften_speed

        # Curvature feedforward
        # delta_ff = arctan(L * curvature)
        ff_steer = math.atan(self.wheelbase * target_wp.curvature)

        # Cross-track correction
        delta_cross = math.atan2(-k_cross * error_state.cross_track_error_m, speed_mag + v_soft)

        if gear == ManeuverGear.FORWARD:
            # Forward Stanley: delta = heading_error + delta_cross + ff
            raw_steer = error_state.heading_error_rad + delta_cross + ff_steer
        else:
            # Reverse Stanley: Ackermann kinematics invert yaw response to front steering
            raw_steer = -error_state.heading_error_rad + delta_cross - ff_steer

        # Saturate target steering angle
        target_steer = max(-self.max_steer_angle_rad, min(self.max_steer_angle_rad, raw_steer))

        # Steering rate limit
        steer_diff = target_steer - current_steer_rad
        dt = self.config.dt
        max_rate = self.config.max_steer_rate_rad_s
        max_delta_per_step = max_rate * dt

        clamped_diff = max(-max_delta_per_step, min(max_delta_per_step, steer_diff))
        rate_cmd = clamped_diff / dt if dt > 0 else 0.0
        commanded_steer = current_steer_rad + clamped_diff

        # Longitudinal PI control
        self._speed_error_integral += error_state.speed_error_m_s * dt
        # Anti-windup
        self._speed_error_integral = max(-2.0, min(2.0, self._speed_error_integral))

        accel_raw = (
            target_wp.acceleration
            + self.config.kp_speed * error_state.speed_error_m_s
            + self.config.ki_speed * self._speed_error_integral
        )
        accel_cmd = max(-self.max_acceleration, min(self.max_acceleration, accel_raw))

        return ControlCommand(
            steering_angle_rad=round(commanded_steer, 4),
            steering_rate_rad_s=round(rate_cmd, 4),
            target_velocity=round(target_wp.velocity, 4),
            acceleration_cmd=round(accel_cmd, 4),
            gear=gear,
            emergency_brake=False,
        )
