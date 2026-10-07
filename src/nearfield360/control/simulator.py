"""Kinematic bicycle vehicle simulator with actuator lag dynamics and noise injection."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.models import ControlCommand, VehicleSimState
from nearfield360.planning.kinematics import AckermannVehicle, normalize_angle
from nearfield360.planning.models import ManeuverGear


class SimulatorNoiseConfig(NamedTuple):
    """Optional Gaussian disturbance noise configuration for simulation testing."""

    pos_std_m: float = 0.0
    heading_std_rad: float = 0.0
    steer_bias_rad: float = 0.0


class VehicleSimulator:
    """Discrete-time Ackermann bicycle model simulator with first-order actuator dynamics."""

    def __init__(
        self,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        control_config: ParkingControlConfig | None = None,
        planner_config: ParkingPlannerConfig | None = None,
        noise_config: SimulatorNoiseConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.control_config = control_config or ParkingControlConfig()
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.noise_config = noise_config or SimulatorNoiseConfig()
        self.vehicle_kinematics = AckermannVehicle(self.planner_config)

        self.wheelbase = self.planner_config.wheelbase
        self.max_steer_angle_rad = self.planner_config.max_steer_angle_rad
        self.max_speed = self.planner_config.max_speed

        self._x = initial_pose[0]
        self._y = initial_pose[1]
        self._heading_rad = normalize_angle(initial_pose[2])
        self._velocity = 0.0
        self._acceleration = 0.0
        self._steer_angle_rad = 0.0
        self._gear = ManeuverGear.FORWARD

        self._rng = np.random.default_rng(seed)

    @property
    def state(self) -> VehicleSimState:
        """Return the current exact vehicle physical state."""
        return VehicleSimState(
            x=round(self._x, 4),
            y=round(self._y, 4),
            heading_rad=round(self._heading_rad, 4),
            velocity=round(self._velocity, 4),
            acceleration=round(self._acceleration, 4),
            steer_angle_rad=round(self._steer_angle_rad, 4),
            gear=self._gear,
        )

    def get_measured_state(self) -> VehicleSimState:
        """Return state corrupted with sensor localization noise for closed-loop testing."""
        if self.noise_config.pos_std_m <= 0.0 and self.noise_config.heading_std_rad <= 0.0:
            return self.state

        noise_x = float(self._rng.normal(0.0, self.noise_config.pos_std_m))
        noise_y = float(self._rng.normal(0.0, self.noise_config.pos_std_m))
        noise_theta = float(self._rng.normal(0.0, self.noise_config.heading_std_rad))

        return VehicleSimState(
            x=round(self._x + noise_x, 4),
            y=round(self._y + noise_y, 4),
            heading_rad=round(normalize_angle(self._heading_rad + noise_theta), 4),
            velocity=round(self._velocity, 4),
            acceleration=round(self._acceleration, 4),
            steer_angle_rad=round(self._steer_angle_rad, 4),
            gear=self._gear,
        )

    def get_footprint_polygon(self) -> list[tuple[float, float]]:
        """Compute current 4-corner footprint bounding box in world/BEV coordinates."""
        return self.vehicle_kinematics.compute_footprint_polygon(
            self._x, self._y, self._heading_rad
        )

    def set_pose(self, pose: tuple[float, float, float], gear: ManeuverGear | None = None) -> None:
        """Reset vehicle pose and optionally gear."""
        self._x = pose[0]
        self._y = pose[1]
        self._heading_rad = normalize_angle(pose[2])
        self._velocity = 0.0
        self._acceleration = 0.0
        self._steer_angle_rad = 0.0
        if gear is not None:
            self._gear = gear

    def set_gear(self, gear: ManeuverGear) -> None:
        """Shift transmission gear (requires vehicle to be stationary)."""
        self._gear = gear
        self._velocity = 0.0
        self._acceleration = 0.0

    def step(self, command: ControlCommand) -> VehicleSimState:
        """Integrate vehicle state forward by dt under the commanded actuator input."""
        dt = self.control_config.dt

        # 1. Update steering angle with first-order actuator lag and rate limit
        biased_steer = command.steering_angle_rad + self.noise_config.steer_bias_rad
        target_steer = max(-self.max_steer_angle_rad, min(self.max_steer_angle_rad, biased_steer))

        tau = max(1e-4, self.control_config.steer_time_constant_s)
        desired_rate = (target_steer - self._steer_angle_rad) / tau
        max_rate = self.control_config.max_steer_rate_rad_s
        actual_rate = max(-max_rate, min(max_rate, desired_rate))

        self._steer_angle_rad += actual_rate * dt
        self._steer_angle_rad = max(
            -self.max_steer_angle_rad, min(self.max_steer_angle_rad, self._steer_angle_rad)
        )

        # 2. Update speed and acceleration
        self._acceleration = command.acceleration_cmd
        new_speed = self._velocity + self._acceleration * dt
        clamped_speed = max(0.0, min(self.max_speed, new_speed))

        if command.emergency_brake and clamped_speed < 0.02:
            clamped_speed = 0.0

        self._velocity = clamped_speed
        self._gear = command.gear

        # 3. Kinematic bicycle integration (midpoint RK2)
        signed_speed = self._velocity if self._gear == ManeuverGear.FORWARD else -self._velocity

        if abs(signed_speed) > 1e-5:
            yaw_rate = (signed_speed / self.wheelbase) * math.tan(self._steer_angle_rad)
            mid_heading = self._heading_rad + 0.5 * yaw_rate * dt
            self._x += signed_speed * math.cos(mid_heading) * dt
            self._y += signed_speed * math.sin(mid_heading) * dt
            self._heading_rad = normalize_angle(self._heading_rad + yaw_rate * dt)

        return self.state
