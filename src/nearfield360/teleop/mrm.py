"""Minimum Risk Maneuver (MRM) safe stop engine conforming to ISO 23374 / SAE J3016."""

from __future__ import annotations

import math
from typing import Any

from nearfield360.config import TeleopConfig
from nearfield360.teleop.models import MRMStage, TeleopWaypoint


class MRMEngine:
    """Finite state machine for deterministic emergency deceleration and failsafe parking lock."""

    def __init__(self, vehicle_id: str, config: TeleopConfig | None = None) -> None:
        self.vehicle_id = vehicle_id
        self.config = config or TeleopConfig()
        self.stage: MRMStage = MRMStage.INACTIVE
        self.trigger_reason: str | None = None
        self.triggered_at_s: float | None = None
        self.initial_speed_mps: float = 0.0
        self.current_speed_mps: float = 0.0
        self.parking_brake_engaged: bool = False
        self.hazard_lights_active: bool = False
        self.distance_traveled_m: float = 0.0

    @property
    def is_active(self) -> bool:
        """Return True if an MRM safe stop procedure has been initiated."""
        return self.stage != MRMStage.INACTIVE

    def trigger_mrm(
        self,
        reason: str,
        current_time_s: float,
        current_speed_mps: float,
        pull_to_shoulder: bool = False,
    ) -> MRMStage:
        """Initiate the Minimum Risk Maneuver procedure."""
        if self.stage == MRMStage.INACTIVE:
            self.trigger_reason = reason
            self.triggered_at_s = current_time_s
            self.initial_speed_mps = max(0.0, current_speed_mps)
            self.current_speed_mps = self.initial_speed_mps
            self.hazard_lights_active = True

            if self.initial_speed_mps <= self.config.safe_stop_standstill_speed_mps:
                self.stage = MRMStage.STANDSTILL_LOCKED
                self.parking_brake_engaged = True
            elif pull_to_shoulder:
                self.stage = MRMStage.PULLING_TO_SHOULDER
            else:
                self.stage = MRMStage.DECELERATING

        return self.stage

    def step(self, current_time_s: float) -> MRMStage:
        """Advance the MRM state machine based on elapsed time and deceleration physics."""
        if self.stage == MRMStage.INACTIVE or self.triggered_at_s is None:
            return MRMStage.INACTIVE

        elapsed = max(0.0, current_time_s - self.triggered_at_s)
        decel = self.config.mrm_deceleration_mps2

        # Physics: v(t) = max(0, v0 - a * t)
        speed = max(0.0, self.initial_speed_mps - decel * elapsed)
        self.current_speed_mps = speed

        # Distance: s(t) = v0*t - 0.5*a*t^2 until standstill
        t_stop = self.initial_speed_mps / decel if decel > 0 else 0.0
        if elapsed < t_stop:
            self.distance_traveled_m = self.initial_speed_mps * elapsed - 0.5 * decel * (elapsed**2)
        else:
            self.distance_traveled_m = (
                0.5 * (self.initial_speed_mps**2) / decel if decel > 0 else 0.0
            )

        # Check standstill transition
        if speed <= self.config.safe_stop_standstill_speed_mps:
            self.current_speed_mps = 0.0
            self.parking_brake_engaged = True
            self.stage = MRMStage.STANDSTILL_LOCKED

        return self.stage

    def generate_safe_stop_trajectory(
        self,
        current_pose: tuple[float, float, float],
        step_dt_s: float = 0.2,
    ) -> list[TeleopWaypoint]:
        """Generate planned forward waypoints decelerating to zero velocity."""
        x0, y0, theta = current_pose
        decel = self.config.mrm_deceleration_mps2
        speed = self.current_speed_mps

        if speed <= self.config.safe_stop_standstill_speed_mps:
            return [TeleopWaypoint(x_m=x0, y_m=y0, target_speed_mps=0.0)]

        waypoints: list[TeleopWaypoint] = []
        t = 0.0
        cur_x = x0
        cur_y = y0
        cur_v = speed

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        while cur_v > self.config.safe_stop_standstill_speed_mps and t < 10.0:
            t += step_dt_s
            cur_v = max(0.0, speed - decel * t)
            ds = cur_v * step_dt_s
            cur_x += ds * cos_t
            cur_y += ds * sin_t
            waypoints.append(
                TeleopWaypoint(
                    x_m=round(cur_x, 3),
                    y_m=round(cur_y, 3),
                    target_speed_mps=round(cur_v, 2),
                )
            )

        # Final standstill point
        if not waypoints or waypoints[-1].target_speed_mps > 0:
            waypoints.append(
                TeleopWaypoint(
                    x_m=round(cur_x, 3),
                    y_m=round(cur_y, 3),
                    target_speed_mps=0.0,
                )
            )

        return waypoints

    def get_status(self) -> dict[str, Any]:
        """Export comprehensive MRM diagnostic status."""
        return {
            "vehicle_id": self.vehicle_id,
            "stage": self.stage.value,
            "is_active": self.is_active,
            "trigger_reason": self.trigger_reason,
            "triggered_at_s": self.triggered_at_s,
            "initial_speed_mps": round(self.initial_speed_mps, 2),
            "current_speed_mps": round(self.current_speed_mps, 2),
            "distance_traveled_m": round(self.distance_traveled_m, 2),
            "parking_brake_engaged": self.parking_brake_engaged,
            "hazard_lights_active": self.hazard_lights_active,
        }
