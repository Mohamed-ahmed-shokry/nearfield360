"""Ramp kinematics, incline pitch dynamics, and chassis breakover ground clearance engine."""

from __future__ import annotations

import math
from typing import Any

from nearfield360.multilevel.models import VehiclePitchState

STANDARD_GRAVITY_MPS2 = 9.80665


class RampDynamicsEngine:
    """Calculates incline pitch dynamics, gravity forces, and underbody breakover clearances."""

    def __init__(
        self,
        default_vehicle_mass_kg: float = 1800.0,
        default_wheelbase_m: float = 2.8,
        default_ground_clearance_m: float = 0.14,
        default_surface_friction: float = 0.80,
    ) -> None:
        self.default_mass = default_vehicle_mass_kg
        self.default_wheelbase = default_wheelbase_m
        self.default_clearance = default_ground_clearance_m
        self.default_friction = default_surface_friction

    def calculate_pitch_dynamics(
        self,
        vehicle_id: str,
        grade_pct: float,
        elevation_m: float,
        vehicle_mass_kg: float | None = None,
        surface_friction: float | None = None,
        wheelbase_m: float | None = None,
        ground_clearance_m: float | None = None,
        min_clearance_buffer_m: float = 0.05,
    ) -> VehiclePitchState:
        """Calculate pitch angle, gravity forces, and chassis breakover clearance."""
        mass = vehicle_mass_kg if vehicle_mass_kg is not None else self.default_mass
        friction = surface_friction if surface_friction is not None else self.default_friction
        wheelbase = wheelbase_m if wheelbase_m is not None else self.default_wheelbase
        clearance = ground_clearance_m if ground_clearance_m is not None else self.default_clearance

        # Pitch slope angle in radians and degrees: theta = arctan(G / 100)
        pitch_rad = math.atan(grade_pct / 100.0)
        pitch_deg = math.degrees(pitch_rad)

        # Longitudinal gravity resistance: F_g = m * g * sin(theta)
        gravity_force_n = mass * STANDARD_GRAVITY_MPS2 * math.sin(pitch_rad)

        # Normal contact force: F_N = m * g * cos(theta)
        normal_force_n = mass * STANDARD_GRAVITY_MPS2 * math.cos(pitch_rad)

        # Adhesion demand: mu_req = tan(theta) = grade_pct / 100
        mu_req = math.tan(pitch_rad)
        traction_margin = friction - mu_req

        # Chassis breakover clearance check over ramp transition crest
        # Apex clearance = clearance - (wheelbase / 2) * tan(delta_theta / 2)
        half_pitch = abs(pitch_rad) / 2.0
        apex_dip_m = (wheelbase / 2.0) * math.tan(half_pitch)
        breakover_clearance_m = max(0.0, clearance - apex_dip_m)
        is_clearance_safe = (
            breakover_clearance_m >= min_clearance_buffer_m and traction_margin > 0.1
        )

        return VehiclePitchState(
            vehicle_id=vehicle_id,
            pitch_angle_deg=round(pitch_deg, 2),
            grade_pct=round(grade_pct, 2),
            elevation_m=round(elevation_m, 2),
            gravity_force_n=round(gravity_force_n, 1),
            normal_force_n=round(normal_force_n, 1),
            traction_margin=round(traction_margin, 3),
            breakover_clearance_m=round(breakover_clearance_m, 4),
            is_clearance_safe=is_clearance_safe,
        )

    def calculate_spiral_kinematics(
        self,
        radius_m: float,
        velocity_mps: float,
        grade_pct: float,
        max_lat_accel_mps2: float = 2.0,
    ) -> dict[str, Any]:
        """Calculate continuous curvature yaw rate, lateral acceleration, and climb rate."""
        if radius_m <= 0.0:
            raise ValueError(f"Spiral ramp radius must be positive (got {radius_m}).")

        # Incline pitch angle
        pitch_rad = math.atan(grade_pct / 100.0)

        # Horizontal curvature velocity: v_horiz = v * cos(pitch)
        v_horiz = velocity_mps * math.cos(pitch_rad)

        # Yaw rate: omega = v_horiz / R
        yaw_rate_radps = v_horiz / radius_m

        # Lateral centripetal acceleration: a_lat = v_horiz^2 / R
        lateral_accel_mps2 = (v_horiz**2) / radius_m

        # Vertical climb speed: dz/dt = v * sin(pitch)
        vertical_speed_mps = velocity_mps * math.sin(pitch_rad)

        # Maximum safe velocity governed by comfortable lateral acceleration limit:
        max_safe_speed_mps = math.sqrt(max_lat_accel_mps2 * radius_m) / max(
            math.cos(pitch_rad), 0.1
        )
        is_speed_safe = velocity_mps <= max_safe_speed_mps

        return {
            "radius_m": round(radius_m, 2),
            "velocity_mps": round(velocity_mps, 2),
            "yaw_rate_radps": round(yaw_rate_radps, 3),
            "lateral_accel_mps2": round(lateral_accel_mps2, 3),
            "vertical_speed_mps": round(vertical_speed_mps, 3),
            "max_safe_speed_mps": round(max_safe_speed_mps, 2),
            "is_speed_safe": is_speed_safe,
        }

    def check_breakover_clearance(
        self,
        grade_change_deg: float,
        wheelbase_m: float | None = None,
        ground_clearance_m: float | None = None,
        min_buffer_m: float = 0.05,
    ) -> tuple[float, bool]:
        """Verify underbody clearance against a discrete grade inflection angle."""
        wheelbase = wheelbase_m if wheelbase_m is not None else self.default_wheelbase
        clearance = ground_clearance_m if ground_clearance_m is not None else self.default_clearance

        delta_rad = math.radians(abs(grade_change_deg))
        apex_dip_m = (wheelbase / 2.0) * math.tan(delta_rad / 2.0)
        remaining_clearance = max(0.0, clearance - apex_dip_m)
        is_safe = remaining_clearance >= min_buffer_m
        return round(remaining_clearance, 4), is_safe
