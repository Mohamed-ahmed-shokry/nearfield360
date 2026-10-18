"""Coordinate transformations and SE(2) frame projections for cooperative perception."""

from __future__ import annotations

import math
from typing import NamedTuple


class Pose2D(NamedTuple):
    """2D planar pose $(x, y, \text{yaw})$. Coordinates in meters and yaw in degrees."""

    x_m: float
    y_m: float
    yaw_deg: float


class Covariance2D(NamedTuple):
    """2x2 symmetric covariance matrix components."""

    xx: float
    yy: float
    xy: float


class SpatiotemporalAligner:
    """Performs spatial coordinate transformations and kinematic time extrapolation."""

    @staticmethod
    def rpu_to_global(
        local_x_m: float,
        local_y_m: float,
        rpu_pose: Pose2D,
    ) -> tuple[float, float]:
        """Transform local RPU sensor coordinates into facility global coordinates."""
        yaw_rad = math.radians(rpu_pose.yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        global_x = rpu_pose.x_m + local_x_m * cos_y - local_y_m * sin_y
        global_y = rpu_pose.y_m + local_x_m * sin_y + local_y_m * cos_y
        return (global_x, global_y)

    @staticmethod
    def global_to_vehicle(
        global_x_m: float,
        global_y_m: float,
        vehicle_pose: Pose2D,
    ) -> tuple[float, float]:
        """Transform facility global coordinates into vehicle-centric body coordinates."""
        dx = global_x_m - vehicle_pose.x_m
        dy = global_y_m - vehicle_pose.y_m
        yaw_rad = math.radians(vehicle_pose.yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        veh_x = dx * cos_y + dy * sin_y
        veh_y = -dx * sin_y + dy * cos_y
        return (veh_x, veh_y)

    @staticmethod
    def rotate_velocity(
        vx_mps: float,
        vy_mps: float,
        angle_deg: float,
    ) -> tuple[float, float]:
        """Rotate a 2D velocity vector by a given angle in degrees."""
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rx = vx_mps * cos_a - vy_mps * sin_a
        ry = vx_mps * sin_a + vy_mps * cos_a
        return (rx, ry)

    @staticmethod
    def rotate_covariance(
        cov: Covariance2D,
        angle_deg: float,
    ) -> Covariance2D:
        """Rotate a 2x2 covariance matrix $R C R^T$ by angle in degrees."""
        rad = math.radians(angle_deg)
        c = math.cos(rad)
        s = math.sin(rad)

        # C' = R * C * R^T
        # [ c -s ] [ xx xy ] [  c  s ]
        # [ s  c ] [ xy yy ] [ -s  c ]
        xx_new = c * (c * cov.xx - s * cov.xy) - s * (c * cov.xy - s * cov.yy)
        yy_new = s * (s * cov.xx + c * cov.xy) + c * (s * cov.xy + c * cov.yy)
        xy_new = s * (c * cov.xx - s * cov.xy) + c * (c * cov.xy - s * cov.yy)

        # Enforce positive semi-definiteness & numerical stability
        return Covariance2D(xx=max(1e-4, xx_new), yy=max(1e-4, yy_new), xy=xy_new)
