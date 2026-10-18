"""Line-of-sight raycasting, pillar occlusion shadows, and blind-spot hazard assessment."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.config import CoopConfig
from nearfield360.coop.alignment import Pose2D
from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    BlindSpotWarning,
    CooperativeTrack,
    WarningAction,
)


def segment_intersects_circle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """Check if line segment [p1, p2] intersects a circle with given center and radius."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - center[0]
    fy = p1[1] - center[1]

    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) <= radius

    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Check if either intersection lies strictly within the segment parameter range (0, 1)
    return (0.01 < t1 < 0.99) or (0.01 < t2 < 0.99) or (t1 <= 0.01 and t2 >= 0.99)


def is_occluded_from_observer(
    observer_pos: tuple[float, float],
    target_pos: tuple[float, float],
    pillars: Sequence[tuple[float, float, float]],
) -> bool:
    """Assess whether target_pos is obscured from observer_pos by any pillar (cx, cy, r)."""
    return any(
        segment_intersects_circle(observer_pos, target_pos, (cx, cy), r) for cx, cy, r in pillars
    )


def compute_pillar_shadow_polygon(
    observer_pos: tuple[float, float],
    pillar: tuple[float, float, float],
    shadow_depth_m: float = 25.0,
) -> list[tuple[float, float]]:
    """Compute 4-vertex shadow quadrilateral cast behind a pillar from the observer vantage."""
    cx, cy, r = pillar
    ox, oy = observer_pos
    dx = cx - ox
    dy = cy - oy
    dist = math.hypot(dx, dy)

    if dist <= r:
        return []

    # Angle to pillar center
    theta = math.atan2(dy, dx)
    # Angular half-spread of tangents
    alpha = math.asin(min(0.99, r / dist))

    # Tangent touch points on pillar circumference
    t1_x = cx + r * math.cos(theta + math.pi / 2.0)
    t1_y = cy + r * math.sin(theta + math.pi / 2.0)
    t2_x = cx + r * math.cos(theta - math.pi / 2.0)
    t2_y = cy + r * math.sin(theta - math.pi / 2.0)

    # Project shadow ray endpoints
    s1_x = t1_x + shadow_depth_m * math.cos(theta + alpha)
    s1_y = t1_y + shadow_depth_m * math.sin(theta + alpha)
    s2_x = t2_x + shadow_depth_m * math.cos(theta - alpha)
    s2_y = t2_y + shadow_depth_m * math.sin(theta - alpha)

    return [(t1_x, t1_y), (s1_x, s1_y), (s2_x, s2_y), (t2_x, t2_y)]


DEFAULT_FACILITY_PILLARS: list[tuple[float, float, float]] = [
    (15.0, 15.0, 0.6),
    (25.0, 15.0, 0.6),
    (15.0, 30.0, 0.6),
    (25.0, 30.0, 0.6),
]


class BlindSpotAssessor:
    """Evaluates line-of-sight occlusion and computes Time-to-Collision for blind-spot hazards."""

    def __init__(
        self,
        pillars: Sequence[tuple[float, float, float]] | None = None,
    ) -> None:
        self._pillars: list[tuple[float, float, float]] = (
            list(pillars) if pillars is not None else list(DEFAULT_FACILITY_PILLARS)
        )

    @property
    def pillars(self) -> list[tuple[float, float, float]]:
        """Return configured structural pillars (cx, cy, radius)."""
        return list(self._pillars)

    def assess_tracks(
        self,
        vehicle_id: str,
        vehicle_pose: Pose2D,
        vehicle_speed_mps: float,
        tracks: Sequence[CooperativeTrack],
        timestamp_s: float,
        config: CoopConfig | None = None,
    ) -> list[BlindSpotWarning]:
        """Assess tracks for occluded collision hazards along the vehicle's projected path."""
        active_cfg = config or CoopConfig()
        ttc_threshold = active_cfg.blind_spot_warning_ttc_s

        veh_x = vehicle_pose.x_m
        veh_y = vehicle_pose.y_m
        yaw_rad = math.radians(vehicle_pose.yaw_deg)
        v_veh_x = vehicle_speed_mps * math.cos(yaw_rad)
        v_veh_y = vehicle_speed_mps * math.sin(yaw_rad)

        warnings: list[BlindSpotWarning] = []

        for trk in tracks:
            # Check occlusion from vehicle vantage
            occluded = is_occluded_from_observer((veh_x, veh_y), (trk.x_m, trk.y_m), self._pillars)
            dist_to_veh = math.hypot(trk.x_m - veh_x, trk.y_m - veh_y)

            # Only evaluate objects within 30m that are occluded
            if dist_to_veh > 30.0 or not occluded:
                continue

            dx = trk.x_m - veh_x
            dy = trk.y_m - veh_y
            dvx = trk.vx_mps - v_veh_x
            dvy = trk.vy_mps - v_veh_y

            v_rel_sq = dvx * dvx + dvy * dvy
            if v_rel_sq < 0.04:
                continue

            # Time to Closest Point of Approach (CPA): t_cpa = - (dr . dv) / |dv|^2
            t_cpa = -(dx * dvx + dy * dvy) / v_rel_sq
            if t_cpa <= 0.05 or t_cpa > ttc_threshold:
                continue

            # Distance at CPA
            miss_dx = dx + dvx * t_cpa
            miss_dy = dy + dvy * t_cpa
            d_cpa = math.hypot(miss_dx, miss_dy)

            # Collision cylinder radius (2.5m vehicle envelope + margin)
            if d_cpa > 2.5:
                continue

            conflict_pt = (veh_x + v_veh_x * t_cpa, veh_y + v_veh_y * t_cpa)

            # Risk classification
            if t_cpa <= 1.0:
                risk = BlindSpotRiskLevel.CRITICAL_BRAKE
                action = WarningAction.EMERGENCY_BRAKE
            elif t_cpa <= 1.8:
                risk = BlindSpotRiskLevel.WARNING
                action = WarningAction.STOP_AND_YIELD
            else:
                risk = BlindSpotRiskLevel.ADVISORY
                action = WarningAction.DECELERATE

            warnings.append(
                BlindSpotWarning(
                    target_vehicle_id=vehicle_id,
                    hazard_object_id=trk.track_id,
                    conflict_point_m=(round(conflict_pt[0], 2), round(conflict_pt[1], 2)),
                    ttc_s=round(t_cpa, 2),
                    risk_level=risk,
                    recommended_action=action,
                    occlusion_source="structural_pillar",
                    timestamp_s=timestamp_s,
                )
            )

        # Sort warnings by TTC ascending (most urgent first)
        warnings.sort(key=lambda w: w.ttc_s)
        return warnings
