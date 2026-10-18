"""Line-of-sight raycasting, pillar occlusion shadows, and blind-spot hazard assessment."""

from __future__ import annotations

import math
from collections.abc import Sequence


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
