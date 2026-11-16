"""BEV semantic occupancy raster vectorization and Douglas-Peucker polyline simplification."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import (
    Lanelet,
    LaneletType,
    ParkingSlotType,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
    TurnDirection,
)


def _point_line_distance(pt: Point2D, start: Point2D, end: Point2D) -> float:
    """Perpendicular distance from pt to the infinite line passing through start and end."""
    dx = end.x - start.x
    dy = end.y - start.y
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-12:
        return pt.distance_to(start)
    # Area of triangle = 0.5 * base * height => height = 2 * Area / base
    # |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)| / sqrt(dx^2 + dy^2)
    numerator = abs(dx * (start.y - pt.y) - (start.x - pt.x) * dy)
    return numerator / math.sqrt(line_len_sq)


def douglas_peucker(points: Sequence[Point2D], epsilon: float) -> list[Point2D]:
    """Simplify polyline using the Ramer-Douglas-Peucker algorithm.

    Reduces the number of vertices in a polyline while preserving shape within
    perpendicular tolerance epsilon.
    """
    if len(points) <= 2:
        return list(points)

    start = points[0]
    end = points[-1]

    d_max = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = _point_line_distance(points[i], start, end)
        if d > d_max:
            d_max = d
            index = i

    if d_max > epsilon:
        # Recursive simplification of both halves
        rec_left = douglas_peucker(points[: index + 1], epsilon)
        rec_right = douglas_peucker(points[index:], epsilon)
        return rec_left[:-1] + rec_right
    return [start, end]


class BEVVectorizer:
    """Converts bird's-eye-view rasters and point contours into simplified metric
    HD map primitives.
    """

    def __init__(self, config: HDMapConfig | None = None) -> None:
        self.config = config or HDMapConfig()

    def simplify_points(
        self,
        coords: Sequence[tuple[float, float]],
        epsilon: float | None = None,
        is_closed: bool = False,
    ) -> Polyline2D:
        """Convert list of (x, y) tuples to a Douglas-Peucker simplified Polyline2D."""
        if len(coords) < 2:
            raise ValueError("Polyline requires at least 2 coordinate points")

        eps = epsilon if epsilon is not None else self.config.douglas_peucker_epsilon_m
        pts = [Point2D(x=float(c[0]), y=float(c[1])) for c in coords]

        if is_closed and len(pts) >= 4:
            # For closed loops, simplify open chain and re-close
            simplified = douglas_peucker(pts, eps)
            if len(simplified) < 3:
                simplified = [pts[0], pts[len(pts) // 2], pts[-1]]
            return Polyline2D(points=simplified, is_closed=True)

        simplified = douglas_peucker(pts, eps)
        if len(simplified) < 2:
            simplified = [pts[0], pts[-1]]
        return Polyline2D(points=simplified, is_closed=is_closed)

    def extract_centerline(
        self,
        left: Polyline2D,
        right: Polyline2D,
        num_samples: int = 20,
    ) -> Polyline2D:
        """Compute the metric centerline between left and right boundary polylines."""
        samples_count = max(2, num_samples)
        center_pts: list[Point2D] = []

        # Parameterize each boundary by cumulative arc length
        l_total = left.length_m
        r_total = right.length_m

        for i in range(samples_count):
            s = i / (samples_count - 1)
            target_l = s * l_total
            target_r = s * r_total

            pl = self._interpolate_polyline(left, target_l)
            pr = self._interpolate_polyline(right, target_r)

            cx = 0.5 * (pl.x + pr.x)
            cy = 0.5 * (pl.y + pr.y)
            center_pts.append(Point2D(x=cx, y=cy))

        simplified = douglas_peucker(center_pts, self.config.douglas_peucker_epsilon_m)
        if len(simplified) < 2:
            simplified = [center_pts[0], center_pts[-1]]
        return Polyline2D(points=simplified, is_closed=False)

    @staticmethod
    def _interpolate_polyline(polyline: Polyline2D, target_dist: float) -> Point2D:
        """Interpolate point along polyline at target distance from the start."""
        pts = polyline.points
        if target_dist <= 0.0:
            return pts[0]

        accum = 0.0
        for i in range(len(pts) - 1):
            seg_len = pts[i].distance_to(pts[i + 1])
            if accum + seg_len >= target_dist or i == len(pts) - 2:
                remain = target_dist - accum
                t = max(0.0, min(1.0, remain / seg_len if seg_len > 1e-12 else 0.0))
                ix = pts[i].x + t * (pts[i + 1].x - pts[i].x)
                iy = pts[i].y + t * (pts[i + 1].y - pts[i].y)
                return Point2D(x=ix, y=iy)
            accum += seg_len
        return pts[-1]

    def build_lanelet(
        self,
        lanelet_id: str,
        left_pts: Sequence[tuple[float, float]],
        right_pts: Sequence[tuple[float, float]],
        speed_limit_mps: float | None = None,
        lanelet_type: LaneletType = LaneletType.PARKING_AISLE,
        turn_direction: TurnDirection = TurnDirection.STRAIGHT,
        predecessor_ids: list[str] | None = None,
        successor_ids: list[str] | None = None,
    ) -> Lanelet:
        """Construct a validated Lanelet corridor with simplified boundaries and centerline."""
        left_poly = self.simplify_points(left_pts, is_closed=False)
        right_poly = self.simplify_points(right_pts, is_closed=False)
        center_poly = self.extract_centerline(left_poly, right_poly)

        speed = (
            speed_limit_mps if speed_limit_mps is not None else self.config.default_speed_limit_mps
        )

        return Lanelet(
            lanelet_id=lanelet_id,
            left_boundary=left_poly,
            right_boundary=right_poly,
            centerline=center_poly,
            speed_limit_mps=speed,
            lanelet_type=lanelet_type,
            turn_direction=turn_direction,
            predecessor_ids=predecessor_ids or [],
            successor_ids=successor_ids or [],
        )

    def extract_parking_space(
        self,
        slot_id: str,
        corners: Sequence[tuple[float, float]],
        slot_type: ParkingSlotType = ParkingSlotType.STANDARD,
    ) -> ParkingSpaceFeature:
        """Extract a metric 4-corner parking space feature with heading and dimensions."""
        if len(corners) != 4:
            raise ValueError(
                f"Parking space requires exactly 4 corner vertices, got {len(corners)}"
            )

        poly = self.simplify_points(corners, epsilon=0.01, is_closed=True)
        pts = poly.points

        # Calculate geometric center
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
        center = Point2D(x=cx, y=cy)

        # Dimension estimation: length along edge 0->1 or 1->2
        d01 = pts[0].distance_to(pts[1])
        d12 = pts[1].distance_to(pts[2])

        if d01 >= d12:
            length = d01
            width = d12
            heading = math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x)
        else:
            length = d12
            width = d01
            heading = math.atan2(pts[2].y - pts[1].y, pts[2].x - pts[1].x)

        return ParkingSpaceFeature(
            slot_id=slot_id,
            boundary=poly,
            center=center,
            heading_rad=heading,
            slot_type=slot_type,
            width_m=width,
            length_m=length,
        )
