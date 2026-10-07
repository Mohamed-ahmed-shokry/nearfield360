"""Continuous swept footprint validation and collision clearance evaluator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import ParkingPlannerConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.kinematics import AckermannVehicle, KinematicWaypoint
from nearfield360.planning.models import TrajectoryWaypoint
from nearfield360.tracking.models import TrackedObstacle


class TrajectoryCollisionResult(NamedTuple):
    """Result of swept footprint collision and clearance evaluation."""

    is_collision_free: bool
    min_clearance_m: float
    collision_waypoint_index: int | None
    collision_reason: str | None


def point_to_polygon_distance(
    px: float,
    py: float,
    poly_corners: list[tuple[float, float]],
) -> float:
    """Compute minimum Euclidean distance from a 2D point to the perimeter of a polygon."""
    n = len(poly_corners)
    if n < 3:
        return 0.0

    min_dist = float("inf")
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = poly_corners[i]
        x2, y2 = poly_corners[j]
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist


class SweptFootprintEvaluator:
    """Evaluates whether vehicle trajectory waypoints collide with obstacles or occupancy."""

    def __init__(
        self,
        vehicle: AckermannVehicle,
        config: ParkingPlannerConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config or ParkingPlannerConfig()

    def evaluate_trajectory(
        self,
        waypoints: Sequence[TrajectoryWaypoint | KinematicWaypoint],
        occupancy: NDArray[np.float64],
        grid: BevGrid,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.50,
        uncertainty: NDArray[np.float64] | None = None,
        uncertainty_threshold: float = 0.20,
    ) -> TrajectoryCollisionResult:
        """Validate all waypoints in a trajectory against BEV occupancy and tracked obstacles."""
        if not waypoints:
            return TrajectoryCollisionResult(
                is_collision_free=True,
                min_clearance_m=10.0,
                collision_waypoint_index=None,
                collision_reason=None,
            )

        # Precompute occupied obstacle mask on grid
        occupied_mask = (occupancy >= danger_threshold).astype(np.uint8)

        # Precompute Euclidean distance field to nearest occupied cell if obstacles exist
        has_occupied_cells = np.any(occupied_mask)
        dist_field_m: NDArray[np.float32] | None = None
        if has_occupied_cells:
            free_mask = (occupied_mask == 0).astype(np.uint8)
            # Distance in pixels to nearest 0 (which are the occupied cells)
            dist_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
            dist_field_m = dist_px * float(grid.resolution)

        overall_min_clearance = float("inf")

        for idx, wp in enumerate(waypoints):
            # Compute footprint polygon corners for this waypoint
            footprint = self.vehicle.compute_footprint_polygon(wp.x, wp.y, wp.heading_rad)

            # Rasterize footprint polygon into BEV grid
            mask = np.zeros(grid.shape, dtype=np.uint8)
            pts_px = np.array(
                [
                    [
                        round((pt[0] - grid.x_min) / grid.resolution),
                        round((pt[1] - grid.y_min) / grid.resolution),
                    ]
                    for pt in footprint
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts_px], 1)
            footprint_mask = mask > 0

            # 1. Check occupancy grid collision
            footprint_cells = int(np.count_nonzero(footprint_mask))
            if footprint_cells > 0:
                intrusion_count = int(np.count_nonzero(occupied_mask[footprint_mask]))
                # Intrusion threshold: > 2 occupied cells or > 1% of footprint cells
                if intrusion_count >= 2 or (intrusion_count / footprint_cells) > 0.01:
                    return TrajectoryCollisionResult(
                        is_collision_free=False,
                        min_clearance_m=0.0,
                        collision_waypoint_index=idx,
                        collision_reason=(
                            f"Occupancy intrusion: {intrusion_count} occupied cells in vehicle "
                            f"footprint at waypoint {idx} ({wp.x:.2f}, {wp.y:.2f})"
                        ),
                    )

                # Distance clearance to occupied cells
                if dist_field_m is not None:
                    # Minimum distance from footprint perimeter cells to occupied cells
                    wp_clearance = float(np.min(dist_field_m[footprint_mask]))
                    overall_min_clearance = min(overall_min_clearance, wp_clearance)

            # 2. Check dynamic tracked obstacles
            if obstacles:
                for obs in obstacles:
                    ox, oy = obs.position
                    dist_to_footprint = point_to_polygon_distance(ox, oy, footprint)
                    overall_min_clearance = min(overall_min_clearance, dist_to_footprint)

                    if dist_to_footprint < self.config.collision_margin:
                        margin = self.config.collision_margin
                        return TrajectoryCollisionResult(
                            is_collision_free=False,
                            min_clearance_m=round(dist_to_footprint, 3),
                            collision_waypoint_index=idx,
                            collision_reason=(
                                f"Tracked obstacle collision at ({ox:.2f}, {oy:.2f}) "
                                f"with clearance {dist_to_footprint:.3f}m < {margin}m"
                            ),
                        )

        final_clearance = (
            round(overall_min_clearance, 3) if math.isfinite(overall_min_clearance) else 5.0
        )
        return TrajectoryCollisionResult(
            is_collision_free=True,
            min_clearance_m=final_clearance,
            collision_waypoint_index=None,
            collision_reason=None,
        )
