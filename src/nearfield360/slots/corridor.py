"""Approach corridor kinematics and trajectory feasibility evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import ParkingSlotConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.slots.models import (
    ParkingSlot,
    SlotApproachPath,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle


def point_to_segment_distance(
    pt: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    """Calculate the shortest Euclidean distance from a point to a 2D line segment."""
    px, py = pt
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)

    # Project pt onto segment, clamping projection t in [0, 1]
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def compute_approach_path(
    slot: ParkingSlot,
    config: ParkingSlotConfig,
) -> SlotApproachPath:
    """Compute the entry waypoint, parking target posture, and direction vector."""
    c0, c1, c2, c3 = slot.corners

    # Entrance edge midpoint
    m_entry_x = 0.5 * (c0.x + c1.x)
    m_entry_y = 0.5 * (c0.y + c1.y)

    # Back edge midpoint
    m_back_x = 0.5 * (c2.x + c3.x)
    m_back_y = 0.5 * (c2.y + c3.y)

    # Inward unit vector
    inward_x = m_back_x - m_entry_x
    inward_y = m_back_y - m_entry_y
    norm = math.hypot(inward_x, inward_y)

    if norm > 1e-6:
        u_x = inward_x / norm
        u_y = inward_y / norm
    else:
        u_x = math.cos(slot.heading_rad)
        u_y = math.sin(slot.heading_rad)

    entry_heading = math.atan2(u_y, u_x)

    # Outside approach waypoint (offset backward along inward vector)
    entry_x = m_entry_x - config.approach_lead_distance * u_x
    entry_y = m_entry_y - config.approach_lead_distance * u_y

    # Target parked stopping center inside slot
    target_x = 0.5 * (m_entry_x + m_back_x)
    target_y = 0.5 * (m_entry_y + m_back_y)

    maneuver_len = math.hypot(target_x - entry_x, target_y - entry_y)

    # Baseline geometric clearance from slot boundaries
    base_clearance = max(0.0, 0.5 * (slot.width_m - config.vehicle_width))

    return SlotApproachPath(
        entry_point=(round(entry_x, 3), round(entry_y, 3)),
        target_point=(round(target_x, 3), round(target_y, 3)),
        entry_heading_rad=round(entry_heading, 4),
        maneuver_length_m=round(maneuver_len, 3),
        clearance_margin_m=round(base_clearance, 3),
        is_feasible=False,  # Evaluated by ApproachCorridorEvaluator
    )


class ApproachCorridorEvaluator:
    """Evaluates whether an ego vehicle can safely maneuver into a parking slot."""

    def __init__(self, config: ParkingSlotConfig | None = None) -> None:
        self.config = config or ParkingSlotConfig()

    def evaluate_corridor(
        self,
        slot: ParkingSlot,
        path: SlotApproachPath,
        occupancy: NDArray[np.float64],
        grid: BevGrid,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """Check collision clearance along approach corridor.

        Returns (is_feasible, clearance_margin_m).
        """
        # A slot must be vacant to be candidate for approach maneuver
        if slot.status != SlotOccupancyStatus.VACANT:
            return (False, 0.0)

        # Build corridor polygon
        p_entry = path.entry_point
        p_target = path.target_point
        dx = p_target[0] - p_entry[0]
        dy = p_target[1] - p_entry[1]
        length = math.hypot(dx, dy)

        if length <= 1e-6:
            return (False, 0.0)

        # Unit vector along approach and orthogonal vector
        u_fwd_x = dx / length
        u_fwd_y = dy / length
        u_lat_x = -u_fwd_y
        u_lat_y = u_fwd_x

        half_corridor_w = 0.5 * self.config.vehicle_width + self.config.safety_margin

        # 4 corners of the swept vehicle corridor
        poly_pts = [
            (p_entry[0] + half_corridor_w * u_lat_x, p_entry[1] + half_corridor_w * u_lat_y),
            (p_entry[0] - half_corridor_w * u_lat_x, p_entry[1] - half_corridor_w * u_lat_y),
            (p_target[0] - half_corridor_w * u_lat_x, p_target[1] - half_corridor_w * u_lat_y),
            (p_target[0] + half_corridor_w * u_lat_x, p_target[1] + half_corridor_w * u_lat_y),
        ]

        # Rasterize corridor polygon on the BEV grid
        mask = np.zeros(grid.shape, dtype=np.uint8)
        pts_px = np.array(
            [
                [
                    round((pt[0] - grid.x_min) / grid.resolution),
                    round((pt[1] - grid.y_min) / grid.resolution),
                ]
                for pt in poly_pts
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts_px], 1)
        corridor_mask = mask > 0

        # Check occupancy grid intrusion
        corridor_cells = int(np.count_nonzero(corridor_mask))
        if corridor_cells > 0:
            occupied_count = int(np.count_nonzero(occupancy[corridor_mask] >= danger_threshold))
            # Trigger collision if obstacle cells exist in corridor (> 3 cells or > 1% area)
            if occupied_count >= 3 or (occupied_count / corridor_cells) > 0.01:
                return (False, 0.0)

        # Evaluate lateral clearance against tracked obstacles
        min_clearance = path.clearance_margin_m
        if obstacles:
            for obs in obstacles:
                ox, oy = obs.position
                dist_to_centerline = point_to_segment_distance((ox, oy), p_entry, p_target)
                obs_clearance = dist_to_centerline - 0.5 * self.config.vehicle_width
                min_clearance = min(min_clearance, max(0.0, obs_clearance))
                if obs_clearance < self.config.safety_margin:
                    return (False, round(min_clearance, 3))

        is_feasible = min_clearance >= self.config.safety_margin
        return (is_feasible, round(min_clearance, 3))

    def evaluate_slot(
        self,
        slot: ParkingSlot,
        occupancy: NDArray[np.float64],
        grid: BevGrid,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.5,
    ) -> ParkingSlot:
        """Compute approach path and evaluate corridor feasibility for a parking slot."""
        path = compute_approach_path(slot, self.config)
        is_feasible, clearance = self.evaluate_corridor(
            slot=slot,
            path=path,
            occupancy=occupancy,
            grid=grid,
            obstacles=obstacles,
            danger_threshold=danger_threshold,
        )

        updated_path = path.model_copy(
            update={
                "is_feasible": is_feasible,
                "clearance_margin_m": clearance,
            }
        )
        return slot.model_copy(update={"approach_path": updated_path})

    def evaluate_slots(
        self,
        slots: Sequence[ParkingSlot],
        occupancy: NDArray[np.float64],
        grid: BevGrid,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.5,
    ) -> list[ParkingSlot]:
        """Evaluate approach corridors for a sequence of parking slots."""
        return [
            self.evaluate_slot(
                slot=slot,
                occupancy=occupancy,
                grid=grid,
                obstacles=obstacles,
                danger_threshold=danger_threshold,
            )
            for slot in slots
        ]


__all__ = [
    "ApproachCorridorEvaluator",
    "compute_approach_path",
    "point_to_segment_distance",
]
