"""Geometric parking slot detection and fitting engine from BEV evidence."""

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
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle


def order_slot_corners(
    raw_corners: Sequence[tuple[float, float]],
    reference_point: tuple[float, float] = (0.0, 0.0),
) -> tuple[ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner]:
    """Order 4 polygon corners into [c0, c1, c2, c3] convention.

    Convention:
    - [c0, c1]: entrance edge closest to reference_point (ego vehicle origin).
    - [c2, c3]: back edge farthest from reference_point.
    - Ordered such that facing inward from entrance:
      c0 is entrance-left, c1 is entrance-right, c2 is back-right, c3 is back-left.
    """
    if len(raw_corners) != 4:
        raise ValueError("raw_corners must contain exactly 4 points")

    pts = np.asarray(raw_corners, dtype=np.float64)

    # 4 possible edges: (0,1)-(2,3), (1,2)-(3,0), etc.
    # Find convex hull / circular order first:
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered_circle = pts[np.argsort(angles)]

    # Calculate distance of each edge midpoint to reference_point
    min_dist = float("inf")
    entrance_idx = 0
    ref = np.array(reference_point, dtype=np.float64)

    for i in range(4):
        p_a = ordered_circle[i]
        p_b = ordered_circle[(i + 1) % 4]
        mid = 0.5 * (p_a + p_b)
        dist = float(np.linalg.norm(mid - ref))
        if dist < min_dist:
            min_dist = dist
            entrance_idx = i

    # Edge i -> i+1 is entrance
    p0_cand = ordered_circle[entrance_idx]
    p1_cand = ordered_circle[(entrance_idx + 1) % 4]
    p2_cand = ordered_circle[(entrance_idx + 2) % 4]
    p3_cand = ordered_circle[(entrance_idx + 3) % 4]

    # Inward vector points from entrance midpoint to back midpoint
    entrance_mid = 0.5 * (p0_cand + p1_cand)
    back_mid = 0.5 * (p3_cand + p2_cand)
    inward = back_mid - entrance_mid

    # Cross product of inward vector with (p1 - p0) to ensure p0 is entrance-left
    # inward_x * (p1_y - p0_y) - inward_y * (p1_x - p0_x)
    edge_vec = p1_cand - p0_cand
    cross = inward[0] * edge_vec[1] - inward[1] * edge_vec[0]

    if cross > 0:
        c0, c1, c2, c3 = p0_cand, p1_cand, p2_cand, p3_cand
    else:
        c0, c1, c2, c3 = p1_cand, p0_cand, p3_cand, p2_cand

    return (
        ParkingSlotCorner(x=float(c0[0]), y=float(c0[1])),
        ParkingSlotCorner(x=float(c1[0]), y=float(c1[1])),
        ParkingSlotCorner(x=float(c2[0]), y=float(c2[1])),
        ParkingSlotCorner(x=float(c3[0]), y=float(c3[1])),
    )


def classify_slot_type(heading_rad: float) -> ParkingSlotType:
    """Classify slot geometry based on its inward entry heading relative to vehicle frame."""
    norm_heading = abs(math.atan2(math.sin(heading_rad), math.cos(heading_rad)))
    # In vehicle frame, X is forward, Y is left.
    # Heading near 0 or pi is parallel to driving lane.
    # Heading near pi/2 (90 deg) is perpendicular to driving lane.
    parallel_margin = math.radians(25)
    perpendicular_margin = math.radians(25)

    if norm_heading <= parallel_margin or norm_heading >= (math.pi - parallel_margin):
        return ParkingSlotType.PARALLEL
    elif abs(norm_heading - (math.pi / 2)) <= perpendicular_margin:
        return ParkingSlotType.PERPENDICULAR
    else:
        return ParkingSlotType.SLANTED


def polygon_iou(poly1: list[tuple[float, float]], poly2: list[tuple[float, float]]) -> float:
    """Calculate approximate 2D IoU between two convex polygons via axis-aligned bounds & area."""
    pts1 = np.asarray(poly1)
    pts2 = np.asarray(poly2)

    min1, max1 = pts1.min(axis=0), pts1.max(axis=0)
    min2, max2 = pts2.min(axis=0), pts2.max(axis=0)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    if np.any(inter_max <= inter_min):
        return 0.0

    inter_area = float((inter_max[0] - inter_min[0]) * (inter_max[1] - inter_min[1]))
    area1 = float((max1[0] - min1[0]) * (max1[1] - min1[1]))
    area2 = float((max2[0] - min2[0]) * (max2[1] - min2[1]))

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def filter_and_suppress_slots(
    slots: list[ParkingSlot], iou_threshold: float = 0.35
) -> list[ParkingSlot]:
    """Non-maximum suppression for overlapping parking slot proposals."""
    if not slots:
        return []

    # Sort descending by confidence
    sorted_slots = sorted(slots, key=lambda s: s.confidence, reverse=True)
    kept: list[ParkingSlot] = []

    for cand in sorted_slots:
        overlap = False
        for existing in kept:
            if polygon_iou(cand.polygon_xy, existing.polygon_xy) >= iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(cand)

    return kept


class ParkingSlotDetector:
    """Detects metric 3D parking slots from BEV markings and obstacle gaps."""

    def __init__(self, config: ParkingSlotConfig | None = None) -> None:
        self.config = config or ParkingSlotConfig()

    def detect_slots_from_markings(
        self,
        markings_mask: NDArray[np.uint8],
        grid: BevGrid,
    ) -> list[ParkingSlot]:
        """Delineate candidate parking slots from BEV binary road markings / curb evidence."""
        if markings_mask.shape != grid.shape:
            raise ValueError(
                f"markings_mask shape {markings_mask.shape} does not match grid {grid.shape}"
            )

        slots: list[ParkingSlot] = []

        # Find connected components / contours of marking enclosures
        contours, _ = cv2.findContours(markings_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours):
            if len(cnt) < 4:
                continue

            rect = cv2.minAreaRect(cnt)
            (_cx_px, _cy_px), (w_px, h_px), _angle = rect

            dim1 = w_px * grid.resolution
            dim2 = h_px * grid.resolution

            # Check if dimensions can form a parking slot
            width_cand = min(dim1, dim2)
            length_cand = max(dim1, dim2)

            if not (
                self.config.min_slot_width <= width_cand <= self.config.max_slot_width
                and self.config.min_slot_length <= length_cand <= self.config.max_slot_length
            ):
                continue

            # Extract 4 corners in pixel coords
            box_px = cv2.boxPoints(rect)
            # Map to metric world coordinates: col -> X, row -> Y
            metric_corners = [
                (
                    grid.x_min + float(pt[0]) * grid.resolution,
                    grid.y_min + float(pt[1]) * grid.resolution,
                )
                for pt in box_px
            ]

            corners = order_slot_corners(metric_corners)
            c0, c1, c2, c3 = corners

            # Inward entry vector from entrance to back
            inward_x = 0.5 * (c2.x + c3.x) - 0.5 * (c0.x + c1.x)
            inward_y = 0.5 * (c2.y + c3.y) - 0.5 * (c0.y + c1.y)
            heading_rad = math.atan2(inward_y, inward_x)

            center_x = float(np.mean([c.x for c in corners]))
            center_y = float(np.mean([c.y for c in corners]))
            slot_type = classify_slot_type(heading_rad)

            # Confidence based on fill ratio of expected slot area
            contour_area_m2 = cv2.contourArea(cnt) * (grid.resolution**2)
            rect_area_m2 = width_cand * length_cand
            fill_ratio = min(1.0, contour_area_m2 / max(rect_area_m2, 1e-3))
            confidence = max(self.config.min_confidence, min(1.0, fill_ratio * 1.2))

            slots.append(
                ParkingSlot(
                    slot_id=f"marking_slot_{idx:02d}",
                    slot_type=slot_type,
                    corners=corners,
                    center=(center_x, center_y),
                    heading_rad=heading_rad,
                    width_m=round(width_cand, 2),
                    length_m=round(length_cand, 2),
                    status=SlotOccupancyStatus.UNCERTAIN,
                    confidence=round(confidence, 3),
                )
            )

        return slots

    def detect_slots_from_obstacle_gaps(
        self,
        obstacles: Sequence[TrackedObstacle],
        grid: BevGrid,
    ) -> list[ParkingSlot]:
        """Synthesize candidate parking slots in free-space gaps between parked vehicles."""
        if len(obstacles) < 2:
            return []

        slots: list[ParkingSlot] = []
        # Separate into left (Y > 0) and right (Y < 0) sides
        left_obs = [obs for obs in obstacles if obs.position[1] > 0.5 and obs.speed < 0.5]
        right_obs = [obs for obs in obstacles if obs.position[1] < -0.5 and obs.speed < 0.5]

        for side_name, side_obs in [("left", left_obs), ("right", right_obs)]:
            if len(side_obs) < 2:
                continue

            # Sort longitudinally along X
            sorted_obs = sorted(side_obs, key=lambda o: o.position[0])

            for i in range(len(sorted_obs) - 1):
                obs_a = sorted_obs[i]
                obs_b = sorted_obs[i + 1]

                x_a, y_a = obs_a.position
                x_b, y_b = obs_b.position

                gap_x = abs(x_b - x_a) - 2.0  # approximate vehicle footprint clearance
                avg_y = 0.5 * (y_a + y_b)

                # Check if gap fits a perpendicular slot (width along X, depth along Y)
                if self.config.min_slot_width <= gap_x <= self.config.max_slot_width:
                    slot_w = gap_x
                    slot_l = 5.0  # standard perpendicular parking depth
                    center_x = 0.5 * (x_a + x_b)

                    # Slot extends outward from vehicle path
                    if side_name == "left":
                        y_mouth = max(0.5, avg_y - 0.5 * slot_l)
                        y_back = y_mouth + slot_l
                        corners_xy = [
                            (center_x - 0.5 * slot_w, y_mouth),
                            (center_x + 0.5 * slot_w, y_mouth),
                            (center_x + 0.5 * slot_w, y_back),
                            (center_x - 0.5 * slot_w, y_back),
                        ]
                    else:
                        y_mouth = min(-0.5, avg_y + 0.5 * slot_l)
                        y_back = y_mouth - slot_l
                        corners_xy = [
                            (center_x - 0.5 * slot_w, y_mouth),
                            (center_x + 0.5 * slot_w, y_mouth),
                            (center_x + 0.5 * slot_w, y_back),
                            (center_x - 0.5 * slot_w, y_back),
                        ]

                    corners = order_slot_corners(corners_xy)
                    c0, c1, c2, c3 = corners
                    inward_x = 0.5 * (c2.x + c3.x) - 0.5 * (c0.x + c1.x)
                    inward_y = 0.5 * (c2.y + c3.y) - 0.5 * (c0.y + c1.y)
                    heading_rad = math.atan2(inward_y, inward_x)

                    slots.append(
                        ParkingSlot(
                            slot_id=f"gap_slot_{side_name}_{i:02d}",
                            slot_type=ParkingSlotType.PERPENDICULAR,
                            corners=corners,
                            center=(center_x, 0.5 * (y_mouth + y_back)),
                            heading_rad=heading_rad,
                            width_m=round(slot_w, 2),
                            length_m=round(slot_l, 2),
                            status=SlotOccupancyStatus.UNCERTAIN,
                            confidence=0.85,
                        )
                    )

                # Or parallel parking gap (length along X, width along Y)
                elif self.config.min_slot_length <= gap_x <= self.config.max_slot_length:
                    slot_l = gap_x
                    slot_w = 2.4
                    center_x = 0.5 * (x_a + x_b)
                    center_y = avg_y

                    corners_xy = [
                        (center_x - 0.5 * slot_l, center_y - 0.5 * slot_w),
                        (center_x - 0.5 * slot_l, center_y + 0.5 * slot_w),
                        (center_x + 0.5 * slot_l, center_y + 0.5 * slot_w),
                        (center_x + 0.5 * slot_l, center_y - 0.5 * slot_w),
                    ]
                    corners = order_slot_corners(corners_xy)
                    c0, c1, c2, c3 = corners
                    inward_x = 0.5 * (c2.x + c3.x) - 0.5 * (c0.x + c1.x)
                    inward_y = 0.5 * (c2.y + c3.y) - 0.5 * (c0.y + c1.y)
                    heading_rad = math.atan2(inward_y, inward_x)

                    slots.append(
                        ParkingSlot(
                            slot_id=f"parallel_gap_{side_name}_{i:02d}",
                            slot_type=ParkingSlotType.PARALLEL,
                            corners=corners,
                            center=(center_x, center_y),
                            heading_rad=heading_rad,
                            width_m=round(slot_w, 2),
                            length_m=round(slot_l, 2),
                            status=SlotOccupancyStatus.UNCERTAIN,
                            confidence=0.80,
                        )
                    )

        return slots

    def detect_slots(
        self,
        grid: BevGrid,
        markings_mask: NDArray[np.uint8] | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
    ) -> list[ParkingSlot]:
        """Detect and unify candidate parking slots from markings and obstacle gaps."""
        all_candidates: list[ParkingSlot] = []

        if markings_mask is not None:
            marking_slots = self.detect_slots_from_markings(markings_mask, grid)
            all_candidates.extend(marking_slots)

        if obstacles:
            gap_slots = self.detect_slots_from_obstacle_gaps(obstacles, grid)
            all_candidates.extend(gap_slots)

        # Filter by confidence threshold
        confident = [s for s in all_candidates if s.confidence >= self.config.min_confidence]

        # Apply Non-Maximum Suppression to remove overlapping duplicates
        return filter_and_suppress_slots(confident)


__all__ = [
    "ParkingSlotDetector",
    "classify_slot_type",
    "filter_and_suppress_slots",
    "order_slot_corners",
    "polygon_iou",
]
