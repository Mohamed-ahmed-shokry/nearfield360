"""BEV visualization engine for parking trajectory plans and vehicle maneuvers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.kinematics import AckermannVehicle
from nearfield360.planning.models import (
    ManeuverGear,
    ParkingTrajectoryPlan,
)
from nearfield360.slots.models import ParkingSlot, SlotOccupancyStatus

# Color constants in BGR format
_GEAR_COLORS: dict[ManeuverGear, tuple[int, int, int]] = {
    ManeuverGear.FORWARD: (255, 165, 0),  # Cyan-blue in BGR
    ManeuverGear.REVERSE: (0, 140, 255),  # Deep orange
}

_STATUS_COLORS: dict[SlotOccupancyStatus, tuple[int, int, int]] = {
    SlotOccupancyStatus.VACANT: (0, 200, 0),
    SlotOccupancyStatus.OCCUPIED: (0, 0, 220),
    SlotOccupancyStatus.UNCERTAIN: (0, 215, 255),
}


def render_parking_plan_bev_overlay(
    grid: BevGrid,
    plan: ParkingTrajectoryPlan,
    *,
    slots: Sequence[ParkingSlot] | None = None,
    base_canvas: NDArray[np.uint8] | None = None,
    vehicle: AckermannVehicle | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render planned parking trajectory, vehicle footprint envelopes, and slots onto BEV canvas."""
    h, w = grid.shape
    canvas = (
        base_canvas.copy() if base_canvas is not None else np.full((h, w, 3), 32, dtype=np.uint8)
    )
    veh = vehicle or AckermannVehicle()

    # 1. Render slots if provided
    if slots:
        slot_overlay = canvas.copy()
        for slot in slots:
            scolor = _STATUS_COLORS.get(slot.status, (150, 150, 150))
            pts_px = [
                (
                    round((c.x - grid.x_min) / grid.resolution),
                    round((c.y - grid.y_min) / grid.resolution),
                )
                for c in slot.corners
            ]
            poly_arr = np.array(pts_px, dtype=np.int32)
            cv2.fillPoly(slot_overlay, [poly_arr], scolor)
            cv2.polylines(canvas, [poly_arr], isClosed=True, color=scolor, thickness=2)
        cv2.addWeighted(slot_overlay, 0.25, canvas, 0.75, 0, canvas)

    # 2. Render trajectory path segments by gear
    for seg in plan.segments:
        gcolor = _GEAR_COLORS.get(seg.gear, (255, 255, 255))
        if len(seg.waypoints) >= 2:
            pts_px = [
                (
                    round((wp.x - grid.x_min) / grid.resolution),
                    round((wp.y - grid.y_min) / grid.resolution),
                )
                for wp in seg.waypoints
            ]
            pts_arr = np.array(pts_px, dtype=np.int32)
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=gcolor, thickness=3)

    # 3. Render vehicle footprint bounding boxes at key poses:
    # Start pose, segment transition poses, and final target pose
    key_poses: list[tuple[float, float, float, tuple[int, int, int]]] = [
        (*plan.start_pose, (0, 255, 255)),  # Yellow at start
    ]
    for seg in plan.segments:
        if seg.waypoints:
            last_wp = seg.waypoints[-1]
            key_poses.append((last_wp.x, last_wp.y, last_wp.heading_rad, (0, 215, 255)))
    key_poses.append((*plan.target_pose, (0, 255, 0)))  # Green at final dock

    for kx, ky, kth, kcolor in key_poses:
        corners = veh.compute_footprint_polygon(kx, ky, kth)
        c_px = [
            (
                round((cx - grid.x_min) / grid.resolution),
                round((cy - grid.y_min) / grid.resolution),
            )
            for cx, cy in corners
        ]
        c_arr = np.array(c_px, dtype=np.int32)
        cv2.polylines(canvas, [c_arr], isClosed=True, color=kcolor, thickness=1)

        # Draw vehicle heading arrow from rear axle to front
        head_len = veh.wheelbase * 0.5
        hx = kx + head_len * math.cos(kth)
        hy = ky + head_len * math.sin(kth)
        p1 = (
            round((kx - grid.x_min) / grid.resolution),
            round((ky - grid.y_min) / grid.resolution),
        )
        p2 = (
            round((hx - grid.x_min) / grid.resolution),
            round((hy - grid.y_min) / grid.resolution),
        )
        cv2.arrowedLine(canvas, p1, p2, kcolor, 1, tipLength=0.3)

    # 4. Info badge
    badge = (
        f"Plan: {plan.slot_id} ({plan.slot_type}) | "
        f"Len: {plan.total_length_m:.1f}m | "
        f"Dur: {plan.total_duration_s:.1f}s | "
        f"Gears: {plan.gear_switches + 1} | "
        f"Clear: {plan.min_clearance_m:.2f}m"
    )
    cv2.putText(
        canvas,
        badge,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
