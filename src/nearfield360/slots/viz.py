"""BEV visualization engine for delineated parking slots and approach corridors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.geometry.bev import BevGrid
from nearfield360.slots.models import (
    ParkingSlot,
    SlotOccupancyStatus,
)

# Colors in BGR format
_STATUS_COLORS: dict[SlotOccupancyStatus, tuple[int, int, int]] = {
    SlotOccupancyStatus.VACANT: (0, 200, 0),  # Bright green
    SlotOccupancyStatus.OCCUPIED: (0, 0, 220),  # Red
    SlotOccupancyStatus.UNCERTAIN: (0, 215, 255),  # Amber / Yellow
}


def render_slots_bev_overlay(
    grid: BevGrid,
    slots: Sequence[ParkingSlot],
    *,
    base_canvas: NDArray[np.uint8] | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render delineated parking slots, status tints, and approach paths on a BEV canvas.

    Args:
        grid: BevGrid defining metric extent and cell resolution.
        slots: ParkingSlot instances to visualize.
        base_canvas: Optional pre-existing BGR BEV image (e.g. occupancy map).
        output_path: Optional destination Path to save PNG image.

    Returns:
        Rendered BGR image array with shape (*grid.shape, 3) and dtype uint8.
    """
    h, w = grid.shape
    if base_canvas is not None:
        canvas = base_canvas.copy()
    else:
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)

    overlay = canvas.copy()

    for slot in slots:
        status_color = _STATUS_COLORS.get(slot.status, (180, 180, 180))

        # Convert corners to pixel coordinates (col, row)
        pts_px = []
        for c in slot.corners:
            col = round((c.x - grid.x_min) / grid.resolution)
            row = round((c.y - grid.y_min) / grid.resolution)
            pts_px.append((col, row))

        poly_arr = np.array(pts_px, dtype=np.int32)

        # Draw filled semi-transparent polygon on overlay
        cv2.fillPoly(overlay, [poly_arr], status_color)

        # Draw solid contour outline
        cv2.polylines(canvas, [poly_arr], isClosed=True, color=status_color, thickness=2)

        # Emphasize entrance mouth [c0 -> c1] with thick line
        cv2.line(canvas, pts_px[0], pts_px[1], (255, 255, 255), thickness=3)

        # Center and text annotation
        cx_col = round((slot.center[0] - grid.x_min) / grid.resolution)
        cy_row = round((slot.center[1] - grid.y_min) / grid.resolution)

        label = f"{slot.slot_id} ({slot.slot_type[:4]})"
        cv2.putText(
            canvas,
            label,
            (cx_col - 20, cy_row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw approach path and entry vector if available
        if slot.approach_path is not None:
            path = slot.approach_path
            ep_col = round((path.entry_point[0] - grid.x_min) / grid.resolution)
            ep_row = round((path.entry_point[1] - grid.y_min) / grid.resolution)
            tp_col = round((path.target_point[0] - grid.x_min) / grid.resolution)
            tp_row = round((path.target_point[1] - grid.y_min) / grid.resolution)

            # Arrow color: Cyan if feasible, Orange if infeasible
            path_color = (255, 255, 0) if path.is_feasible else (0, 140, 255)

            # Draw approach vector arrow
            cv2.arrowedLine(
                canvas,
                (ep_col, ep_row),
                (tp_col, tp_row),
                path_color,
                thickness=2,
                tipLength=0.2,
            )

            # Draw entry waypoint circle
            cv2.circle(canvas, (ep_col, ep_row), 3, (255, 255, 255), -1)

    # Blend overlay with canvas for transparency
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Draw ego vehicle footprint at origin (0, 0)
    ego_col = round((0.0 - grid.x_min) / grid.resolution)
    ego_row = round((0.0 - grid.y_min) / grid.resolution)
    cv2.circle(canvas, (ego_col, ego_row), 4, (0, 255, 255), -1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas


__all__ = ["render_slots_bev_overlay"]
