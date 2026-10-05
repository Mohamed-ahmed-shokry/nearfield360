"""Slot occupancy and Bayesian uncertainty classification engine."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import ParkingSlotConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.slots.models import ParkingSlot, SlotOccupancyStatus
from nearfield360.tracking.models import TrackedObstacle


def rasterize_slot_mask(slot: ParkingSlot, grid: BevGrid) -> NDArray[np.bool_]:
    """Rasterize the convex 4-corner slot polygon into a boolean mask over the BEV grid.

    Image coordinates for OpenCV rasterization use (col, row) order:
    col = (x - x_min) / resolution
    row = (y - y_min) / resolution
    """
    mask = np.zeros(grid.shape, dtype=np.uint8)
    poly_xy = slot.polygon_xy

    # Convert vehicle-frame (x, y) coordinates to BEV grid pixel (col, row)
    pts = np.array(
        [
            [
                round((p[0] - grid.x_min) / grid.resolution),
                round((p[1] - grid.y_min) / grid.resolution),
            ]
            for p in poly_xy
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, [pts], 1)
    return mask > 0


class SlotOccupancyClassifier:
    """Evaluates parking slot vacancy using BEV occupancy and Bayesian uncertainty evidence."""

    def __init__(self, config: ParkingSlotConfig | None = None) -> None:
        self.config = config or ParkingSlotConfig()

    def classify_slot(
        self,
        slot: ParkingSlot,
        occupancy: NDArray[np.float64],
        uncertainty: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.5,
    ) -> ParkingSlot:
        """Classify a single parking slot into VACANT, OCCUPIED, or UNCERTAIN."""
        if grid is None:
            grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.05)

        if occupancy.shape != grid.shape:
            raise ValueError(
                f"occupancy shape {occupancy.shape} does not match grid shape {grid.shape}"
            )
        if uncertainty is not None and uncertainty.shape != grid.shape:
            raise ValueError(
                f"uncertainty shape {uncertainty.shape} does not match grid shape {grid.shape}"
            )

        mask = rasterize_slot_mask(slot, grid)
        total_cells = int(np.count_nonzero(mask))

        if total_cells == 0:
            # Slot is entirely outside the observable BEV grid
            return slot.model_copy(
                update={
                    "status": SlotOccupancyStatus.UNCERTAIN,
                    "occupancy_ratio": 0.0,
                    "uncertainty_ratio": 1.0,
                }
            )

        slot_occ = occupancy[mask]
        occupied_cells = int(np.count_nonzero(slot_occ >= danger_threshold))
        occupancy_ratio = float(occupied_cells / total_cells)

        uncertainty_ratio = 0.0
        if uncertainty is not None:
            slot_unc = uncertainty[mask]
            # Cells with variance >= 0.15 (theoretical max variance is 0.25)
            uncertain_cells = int(np.count_nonzero(slot_unc >= 0.15))
            uncertainty_ratio = float(uncertain_cells / total_cells)

        # Check for dynamic obstacle intrusion
        obstacle_inside = False
        if obstacles:
            for obs in obstacles:
                ox, oy = obs.position
                if slot.contains_point(ox, oy):
                    obstacle_inside = True
                    break

        # Determine occupancy status
        if obstacle_inside or occupancy_ratio >= self.config.occupied_ratio_threshold:
            status = SlotOccupancyStatus.OCCUPIED
            if obstacle_inside:
                occupancy_ratio = max(occupancy_ratio, self.config.occupied_ratio_threshold)
        elif uncertainty_ratio >= self.config.uncertain_ratio_threshold:
            status = SlotOccupancyStatus.UNCERTAIN
        else:
            status = SlotOccupancyStatus.VACANT

        return slot.model_copy(
            update={
                "status": status,
                "occupancy_ratio": round(occupancy_ratio, 4),
                "uncertainty_ratio": round(uncertainty_ratio, 4),
            }
        )

    def classify_slots(
        self,
        slots: Sequence[ParkingSlot],
        occupancy: NDArray[np.float64],
        uncertainty: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        danger_threshold: float = 0.5,
    ) -> list[ParkingSlot]:
        """Classify a sequence of parking slots."""
        return [
            self.classify_slot(
                slot=slot,
                occupancy=occupancy,
                uncertainty=uncertainty,
                grid=grid,
                obstacles=obstacles,
                danger_threshold=danger_threshold,
            )
            for slot in slots
        ]


__all__ = ["SlotOccupancyClassifier", "rasterize_slot_mask"]
