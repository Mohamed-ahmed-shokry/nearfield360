"""Unit tests for approach corridor kinematics and collision feasibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nearfield360.config import ParkingSlotConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.slots.corridor import (
    ApproachCorridorEvaluator,
    compute_approach_path,
    point_to_segment_distance,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle, TrackState


def test_point_to_segment_distance() -> None:
    # Segment from (0, 0) to (10, 0)
    seg_start = (0.0, 0.0)
    seg_end = (10.0, 0.0)

    # Point directly above midpoint
    assert point_to_segment_distance((5.0, 3.0), seg_start, seg_end) == pytest.approx(3.0)

    # Point to the left of start
    assert point_to_segment_distance((-4.0, 3.0), seg_start, seg_end) == pytest.approx(5.0)

    # Point to the right of end
    assert point_to_segment_distance((14.0, 3.0), seg_start, seg_end) == pytest.approx(5.0)

    # Degenerate zero-length segment
    assert point_to_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)


def test_compute_approach_path() -> None:
    # Slot with entrance at y=1.0 (between x=1 and x=3.5), depth towards y=6.0
    c0 = ParkingSlotCorner(x=1.0, y=1.0)
    c1 = ParkingSlotCorner(x=3.5, y=1.0)
    c2 = ParkingSlotCorner(x=3.5, y=6.0)
    c3 = ParkingSlotCorner(x=1.0, y=6.0)

    slot = ParkingSlot(
        slot_id="slot_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(2.25, 3.5),
        heading_rad=math.pi / 2,
        width_m=2.5,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
    )

    config = ParkingSlotConfig(
        approach_lead_distance=1.5,
        vehicle_width=1.8,
        safety_margin=0.25,
    )

    path = compute_approach_path(slot, config)

    # Entrance midpoint is (2.25, 1.0). Inward vector is (0, 1).
    # Entry point should be (2.25, 1.0 - 1.5) = (2.25, -0.5)
    assert path.entry_point == (2.25, -0.5)
    # Target center is (2.25, 3.5)
    assert path.target_point == (2.25, 3.5)
    # Maneuver length = 3.5 - (-0.5) = 4.0
    assert path.maneuver_length_m == pytest.approx(4.0)
    # Base clearance = 0.5 * (2.5 - 1.8) = 0.35
    assert path.clearance_margin_m == pytest.approx(0.35)


def test_corridor_evaluator_feasible_and_infeasible() -> None:
    grid = BevGrid(x_min=-5.0, x_max=10.0, y_min=-5.0, y_max=10.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    c0 = ParkingSlotCorner(x=1.0, y=1.0)
    c1 = ParkingSlotCorner(x=3.5, y=1.0)
    c2 = ParkingSlotCorner(x=3.5, y=6.0)
    c3 = ParkingSlotCorner(x=1.0, y=6.0)

    slot_vacant = ParkingSlot(
        slot_id="slot_v",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(2.25, 3.5),
        heading_rad=math.pi / 2,
        width_m=2.5,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
    )

    evaluator = ApproachCorridorEvaluator(ParkingSlotConfig(vehicle_width=1.8, safety_margin=0.25))

    # 1. Feasible corridor
    res = evaluator.evaluate_slot(slot_vacant, occupancy, grid)
    assert res.approach_path is not None
    assert res.approach_path.is_feasible is True
    assert res.approach_path.clearance_margin_m >= 0.25

    # 2. Infeasible when slot is occupied
    slot_occupied = slot_vacant.model_copy(update={"status": SlotOccupancyStatus.OCCUPIED})
    res_occ = evaluator.evaluate_slot(slot_occupied, occupancy, grid)
    assert res_occ.approach_path is not None
    assert res_occ.approach_path.is_feasible is False

    # 3. Blocked corridor by high occupancy grid cells in front of entrance
    # Entrance is at (2.25, 1.0). Block cells around (2.25, 0.0)
    row_idx = round((0.0 - grid.y_min) / grid.resolution)
    col_idx = round((2.25 - grid.x_min) / grid.resolution)
    occupancy[row_idx - 2 : row_idx + 2, col_idx - 2 : col_idx + 2] = 0.9

    res_blocked = evaluator.evaluate_slot(slot_vacant, occupancy, grid)
    assert res_blocked.approach_path is not None
    assert res_blocked.approach_path.is_feasible is False

    # 4. Flanking obstacle encroaching within safety margin
    occupancy.fill(0.0)  # Reset grid
    # Corridor centerline is at x=2.25. Vehicle half width is 0.9.
    # An obstacle at x=3.1 is dist 0.85 from centerline -> clearance is -0.05 m < safety_margin
    encroaching_obs = TrackedObstacle(
        track_id=1,
        class_id=0,
        class_name="person",
        state=TrackState.CONFIRMED,
        position=(3.0, 0.5),  # very close to entry corridor
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=5,
        age=5,
        time_since_update=0,
    )
    res_encroached = evaluator.evaluate_slot(
        slot_vacant, occupancy, grid, obstacles=[encroaching_obs]
    )
    assert res_encroached.approach_path is not None
    assert res_encroached.approach_path.is_feasible is False
