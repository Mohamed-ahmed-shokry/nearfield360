"""Unit tests for geometric slot detection and occupancy classification."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from nearfield360.config import ParkingSlotConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.slots.classifier import SlotOccupancyClassifier, rasterize_slot_mask
from nearfield360.slots.detector import (
    ParkingSlotDetector,
    classify_slot_type,
    filter_and_suppress_slots,
    order_slot_corners,
    polygon_iou,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle, TrackState


def test_classify_slot_type() -> None:
    # 0 or pi -> parallel
    assert classify_slot_type(0.0) == ParkingSlotType.PARALLEL
    assert classify_slot_type(math.pi) == ParkingSlotType.PARALLEL

    # pi/2 or -pi/2 -> perpendicular
    assert classify_slot_type(math.pi / 2) == ParkingSlotType.PERPENDICULAR
    assert classify_slot_type(-math.pi / 2) == ParkingSlotType.PERPENDICULAR

    # pi/4 -> slanted
    assert classify_slot_type(math.pi / 4) == ParkingSlotType.SLANTED


def test_order_slot_corners_and_entrance() -> None:
    # A slot located at x in [2, 5], y in [1, 3]
    # Closest edge to vehicle origin (0, 0) is at x=2 or y=1
    raw = [(5.0, 3.0), (2.0, 1.0), (5.0, 1.0), (2.0, 3.0)]
    corners = order_slot_corners(raw, reference_point=(0.0, 0.0))

    assert len(corners) == 4
    # The entrance corners must be closer to (0, 0) than the back corners
    d_front = math.hypot(corners[0].x, corners[0].y)
    d_back = math.hypot(corners[2].x, corners[2].y)
    assert d_front < d_back


def test_polygon_iou_and_suppression() -> None:
    poly1 = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]
    poly2 = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]  # Identical
    poly3 = [(5.0, 5.0), (7.0, 5.0), (7.0, 9.0), (5.0, 9.0)]  # Disjoint

    assert pytest.approx(polygon_iou(poly1, poly2), rel=1e-3) == 1.0
    assert polygon_iou(poly1, poly3) == 0.0

    c0 = ParkingSlotCorner(x=0.0, y=0.0)
    c1 = ParkingSlotCorner(x=2.0, y=0.0)
    c2 = ParkingSlotCorner(x=2.0, y=4.0)
    c3 = ParkingSlotCorner(x=0.0, y=4.0)

    slot_high = ParkingSlot(
        slot_id="s1",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(1.0, 2.0),
        heading_rad=math.pi / 2,
        width_m=2.0,
        length_m=4.0,
        confidence=0.9,
    )
    slot_low = ParkingSlot(
        slot_id="s2",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(1.0, 2.0),
        heading_rad=math.pi / 2,
        width_m=2.0,
        length_m=4.0,
        confidence=0.6,
    )

    kept = filter_and_suppress_slots([slot_low, slot_high], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].slot_id == "s1"


def test_detect_slots_from_markings() -> None:
    grid = BevGrid(x_min=-5.0, x_max=10.0, y_min=-5.0, y_max=5.0, resolution=0.05)
    mask = np.zeros(grid.shape, dtype=np.uint8)

    # Draw a synthetic rectangular parking slot in mask
    # Slot of size 2.5m x 5.0m centered at x=3.0, y=2.0
    # In pixels:
    # col = (x - x_min) / 0.05 = (3.0 - (-5.0)) / 0.05 = 160
    # row = (y - y_min) / 0.05 = (2.0 - (-5.0)) / 0.05 = 140
    # width_px = 2.5 / 0.05 = 50, length_px = 5.0 / 0.05 = 100
    cv2.rectangle(mask, (135, 90), (185, 190), 255, -1)

    detector = ParkingSlotDetector()
    slots = detector.detect_slots_from_markings(mask, grid)

    assert len(slots) >= 1
    slot = slots[0]
    assert 2.0 <= slot.width_m <= 3.8
    assert 4.0 <= slot.length_m <= 8.0
    assert slot.confidence >= 0.4


def test_detect_slots_from_obstacle_gaps() -> None:
    grid = BevGrid(x_min=-5.0, x_max=15.0, y_min=-5.0, y_max=5.0, resolution=0.05)

    # Two parked vehicles on the left side (y=3.0) separated by 4.5 metres (gap_x = 2.5m)
    obs1 = TrackedObstacle(
        track_id=1,
        class_id=0,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(1.0, 3.0),
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=5,
        age=5,
        time_since_update=0,
    )
    obs2 = TrackedObstacle(
        track_id=2,
        class_id=0,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(5.5, 3.0),
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=5,
        age=5,
        time_since_update=0,
    )

    detector = ParkingSlotDetector()
    slots = detector.detect_slots_from_obstacle_gaps([obs1, obs2], grid)

    assert len(slots) == 1
    slot = slots[0]
    assert slot.slot_type == ParkingSlotType.PERPENDICULAR
    assert slot.width_m == pytest.approx(2.5, abs=0.1)


def test_rasterize_slot_mask_and_classify_occupancy() -> None:
    grid = BevGrid(x_min=-5.0, x_max=10.0, y_min=-5.0, y_max=5.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)
    uncertainty = np.full(grid.shape, 0.02, dtype=np.float64)

    c0 = ParkingSlotCorner(x=1.0, y=1.0)
    c1 = ParkingSlotCorner(x=3.5, y=1.0)
    c2 = ParkingSlotCorner(x=3.5, y=6.0)
    c3 = ParkingSlotCorner(x=1.0, y=6.0)

    slot = ParkingSlot(
        slot_id="slot_test",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(2.25, 3.5),
        heading_rad=math.pi / 2,
        width_m=2.5,
        length_m=5.0,
    )

    mask = rasterize_slot_mask(slot, grid)
    assert np.count_nonzero(mask) > 100

    classifier = SlotOccupancyClassifier(ParkingSlotConfig())

    # 1. Vacant classification (clean free grid)
    vacant_slot = classifier.classify_slot(slot, occupancy, uncertainty, grid)
    assert vacant_slot.status == SlotOccupancyStatus.VACANT
    assert vacant_slot.occupancy_ratio == 0.0

    # 2. Occupied classification by high occupancy grid cells
    occupancy[mask] = 0.8
    occupied_slot = classifier.classify_slot(slot, occupancy, uncertainty, grid)
    assert occupied_slot.status == SlotOccupancyStatus.OCCUPIED
    assert occupied_slot.occupancy_ratio == 1.0

    # 3. Uncertain classification by high Bayesian uncertainty
    occupancy[mask] = 0.0
    uncertainty[mask] = 0.20  # > 0.15 threshold
    uncertain_slot = classifier.classify_slot(slot, occupancy, uncertainty, grid)
    assert uncertain_slot.status == SlotOccupancyStatus.UNCERTAIN
    assert uncertain_slot.uncertainty_ratio == 1.0

    # 4. Obstacle intrusion classification
    uncertainty[mask] = 0.02
    obs = TrackedObstacle(
        track_id=10,
        class_id=0,
        class_name="person",
        state=TrackState.CONFIRMED,
        position=(2.25, 3.5),  # right at slot center
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=3,
        age=3,
        time_since_update=0,
    )
    obs_occupied_slot = classifier.classify_slot(
        slot, occupancy, uncertainty, grid, obstacles=[obs]
    )
    assert obs_occupied_slot.status == SlotOccupancyStatus.OCCUPIED
