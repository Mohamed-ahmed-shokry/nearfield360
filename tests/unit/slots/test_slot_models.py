"""Unit tests for parking slot data models and geometric methods."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotApproachPath,
    SlotDetectionReport,
    SlotDetectionSummary,
    SlotOccupancyStatus,
)


def test_parking_slot_corner_and_xy_property() -> None:
    corner = ParkingSlotCorner(x=2.5, y=-1.2, z=0.0)
    assert corner.xy == (2.5, -1.2)
    assert corner.z == 0.0


def test_parking_slot_corner_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        ParkingSlotCorner(x=float("nan"), y=1.0)


def test_slot_approach_path_validation() -> None:
    path = SlotApproachPath(
        entry_point=(2.0, 3.0),
        target_point=(2.0, 5.5),
        entry_heading_rad=math.pi / 2,
        maneuver_length_m=2.5,
        clearance_margin_m=0.35,
        is_feasible=True,
    )
    assert path.entry_point == (2.0, 3.0)
    assert path.target_point == (2.0, 5.5)
    assert path.is_feasible is True
    assert path.clearance_margin_m == 0.35


def test_slot_approach_path_rejects_invalid_point() -> None:
    with pytest.raises(ValidationError):
        SlotApproachPath(
            entry_point=(float("inf"), 0.0),
            target_point=(0.0, 0.0),
            entry_heading_rad=0.0,
            maneuver_length_m=1.0,
            clearance_margin_m=0.1,
            is_feasible=False,
        )


def test_parking_slot_geometry_properties() -> None:
    # Rectangle of width 2.5m along X, length 5.0m along Y
    # Corners: (0, 0), (2.5, 0), (2.5, 5.0), (0, 5.0)
    c0 = ParkingSlotCorner(x=0.0, y=0.0)
    c1 = ParkingSlotCorner(x=2.5, y=0.0)
    c2 = ParkingSlotCorner(x=2.5, y=5.0)
    c3 = ParkingSlotCorner(x=0.0, y=5.0)

    slot = ParkingSlot(
        slot_id="slot_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(c0, c1, c2, c3),
        center=(1.25, 2.5),
        heading_rad=math.pi / 2,
        width_m=2.5,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        occupancy_ratio=0.05,
        uncertainty_ratio=0.02,
        confidence=0.95,
    )

    assert slot.slot_id == "slot_01"
    assert slot.slot_type == ParkingSlotType.PERPENDICULAR
    assert len(slot.polygon_xy) == 4
    assert pytest.approx(slot.area_m2, rel=1e-3) == 12.5

    # Point inclusion checks
    assert slot.contains_point(1.25, 2.5) is True  # Center
    assert slot.contains_point(0.5, 1.0) is True  # Inside
    assert slot.contains_point(3.0, 2.5) is False  # Outside right
    assert slot.contains_point(1.25, -1.0) is False  # Outside bottom


def test_parking_slot_rejects_invalid_corner_count() -> None:
    c0 = ParkingSlotCorner(x=0.0, y=0.0)
    c1 = ParkingSlotCorner(x=2.0, y=0.0)
    c2 = ParkingSlotCorner(x=2.0, y=5.0)

    with pytest.raises(ValidationError):
        ParkingSlot(
            slot_id="bad_slot",
            slot_type=ParkingSlotType.PARALLEL,
            corners=(c0, c1, c2),  # type: ignore[arg-type]
            center=(1.0, 2.5),
            heading_rad=0.0,
            width_m=2.0,
            length_m=5.0,
        )


def test_slot_detection_report_serialization() -> None:
    c0 = ParkingSlotCorner(x=1.0, y=2.0)
    c1 = ParkingSlotCorner(x=3.0, y=2.0)
    c2 = ParkingSlotCorner(x=3.0, y=7.0)
    c3 = ParkingSlotCorner(x=1.0, y=7.0)

    slot = ParkingSlot(
        slot_id="slot_02",
        slot_type=ParkingSlotType.PARALLEL,
        corners=(c0, c1, c2, c3),
        center=(2.0, 4.5),
        heading_rad=0.0,
        width_m=2.0,
        length_m=5.0,
        status=SlotOccupancyStatus.OCCUPIED,
        occupancy_ratio=0.75,
    )

    summary = SlotDetectionSummary(
        total_slots=1,
        vacant_slots=0,
        occupied_slots=1,
        uncertain_slots=0,
        parallel_slots=1,
        perpendicular_slots=0,
        slanted_slots=0,
        feasible_approaches=0,
    )

    report = SlotDetectionReport(
        frame_id="00042",
        camera_sources=("FV", "RV"),
        slots=[slot],
        summary=summary,
        metadata={"note": "test"},
    )

    dumped = report.model_dump()
    assert dumped["frame_id"] == "00042"
    assert dumped["camera_sources"] == ("FV", "RV")
    assert len(dumped["slots"]) == 1
    assert dumped["slots"][0]["status"] == "occupied"
    assert dumped["summary"]["occupied_slots"] == 1
