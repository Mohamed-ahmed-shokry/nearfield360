"""Unit tests for FacilityBuilder and benchmark facility generation."""

from __future__ import annotations

import pytest

from nearfield360.mapping.builder import (
    FacilityBuilder,
    build_benchmark_garage,
    build_surface_lot,
)
from nearfield360.mapping.models import (
    SlotBayType,
    SlotReservationStatus,
    WaypointType,
)


def test_fluent_facility_builder() -> None:
    builder = FacilityBuilder()
    builder.add_waypoint("wp_0", 0.0, 0.0, waypoint_type=WaypointType.ENTRY)
    builder.add_waypoint("wp_1", 10.0, 0.0, waypoint_type=WaypointType.EXIT)
    builder.add_lane(
        "lane_main",
        start_waypoint_id="wp_0",
        end_waypoint_id="wp_1",
        centerline_points=[(0.0, 0.0), (10.0, 0.0)],
    )
    builder.add_slot(
        slot_id="slot_1",
        slot_type=SlotBayType.PERPENDICULAR,
        center=(5.0, 3.0),
        heading_rad=1.57,
        width_m=2.5,
        length_m=5.0,
        access_lane_id="lane_main",
        access_waypoint_id="wp_0",
    )
    builder.add_obstacle("wall_1", [(0.0, 7.0), (10.0, 7.0)])

    # Build with automatic bounds calculation
    facility = builder.build(map_id="test_fluent", name="Fluent Facility")
    assert facility.map_id == "test_fluent"
    assert len(facility.lanes) == 1
    assert len(facility.slots) == 1
    assert len(facility.waypoints) == 2
    assert len(facility.obstacles) == 1
    assert facility.bounds.contains(5.0, 3.0)
    assert facility.validate_integrity() == []


def test_builder_rejects_invalid_topology() -> None:
    builder = FacilityBuilder()
    builder.set_bounds(0.0, 20.0, 0.0, 20.0)
    builder.add_waypoint("wp_0", 0.0, 0.0)
    # lane refers to missing end waypoint
    builder.add_lane(
        "lane_broken",
        start_waypoint_id="wp_0",
        end_waypoint_id="wp_missing",
        centerline_points=[(0.0, 0.0), (5.0, 0.0)],
    )
    with pytest.raises(ValueError, match="Facility integrity errors"):
        builder.build(map_id="bad", name="Bad")


def test_build_benchmark_garage() -> None:
    garage = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=(1, 3))
    assert garage.map_id == "benchmark_garage_avp"
    assert garage.facility_type == "indoor_garage"
    assert len(garage.lanes) > 5
    assert len(garage.slots) == 8
    assert len(garage.obstacles) >= 4  # Perimeter walls

    # Check vacant vs occupied status
    vacant_slots = [s for s in garage.slots if s.status == SlotReservationStatus.VACANT]
    occupied_slots = [s for s in garage.slots if s.status == SlotReservationStatus.OCCUPIED]
    assert len(occupied_slots) == 2
    assert len(vacant_slots) == 6

    # Verify complete geometric and topological integrity
    assert garage.validate_integrity() == []


def test_build_surface_lot() -> None:
    lot = build_surface_lot(bay_count=6, vacant_count=4)
    assert lot.map_id == "benchmark_surface_lot"
    assert lot.facility_type == "surface_lot"
    assert len(lot.slots) == 6
    assert len(lot.lanes) == 2

    vacant = [s for s in lot.slots if s.status == SlotReservationStatus.VACANT]
    assert len(vacant) == 4

    assert lot.validate_integrity() == []
