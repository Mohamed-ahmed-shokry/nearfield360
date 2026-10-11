"""Unit tests for Dynamic Slot Dispatcher and reservation management."""

from __future__ import annotations

import pytest

from nearfield360.config import FleetConfig
from nearfield360.fleet.dispatcher import SlotAllocationError, SlotDispatcher
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import SlotReservationStatus


def test_slot_dispatcher_initialization() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=())
    dispatcher = SlotDispatcher(facility)

    assert dispatcher.total_slots == 8
    summary = dispatcher.get_occupancy_summary(current_time_s=0.0)
    assert summary.total_slots == 8
    assert summary.vacant_slots == 8
    assert summary.reserved_slots == 0
    assert summary.occupied_slots == 0


def test_slot_dispatcher_allocate_and_confirm() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=())
    dispatcher = SlotDispatcher(facility)

    # Allocate slot for vehicle 1
    asg1 = dispatcher.allocate_slot(
        vehicle_id="veh_01",
        vehicle_pose=(0.0, 0.0, 0.0),
        current_time_s=10.0,
    )
    assert asg1.vehicle_id == "veh_01"
    assert asg1.assigned_slot_id in [s.slot_id for s in facility.slots]
    assert dispatcher.get_slot_status(asg1.assigned_slot_id) == SlotReservationStatus.RESERVED

    # Re-allocating for same vehicle returns existing active assignment
    asg1_dup = dispatcher.allocate_slot(
        vehicle_id="veh_01",
        vehicle_pose=(0.0, 0.0, 0.0),
        current_time_s=15.0,
    )
    assert asg1_dup.assignment_id == asg1.assignment_id

    # Confirm reservation
    ok = dispatcher.confirm_reservation("veh_01", asg1.assigned_slot_id)
    assert ok is True

    # Complete parking
    ok_complete = dispatcher.complete_parking("veh_01", asg1.assigned_slot_id)
    assert ok_complete is True
    assert dispatcher.get_slot_status(asg1.assigned_slot_id) == SlotReservationStatus.OCCUPIED

    summary = dispatcher.get_occupancy_summary(current_time_s=20.0)
    assert summary.vacant_slots == 7
    assert summary.occupied_slots == 1


def test_slot_dispatcher_lease_expiration() -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    config = FleetConfig(reservation_lease_duration_s=30.0)
    dispatcher = SlotDispatcher(facility, config=config)

    asg = dispatcher.allocate_slot(
        vehicle_id="veh_slow",
        vehicle_pose=(0.0, 0.0, 0.0),
        current_time_s=0.0,
    )
    slot_id = asg.assigned_slot_id
    assert dispatcher.get_slot_status(slot_id) == SlotReservationStatus.RESERVED

    # Advance time beyond lease duration without confirmation
    expired = dispatcher.clean_expired_leases(current_time_s=35.0)
    assert slot_id in expired
    assert dispatcher.get_slot_status(slot_id) == SlotReservationStatus.VACANT


def test_slot_dispatcher_full_capacity_rejection() -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    dispatcher = SlotDispatcher(facility)

    # Allocate all slots
    asg1 = dispatcher.allocate_slot("v1", (0.0, 0.0, 0.0), 0.0)
    asg2 = dispatcher.allocate_slot("v2", (0.0, 0.0, 0.0), 0.0)
    assert asg1.assigned_slot_id != asg2.assigned_slot_id

    # 3rd allocation must raise SlotAllocationError
    with pytest.raises(SlotAllocationError, match="No vacant parking slots"):
        dispatcher.allocate_slot("v3", (0.0, 0.0, 0.0), 0.0)


def test_slot_dispatcher_release_slot() -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    dispatcher = SlotDispatcher(facility)

    asg = dispatcher.allocate_slot("v1", (0.0, 0.0, 0.0), 0.0)
    slot_id = asg.assigned_slot_id

    # Release slot
    released = dispatcher.release_slot("v1", slot_id)
    assert released is True
    assert dispatcher.get_slot_status(slot_id) == SlotReservationStatus.VACANT
