"""Unit tests for vehicle elevator dispatcher, hoisting lifecycle, and safety interlocks."""

import pytest

from nearfield360.multilevel.elevator import VehicleElevatorDispatcher
from nearfield360.multilevel.models import ElevatorState, FloorLevel


def test_elevator_registration_and_validation() -> None:
    dispatcher = VehicleElevatorDispatcher(max_cabin_weight_kg=3500.0)

    f1 = FloorLevel(floor_id="L1", name="Ground", elevation_m=0.0, bays_count=10, available_bays=5)
    f2 = FloorLevel(floor_id="L2", name="Deck", elevation_m=3.5, bays_count=10, available_bays=5)
    dispatcher.register_floor(f1)
    dispatcher.register_floor(f2)

    dispatcher.register_elevator(elevator_id="lift-01", initial_floor_id="L1")
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.current_floor_id == "L1"
    assert cabin.state == ElevatorState.IDLE

    # Same floor call rejected
    with pytest.raises(ValueError, match="must be distinct"):
        dispatcher.request_elevator("veh-01", "L1", "L1")

    # Overweight vehicle rejected
    with pytest.raises(ValueError, match="exceeds elevator capacity"):
        dispatcher.request_elevator("veh-heavy", "L1", "L2", vehicle_weight_kg=4500.0)

    # Unknown floor rejected
    with pytest.raises(KeyError, match="must be registered"):
        dispatcher.request_elevator("veh-01", "L1", "L_UNKNOWN")


def test_elevator_interlock_gates_and_lifecycle() -> None:
    dispatcher = VehicleElevatorDispatcher(
        door_dwell_s=1.0,  # Fast 1.0s dwell for test
        hoist_speed_mps=2.0,  # 2.0 m/s hoist speed
        leveling_tolerance_m=0.015,
    )
    f1 = FloorLevel(floor_id="L1", name="Ground", elevation_m=0.0, bays_count=10, available_bays=5)
    f2 = FloorLevel(floor_id="L2", name="Upper", elevation_m=4.0, bays_count=10, available_bays=5)
    dispatcher.register_floor(f1)
    dispatcher.register_floor(f2)

    dispatcher.register_elevator("lift-01", initial_floor_id="L1")

    # While IDLE, drive-in is not permitted
    assert dispatcher.is_drive_in_permitted("lift-01") is False

    # Request call L1 -> L2
    dispatcher.request_elevator("veh-01", origin_floor_id="L1", destination_floor_id="L2")

    # Initial state should transition to DOORS_OPENING
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.DOORS_OPENING
    assert dispatcher.is_drive_in_permitted("lift-01") is False

    # Step past door open dwell (1.0s)
    events = dispatcher.step(dt_s=1.2)
    assert any(e["event"] == "doors_opened_loading" for e in events)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.LOADING
    # Now drive-in is safely permitted!
    assert dispatcher.is_drive_in_permitted("lift-01") is True

    # Step past boarding dwell (1.0s) -> vehicle boards, enters HOISTING
    events = dispatcher.step(dt_s=1.2)
    assert any(e["event"] == "vehicle_boarded" for e in events)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.HOISTING
    assert cabin.occupied_vehicle_id == "veh-01"
    assert dispatcher.is_drive_in_permitted("lift-01") is False

    # Step hoisting travel: distance 4.0m / 2.0 m/s = 2.0s
    dispatcher.step(dt_s=1.0)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.HOISTING

    # Step arrival at L2
    events = dispatcher.step(dt_s=1.2)
    assert any(e["event"] == "arrived_at_destination" for e in events)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.DOORS_OPENING
    assert cabin.current_floor_id == "L2"

    # Step door open for unloading
    events = dispatcher.step(dt_s=1.2)
    assert any(e["event"] == "doors_opened_unloading" for e in events)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.UNLOADING
    # Drive-out permitted!
    assert dispatcher.is_drive_out_permitted("lift-01") is True

    # Step vehicle egress dwell
    events = dispatcher.step(dt_s=1.2)
    assert any(e["event"] == "cycle_completed" for e in events)
    cabin = dispatcher.get_cabin_state("lift-01")
    assert cabin.state == ElevatorState.IDLE
    assert cabin.occupied_vehicle_id is None
    assert dispatcher.total_cycles_completed == 1


def test_closest_elevator_dispatch() -> None:
    dispatcher = VehicleElevatorDispatcher(hoist_speed_mps=1.0)
    f1 = FloorLevel(floor_id="L1", name="Ground", elevation_m=0.0, bays_count=10, available_bays=5)
    f2 = FloorLevel(floor_id="L2", name="Mid", elevation_m=3.5, bays_count=10, available_bays=5)
    f3 = FloorLevel(floor_id="L3", name="Top", elevation_m=7.0, bays_count=10, available_bays=5)
    dispatcher.register_floor(f1)
    dispatcher.register_floor(f2)
    dispatcher.register_floor(f3)

    dispatcher.register_elevator("lift-far", initial_floor_id="L3")  # z = 7.0
    dispatcher.register_elevator("lift-near", initial_floor_id="L2")  # z = 3.5

    # Request from L1 (z = 0.0) -> lift-near is closer than lift-far
    dispatcher.request_elevator("veh-dispatch", origin_floor_id="L1", destination_floor_id="L2")

    cabin_near = dispatcher.get_cabin_state("lift-near")
    cabin_far = dispatcher.get_cabin_state("lift-far")

    assert cabin_near.state == ElevatorState.TRANSITING
    assert cabin_far.state == ElevatorState.IDLE
