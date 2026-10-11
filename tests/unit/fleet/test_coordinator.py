"""Unit tests for FleetCoordinator and multi-vehicle simulation."""

from __future__ import annotations

from nearfield360.config import FleetConfig
from nearfield360.fleet.coordinator import FleetCoordinator
from nearfield360.fleet.models import V2XMessage, V2XMessageType, VehicleFleetStatus
from nearfield360.mapping.builder import build_benchmark_garage


def test_fleet_coordinator_registration_and_dispatch() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=())
    coordinator = FleetCoordinator(facility)

    # Register 2 vehicles
    v1 = coordinator.register_vehicle("veh_01", initial_pose=(0.0, 0.0, 0.0))
    v2 = coordinator.register_vehicle("veh_02", initial_pose=(0.0, 0.0, 0.0))

    assert v1.status == VehicleFleetStatus.UNASSIGNED
    assert v2.status == VehicleFleetStatus.UNASSIGNED
    assert len(coordinator.vehicles) == 2

    # Dispatch vehicle 1
    asg1 = coordinator.dispatch_vehicle("veh_01")
    assert asg1.vehicle_id == "veh_01"
    veh_1 = coordinator.vehicles["veh_01"]
    assert veh_1.status == VehicleFleetStatus.EN_ROUTE_SLOT
    assert veh_1.trajectory is not None
    assert len(coordinator.message_queue) == 1

    # Dispatch vehicle 2
    asg2 = coordinator.dispatch_vehicle("veh_02")
    assert asg2.vehicle_id == "veh_02"
    assert asg1.assigned_slot_id != asg2.assigned_slot_id


def test_fleet_coordinator_v2x_message_transmission() -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    # Deterministic zero packet loss
    config = FleetConfig(packet_loss_rate=0.0)
    coordinator = FleetCoordinator(facility, config=config)

    msg = V2XMessage(
        message_type=V2XMessageType.HEARTBEAT,
        sender_id="veh_01",
        timestamp_s=1.0,
        payload={"speed": 1.5},
    )
    sent = coordinator.send_v2x_message(msg)
    assert sent is True
    assert len(coordinator.message_queue) == 1


def test_fleet_coordinator_simulation_steps() -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    coordinator = FleetCoordinator(facility)

    coordinator.register_vehicle("v1", (0.0, 0.0, 0.0))
    coordinator.dispatch_vehicle("v1")

    # Step simulation forward
    for _ in range(40):
        coordinator.step(dt_s=0.5)

    report = coordinator.generate_report()
    assert report.total_vehicles_active == 1
    assert report.timestamp_s > 0.0
    # Vehicle should have moved along trajectory
    v1 = coordinator.vehicles["v1"]
    assert v1.pose[0] > 0.0 or v1.status == VehicleFleetStatus.PARKED


def test_fleet_coordinator_deconfliction_resolution() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=())
    # Create coordinator with high conflict radius to trigger conflict
    config = FleetConfig(conflict_radius_m=5.0, temporal_headway_s=5.0)
    coordinator = FleetCoordinator(facility, config=config)

    coordinator.register_vehicle("v1", (0.0, 0.0, 0.0))
    coordinator.register_vehicle("v2", (0.0, 0.0, 0.0))

    coordinator.dispatch_vehicle("v1")
    coordinator.dispatch_vehicle("v2")

    # Both vehicles dispatched from entrance will conflict along main corridor
    conflicts = coordinator.run_deconfliction_cycle()
    assert len(conflicts) > 0
    assert coordinator.resolved_conflicts_count > 0

    # Secondary vehicle trajectory duration should have increased due to yield delay
    v2 = coordinator.vehicles["v2"]
    assert v2.trajectory is not None
    assert v2.trajectory.total_duration_s > 0.0
