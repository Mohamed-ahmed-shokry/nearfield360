"""Centralized Cloud Fleet Coordinator and multi-vehicle discrete-time simulator."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from nearfield360.config import FleetConfig
from nearfield360.fleet.deconfliction import DeconflictionEngine, build_trajectory_profile
from nearfield360.fleet.dispatcher import SlotDispatcher
from nearfield360.fleet.models import (
    ConflictPoint,
    DispatchAssignment,
    FleetDispatchReport,
    TrajectoryProfile,
    V2XMessage,
    V2XMessageType,
    VehicleFleetStatus,
)
from nearfield360.mapping.models import FacilityMap, SlotReservationStatus
from nearfield360.mapping.router import GlobalRouter


@dataclass
class SimulatedVehicle:
    """State of an autonomous vehicle simulated within the multi-agent fleet environment."""

    vehicle_id: str
    pose: tuple[float, float, float]
    speed_mps: float = 0.0
    status: VehicleFleetStatus = VehicleFleetStatus.UNASSIGNED
    assignment: DispatchAssignment | None = None
    trajectory: TrajectoryProfile | None = None
    current_wp_idx: int = 0
    wait_time_s: float = 0.0
    battery_pct: float = 100.0


class FleetCoordinator:
    """Coordinates fleet dispatch, multi-vehicle navigation, and V2X messaging."""

    def __init__(
        self,
        facility: FacilityMap,
        config: FleetConfig | None = None,
        router: GlobalRouter | None = None,
        dispatcher: SlotDispatcher | None = None,
        deconfliction: DeconflictionEngine | None = None,
        random_seed: int = 42,
    ) -> None:
        self.facility = facility
        self.config = config or FleetConfig()
        self.router = router or GlobalRouter(facility)
        self.dispatcher = dispatcher or SlotDispatcher(facility, self.config, self.router)
        self.deconfliction = deconfliction or DeconflictionEngine(self.config)
        self.rng = random.Random(random_seed)  # noqa: S311

        self.vehicles: dict[str, SimulatedVehicle] = {}
        self.active_conflicts: list[ConflictPoint] = []
        self.resolved_conflicts_count: int = 0
        self.message_queue: list[V2XMessage] = []
        self.current_time_s: float = 0.0

    def register_vehicle(
        self,
        vehicle_id: str,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        battery_pct: float = 100.0,
    ) -> SimulatedVehicle:
        """Register a vehicle into the fleet."""
        veh = SimulatedVehicle(
            vehicle_id=vehicle_id,
            pose=initial_pose,
            battery_pct=battery_pct,
            status=VehicleFleetStatus.UNASSIGNED,
        )
        self.vehicles[vehicle_id] = veh
        return veh

    def dispatch_vehicle(
        self,
        vehicle_id: str,
        target_slot_id: str | None = None,
    ) -> DispatchAssignment:
        """Assign target parking bay and compute global route for a vehicle."""
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            msg = f"Vehicle {vehicle_id} not registered in fleet"
            raise ValueError(msg)

        if target_slot_id is not None:
            # Explicit slot assignment
            slot = self.facility.get_slot(target_slot_id)
            if slot is None:
                msg = f"Target slot {target_slot_id} not found in facility"
                raise ValueError(msg)
            # Reserve in dispatcher
            self.dispatcher.set_slot_status(target_slot_id, SlotReservationStatus.RESERVED)
            route = self.router.plan(veh.pose, target_slot_id=target_slot_id)
            asg = DispatchAssignment(
                assignment_id=f"asg_{len(self.vehicles):04d}",
                vehicle_id=vehicle_id,
                assigned_slot_id=target_slot_id,
                route_waypoint_ids=[f"RW_{i}" for i in range(len(route.waypoints))],
                issued_at_s=self.current_time_s,
                lease_expires_at_s=self.current_time_s + self.config.reservation_lease_duration_s,
                target_arrival_s=self.current_time_s + route.estimated_duration_s,
            )
        else:
            asg = self.dispatcher.allocate_slot(vehicle_id, veh.pose, self.current_time_s)
            route = self.router.plan(veh.pose, target_slot_id=asg.assigned_slot_id)

        veh.assignment = asg
        veh.status = VehicleFleetStatus.EN_ROUTE_SLOT
        veh.current_wp_idx = 0

        # Build 4D trajectory profile
        veh.trajectory = build_trajectory_profile(
            vehicle_id,
            route,
            start_time_s=self.current_time_s,
            temporal_headway_s=self.config.temporal_headway_s,
        )

        # Transmit V2X Dispatch Assignment Message
        self.send_v2x_message(
            V2XMessage(
                message_type=V2XMessageType.DISPATCH_ASSIGNMENT,
                sender_id="cloud_dispatcher",
                recipient_id=vehicle_id,
                timestamp_s=self.current_time_s,
                payload=asg.model_dump(mode="json"),
            )
        )
        return asg

    def send_v2x_message(self, message: V2XMessage) -> bool:
        """Simulate sending a V2X message over stochastic wireless channel."""
        # Check simulated packet loss
        if self.rng.random() < self.config.packet_loss_rate:
            return False  # Dropped packet

        self.message_queue.append(message)
        return True

    def run_deconfliction_cycle(self) -> list[ConflictPoint]:
        """Detect conflicts across all active vehicles and apply yield delays."""
        active_profiles: list[TrajectoryProfile] = [
            v.trajectory
            for v in self.vehicles.values()
            if v.trajectory is not None and v.status == VehicleFleetStatus.EN_ROUTE_SLOT
        ]
        if len(active_profiles) < 2:
            self.active_conflicts = []
            return []

        conflicts = self.deconfliction.detect_conflicts(active_profiles)
        self.active_conflicts = conflicts

        for c in conflicts:
            # Yield secondary vehicle
            secondary_veh = self.vehicles.get(c.secondary_vehicle_id)
            if secondary_veh and secondary_veh.trajectory:
                secondary_veh.trajectory = self.deconfliction.adjust_trajectory_for_yield(
                    secondary_veh.trajectory,
                    c,
                    hold_duration_s=self.config.temporal_headway_s,
                )
                self.resolved_conflicts_count += 1
                c.resolved = True

        return conflicts

    def step(self, dt_s: float = 0.5) -> None:
        """Advance the multi-agent simulation forward by dt_s."""
        self.current_time_s += dt_s

        # Run deconfliction cycle periodically
        self.run_deconfliction_cycle()

        for veh in self.vehicles.values():
            if veh.status != VehicleFleetStatus.EN_ROUTE_SLOT or veh.trajectory is None:
                continue

            # Check if vehicle has arrived at goal
            if veh.current_wp_idx >= len(veh.trajectory.waypoints):
                veh.status = VehicleFleetStatus.PARKED
                veh.speed_mps = 0.0
                if veh.assignment:
                    self.dispatcher.complete_parking(
                        veh.vehicle_id, veh.assignment.assigned_slot_id
                    )
                continue

            target_wp = veh.trajectory.waypoints[veh.current_wp_idx]
            # Check if held back by time schedule
            if self.current_time_s < target_wp.t_expected_s:
                veh.speed_mps = 0.0
                veh.wait_time_s += dt_s
                continue

            # Move towards waypoint
            curr_x, curr_y, _ = veh.pose
            dx = target_wp.x - curr_x
            dy = target_wp.y - curr_y
            dist = math.hypot(dx, dy)

            speed = max(0.5, target_wp.speed_target_mps)
            veh.speed_mps = speed
            travel_dist = speed * dt_s

            if travel_dist >= dist:
                veh.pose = (target_wp.x, target_wp.y, target_wp.heading_rad)
                veh.current_wp_idx += 1
            else:
                heading = math.atan2(dy, dx)
                new_x = curr_x + travel_dist * math.cos(heading)
                new_y = curr_y + travel_dist * math.sin(heading)
                veh.pose = (new_x, new_y, heading)

    def generate_report(self) -> FleetDispatchReport:
        """Generate comprehensive fleet coordination telemetry report."""
        occ_summary = self.dispatcher.get_occupancy_summary(self.current_time_s)
        assignments = [v.assignment for v in self.vehicles.values() if v.assignment is not None]
        avg_wait = sum(v.wait_time_s for v in self.vehicles.values()) / max(1, len(self.vehicles))

        return FleetDispatchReport(
            facility_id=self.facility.map_id,
            timestamp_s=self.current_time_s,
            total_vehicles_active=len(self.vehicles),
            occupancy_summary=occ_summary,
            active_assignments=assignments,
            active_conflicts=self.active_conflicts,
            resolved_conflicts_count=self.resolved_conflicts_count,
            average_wait_time_s=avg_wait,
        )
