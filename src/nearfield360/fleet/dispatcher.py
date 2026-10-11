"""Dynamic parking slot allocation, reservation lease management, and facility dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nearfield360.config import FleetConfig
from nearfield360.fleet.models import (
    DispatchAssignment,
    FacilityOccupancySummary,
)
from nearfield360.mapping.models import (
    FacilityMap,
    FacilitySlot,
    SlotBayType,
    SlotReservationStatus,
)
from nearfield360.mapping.router import GlobalRouter


class SlotAllocationError(RuntimeError):
    """Raised when slot assignment cannot be fulfilled."""


@dataclass
class SlotReservationLease:
    """Active reservation lease linking a parking slot to a designated vehicle."""

    slot_id: str
    vehicle_id: str
    granted_at_s: float
    expires_at_s: float
    confirmed: bool = False


class SlotDispatcher:
    """Manages slot allocations, lease expiries, and optimal target assignment."""

    def __init__(
        self,
        facility: FacilityMap,
        config: FleetConfig | None = None,
        router: GlobalRouter | None = None,
    ) -> None:
        self.facility = facility
        self.config = config or FleetConfig()
        self.router = router or GlobalRouter(facility)

        # Tracked slot states (overriding static map statuses if modified dynamically)
        self._slot_statuses: dict[str, SlotReservationStatus] = {
            slot.slot_id: slot.status for slot in self.facility.slots
        }
        # Active leases: slot_id -> lease
        self._leases: dict[str, SlotReservationLease] = {}
        # Vehicle active assignments: vehicle_id -> DispatchAssignment
        self._vehicle_assignments: dict[str, DispatchAssignment] = {}
        # Next assignment counter
        self._assignment_seq: int = 1

    @property
    def total_slots(self) -> int:
        return len(self.facility.slots)

    def get_slot_status(self, slot_id: str) -> SlotReservationStatus:
        return self._slot_statuses.get(slot_id, SlotReservationStatus.BLOCKED)

    def set_slot_status(self, slot_id: str, status: SlotReservationStatus) -> None:
        if slot_id in self._slot_statuses:
            self._slot_statuses[slot_id] = status
            if status in (SlotReservationStatus.VACANT, SlotReservationStatus.BLOCKED):
                # Clean up lease if any
                self._leases.pop(slot_id, None)

    def clean_expired_leases(self, current_time_s: float) -> list[str]:
        """Release reservation leases that exceeded lease_expires_at_s without confirmation."""
        expired_slots: list[str] = []
        for slot_id, lease in list(self._leases.items()):
            if not lease.confirmed and current_time_s > lease.expires_at_s:
                expired_slots.append(slot_id)
                self._leases.pop(slot_id, None)
                self._slot_statuses[slot_id] = SlotReservationStatus.VACANT
                # Remove vehicle assignment
                self._vehicle_assignments.pop(lease.vehicle_id, None)
        return expired_slots

    def allocate_slot(
        self,
        vehicle_id: str,
        vehicle_pose: tuple[float, float, float],
        current_time_s: float,
        preferred_type: SlotBayType | None = None,
        priority: int = 1,
    ) -> DispatchAssignment:
        """Find the optimal vacant parking slot and issue a reservation lease."""
        self.clean_expired_leases(current_time_s)

        # Check if vehicle already has an active assignment
        if vehicle_id in self._vehicle_assignments:
            existing = self._vehicle_assignments[vehicle_id]
            if existing.lease_expires_at_s > current_time_s:
                return existing

        # Find candidates
        candidates: list[FacilitySlot] = [
            slot
            for slot in self.facility.slots
            if self._slot_statuses.get(slot.slot_id) == SlotReservationStatus.VACANT
            and (preferred_type is None or slot.slot_type == preferred_type)
        ]

        if not candidates:
            msg = f"No vacant parking slots available in facility {self.facility.map_id}"
            raise SlotAllocationError(msg)

        # Score candidates: combine Euclidean distance and aisle density / turn count
        scored_candidates: list[tuple[float, FacilitySlot, list[str]]] = []
        vx, vy, _ = vehicle_pose

        for slot in candidates:
            # Quick distance heuristic
            dist = math.hypot(slot.center[0] - vx, slot.center[1] - vy)
            try:
                # Plan route to test connectivity and obtain waypoint IDs
                route = self.router.plan(vehicle_pose, target_slot_id=slot.slot_id)
                effective_cost = route.total_length_m + (0.5 * len(route.waypoints))
                wp_ids = [f"RW_{i}" for i in range(len(route.waypoints))]
            except Exception:
                # If route planner fails, penalize heavily
                effective_cost = dist + 1000.0
                wp_ids = []

            scored_candidates.append((effective_cost, slot, wp_ids))

        scored_candidates.sort(key=lambda item: item[0])
        best_cost, best_slot, route_wp_ids = scored_candidates[0]

        # Create lease
        lease_duration = self.config.reservation_lease_duration_s
        expires_at = current_time_s + lease_duration
        lease = SlotReservationLease(
            slot_id=best_slot.slot_id,
            vehicle_id=vehicle_id,
            granted_at_s=current_time_s,
            expires_at_s=expires_at,
            confirmed=False,
        )
        self._leases[best_slot.slot_id] = lease
        self._slot_statuses[best_slot.slot_id] = SlotReservationStatus.RESERVED

        assignment = DispatchAssignment(
            assignment_id=f"asg_{self._assignment_seq:04d}",
            vehicle_id=vehicle_id,
            assigned_slot_id=best_slot.slot_id,
            route_waypoint_ids=route_wp_ids,
            issued_at_s=current_time_s,
            lease_expires_at_s=expires_at,
            target_arrival_s=current_time_s + (best_cost / 1.5),
            priority_level=priority,
        )
        self._assignment_seq += 1
        self._vehicle_assignments[vehicle_id] = assignment
        return assignment

    def confirm_reservation(self, vehicle_id: str, slot_id: str) -> bool:
        """Confirm vehicle arrival or lock for parking maneuver."""
        lease = self._leases.get(slot_id)
        if lease and lease.vehicle_id == vehicle_id:
            lease.confirmed = True
            return True
        return False

    def complete_parking(self, vehicle_id: str, slot_id: str) -> bool:
        """Mark slot as occupied once vehicle successfully docks."""
        lease = self._leases.get(slot_id)
        if lease and lease.vehicle_id == vehicle_id:
            self._slot_statuses[slot_id] = SlotReservationStatus.OCCUPIED
            self._leases.pop(slot_id, None)
            self._vehicle_assignments.pop(vehicle_id, None)
            return True
        return False

    def release_slot(self, vehicle_id: str, slot_id: str) -> bool:
        """Release a slot when vehicle departs or cancels."""
        if slot_id in self._slot_statuses:
            self._slot_statuses[slot_id] = SlotReservationStatus.VACANT
            self._leases.pop(slot_id, None)
            self._vehicle_assignments.pop(vehicle_id, None)
            return True
        return False

    def get_occupancy_summary(self, current_time_s: float) -> FacilityOccupancySummary:
        """Aggregate current occupancy status across all bays."""
        self.clean_expired_leases(current_time_s)

        vacant = 0
        reserved = 0
        occupied = 0
        blocked = 0
        allocations: dict[str, str] = {}

        for slot_id, status in self._slot_statuses.items():
            if status == SlotReservationStatus.VACANT:
                vacant += 1
            elif status == SlotReservationStatus.RESERVED:
                reserved += 1
                lease = self._leases.get(slot_id)
                if lease:
                    allocations[slot_id] = lease.vehicle_id
            elif status == SlotReservationStatus.OCCUPIED:
                occupied += 1
            elif status == SlotReservationStatus.BLOCKED:
                blocked += 1

        return FacilityOccupancySummary(
            timestamp_s=current_time_s,
            total_slots=len(self._slot_statuses),
            vacant_slots=vacant,
            reserved_slots=reserved,
            occupied_slots=occupied,
            blocked_slots=blocked,
            slot_allocations=allocations,
        )
