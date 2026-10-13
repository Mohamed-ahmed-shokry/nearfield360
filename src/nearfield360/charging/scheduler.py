"""EV Charging queue scheduler, bay assignment, and relocation queue manager."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargingSession,
    ChargingSessionStatus,
    RelocationOrder,
)
from nearfield360.config import ChargingConfig
from nearfield360.mapping.models import FacilityMap


@dataclass
class ChargingRequest:
    """Incoming vehicle request queued for charging."""

    vehicle_id: str
    arrival_time_s: float
    current_soc: float
    target_soc: float = 0.80
    battery_capacity_kwh: float = 75.0
    max_charge_acceptance_kw: float = 100.0
    priority_override: int | None = None


class ChargingScheduler:
    """Manages EV charging queues, bay assignments, and multi-stage relocation dispatches.

    Prioritizes waiting vehicles based on critical battery urgency, arrival time, and charging
    compatibility. Once a vehicle reaches its target SoC, it monitors dwell times and generates
    relocation orders to move the vehicle to a standard staging parking slot.
    """

    def __init__(
        self,
        config: ChargingConfig | None = None,
        facility_map: FacilityMap | None = None,
    ) -> None:
        self.config = config or ChargingConfig()
        self.facility_map = facility_map
        self._queue: list[ChargingRequest] = []
        self._bays: dict[str, ChargerBay] = {}
        self._sessions: dict[str, ChargingSession] = {}
        self._relocations: list[RelocationOrder] = []

    @property
    def queue(self) -> list[ChargingRequest]:
        """Current waiting queue of vehicles."""
        return list(self._queue)

    @property
    def active_sessions(self) -> list[ChargingSession]:
        """All active, pending, or completed sessions."""
        return list(self._sessions.values())

    @property
    def relocation_orders(self) -> list[RelocationOrder]:
        """Pending vehicle relocation orders."""
        return list(self._relocations)

    def register_charger_bays(self, bays: Sequence[ChargerBay]) -> None:
        """Register facility charger bays."""
        for b in bays:
            self._bays[b.bay_id] = b

    def enqueue_request(self, request: ChargingRequest) -> None:
        """Enqueue an EV charging request."""
        # Avoid duplicate requests for the same vehicle
        if any(r.vehicle_id == request.vehicle_id for r in self._queue):
            return
        if any(
            s.vehicle_id == request.vehicle_id
            and s.status
            in (ChargingSessionStatus.ACTIVE_CHARGING, ChargingSessionStatus.PENDING_DOCKING)
            for s in self._sessions.values()
        ):
            return
        self._queue.append(request)

    def _score_request_priority(self, req: ChargingRequest, current_time_s: float) -> float:
        """Compute sorting priority score (higher score = served earlier)."""
        if req.priority_override is not None:
            return float(req.priority_override) * 100.0

        # Critical battery boost
        urgency = 0.0
        if req.current_soc < self.config.critical_soc_threshold:
            urgency += 50.0

        # Deficit score
        deficit = max(0.0, req.target_soc - req.current_soc)
        urgency += deficit * 20.0

        # Waiting time score (0.1 pt per second waiting)
        wait_time = max(0.0, current_time_s - req.arrival_time_s)
        urgency += wait_time * 0.1

        return urgency

    def schedule_assignments(self, current_time_s: float) -> list[ChargingSession]:
        """Match highest priority queued vehicles to available compatible charger bays."""
        if not self._queue:
            return []

        # Sort queue by priority descending
        self._queue.sort(
            key=lambda r: self._score_request_priority(r, current_time_s),
            reverse=True,
        )

        assigned_sessions: list[ChargingSession] = []
        unassigned_queue: list[ChargingRequest] = []

        available_bays = [b for b in self._bays.values() if b.status == ChargerPortStatus.AVAILABLE]

        for req in self._queue:
            if not available_bays:
                unassigned_queue.append(req)
                continue

            # Pick best matching bay (e.g. prefer DC fast if vehicle accepts high rate)
            best_bay = available_bays.pop(0)

            # Create session
            session_id = f"SESS_{uuid.uuid4().hex[:8]}"
            session = ChargingSession(
                session_id=session_id,
                vehicle_id=req.vehicle_id,
                bay_id=best_bay.bay_id,
                start_time_s=current_time_s,
                last_update_time_s=current_time_s,
                start_soc=req.current_soc,
                current_soc=req.current_soc,
                target_soc=req.target_soc,
                allocated_power_kw=min(best_bay.max_power_rating_kw, req.max_charge_acceptance_kw),
                status=ChargingSessionStatus.PENDING_DOCKING,
            )

            # Update bay status
            object.__setattr__(best_bay, "status", ChargerPortStatus.OCCUPIED_CHARGING)
            object.__setattr__(best_bay, "occupied_by_vehicle_id", req.vehicle_id)

            self._sessions[session_id] = session
            assigned_sessions.append(session)

        self._queue = unassigned_queue
        return assigned_sessions

    def check_completions_and_relocations(
        self,
        current_time_s: float,
        staging_slots: Sequence[str] | None = None,
    ) -> list[RelocationOrder]:
        """Check for completed sessions and generate relocation orders to staging slots."""
        new_relocations: list[RelocationOrder] = []
        available_staging = list(staging_slots or [])

        # If no explicit staging slots provided and facility map exists, find regular vacant slots
        if not available_staging and self.facility_map:
            charging_slot_ids = {b.slot_id for b in self._bays.values()}
            available_staging = [
                s.slot_id
                for s in self.facility_map.slots
                if s.slot_id not in charging_slot_ids and s.status.value == "vacant"
            ]

        for session in list(self._sessions.values()):
            bay = self._bays.get(session.bay_id)
            if not bay:
                continue

            # Check if target SoC reached
            if session.current_soc >= session.target_soc and session.status in (
                ChargingSessionStatus.ACTIVE_CHARGING,
                ChargingSessionStatus.TOPPING_OFF,
            ):
                session.status = ChargingSessionStatus.COMPLETED
                session.completion_time_s = current_time_s
                object.__setattr__(
                    bay, "status", ChargerPortStatus.OCCUPIED_COMPLETE_WAITING_RELOCATION
                )

            # If completed and waiting relocation, trigger relocation order
            if session.status == ChargingSessionStatus.COMPLETED and not any(
                r.vehicle_id == session.vehicle_id for r in self._relocations
            ):
                target_slot = (
                    available_staging.pop(0) if available_staging else "STAGING_STAGING_01"
                )
                order = RelocationOrder(
                    order_id=f"RELOC_{uuid.uuid4().hex[:8]}",
                    vehicle_id=session.vehicle_id,
                    source_bay_id=session.bay_id,
                    target_slot_id=target_slot,
                    created_at_s=current_time_s,
                    reason="target_soc_reached",
                    urgency_priority=2,
                )
                self._relocations.append(order)
                new_relocations.append(order)

        return new_relocations

    def complete_relocation(self, order_id: str) -> None:
        """Mark a relocation order complete and free the charging bay for subsequent sessions."""
        matching = [r for r in self._relocations if r.order_id == order_id]
        if not matching:
            return

        order = matching[0]
        self._relocations.remove(order)

        # Free bay
        bay = self._bays.get(order.source_bay_id)
        if bay:
            object.__setattr__(bay, "status", ChargerPortStatus.AVAILABLE)
            object.__setattr__(bay, "occupied_by_vehicle_id", None)
            object.__setattr__(bay, "current_power_kw", 0.0)
