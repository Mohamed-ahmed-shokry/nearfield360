"""VIP express corridor reservation and traffic preemption engine."""

from __future__ import annotations

from nearfield360.tenant.models import VIPExpressCorridor


class CorridorConflictError(ValueError):
    """Raised when a corridor reservation conflicts with an active higher or equal priority one."""


class VIPCorridorManager:
    """Manages dedicated VIP express corridors and right-of-way preemption on facility lanes."""

    def __init__(self, corridors: list[VIPExpressCorridor] | None = None) -> None:
        self._corridors: dict[str, VIPExpressCorridor] = {}
        if corridors:
            for c in corridors:
                self._corridors[c.corridor_id] = c

    @property
    def corridors(self) -> list[VIPExpressCorridor]:
        """Return all registered VIP express corridors."""
        return list(self._corridors.values())

    def register_corridor(
        self,
        corridor_id: str,
        lane_ids: list[str],
        default_duration_s: float = 60.0,
    ) -> VIPExpressCorridor:
        """Register a new express corridor geometry."""
        corridor = VIPExpressCorridor(
            corridor_id=corridor_id,
            lane_ids=lane_ids,
            duration_s=default_duration_s,
        )
        self._corridors[corridor_id] = corridor
        return corridor

    def reserve_corridor(
        self,
        corridor_id: str,
        vehicle_id: str,
        start_time_s: float,
        duration_s: float | None = None,
        priority_level: int = 1,
    ) -> VIPExpressCorridor:
        """Reserve a corridor for a vehicle, preempting lower-priority reservations if necessary."""
        if corridor_id not in self._corridors:
            raise KeyError(f"Corridor '{corridor_id}' is not registered.")

        target = self._corridors[corridor_id]
        dur = duration_s if duration_s is not None else target.duration_s

        # Check lane overlaps with other active corridors
        for other in self._corridors.values():
            if not other.is_active_at(start_time_s):
                continue

            shared_lanes = set(target.lane_ids).intersection(set(other.lane_ids))
            if shared_lanes:
                if other.priority_level >= priority_level:
                    raise CorridorConflictError(
                        f"Lane(s) {sorted(shared_lanes)} occupied by "
                        f"corridor '{other.corridor_id}' "
                        f"(priority {other.priority_level} >= {priority_level})."
                    )
                # Lower priority: preempt existing reservation
                preempted = other.model_copy(
                    update={
                        "active_vehicle_id": None,
                        "is_preempting": False,
                    }
                )
                self._corridors[other.corridor_id] = preempted

        is_preempting = priority_level > 1
        updated = target.model_copy(
            update={
                "active_vehicle_id": vehicle_id,
                "reservation_start_s": start_time_s,
                "duration_s": dur,
                "is_preempting": is_preempting,
                "priority_level": priority_level,
            }
        )
        self._corridors[corridor_id] = updated
        return updated

    def release_corridor(self, corridor_id: str) -> None:
        """Release active reservation on corridor."""
        if corridor_id in self._corridors:
            corridor = self._corridors[corridor_id]
            updated = corridor.model_copy(
                update={
                    "active_vehicle_id": None,
                    "is_preempting": False,
                }
            )
            self._corridors[corridor_id] = updated

    def get_active_corridors(self, current_time_s: float) -> list[VIPExpressCorridor]:
        """Return all corridors with an active reservation at given timestamp."""
        return [c for c in self._corridors.values() if c.is_active_at(current_time_s)]

    def is_lane_reserved(
        self, lane_id: str, current_time_s: float
    ) -> tuple[bool, str | None]:
        """Check whether a lane is currently reserved by an active express corridor.

        Returns:
            Tuple of (is_reserved, active_vehicle_id).
        """
        for c in self.get_active_corridors(current_time_s):
            if lane_id in c.lane_ids:
                return True, c.active_vehicle_id
        return False, None
