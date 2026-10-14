"""Pedestrian safety interlock engine and motion clearance debounce monitor."""

from __future__ import annotations

from nearfield360.config import CurbsideConfig
from nearfield360.curbside.models import PedestrianDetection, SafetyInterlockStatus


class InterlockViolationError(PermissionError):
    """Raised when an autonomous motion attempt is made while a safety interlock is engaged."""


class SafetyInterlockEngine:
    """Evaluates vehicle door, trunk, and pedestrian proximity interlocks for AVP handovers."""

    def __init__(self, config: CurbsideConfig | None = None) -> None:
        self.config = config or CurbsideConfig()
        self._doors_open: dict[str, bool] = {}
        self._trunk_open: dict[str, bool] = {}
        self._last_pedestrian_s: dict[str, float] = {}

    def set_door_state(self, vehicle_id: str, is_open: bool) -> None:
        """Update door open/closed latch state for vehicle."""
        self._doors_open[vehicle_id] = is_open

    def set_trunk_state(self, vehicle_id: str, is_open: bool) -> None:
        """Update trunk open/closed latch state for vehicle."""
        self._trunk_open[vehicle_id] = is_open

    def update_pedestrians(
        self,
        vehicle_id: str,
        pedestrians: list[PedestrianDetection],
        current_time_s: float,
    ) -> list[PedestrianDetection]:
        """Evaluate proximity detections and mark critical zone violations."""
        annotated: list[PedestrianDetection] = []
        any_critical = False

        for ped in pedestrians:
            is_critical = ped.distance_m < self.config.pedestrian_safety_radius_m
            if is_critical:
                any_critical = True
            annotated.append(ped.model_copy(update={"in_critical_zone": is_critical}))

        if any_critical:
            self._last_pedestrian_s[vehicle_id] = current_time_s

        return annotated

    def evaluate_interlock(
        self,
        vehicle_id: str,
        current_time_s: float,
    ) -> tuple[bool, SafetyInterlockStatus]:
        """Evaluate motion permission and determine active interlock state.

        Returns:
            Tuple of (is_motion_permitted, status).
        """
        if self._doors_open.get(vehicle_id, False):
            return False, SafetyInterlockStatus.ENGAGED_DOOR_OPEN

        if self._trunk_open.get(vehicle_id, False):
            return False, SafetyInterlockStatus.ENGAGED_TRUNK_OPEN

        last_ped_s = self._last_pedestrian_s.get(vehicle_id)
        if last_ped_s is not None:
            elapsed_since_ped = current_time_s - last_ped_s
            if elapsed_since_ped < self.config.interlock_clearance_hold_s:
                return False, SafetyInterlockStatus.ENGAGED_PEDESTRIAN_PROXIMITY

        return True, SafetyInterlockStatus.DISENGAGED_CLEAR

    def assert_motion_permitted(self, vehicle_id: str, current_time_s: float) -> None:
        """Raise InterlockViolationError if autonomous motion is currently inhibited."""
        permitted, status = self.evaluate_interlock(vehicle_id, current_time_s)
        if not permitted:
            raise InterlockViolationError(
                f"Autonomous motion prohibited for vehicle '{vehicle_id}': "
                f"safety interlock active ({status.value})."
            )
