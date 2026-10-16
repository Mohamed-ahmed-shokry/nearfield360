"""Dynamic hazard containment, EV thermal runaway detection, and fire door interlocks."""

from __future__ import annotations

import math
from collections.abc import Mapping

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    VehicleEvacuationStatus,
)


class HazardContainmentEngine:
    """Manages spatial hazard exclusion zones, thermal alerts, and barrier closures."""

    def __init__(self, config: EmergencyConfig | None = None) -> None:
        self.config = config or EmergencyConfig()
        self._hazards: dict[str, HazardZone] = {}
        self._fire_doors: dict[str, FireDoorInterlock] = {}

    @property
    def hazards(self) -> Mapping[str, HazardZone]:
        """Read-only view of active hazard zones."""
        return self._hazards

    @property
    def fire_doors(self) -> Mapping[str, FireDoorInterlock]:
        """Read-only view of registered fire barrier doors."""
        return self._fire_doors

    def add_hazard(self, hazard: HazardZone) -> None:
        """Register or update an active hazard zone."""
        self._hazards[hazard.hazard_id] = hazard

    def remove_hazard(self, hazard_id: str) -> bool:
        """Deregister an extinguished or neutralized hazard zone."""
        return self._hazards.pop(hazard_id, None) is not None

    def clear_hazards(self) -> None:
        """Clear all registered hazard zones."""
        self._hazards.clear()

    def report_ev_temperature(
        self,
        vehicle_id: str,
        bay_id: str,
        center_xy: tuple[float, float],
        temp_c: float,
        floor_id: str = "L1",
        timestamp_s: float = 0.0,
    ) -> HazardZone | None:
        """Evaluate EV battery pack temperature and initiate thermal containment if critical."""
        if temp_c < self.config.thermal_runaway_temp_c:
            return None

        hazard_id = f"thermal-{vehicle_id}"
        hazard = HazardZone(
            hazard_id=hazard_id,
            emergency_type=EmergencyType.EV_THERMAL_RUNAWAY,
            severity=EmergencySeverity.PARTIAL_EVACUATION,
            center_x=center_xy[0],
            center_y=center_xy[1],
            center_z=0.0,
            radius_m=self.config.hazard_buffer_radius_m,
            floor_id=floor_id,
            active=True,
            reported_at_s=timestamp_s,
            description=(
                f"EV battery thermal runaway on vehicle '{vehicle_id}' "
                f"at bay '{bay_id}' ({temp_c:.1f}°C >= {self.config.thermal_runaway_temp_c:.1f}°C)"
            ),
        )
        self.add_hazard(hazard)
        return hazard

    def register_fire_door(self, door: FireDoorInterlock) -> None:
        """Register a facility fire door or rolling smoke curtain."""
        self._fire_doors[door.door_id] = door

    def update_fire_door_status(
        self,
        door_id: str,
        status: FireDoorStatus,
        time_to_close_s: float | None = None,
    ) -> bool:
        """Update operational status and countdown for a fire door."""
        door = self._fire_doors.get(door_id)
        if door is None:
            return False

        new_time = door.time_to_close_s if time_to_close_s is None else time_to_close_s
        self._fire_doors[door_id] = FireDoorInterlock(
            door_id=door.door_id,
            floor_id=door.floor_id,
            location_x=door.location_x,
            location_y=door.location_y,
            width_m=door.width_m,
            status=status,
            time_to_close_s=max(0.0, new_time),
            is_blocked=door.is_blocked,
        )
        return True

    def is_point_inside_hazard(
        self,
        x: float,
        y: float,
        floor_id: str = "L1",
        buffer_m: float = 0.0,
    ) -> bool:
        """Check if 2D coordinates violate any active hazard exclusion boundary."""
        for hz in self._hazards.values():
            if not hz.active or hz.floor_id != floor_id:
                continue
            effective_radius = hz.radius_m + buffer_m
            dist = math.hypot(x - hz.center_x, y - hz.center_y)
            if dist <= effective_radius:
                return True
        return False

    def classify_vehicle_threat(
        self,
        vehicle_id: str,
        pose: tuple[float, float],
        floor_id: str = "L1",
    ) -> tuple[VehicleEvacuationStatus, str | None, float]:
        """Classify proximity threat level of a vehicle relative to active hazards."""
        if not self._hazards:
            return VehicleEvacuationStatus.STATIONARY_SAFE, None, float("inf")

        vx, vy = pose
        min_dist = float("inf")
        worst_hazard: HazardZone | None = None

        for hz in self._hazards.values():
            if not hz.active or hz.floor_id != floor_id:
                continue
            dist = math.hypot(vx - hz.center_x, vy - hz.center_y)
            if dist < min_dist:
                min_dist = dist
                worst_hazard = hz

        if worst_hazard is None:
            return VehicleEvacuationStatus.STATIONARY_SAFE, None, float("inf")

        if min_dist <= worst_hazard.radius_m:
            msg = (
                f"Vehicle '{vehicle_id}' inside hazard zone '{worst_hazard.hazard_id}' "
                f"({min_dist:.1f}m <= {worst_hazard.radius_m:.1f}m)"
            )
            return VehicleEvacuationStatus.TRAPPED_IN_HAZARD, msg, min_dist

        buffer_zone = worst_hazard.radius_m + self.config.hazard_buffer_radius_m
        if min_dist <= buffer_zone:
            msg = (
                f"Vehicle '{vehicle_id}' within hazard buffer of '{worst_hazard.hazard_id}' "
                f"({min_dist:.1f}m <= {buffer_zone:.1f}m)"
            )
            return VehicleEvacuationStatus.EVACUATING, msg, min_dist

        return VehicleEvacuationStatus.STATIONARY_SAFE, None, min_dist

    def can_transit_fire_door(
        self,
        door_id: str,
        estimated_transit_time_s: float,
    ) -> tuple[bool, str]:
        """Validate if a vehicle can safely transit an active fire door aperture."""
        door = self._fire_doors.get(door_id)
        if door is None:
            return True, f"Door '{door_id}' not found; assuming open passage"

        if door.status == FireDoorStatus.LOCKED_SHUT:
            return False, f"Fire door '{door_id}' is locked shut"

        if (
            door.status == FireDoorStatus.CLOSING
            and estimated_transit_time_s > door.time_to_close_s
        ):
            return (
                False,
                f"Transit time {estimated_transit_time_s:.1f}s exceeds remaining closure "
                f"countdown {door.time_to_close_s:.1f}s for door '{door_id}'",
            )

        return True, f"Clearance granted for fire door '{door_id}'"
