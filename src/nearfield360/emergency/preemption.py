"""Emergency Vehicle Preemption (EVP) arbiter and corridor clearance engine."""

from __future__ import annotations

import math
from collections.abc import Mapping

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.models import EmergencyVehicle, VehicleEvacuationStatus


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate the Euclidean distance from a 2D point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    segment_len_sq = dx * dx + dy * dy
    if segment_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / segment_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class EmergencyPreemptionArbiter:
    """Arbitrates right-of-way preemption for approaching first responders."""

    def __init__(self, config: EmergencyConfig | None = None) -> None:
        self.config = config or EmergencyConfig()
        self._responders: dict[str, EmergencyVehicle] = {}

    @property
    def responders(self) -> Mapping[str, EmergencyVehicle]:
        """Read-only view of registered emergency vehicles."""
        return self._responders

    def register_responder(self, responder: EmergencyVehicle) -> None:
        """Register or update an active first responder vehicle."""
        self._responders[responder.vehicle_id] = responder

    def remove_responder(self, vehicle_id: str) -> bool:
        """Deregister an emergency responder when clear of the facility."""
        return self._responders.pop(vehicle_id, None) is not None

    def clear(self) -> None:
        """Clear all registered emergency responders."""
        self._responders.clear()

    def is_preemption_active(self) -> bool:
        """Return True if any active emergency responder is currently registered."""
        return bool(self._responders)

    def evaluate_vehicle(
        self,
        vehicle_id: str,
        pose: tuple[float, float],
        floor_id: str = "L1",
    ) -> tuple[VehicleEvacuationStatus, str | None]:
        """Evaluate whether a vehicle must yield to approaching emergency units."""
        if not self._responders:
            return VehicleEvacuationStatus.STATIONARY_SAFE, None

        vx, vy = pose
        radius = self.config.preemption_radius_m

        for resp in self._responders.values():
            dist_to_responder = math.hypot(vx - resp.x, vy - resp.y)
            if dist_to_responder <= radius:
                msg = (
                    f"Yielding right-of-way to {resp.responder_type.value} "
                    f"'{resp.vehicle_id}' (distance {dist_to_responder:.1f}m <= {radius:.1f}m)"
                )
                return VehicleEvacuationStatus.HOLD_YIELDING, msg

            # Evaluate corridor along responder trajectory to its destination
            has_dest = abs(resp.target_x - resp.x) > 1e-3 or abs(resp.target_y - resp.y) > 1e-3
            if has_dest:
                dist_to_corridor = point_to_segment_distance(
                    vx, vy, resp.x, resp.y, resp.target_x, resp.target_y
                )
                corridor_buf = radius * 0.5
                if dist_to_corridor <= corridor_buf:
                    msg = (
                        f"In corridor path of {resp.responder_type.value} '{resp.vehicle_id}' "
                        f"(offset {dist_to_corridor:.1f}m <= {corridor_buf:.1f}m)"
                    )
                    return VehicleEvacuationStatus.HOLD_YIELDING, msg

        return VehicleEvacuationStatus.STATIONARY_SAFE, None

    def evaluate_fleet(
        self,
        vehicles: Mapping[str, tuple[float, float]],
        floor_id: str = "L1",
    ) -> dict[str, tuple[VehicleEvacuationStatus, str | None]]:
        """Evaluate preemption status across all active facility vehicles."""
        return {
            vid: self.evaluate_vehicle(vid, pose, floor_id=floor_id)
            for vid, pose in vehicles.items()
        }
