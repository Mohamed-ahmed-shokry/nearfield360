"""Roadside Perception Unit (RPU) station manager and geometric FOV profiler."""

from __future__ import annotations

import math

from nearfield360.coop.models import (
    RPUSensorState,
)


class RPUSensorManager:
    """Manages stationary roadside perception units (RPUs) and computes sensor coverage."""

    def __init__(self) -> None:
        self._stations: dict[str, RPUSensorState] = {}
        self._sequence_counters: dict[str, int] = {}

    def register_station(self, station: RPUSensorState) -> None:
        """Register or update a stationary roadside perception station."""
        self._stations[station.station_id] = station
        if station.station_id not in self._sequence_counters:
            self._sequence_counters[station.station_id] = 0

    def get_station(self, station_id: str) -> RPUSensorState:
        """Retrieve registered RPU state; raises KeyError if not found."""
        if station_id not in self._stations:
            raise KeyError(f"RPU station '{station_id}' is not registered.")
        return self._stations[station_id]

    def list_stations(self) -> list[RPUSensorState]:
        """Return all registered stations sorted by identifier."""
        return sorted(self._stations.values(), key=lambda s: s.station_id)

    def remove_station(self, station_id: str) -> bool:
        """Remove a station; returns True if station existed."""
        if station_id in self._stations:
            del self._stations[station_id]
            self._sequence_counters.pop(station_id, None)
            return True
        return False

    def is_point_in_fov(self, station_id: str, point: tuple[float, float]) -> bool:
        """Check if an $(x, y)$ coordinate falls within the station's horizontal FOV cone."""
        station = self.get_station(station_id)
        if not station.is_operational:
            return False

        sx, sy, _ = station.position_m
        px, py = point
        dx = px - sx
        dy = py - sy
        distance = math.hypot(dx, dy)

        if distance > station.range_max_m or distance < 0.1:
            return False

        angle_rad = math.atan2(dy, dx)
        yaw_rad = math.radians(station.yaw_deg)
        diff_rad = (angle_rad - yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
        half_fov_rad = math.radians(station.fov_azimuth_deg / 2.0)

        return abs(diff_rad) <= half_fov_rad

    def get_fov_polygon(
        self, station_id: str, num_arc_points: int = 16
    ) -> list[tuple[float, float]]:
        """Compute 2D polygon vertices representing the RPU sensor coverage sector."""
        station = self.get_station(station_id)
        sx, sy, _ = station.position_m
        yaw_rad = math.radians(station.yaw_deg)
        half_fov_rad = math.radians(station.fov_azimuth_deg / 2.0)
        start_angle = yaw_rad - half_fov_rad
        end_angle = yaw_rad + half_fov_rad

        polygon: list[tuple[float, float]] = [(sx, sy)]
        steps = max(4, num_arc_points)
        for i in range(steps + 1):
            theta = start_angle + (end_angle - start_angle) * (i / steps)
            rx = sx + station.range_max_m * math.cos(theta)
            ry = sy + station.range_max_m * math.sin(theta)
            polygon.append((rx, ry))

        return polygon
