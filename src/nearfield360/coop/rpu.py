"""Roadside Perception Unit (RPU) station manager and geometric FOV profiler."""

from __future__ import annotations

import math

from nearfield360.coop.models import (
    CollectivePerceptionMessage,
    CPMPerceivedObject,
    CPMStationType,
    RPUSensorState,
    RPUSensorType,
)


class RPUSensorManager:
    """Manages stationary roadside perception units (RPUs) and computes sensor coverage."""

    def __init__(self) -> None:
        self._stations: dict[str, RPUSensorState] = {}
        self._sequence_counters: dict[str, int] = {}
        self._perceived_objects: dict[str, list[CPMPerceivedObject]] = {}

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

    def add_perceived_object(self, station_id: str, obj: CPMPerceivedObject) -> None:
        """Register an observed object detected by a specific RPU station."""
        if station_id not in self._stations:
            raise KeyError(f"RPU station '{station_id}' is not registered.")
        if obj.station_id != station_id:
            msg = (
                f"Object station_id '{obj.station_id}' does not match "
                f"target station '{station_id}'."
            )
            raise ValueError(msg)
        self._perceived_objects.setdefault(station_id, []).append(obj)

    def get_perceived_objects(self, station_id: str) -> list[CPMPerceivedObject]:
        """Return all active perceived objects registered for a station."""
        return list(self._perceived_objects.get(station_id, []))

    def clear_perceptions(self, station_id: str | None = None) -> None:
        """Clear active perceptions for one station or all stations if None."""
        if station_id is None:
            self._perceived_objects.clear()
        elif station_id in self._perceived_objects:
            self._perceived_objects[station_id].clear()

    def generate_cpm(
        self, station_id: str, timestamp_s: float, filter_fov: bool = False
    ) -> CollectivePerceptionMessage:
        """Construct an ETSI Collective Perception Message (CPM) for a given station."""
        station = self.get_station(station_id)
        current_seq = self._sequence_counters.get(station_id, 0)
        self._sequence_counters[station_id] = current_seq + 1

        raw_objects = self._perceived_objects.get(station_id, [])
        if filter_fov:
            perceptions = [
                obj for obj in raw_objects if self.is_point_in_fov(station_id, (obj.x_m, obj.y_m))
            ]
        else:
            perceptions = list(raw_objects)

        sx, sy, sz = station.position_m
        return CollectivePerceptionMessage(
            station_id=station_id,
            station_type=CPMStationType.INFRASTRUCTURE_RPU,
            sequence_number=current_seq,
            timestamp_s=timestamp_s,
            reference_x_m=sx,
            reference_y_m=sy,
            reference_z_m=sz,
            perceived_objects=perceptions,
        )

    @classmethod
    def create_default_demonstrator_network(cls) -> RPUSensorManager:
        """Factory creating a realistic multi-station cooperative facility network."""
        mgr = cls()

        # Station 1: LiDAR monitoring 90-degree blind corner between Main Aisle and West Corridor
        mgr.register_station(
            RPUSensorState(
                station_id="RPU-BLIND-CORNER-1",
                sensor_type=RPUSensorType.LIDAR,
                position_m=(20.0, 15.0, 4.0),
                yaw_deg=225.0,
                range_max_m=35.0,
                fov_azimuth_deg=120.0,
            )
        )

        # Station 2: Radar monitoring spiral ramp crest and floor transition
        mgr.register_station(
            RPUSensorState(
                station_id="RPU-RAMP-CREST-1",
                sensor_type=RPUSensorType.RADAR,
                position_m=(35.0, 40.0, 3.5),
                yaw_deg=180.0,
                range_max_m=45.0,
                fov_azimuth_deg=90.0,
            )
        )

        # Station 3: Overhead Fisheye Camera monitoring facility access gateway
        mgr.register_station(
            RPUSensorState(
                station_id="RPU-GATE-OVERHEAD-1",
                sensor_type=RPUSensorType.FISHEYE_CAMERA,
                position_m=(0.0, 10.0, 4.5),
                yaw_deg=0.0,
                range_max_m=25.0,
                fov_azimuth_deg=180.0,
            )
        )

        return mgr
