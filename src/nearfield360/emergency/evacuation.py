"""Multi-wave facility emergency evacuation planner and egress route synthesizer."""

from __future__ import annotations

import math
from collections.abc import Mapping

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.containment import HazardContainmentEngine
from nearfield360.emergency.models import (
    EvacuationRoute,
    EvacuationWave,
    EvacuationWaypoint,
)


class FacilityEvacuationPlanner:
    """Plans prioritized, hazard-avoiding egress waves and vehicle escape trajectories."""

    def __init__(
        self,
        containment_engine: HazardContainmentEngine | None = None,
        config: EmergencyConfig | None = None,
    ) -> None:
        self.containment = containment_engine or HazardContainmentEngine(config)
        self.config = config or self.containment.config

    def plan_evacuation_route(
        self,
        vehicle_id: str,
        origin_bay_id: str,
        start_pose: tuple[float, float],
        exit_portal_id: str,
        exit_pose: tuple[float, float],
        floor_id: str = "L1",
    ) -> EvacuationRoute:
        """Synthesize an emergency egress route from current bay to designated exit portal."""
        sx, sy = start_pose
        ex, ey = exit_pose
        speed = self.config.evacuation_speed_limit_mps

        # Sample intermediate path points to evaluate hazard clearance
        dist_direct = math.hypot(ex - sx, ey - sy)
        steps = max(2, int(dist_direct / 2.0))
        waypoints: list[EvacuationWaypoint] = []
        min_hazard_dist = float("inf")
        is_cleared = True

        for i in range(steps + 1):
            alpha = float(i) / float(steps)
            wx = sx + alpha * (ex - sx)
            wy = sy + alpha * (ey - sy)
            waypoints.append(
                EvacuationWaypoint(
                    x_m=wx,
                    y_m=wy,
                    z_m=0.0,
                    speed_limit_mps=speed,
                )
            )

            # Check clearance against hazards
            for hz in self.containment.hazards.values():
                if not hz.active or hz.floor_id != floor_id:
                    continue
                d = math.hypot(wx - hz.center_x, wy - hz.center_y)
                if d < min_hazard_dist:
                    min_hazard_dist = d
                if d <= hz.radius_m:
                    is_cleared = False

        transit_time = dist_direct / speed if speed > 0 else 0.0

        return EvacuationRoute(
            route_id=f"evac-{vehicle_id}-{exit_portal_id}",
            vehicle_id=vehicle_id,
            origin_bay_id=origin_bay_id,
            exit_portal_id=exit_portal_id,
            waypoints=waypoints,
            total_length_m=round(dist_direct, 2),
            estimated_transit_time_s=round(transit_time, 2),
            min_hazard_distance_m=round(min_hazard_dist, 2)
            if min_hazard_dist != float("inf")
            else 999.0,
            is_cleared=is_cleared,
        )

    def select_best_exit(
        self,
        start_pose: tuple[float, float],
        exit_portals: Mapping[str, tuple[float, float]],
        floor_id: str = "L1",
    ) -> tuple[str, tuple[float, float]]:
        """Select closest exit portal that does not route through active hazards."""
        sx, sy = start_pose
        best_portal = ""
        best_pose = (0.0, 0.0)
        best_cost = float("inf")

        for portal_id, portal_pose in exit_portals.items():
            px, py = portal_pose
            dist = math.hypot(px - sx, py - sy)

            # Heavily penalize exit portals that are itself inside a hazard
            penalty = 0.0
            if self.containment.is_point_inside_hazard(px, py, floor_id=floor_id):
                penalty = 1000.0

            total_cost = dist + penalty
            if total_cost < best_cost:
                best_cost = total_cost
                best_portal = portal_id
                best_pose = portal_pose

        return best_portal, best_pose

    def schedule_evacuation_waves(
        self,
        vehicles: Mapping[str, tuple[str, tuple[float, float]]],
        exit_portals: Mapping[str, tuple[float, float]],
        floor_id: str = "L1",
        batch_size: int = 2,
    ) -> list[EvacuationWave]:
        """Schedule vehicles into prioritized departing waves staggered by delay intervals."""
        if not vehicles or not exit_portals:
            return []

        # Calculate distance of each vehicle to nearest active hazard
        urgency_list: list[tuple[float, str]] = []
        for vid, (_, pose) in vehicles.items():
            _, _, dist_to_hazard = self.containment.classify_vehicle_threat(
                vid, pose, floor_id=floor_id
            )
            # Smaller distance to hazard = higher evacuation urgency
            urgency_list.append((dist_to_hazard, vid))

        urgency_list.sort(key=lambda item: item[0])

        waves: list[EvacuationWave] = []
        total_veh = len(urgency_list)
        stagger = self.config.stagger_interval_s

        portal_keys = list(exit_portals.keys())
        batches = [
            [item[1] for item in urgency_list[i : i + batch_size]]
            for i in range(0, total_veh, batch_size)
        ]

        for wave_idx, batch_vids in enumerate(batches):
            assigned_portal = portal_keys[wave_idx % len(portal_keys)]
            waves.append(
                EvacuationWave(
                    wave_index=wave_idx,
                    vehicle_ids=batch_vids,
                    dispatch_time_s=round(wave_idx * stagger, 2),
                    exit_portal_id=assigned_portal,
                )
            )

        return waves

    def plan_facility_evacuation(
        self,
        vehicles: Mapping[str, tuple[str, tuple[float, float]]],
        exit_portals: Mapping[str, tuple[float, float]],
        floor_id: str = "L1",
    ) -> tuple[list[EvacuationWave], dict[str, EvacuationRoute]]:
        """Synthesize end-to-end multi-wave egress schedule and individual route trajectories."""
        waves = self.schedule_evacuation_waves(vehicles, exit_portals, floor_id=floor_id)
        routes: dict[str, EvacuationRoute] = {}

        for vid, (bay_id, pose) in vehicles.items():
            portal_id, portal_pose = self.select_best_exit(pose, exit_portals, floor_id=floor_id)
            route = self.plan_evacuation_route(
                vid, bay_id, pose, portal_id, portal_pose, floor_id=floor_id
            )
            routes[vid] = route

        return waves, routes
