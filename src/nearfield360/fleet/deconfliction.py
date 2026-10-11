"""Multi-agent spatiotemporal trajectory deconfliction and priority token scheduling."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.config import FleetConfig
from nearfield360.fleet.models import (
    ConflictPoint,
    ConflictResolutionAction,
    SpatiotemporalWaypoint,
    TrajectoryProfile,
)
from nearfield360.mapping.models import GlobalRoute


def build_trajectory_profile(
    vehicle_id: str,
    route: GlobalRoute,
    start_time_s: float,
    temporal_headway_s: float = 2.0,
    safety_margin_m: float = 1.5,
) -> TrajectoryProfile:
    """Discretize a GlobalRoute into 4D spatiotemporal waypoints with estimated arrival times."""
    sp_waypoints: list[SpatiotemporalWaypoint] = []
    current_time = start_time_s

    for i, wp in enumerate(route.waypoints):
        if i > 0:
            prev = route.waypoints[i - 1]
            dist = math.hypot(wp.x - prev.x, wp.y - prev.y)
            speed = max(0.5, wp.target_speed_mps)
            dt = dist / speed
            current_time += dt

        sp_waypoints.append(
            SpatiotemporalWaypoint(
                x=wp.x,
                y=wp.y,
                heading_rad=wp.heading_rad,
                t_expected_s=current_time,
                t_min_s=max(0.0, current_time - (temporal_headway_s / 2.0)),
                t_max_s=current_time + (temporal_headway_s / 2.0),
                speed_target_mps=wp.target_speed_mps,
            )
        )

    duration = current_time - start_time_s
    return TrajectoryProfile(
        vehicle_id=vehicle_id,
        waypoints=sp_waypoints,
        total_duration_s=duration,
        safety_margin_m=safety_margin_m,
    )


class DeconflictionEngine:
    """Detects multi-vehicle 4D space-time trajectory conflicts and computes yield actions."""

    def __init__(self, config: FleetConfig | None = None) -> None:
        self.config = config or FleetConfig()
        self._conflict_seq: int = 1

    def detect_conflicts(
        self,
        profiles: Sequence[TrajectoryProfile],
    ) -> list[ConflictPoint]:
        """Examine pairwise vehicle trajectory profiles for space-time overlap."""
        conflicts: list[ConflictPoint] = []
        n = len(profiles)

        for i in range(n):
            for j in range(i + 1, n):
                p1 = profiles[i]
                p2 = profiles[j]
                pair_conflicts = self._check_trajectory_pair(p1, p2)
                conflicts.extend(pair_conflicts)

        return conflicts

    def _check_trajectory_pair(
        self,
        p1: TrajectoryProfile,
        p2: TrajectoryProfile,
    ) -> list[ConflictPoint]:
        """Check for intersections between two vehicle trajectories."""
        conflicts: list[ConflictPoint] = []
        conflict_thresh_m = self.config.conflict_radius_m
        temporal_headway = self.config.temporal_headway_s

        for w1 in p1.waypoints:
            for w2 in p2.waypoints:
                # Spatial distance
                dist = math.hypot(w1.x - w2.x, w1.y - w2.y)
                if dist < conflict_thresh_m:
                    # Check temporal overlap
                    # Two vehicles conflict if arrival delta is less than headway
                    dt = abs(w1.t_expected_s - w2.t_expected_s)
                    if dt < temporal_headway:
                        # Conflict identified
                        conflict_loc = ((w1.x + w2.x) / 2.0, (w1.y + w2.y) / 2.0)
                        earliest = min(w1.t_expected_s, w2.t_expected_s)
                        latest = max(w1.t_expected_s, w2.t_expected_s)

                        # Determine right-of-way priority
                        action1, action2 = self._resolve_priority(p1, w1, p2, w2)

                        cid = f"conf_{self._conflict_seq:04d}"
                        self._conflict_seq += 1

                        conflict = ConflictPoint(
                            conflict_id=cid,
                            primary_vehicle_id=p1.vehicle_id,
                            secondary_vehicle_id=p2.vehicle_id,
                            conflict_location=conflict_loc,
                            earliest_arrival_s=earliest,
                            latest_departure_s=latest + temporal_headway,
                            distance_between_trajectories_m=dist,
                            action_primary=action1,
                            action_secondary=action2,
                            resolved=False,
                        )
                        conflicts.append(conflict)
                        # We return only the earliest conflict point along this trajectory encounter
                        return conflicts

        return conflicts

    def _resolve_priority(
        self,
        p1: TrajectoryProfile,
        w1: SpatiotemporalWaypoint,
        p2: TrajectoryProfile,
        w2: SpatiotemporalWaypoint,
    ) -> tuple[ConflictResolutionAction, ConflictResolutionAction]:
        """Assign right-of-way and yielding resolution actions according to configured policy."""
        policy = self.config.priority_policy

        if policy == "first_come_first_served":
            # Vehicle that arrives earlier gets priority
            if w1.t_expected_s <= w2.t_expected_s:
                return ConflictResolutionAction.PROCEED, ConflictResolutionAction.YIELD_HOLD
            return ConflictResolutionAction.YIELD_HOLD, ConflictResolutionAction.PROCEED

        if policy == "distance_to_conflict":
            # Vehicle closer to the conflict point proceeds
            # Approximate by index along waypoints
            idx1 = p1.waypoints.index(w1) if w1 in p1.waypoints else 0
            idx2 = p2.waypoints.index(w2) if w2 in p2.waypoints else 0
            if idx1 <= idx2:
                return ConflictResolutionAction.PROCEED, ConflictResolutionAction.YIELD_HOLD
            return ConflictResolutionAction.YIELD_HOLD, ConflictResolutionAction.PROCEED

        # Default mission priority: vehicle 1 proceeds
        return ConflictResolutionAction.PROCEED, ConflictResolutionAction.SLOW_DOWN

    def adjust_trajectory_for_yield(
        self,
        profile: TrajectoryProfile,
        conflict: ConflictPoint,
        hold_duration_s: float | None = None,
    ) -> TrajectoryProfile:
        """Delay or slow down a yielding vehicle's trajectory after a conflict point."""
        hold_time = hold_duration_s or self.config.temporal_headway_s
        new_wps: list[SpatiotemporalWaypoint] = []
        delayed = False

        for wp in profile.waypoints:
            dist_to_conflict = math.hypot(
                wp.x - conflict.conflict_location[0],
                wp.y - conflict.conflict_location[1],
            )
            if dist_to_conflict < self.config.conflict_radius_m:
                delayed = True

            t_exp = wp.t_expected_s + (hold_time if delayed else 0.0)
            t_min = wp.t_min_s + (hold_time if delayed else 0.0)
            t_max = wp.t_max_s + (hold_time if delayed else 0.0)
            speed = (
                min(wp.speed_target_mps, self.config.max_conflict_speed_mps)
                if delayed
                else wp.speed_target_mps
            )

            new_wps.append(
                SpatiotemporalWaypoint(
                    x=wp.x,
                    y=wp.y,
                    heading_rad=wp.heading_rad,
                    t_expected_s=t_exp,
                    t_min_s=t_min,
                    t_max_s=t_max,
                    speed_target_mps=speed,
                )
            )

        return TrajectoryProfile(
            vehicle_id=profile.vehicle_id,
            waypoints=new_wps,
            total_duration_s=profile.total_duration_s + hold_time,
            safety_margin_m=profile.safety_margin_m,
        )
