"""Dynamic parking trajectory replanning and multi-stage recovery maneuver generator."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nearfield360.config import MissionConfig, ParkingPlannerConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.mission.models import RecoveryManeuver
from nearfield360.planning.collision import (
    SweptFootprintEvaluator,
    point_to_polygon_distance,
)
from nearfield360.planning.kinematics import (
    AckermannVehicle,
    KinematicWaypoint,
    normalize_angle,
)
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
)
from nearfield360.planning.planner import (
    ParkingTrajectoryPlanner,
    profile_segment_speed,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotType,
)
from nearfield360.tracking.models import TrackedObstacle


class ParkingReplanner:
    """Computes online recovery trajectories when a parking maneuver is blocked or deviated."""

    def __init__(
        self,
        mission_config: MissionConfig | None = None,
        planner_config: ParkingPlannerConfig | None = None,
    ) -> None:
        self.mission_config = mission_config or MissionConfig()
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.vehicle = AckermannVehicle(self.planner_config)
        self.planner = ParkingTrajectoryPlanner(self.planner_config)
        self.evaluator = SweptFootprintEvaluator(self.vehicle, self.planner_config)

    def _check_injected_collision(
        self,
        waypoints: list[KinematicWaypoint],
        injected_obstacle: tuple[float, float, float] | None,
    ) -> tuple[bool, float]:
        """Check waypoints against injected obstacle (x, y, radius)."""
        if injected_obstacle is None:
            return (False, 5.0)

        ox, oy, o_radius = injected_obstacle
        min_clearance = float("inf")

        for wp in waypoints:
            footprint = self.vehicle.compute_footprint_polygon(wp.x, wp.y, wp.heading_rad)
            dist_to_center = point_to_polygon_distance(ox, oy, footprint)
            clearance = max(0.0, dist_to_center - o_radius)
            min_clearance = min(min_clearance, clearance)
            if clearance < self.planner_config.collision_margin:
                return (True, round(clearance, 3))

        return (False, round(min_clearance, 3))

    def _check_obstacles_clearance(
        self,
        waypoints: list[KinematicWaypoint],
        obstacles: Sequence[TrackedObstacle] | None,
    ) -> tuple[bool, float]:
        """Check distance from waypoints footprint to dynamic obstacles."""
        if not obstacles:
            return (False, 5.0)

        min_clearance = float("inf")
        for wp in waypoints:
            footprint = self.vehicle.compute_footprint_polygon(wp.x, wp.y, wp.heading_rad)
            for obs in obstacles:
                ox, oy = obs.position
                dist = point_to_polygon_distance(ox, oy, footprint)
                min_clearance = min(min_clearance, dist)
                if dist < self.planner_config.collision_margin:
                    return (True, round(dist, 3))

        return (False, round(min_clearance, 3))

    def replan_maneuver(
        self,
        current_pose: tuple[float, float, float],
        current_gear: ManeuverGear,
        target_slot: ParkingSlot,
        replan_count: int = 1,
        trigger_reason: str = "Obstacle blockage detected",
        occupancy: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        injected_obstacle: tuple[float, float, float] | None = None,
    ) -> RecoveryManeuver:
        """Generate a valid, collision-free recovery plan to complete the parking mission."""
        replan_id = f"replan_{target_slot.slot_id}_{replan_count}"

        if replan_count > self.mission_config.max_replans:
            return RecoveryManeuver(
                replan_id=replan_id,
                trigger_reason=f"Max replans ({self.mission_config.max_replans}) exceeded",
                start_pose=current_pose,
                target_slot_id=target_slot.slot_id,
                is_successful=False,
                plan=None,
                clearance_m=0.0,
            )

        target_pose = self.planner._rear_axle_target_pose(target_slot)
        candidates: list[list[tuple[ManeuverGear, ManeuverPhase, list[KinematicWaypoint]]]] = []

        # Candidate 1: Pull-out adjustment maneuver (Forward pull-out then Reverse dock)
        pull_dist = self.mission_config.replan_pull_out_dist_m
        heading_err = normalize_angle(target_pose[2] - current_pose[2])
        curv = math.copysign(
            min(abs(heading_err) / max(0.5, pull_dist), self.vehicle.max_curvature),
            heading_err,
        )
        wps_pull = self.vehicle.generate_arc(
            current_pose,
            curvature=curv,
            arc_length_m=pull_dist,
            gear=ManeuverGear.FORWARD,
            step_size=self.planner_config.step_size,
        )
        if wps_pull:
            fwd_pose = (wps_pull[-1].x, wps_pull[-1].y, wps_pull[-1].heading_rad)
            dock_dist = math.hypot(target_pose[0] - fwd_pose[0], target_pose[1] - fwd_pose[1])
            dock_len = max(0.5, min(4.5, dock_dist))
            wps_dock = self.vehicle.generate_straight(
                fwd_pose,
                length_m=dock_len,
                gear=ManeuverGear.REVERSE,
                step_size=self.planner_config.step_size,
            )
            candidates.append(
                [
                    (ManeuverGear.FORWARD, ManeuverPhase.ALIGN, wps_pull),
                    (ManeuverGear.REVERSE, ManeuverPhase.DOCK, wps_dock),
                ]
            )

        # Candidate 2: Direct replan from current pose using slot geometry planner
        if target_slot.slot_type == ParkingSlotType.PARALLEL:
            candidates.append(self.planner._plan_parallel(current_pose, target_slot))
        elif target_slot.slot_type == ParkingSlotType.PERPENDICULAR:
            candidates.append(self.planner._plan_perpendicular(current_pose, target_slot))
        else:
            candidates.append(self.planner._plan_slanted(current_pose, target_slot))

        # Evaluate candidate plans
        for candidate_raw in candidates:
            all_raw_wps: list[KinematicWaypoint] = []
            for _, _, wps in candidate_raw:
                all_raw_wps.extend(wps)

            if not all_raw_wps:
                continue

            # Check injected obstacle collision
            is_inj_col, inj_clearance = self._check_injected_collision(
                all_raw_wps, injected_obstacle
            )
            if is_inj_col:
                continue

            # Check dynamic obstacle tracks
            is_obs_col, obs_clearance = self._check_obstacles_clearance(all_raw_wps, obstacles)
            if is_obs_col:
                continue

            # Check occupancy grid if available
            eval_clearance = min(inj_clearance, obs_clearance)
            if occupancy is not None and grid is not None:
                eval_res = self.evaluator.evaluate_trajectory(
                    all_raw_wps,
                    occupancy=occupancy,
                    grid=grid,
                    obstacles=obstacles,
                    danger_threshold=self.planner_config.danger_threshold,
                    uncertainty_threshold=self.planner_config.uncertainty_threshold,
                )
                if not eval_res.is_collision_free:
                    continue
                eval_clearance = min(eval_clearance, eval_res.min_clearance_m)

            # Build timed segments with trapezoidal speed profiler
            timed_segments: list[ManeuverSegment] = []
            curr_time = 0.0
            gear_switches = 0
            prev_gear: ManeuverGear | None = None
            max_c = 0.0

            for s_idx, (gear, phase, wps) in enumerate(candidate_raw):
                if prev_gear is not None and gear != prev_gear:
                    gear_switches += 1
                prev_gear = gear

                for wp in wps:
                    max_c = max(max_c, abs(wp.curvature))

                seg, curr_time = profile_segment_speed(
                    wps,
                    gear=gear,
                    phase=phase,
                    segment_index=s_idx,
                    t_start=curr_time,
                    max_speed=self.planner_config.max_speed,
                    max_accel=self.planner_config.max_acceleration,
                )
                timed_segments.append(seg)

            total_len = sum(seg.length_m for seg in timed_segments)

            recovery_plan = ParkingTrajectoryPlan(
                plan_id=replan_id,
                slot_id=target_slot.slot_id,
                slot_type=target_slot.slot_type,
                status=PlanStatus.SUCCESS,
                start_pose=current_pose,
                target_pose=target_pose,
                total_length_m=round(total_len, 3),
                total_duration_s=round(curr_time, 3),
                gear_switches=gear_switches,
                max_curvature=round(max_c, 4),
                min_clearance_m=round(eval_clearance, 3),
                is_executable=True,
                segments=timed_segments,
            )

            return RecoveryManeuver(
                replan_id=replan_id,
                trigger_reason=trigger_reason,
                start_pose=current_pose,
                target_slot_id=target_slot.slot_id,
                is_successful=True,
                plan=recovery_plan,
                clearance_m=round(eval_clearance, 3),
            )

        # If all candidates collide or are unfeasible
        return RecoveryManeuver(
            replan_id=replan_id,
            trigger_reason="No collision-free recovery trajectory feasible around obstacles",
            start_pose=current_pose,
            target_slot_id=target_slot.slot_id,
            is_successful=False,
            plan=None,
            clearance_m=0.0,
        )
