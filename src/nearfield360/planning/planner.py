"""Multi-stage autonomous parking maneuver planner and speed profiler."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nearfield360.config import ParkingPlannerConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.collision import SweptFootprintEvaluator
from nearfield360.planning.kinematics import (
    AckermannVehicle,
    KinematicWaypoint,
    normalize_angle,
)
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingPlanReport,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotType,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle


def profile_segment_speed(
    raw_waypoints: list[KinematicWaypoint],
    gear: ManeuverGear,
    phase: ManeuverPhase,
    segment_index: int,
    t_start: float,
    max_speed: float,
    max_accel: float,
) -> tuple[ManeuverSegment, float]:
    """Generate time-parameterized waypoints with a smooth trapezoidal speed profile."""
    if not raw_waypoints:
        return (
            ManeuverSegment(
                segment_index=segment_index,
                phase=phase,
                gear=gear,
                length_m=0.0,
                duration_s=0.0,
                waypoints=[],
            ),
            t_start,
        )

    total_len = raw_waypoints[-1].distance_m
    if total_len <= 1e-4:
        wp = TrajectoryWaypoint(
            x=raw_waypoints[0].x,
            y=raw_waypoints[0].y,
            heading_rad=raw_waypoints[0].heading_rad,
            curvature=raw_waypoints[0].curvature,
            velocity=0.0,
            acceleration=0.0,
            gear=gear,
            t=round(t_start, 3),
            distance_m=0.0,
        )
        return (
            ManeuverSegment(
                segment_index=segment_index,
                phase=phase,
                gear=gear,
                length_m=0.0,
                duration_s=0.0,
                waypoints=[wp],
            ),
            t_start,
        )

    # Accelerate up to max_speed, then decelerate to 0 at the end
    # d_accel = v^2 / (2 * a)
    d_accel = (max_speed * max_speed) / (2.0 * max_accel)
    actual_top_speed = max_speed if total_len >= 2.0 * d_accel else math.sqrt(max_accel * total_len)

    waypoints: list[TrajectoryWaypoint] = []
    curr_t = t_start
    prev_s = 0.0

    for kwp in raw_waypoints:
        s = kwp.distance_m
        ds = s - prev_s

        # Speed profile: v(s)
        if s <= 0.5 * total_len:
            v = math.sqrt(max(0.01, 2.0 * max_accel * s))
            accel = max_accel if v < actual_top_speed else 0.0
        else:
            dist_to_end = max(0.0, total_len - s)
            v = math.sqrt(max(0.01, 2.0 * max_accel * dist_to_end))
            accel = -max_accel

        v = min(v, actual_top_speed)
        v = max(0.1, v)  # Minimum creep speed for finite dt calculation

        if ds > 1e-6:
            dt = ds / v
            curr_t += dt

        waypoints.append(
            TrajectoryWaypoint(
                x=kwp.x,
                y=kwp.y,
                heading_rad=kwp.heading_rad,
                curvature=kwp.curvature,
                velocity=round(v if (0 < s < total_len) else 0.0, 3),
                acceleration=round(accel if (0 < s < total_len) else 0.0, 3),
                gear=gear,
                t=round(curr_t, 3),
                distance_m=round(s, 3),
            )
        )
        prev_s = s

    duration = curr_t - t_start
    seg = ManeuverSegment(
        segment_index=segment_index,
        phase=phase,
        gear=gear,
        length_m=round(total_len, 3),
        duration_s=round(duration, 3),
        waypoints=waypoints,
    )
    return seg, round(curr_t, 3)


class ParkingTrajectoryPlanner:
    """Synthesizes kinematically feasible, collision-free parking trajectories."""

    def __init__(self, config: ParkingPlannerConfig | None = None) -> None:
        self.config = config or ParkingPlannerConfig()
        self.vehicle = AckermannVehicle(self.config)
        self.evaluator = SweptFootprintEvaluator(self.vehicle, self.config)

    def _rear_axle_target_pose(self, slot: ParkingSlot) -> tuple[float, float, float]:
        """Compute the target rear-axle center pose when parked inside the slot."""
        c0, c1, c2, c3 = slot.corners
        # Entrance midpoint and back midpoint
        m_entry_x = 0.5 * (c0.x + c1.x)
        m_entry_y = 0.5 * (c0.y + c1.y)
        m_back_x = 0.5 * (c2.x + c3.x)
        m_back_y = 0.5 * (c2.y + c3.y)

        # Inward heading vector
        inward_x = m_back_x - m_entry_x
        inward_y = m_back_y - m_entry_y
        norm = math.hypot(inward_x, inward_y)
        heading = math.atan2(inward_y, inward_x) if norm > 1e-6 else slot.heading_rad

        # Geometric center of the slot
        center_x = 0.5 * (m_entry_x + m_back_x)
        center_y = 0.5 * (m_entry_y + m_back_y)

        # Offset from vehicle body center to rear axle:
        # Body length = wheelbase + front_overhang + rear_overhang
        # Distance from rear axle to body center = (wheelbase + front_overhang - rear_overhang) / 2
        body_center_offset = 0.5 * (
            self.vehicle.wheelbase + self.vehicle.front_overhang - self.vehicle.rear_overhang
        )

        axle_x = center_x - body_center_offset * math.cos(heading)
        axle_y = center_y - body_center_offset * math.sin(heading)

        return (round(axle_x, 3), round(axle_y, 3), round(normalize_angle(heading), 4))

    def _plan_parallel(
        self,
        start_pose: tuple[float, float, float],
        slot: ParkingSlot,
    ) -> list[tuple[ManeuverGear, ManeuverPhase, list[KinematicWaypoint]]]:
        """Generate 2-arc reverse S-turn parallel parking maneuver."""
        target_pose = self._rear_axle_target_pose(slot)
        r_min = self.vehicle.min_turn_radius * 1.05  # Slight margin above kinematic minimum

        # Target relative to start orientation
        x0, y0, t0 = start_pose
        xf, yf, _tf = target_pose
        dx = xf - x0
        dy = yf - y0
        sin0 = math.sin(t0)
        cos0 = math.cos(t0)
        ly = -dx * sin0 + dy * cos0

        turn_dir = -1.0 if ly < 0 else 1.0
        abs_ly = abs(ly)

        # Clamping lateral displacement
        cos_alpha = max(-1.0, min(1.0, 1.0 - abs_ly / (2.0 * r_min)))
        alpha = math.acos(cos_alpha)
        if alpha < 0.05:
            alpha = 0.35  # Fallback minimum turning angle

        arc_len = r_min * alpha

        # Phase 1: Reverse turn towards slot
        curv1 = turn_dir / r_min
        wps1 = self.vehicle.generate_arc(
            start_pose,
            curv1,
            arc_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )
        mid_pose = (wps1[-1].x, wps1[-1].y, wps1[-1].heading_rad)

        # Phase 2: Reverse counter-turn to straighten parallel to slot
        curv2 = -turn_dir / r_min
        wps2 = self.vehicle.generate_arc(
            mid_pose,
            curv2,
            arc_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )
        end_pose = (wps2[-1].x, wps2[-1].y, wps2[-1].heading_rad)

        # Phase 3: Forward pull-in straight dock if needed to center
        dist_to_target = math.hypot(target_pose[0] - end_pose[0], target_pose[1] - end_pose[1])
        pull_len = min(1.5, max(0.2, dist_to_target))
        wps3 = self.vehicle.generate_straight(
            end_pose,
            pull_len,
            ManeuverGear.FORWARD,
            step_size=self.config.step_size,
        )

        return [
            (ManeuverGear.REVERSE, ManeuverPhase.STEER_IN, wps1),
            (ManeuverGear.REVERSE, ManeuverPhase.ALIGN, wps2),
            (ManeuverGear.FORWARD, ManeuverPhase.FINAL_ALIGN, wps3),
        ]

    def _plan_perpendicular(
        self,
        start_pose: tuple[float, float, float],
        slot: ParkingSlot,
    ) -> list[tuple[ManeuverGear, ManeuverPhase, list[KinematicWaypoint]]]:
        """Generate reverse 90-degree dock maneuver into perpendicular slot."""
        target_pose = self._rear_axle_target_pose(slot)
        r_min = self.vehicle.min_turn_radius * 1.05

        # Approach orientation vs target inward heading
        t0 = start_pose[2]
        tf = target_pose[2]
        delta_theta = normalize_angle(tf - t0)

        turn_dir = 1.0 if delta_theta > 0 else -1.0
        turn_angle = abs(delta_theta)
        if turn_angle < 0.1:
            turn_angle = math.pi * 0.5  # Standard 90 deg

        arc_len = r_min * turn_angle

        # Phase 1: Reverse sweep turn
        curv = turn_dir / r_min
        wps1 = self.vehicle.generate_arc(
            start_pose,
            curv,
            arc_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )
        mid_pose = (wps1[-1].x, wps1[-1].y, wps1[-1].heading_rad)

        # Phase 2: Reverse straight line dock into slot
        dock_dist = math.hypot(target_pose[0] - mid_pose[0], target_pose[1] - mid_pose[1])
        dock_len = max(0.5, min(3.5, dock_dist))
        wps2 = self.vehicle.generate_straight(
            mid_pose,
            dock_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )

        return [
            (ManeuverGear.REVERSE, ManeuverPhase.STEER_IN, wps1),
            (ManeuverGear.REVERSE, ManeuverPhase.DOCK, wps2),
        ]

    def _plan_slanted(
        self,
        start_pose: tuple[float, float, float],
        slot: ParkingSlot,
    ) -> list[tuple[ManeuverGear, ManeuverPhase, list[KinematicWaypoint]]]:
        """Generate angled reverse dock maneuver into slanted slot."""
        target_pose = self._rear_axle_target_pose(slot)
        r_min = self.vehicle.min_turn_radius * 1.05

        t0 = start_pose[2]
        tf = target_pose[2]
        delta_theta = normalize_angle(tf - t0)
        turn_dir = 1.0 if delta_theta > 0 else -1.0
        turn_angle = max(0.2, abs(delta_theta))

        arc_len = r_min * turn_angle
        curv = turn_dir / r_min

        wps1 = self.vehicle.generate_arc(
            start_pose,
            curv,
            arc_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )
        mid_pose = (wps1[-1].x, wps1[-1].y, wps1[-1].heading_rad)

        dock_dist = math.hypot(target_pose[0] - mid_pose[0], target_pose[1] - mid_pose[1])
        dock_len = max(0.5, min(3.0, dock_dist))
        wps2 = self.vehicle.generate_straight(
            mid_pose,
            dock_len,
            ManeuverGear.REVERSE,
            step_size=self.config.step_size,
        )

        return [
            (ManeuverGear.REVERSE, ManeuverPhase.STEER_IN, wps1),
            (ManeuverGear.REVERSE, ManeuverPhase.DOCK, wps2),
        ]

    def plan_parking(
        self,
        slots: Sequence[ParkingSlot],
        occupancy: NDArray[np.float64],
        grid: BevGrid,
        obstacles: Sequence[TrackedObstacle] | None = None,
        target_slot_id: str | None = None,
        start_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        uncertainty: NDArray[np.float64] | None = None,
    ) -> ParkingPlanReport:
        """Find the optimal vacant slot and plan a collision-free parking trajectory."""
        # 1. Filter candidate vacant slots
        vacant_slots = [s for s in slots if s.status == SlotOccupancyStatus.VACANT]
        if target_slot_id is not None:
            vacant_slots = [s for s in vacant_slots if s.slot_id == target_slot_id]

        if not vacant_slots:
            return ParkingPlanReport(
                selected_slot_id=target_slot_id,
                candidate_slots_evaluated=len(slots),
                status=PlanStatus.NO_VACANT_SLOT,
                total_length_m=0.0,
                total_duration_s=0.0,
                gear_switches=0,
                min_clearance_m=0.0,
                is_executable=False,
                plan=None,
            )

        # 2. Sort candidates: feasible approach corridors first, then clearance margin
        sorted_slots = sorted(
            vacant_slots,
            key=lambda s: (
                1 if (s.approach_path and s.approach_path.is_feasible) else 0,
                s.approach_path.clearance_margin_m if s.approach_path else 0.0,
            ),
            reverse=True,
        )

        # 3. Evaluate candidate plans in priority order
        for candidate in sorted_slots:
            target_pose = self._rear_axle_target_pose(candidate)

            # Generate raw kinematic maneuver segments
            if candidate.slot_type == ParkingSlotType.PARALLEL:
                raw_segments = self._plan_parallel(start_pose, candidate)
            elif candidate.slot_type == ParkingSlotType.PERPENDICULAR:
                raw_segments = self._plan_perpendicular(start_pose, candidate)
            else:
                raw_segments = self._plan_slanted(start_pose, candidate)

            # Apply speed profiler across segments
            timed_segments: list[ManeuverSegment] = []
            curr_time = 0.0
            all_raw_wps: list[KinematicWaypoint] = []
            gear_switches = 0
            prev_gear: ManeuverGear | None = None

            for s_idx, (gear, phase, wps) in enumerate(raw_segments):
                if prev_gear is not None and gear != prev_gear:
                    gear_switches += 1
                prev_gear = gear
                all_raw_wps.extend(wps)

                seg, curr_time = profile_segment_speed(
                    wps,
                    gear=gear,
                    phase=phase,
                    segment_index=s_idx,
                    t_start=curr_time,
                    max_speed=self.config.max_speed,
                    max_accel=self.config.max_acceleration,
                )
                timed_segments.append(seg)

            # 4. Check collision & swept-footprint clearance
            eval_res = self.evaluator.evaluate_trajectory(
                all_raw_wps,
                occupancy=occupancy,
                grid=grid,
                obstacles=obstacles,
                danger_threshold=self.config.danger_threshold,
                uncertainty=uncertainty,
                uncertainty_threshold=self.config.uncertainty_threshold,
            )

            total_len = sum(seg.length_m for seg in timed_segments)
            total_dur = curr_time
            max_curv = max((abs(wp.curvature) for wp in all_raw_wps), default=0.0)

            if eval_res.is_collision_free:
                plan = ParkingTrajectoryPlan(
                    plan_id=f"plan_{candidate.slot_id}",
                    slot_id=candidate.slot_id,
                    slot_type=candidate.slot_type,
                    status=PlanStatus.SUCCESS,
                    total_length_m=round(total_len, 3),
                    total_duration_s=round(total_dur, 3),
                    gear_switches=gear_switches,
                    max_curvature=round(max_curv, 4),
                    min_clearance_m=eval_res.min_clearance_m,
                    start_pose=start_pose,
                    target_pose=target_pose,
                    is_executable=True,
                    segments=timed_segments,
                )
                return ParkingPlanReport(
                    selected_slot_id=candidate.slot_id,
                    candidate_slots_evaluated=len(vacant_slots),
                    status=PlanStatus.SUCCESS,
                    total_length_m=round(total_len, 3),
                    total_duration_s=round(total_dur, 3),
                    gear_switches=gear_switches,
                    min_clearance_m=eval_res.min_clearance_m,
                    is_executable=True,
                    plan=plan,
                )

        # If all candidates collide or are infeasible
        return ParkingPlanReport(
            selected_slot_id=sorted_slots[0].slot_id if sorted_slots else None,
            candidate_slots_evaluated=len(vacant_slots),
            status=PlanStatus.COLLISION_DETECTED,
            total_length_m=0.0,
            total_duration_s=0.0,
            gear_switches=0,
            min_clearance_m=0.0,
            is_executable=False,
            plan=None,
        )
