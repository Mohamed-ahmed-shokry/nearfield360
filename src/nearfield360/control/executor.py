"""Closed-loop maneuver execution engine and dynamic safety monitoring."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.controller import PathTrackingController
from nearfield360.control.models import (
    ControlCommand,
    ControlPerformanceKPIs,
    ExecutionStatus,
    ManeuverExecutionReport,
    ManeuverExecutionStep,
    TrackingErrorState,
)
from nearfield360.control.simulator import SimulatorNoiseConfig, VehicleSimulator
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.collision import point_to_polygon_distance
from nearfield360.planning.kinematics import normalize_angle
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.tracking.models import TrackedObstacle


class DynamicSafetyMonitor:
    """Evaluates real-time obstacle proximity and emergency braking triggers for the ego vehicle."""

    def __init__(
        self,
        planner_config: ParkingPlannerConfig | None = None,
        control_config: ParkingControlConfig | None = None,
    ) -> None:
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.control_config = control_config or ParkingControlConfig()
        self.collision_margin = self.planner_config.collision_margin

    def evaluate_safety(
        self,
        footprint_corners: list[tuple[float, float]],
        occupancy: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        injected_obstacle: tuple[float, float, float] | None = None,
    ) -> tuple[bool, float, str | None]:
        """Check for collision hazard against occupancy grid, dynamic obstacles, and threats.

        Returns (is_hazardous, min_clearance_m, reason_str).
        """
        min_clearance = float("inf")

        # 1. Check injected obstacle (x, y, radius)
        if injected_obstacle is not None:
            ox, oy, o_radius = injected_obstacle
            dist_to_center = point_to_polygon_distance(ox, oy, footprint_corners)
            clearance = max(0.0, dist_to_center - o_radius)
            min_clearance = min(min_clearance, clearance)
            if clearance < self.collision_margin:
                msg = (
                    f"Injected obstacle collision at ({ox:.2f}, {oy:.2f}) "
                    f"with clearance {clearance:.3f}m"
                )
                return (True, round(clearance, 3), msg)

        # 2. Check tracked dynamic obstacles
        if obstacles:
            for obs in obstacles:
                ox, oy = obs.position
                dist = point_to_polygon_distance(ox, oy, footprint_corners)
                min_clearance = min(min_clearance, dist)
                if dist < self.collision_margin:
                    msg = (
                        f"Dynamic obstacle intrusion at ({ox:.2f}, {oy:.2f}) "
                        f"with clearance {dist:.3f}m"
                    )
                    return (True, round(dist, 3), msg)

        # 3. Check occupancy grid
        if occupancy is not None and grid is not None:
            occupied_mask = (occupancy >= self.planner_config.danger_threshold).astype(np.uint8)
            if np.any(occupied_mask):
                mask = np.zeros(grid.shape, dtype=np.uint8)
                pts_px = np.array(
                    [
                        [
                            round((pt[0] - grid.x_min) / grid.resolution),
                            round((pt[1] - grid.y_min) / grid.resolution),
                        ]
                        for pt in footprint_corners
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_px], 1)
                footprint_mask = mask > 0
                footprint_cells = int(np.count_nonzero(footprint_mask))
                if footprint_cells > 0:
                    intrusion_count = int(np.count_nonzero(occupied_mask[footprint_mask]))
                    if intrusion_count >= 2 or (intrusion_count / footprint_cells) > 0.01:
                        return (
                            True,
                            0.0,
                            f"Occupancy grid intrusion: {intrusion_count} cells occupied",
                        )

        final_clearance = min_clearance if math.isfinite(min_clearance) else 5.0
        return (False, round(final_clearance, 3), None)


class ManeuverExecutor:
    """Orchestrates closed-loop tracking, gear shifts, safety audits, and KPI computation."""

    def __init__(
        self,
        control_config: ParkingControlConfig | None = None,
        planner_config: ParkingPlannerConfig | None = None,
    ) -> None:
        self.control_config = control_config or ParkingControlConfig()
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.controller = PathTrackingController(self.control_config, self.planner_config)
        self.safety_monitor = DynamicSafetyMonitor(self.planner_config, self.control_config)

    def execute_plan(
        self,
        plan: ParkingTrajectoryPlan,
        occupancy: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        noise_config: SimulatorNoiseConfig | None = None,
        injected_obstacle: tuple[float, float, float] | None = None,
        inject_obstacle_at_step: int | None = None,
    ) -> ManeuverExecutionReport:
        """Simulate closed-loop execution of the parking trajectory plan."""
        self.controller.reset()

        if not plan.segments or not plan.is_executable:
            empty_kpis = ControlPerformanceKPIs(
                max_cross_track_error_m=0.0,
                mean_cross_track_error_m=0.0,
                rmse_cross_track_error_m=0.0,
                max_heading_error_rad=0.0,
                mean_heading_error_rad=0.0,
                max_lateral_accel_m_s2=0.0,
                max_jerk_m_s3=0.0,
                docking_error_x_m=0.0,
                docking_error_y_m=0.0,
                docking_error_heading_rad=0.0,
                docking_distance_m=0.0,
                is_docked_successfully=False,
            )
            return ManeuverExecutionReport(
                plan_id=plan.plan_id,
                slot_id=plan.slot_id,
                status=ExecutionStatus.ABORTED_DEVIATION,
                total_steps=0,
                duration_s=0.0,
                kpis=empty_kpis,
                steps=[],
                message="Plan is empty or marked not executable",
            )

        # Initialize vehicle simulator at plan start pose and first segment gear
        first_gear = plan.segments[0].gear
        simulator = VehicleSimulator(
            initial_pose=plan.start_pose,
            control_config=self.control_config,
            planner_config=self.planner_config,
            noise_config=noise_config,
        )
        simulator.set_gear(first_gear)

        dt = self.control_config.dt
        steps: list[ManeuverExecutionStep] = []
        overall_status = ExecutionStatus.ACTIVE
        abort_message = ""

        step_idx = 0
        current_time = 0.0

        max_allowed_steps = max(2000, int(plan.total_duration_s / dt * 4))

        for seg_idx, segment in enumerate(plan.segments):
            if overall_status not in (ExecutionStatus.ACTIVE, ExecutionStatus.SWITCHING_GEARS):
                break

            # Handle gear shift transition if segment gear differs from simulator gear
            if segment.gear != simulator.state.gear:
                # Dwell stationary for gear change (e.g. 4 steps = 0.2s)
                simulator.set_gear(segment.gear)
                self.controller.reset()
                for _ in range(4):
                    step_idx += 1
                    current_time += dt
                    dwell_step = ManeuverExecutionStep(
                        t=round(current_time, 3),
                        step_index=step_idx,
                        segment_index=seg_idx,
                        phase=segment.phase,
                        vehicle_state=simulator.state,
                        command=ControlCommand(
                            steering_angle_rad=simulator.state.steer_angle_rad,
                            gear=segment.gear,
                        ),
                        error=TrackingErrorState(
                            cross_track_error_m=0.0,
                            heading_error_rad=0.0,
                        ),
                        nearest_obstacle_distance_m=5.0,
                        status=ExecutionStatus.SWITCHING_GEARS,
                    )
                    steps.append(dwell_step)

            active_wp_idx = 0
            waypoints = segment.waypoints
            if not waypoints:
                continue

            while active_wp_idx < len(waypoints) and step_idx < max_allowed_steps:
                step_idx += 1
                current_time += dt

                veh_state = simulator.get_measured_state()

                # Find target waypoint index
                active_wp_idx = self.controller.find_target_waypoint_index(
                    veh_state.x,
                    veh_state.y,
                    waypoints,
                    start_idx=active_wp_idx,
                    search_window=20,
                )

                error_state, target_wp = self.controller.compute_tracking_error(
                    veh_state.x,
                    veh_state.y,
                    veh_state.heading_rad,
                    veh_state.velocity,
                    waypoints,
                    active_idx=active_wp_idx,
                    gear=segment.gear,
                )

                # Check watchdog abort limits
                max_cte = self.control_config.max_cross_track_error_m
                max_he = self.control_config.max_heading_error_rad
                if (
                    abs(error_state.cross_track_error_m) > max_cte
                    or abs(error_state.heading_error_rad) > max_he
                ):
                    overall_status = ExecutionStatus.ABORTED_DEVIATION
                    abort_message = (
                        f"Tracking error exceeded tolerance at step {step_idx}: "
                        f"cross-track {error_state.cross_track_error_m:.2f}m "
                        f"(max {self.control_config.max_cross_track_error_m:.2f}m), "
                        f"heading {error_state.heading_error_rad:.2f}rad "
                        f"(max {self.control_config.max_heading_error_rad:.2f}rad)"
                    )
                    break

                # Check obstacle safety clearance
                footprint = simulator.get_footprint_polygon()
                active_injected = (
                    injected_obstacle
                    if inject_obstacle_at_step is None or step_idx >= inject_obstacle_at_step
                    else None
                )

                is_hazardous, min_clearance, reason = self.safety_monitor.evaluate_safety(
                    footprint,
                    occupancy=occupancy,
                    grid=grid,
                    obstacles=obstacles,
                    injected_obstacle=active_injected,
                )

                if is_hazardous:
                    # Trigger emergency brake stop
                    overall_status = ExecutionStatus.EMERGENCY_STOPPED
                    abort_message = reason or "Obstacle collision hazard detected"
                    stop_cmd = self.controller.compute_control_command(
                        current_steer_rad=veh_state.steer_angle_rad,
                        current_velocity=veh_state.velocity,
                        error_state=error_state,
                        target_wp=target_wp,
                        gear=segment.gear,
                        emergency_brake=True,
                    )
                    simulator.step(stop_cmd)
                    step_record = ManeuverExecutionStep(
                        t=round(current_time, 3),
                        step_index=step_idx,
                        segment_index=seg_idx,
                        phase=segment.phase,
                        vehicle_state=simulator.state,
                        command=stop_cmd,
                        error=error_state,
                        nearest_obstacle_distance_m=min_clearance,
                        status=ExecutionStatus.EMERGENCY_STOPPED,
                    )
                    steps.append(step_record)
                    break

                # Normal control step
                cmd = self.controller.compute_control_command(
                    current_steer_rad=veh_state.steer_angle_rad,
                    current_velocity=veh_state.velocity,
                    error_state=error_state,
                    target_wp=target_wp,
                    gear=segment.gear,
                    emergency_brake=False,
                )

                simulator.step(cmd)

                step_record = ManeuverExecutionStep(
                    t=round(current_time, 3),
                    step_index=step_idx,
                    segment_index=seg_idx,
                    phase=segment.phase,
                    vehicle_state=simulator.state,
                    command=cmd,
                    error=error_state,
                    nearest_obstacle_distance_m=min_clearance,
                    status=ExecutionStatus.ACTIVE,
                )
                steps.append(step_record)

                # Segment completion check: near final waypoint of segment
                dist_to_end = math.hypot(
                    veh_state.x - waypoints[-1].x,
                    veh_state.y - waypoints[-1].y,
                )
                if (active_wp_idx >= len(waypoints) - 2 and dist_to_end < 0.15) or (
                    active_wp_idx >= len(waypoints) - 1 and dist_to_end < 0.25
                ):
                    break

        if overall_status == ExecutionStatus.ACTIVE:
            overall_status = ExecutionStatus.COMPLETED
            abort_message = "Maneuver trajectory tracking completed successfully"

        # Compute KPIs
        final_state = simulator.state
        target_pose = plan.target_pose
        docking_dx = final_state.x - target_pose[0]
        docking_dy = final_state.y - target_pose[1]
        docking_dtheta = normalize_angle(final_state.heading_rad - target_pose[2])
        docking_dist = math.hypot(docking_dx, docking_dy)

        is_docked = (
            overall_status == ExecutionStatus.COMPLETED
            and docking_dist <= self.control_config.terminal_dock_tol_xy
            and abs(docking_dtheta) <= self.control_config.terminal_dock_tol_heading
        )

        active_steps = [s for s in steps if s.status == ExecutionStatus.ACTIVE]
        if active_steps:
            cte_list = [abs(s.error.cross_track_error_m) for s in active_steps]
            head_err_list = [abs(s.error.heading_error_rad) for s in active_steps]
            max_cte = max(cte_list)
            mean_cte = sum(cte_list) / len(cte_list)
            rmse_cte = math.sqrt(sum(e**2 for e in cte_list) / len(cte_list))
            max_he = max(head_err_list)
            mean_he = sum(head_err_list) / len(head_err_list)

            # Lat accel: v^2 * tan(delta) / L
            lat_accels = [
                (abs(s.vehicle_state.velocity) ** 2)
                * abs(math.tan(s.vehicle_state.steer_angle_rad))
                / self.planner_config.wheelbase
                for s in active_steps
            ]
            max_lat_accel = max(lat_accels) if lat_accels else 0.0

            # Max jerk: da / dt
            jerks = [
                abs(
                    active_steps[i].vehicle_state.acceleration
                    - active_steps[i - 1].vehicle_state.acceleration
                )
                / dt
                for i in range(1, len(active_steps))
            ]
            max_jerk = max(jerks) if jerks else 0.0
        else:
            max_cte = mean_cte = rmse_cte = max_he = mean_he = max_lat_accel = max_jerk = 0.0

        kpis = ControlPerformanceKPIs(
            max_cross_track_error_m=round(max_cte, 4),
            mean_cross_track_error_m=round(mean_cte, 4),
            rmse_cross_track_error_m=round(rmse_cte, 4),
            max_heading_error_rad=round(max_he, 4),
            mean_heading_error_rad=round(mean_he, 4),
            max_lateral_accel_m_s2=round(max_lat_accel, 4),
            max_jerk_m_s3=round(max_jerk, 4),
            docking_error_x_m=round(docking_dx, 4),
            docking_error_y_m=round(docking_dy, 4),
            docking_error_heading_rad=round(docking_dtheta, 4),
            docking_distance_m=round(docking_dist, 4),
            is_docked_successfully=is_docked,
        )

        return ManeuverExecutionReport(
            plan_id=plan.plan_id,
            slot_id=plan.slot_id,
            status=overall_status,
            total_steps=len(steps),
            duration_s=round(current_time, 3),
            kpis=kpis,
            steps=steps,
            message=abort_message,
        )
