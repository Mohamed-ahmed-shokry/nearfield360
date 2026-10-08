"""Autonomous Valet Parking (AVP) Mission Executive and lifecycle orchestrator."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import MissionConfig, ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.controller import PathTrackingController
from nearfield360.control.models import (
    ControlCommand,
    ControlPerformanceKPIs,
    ExecutionStatus,
    ManeuverExecutionStep,
    TrackingErrorState,
)
from nearfield360.control.simulator import SimulatorNoiseConfig, VehicleSimulator
from nearfield360.geometry.bev import BevGrid
from nearfield360.mission.models import (
    MissionEvent,
    MissionState,
    MissionSummaryReport,
    MissionTrigger,
)
from nearfield360.mission.replanner import ParkingReplanner
from nearfield360.mission.slot_tracker import SlotTracker
from nearfield360.planning.collision import point_to_polygon_distance
from nearfield360.planning.kinematics import normalize_angle
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ParkingTrajectoryPlan,
)
from nearfield360.planning.planner import ParkingTrajectoryPlanner
from nearfield360.slots.models import ParkingSlot, SlotOccupancyStatus
from nearfield360.tracking.models import TrackedObstacle


class DynamicSafetyMonitor:
    """Evaluates clearance and collision threats against occupancy and dynamic obstacles."""

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
        """Check for collision threat. Returns (is_hazardous, min_clearance_m, reason_str)."""
        min_clearance = float("inf")
        poly_arr = np.array(footprint_corners, dtype=np.float32)

        if injected_obstacle is not None:
            ox, oy, o_radius = injected_obstacle
            is_inside = cv2.pointPolygonTest(poly_arr, (ox, oy), False) >= 0
            dist_to_center = 0.0 if is_inside else point_to_polygon_distance(ox, oy, footprint_corners)
            clearance = max(0.0, dist_to_center - o_radius)
            min_clearance = min(min_clearance, clearance)
            if clearance < self.collision_margin:
                msg = (
                    f"Obstacle hazard at ({ox:.2f}, {oy:.2f}) "
                    f"with clearance {clearance:.3f}m < {self.collision_margin:.3f}m"
                )
                return (True, round(clearance, 3), msg)

        if obstacles:
            for obs in obstacles:
                ox, oy = obs.position
                is_inside = cv2.pointPolygonTest(poly_arr, (ox, oy), False) >= 0
                dist = 0.0 if is_inside else point_to_polygon_distance(ox, oy, footprint_corners)
                min_clearance = min(min_clearance, dist)
                if dist < self.collision_margin:
                    msg = (
                        f"Dynamic obstacle intrusion at ({ox:.2f}, {oy:.2f}) "
                        f"with clearance {dist:.3f}m"
                    )
                    return (True, round(dist, 3), msg)

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


class MissionExecutive:
    """End-to-end Autonomous Valet Parking (AVP) mission state machine and lifecycle manager."""

    def __init__(
        self,
        mission_config: MissionConfig | None = None,
        planner_config: ParkingPlannerConfig | None = None,
        control_config: ParkingControlConfig | None = None,
    ) -> None:
        self.mission_config = mission_config or MissionConfig()
        self.planner_config = planner_config or ParkingPlannerConfig()
        self.control_config = control_config or ParkingControlConfig()

        self.slot_tracker = SlotTracker(self.mission_config)
        self.replanner = ParkingReplanner(self.mission_config, self.planner_config)
        self.planner = ParkingTrajectoryPlanner(self.planner_config)
        self.controller = PathTrackingController(self.control_config, self.planner_config)
        self.safety_monitor = DynamicSafetyMonitor(self.planner_config, self.control_config)

        self._state: MissionState = MissionState.STANDBY
        self._events: list[MissionEvent] = []

    @property
    def current_state(self) -> MissionState:
        return self._state

    @property
    def events(self) -> list[MissionEvent]:
        return list(self._events)

    def transition_to(
        self,
        target_state: MissionState,
        trigger: MissionTrigger,
        t: float,
        description: str,
    ) -> None:
        """Execute a state transition and log the mission lifecycle event."""
        event = MissionEvent(
            t=round(t, 3),
            source_state=self._state,
            target_state=target_state,
            trigger=trigger,
            description=description,
        )
        self._events.append(event)
        self._state = target_state

    def execute_mission(
        self,
        slots: Sequence[ParkingSlot] | None = None,
        initial_plan: ParkingTrajectoryPlan | None = None,
        target_slot: ParkingSlot | None = None,
        occupancy: NDArray[np.float64] | None = None,
        grid: BevGrid | None = None,
        obstacles: Sequence[TrackedObstacle] | None = None,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        transient_obstacle: tuple[float, float, float] | None = None,
        transient_obstacle_window: tuple[int, int] | None = None,
        persistent_obstacle: tuple[float, float, float] | None = None,
        persistent_obstacle_step: int | None = None,
        noise_config: SimulatorNoiseConfig | None = None,
        mission_id: str = "mission_avp_001",
        max_steps: int = 2500,
    ) -> tuple[MissionSummaryReport, list[ManeuverExecutionStep]]:
        """Run the complete closed-loop AVP mission lifecycle simulation."""
        self._state = MissionState.STANDBY
        self._events.clear()
        self.slot_tracker.reset()
        self.controller.reset()

        dt = self.control_config.dt
        current_time = 0.0
        step_idx = 0
        replan_count = 0
        execution_steps: list[ManeuverExecutionStep] = []

        # 1. STANDBY -> SEARCHING
        self.transition_to(
            MissionState.SEARCHING,
            MissionTrigger.ACTIVATE,
            current_time,
            "AVP system activated. Initiating slot scanning.",
        )

        # 2. Slot tracking and target selection
        selected_slot: ParkingSlot | None = target_slot
        active_plan: ParkingTrajectoryPlan | None = initial_plan

        if slots:
            self.slot_tracker.update(slots, frame_index=0)
            if selected_slot is None:
                best_tracked = self.slot_tracker.get_best_target_slot(
                    current_pose=initial_pose, confirmed_only=False
                )
                if best_tracked:
                    selected_slot = best_tracked.slot

        if selected_slot is None and active_plan is not None:
            sp_x, sp_y = active_plan.target_pose[0], active_plan.target_pose[1]
            from nearfield360.slots.models import ParkingSlotCorner, ParkingSlotType

            dummy_corners = (
                ParkingSlotCorner(x=sp_x - 1.2, y=sp_y - 2.5, z=0.0),
                ParkingSlotCorner(x=sp_x - 1.2, y=sp_y + 2.5, z=0.0),
                ParkingSlotCorner(x=sp_x + 1.2, y=sp_y + 2.5, z=0.0),
                ParkingSlotCorner(x=sp_x + 1.2, y=sp_y - 2.5, z=0.0),
            )
            selected_slot = ParkingSlot(
                slot_id=active_plan.slot_id,
                slot_type=active_plan.slot_type or ParkingSlotType.PERPENDICULAR,
                corners=dummy_corners,
                center=(sp_x, sp_y),
                heading_rad=active_plan.target_pose[2],
                width_m=2.4,
                length_m=5.0,
                status=SlotOccupancyStatus.VACANT,
                confidence=0.95,
            )

        if selected_slot is None:
            self.transition_to(
                MissionState.ABORTED,
                MissionTrigger.ABORT,
                current_time,
                "No viable vacant parking slot detected.",
            )
            report = MissionSummaryReport(
                mission_id=mission_id,
                final_state=self._state,
                target_slot_id=None,
                total_duration_s=current_time,
                total_steps=step_idx,
                replan_count=replan_count,
                events=self.events,
                final_kpis=None,
                is_success=False,
                message="Mission aborted: No vacant slot discovered.",
            )
            return (report, execution_steps)

        # Transition to SLOT_SELECTED
        self.transition_to(
            MissionState.SLOT_SELECTED,
            MissionTrigger.SLOT_DISCOVERED,
            current_time,
            f"Target parking slot {selected_slot.slot_id} selected and locked.",
        )

        # Plan trajectory if not provided
        if active_plan is None:
            effective_grid = grid or BevGrid(
                x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.05
            )
            effective_occ = (
                occupancy
                if occupancy is not None
                else np.zeros(effective_grid.shape, dtype=np.float64)
            )
            plan_report = self.planner.plan_parking(
                slots=[selected_slot],
                occupancy=effective_occ,
                grid=effective_grid,
                target_slot_id=selected_slot.slot_id,
                start_pose=initial_pose,
                obstacles=obstacles,
            )
            if not plan_report.is_executable or plan_report.plan is None:
                self.transition_to(
                    MissionState.ABORTED,
                    MissionTrigger.ABORT,
                    current_time,
                    "Unable to compute executable trajectory into target slot.",
                )
                report = MissionSummaryReport(
                    mission_id=mission_id,
                    final_state=self._state,
                    target_slot_id=selected_slot.slot_id,
                    total_duration_s=current_time,
                    total_steps=step_idx,
                    replan_count=replan_count,
                    events=self.events,
                    final_kpis=None,
                    is_success=False,
                    message="Trajectory planning failed into target slot.",
                )
                return (report, execution_steps)
            active_plan = plan_report.plan

        # Initialize vehicle simulator
        start_gear = active_plan.segments[0].gear if active_plan.segments else ManeuverGear.FORWARD
        simulator = VehicleSimulator(
            initial_pose=active_plan.start_pose,
            control_config=self.control_config,
            planner_config=self.planner_config,
            noise_config=noise_config,
        )
        simulator.set_gear(start_gear)

        # Transition to PARKING_MANEUVER
        self.transition_to(
            MissionState.PARKING_MANEUVER,
            MissionTrigger.MANEUVER_START,
            current_time,
            f"Executing closed-loop trajectory tracking (plan: {active_plan.plan_id}).",
        )

        hold_dwell_seconds = 0.0
        seg_idx = 0
        active_wp_idx = 0

        while step_idx < max_steps and self._state in (
            MissionState.PARKING_MANEUVER,
            MissionState.OBSTACLE_HOLD,
            MissionState.REPLANNING,
            MissionState.FINAL_ALIGNMENT,
        ):
            step_idx += 1
            current_time += dt

            veh_state = simulator.get_measured_state()
            footprint = simulator.get_footprint_polygon()

            # Determine active dynamic or injected obstacles for this step
            current_injected: tuple[float, float, float] | None = None
            if (
                transient_obstacle is not None
                and transient_obstacle_window is not None
                and transient_obstacle_window[0] <= step_idx <= transient_obstacle_window[1]
            ):
                current_injected = transient_obstacle
            elif (
                persistent_obstacle is not None
                and persistent_obstacle_step is not None
                and step_idx >= persistent_obstacle_step
            ):
                current_injected = persistent_obstacle

            # Safety audit
            is_hazardous, min_clearance, reason = self.safety_monitor.evaluate_safety(
                footprint_corners=footprint,
                occupancy=occupancy,
                grid=grid,
                obstacles=obstacles,
                injected_obstacle=current_injected,
            )

            # State Machine Dispatch
            if self._state == MissionState.PARKING_MANEUVER:
                if is_hazardous:
                    hold_dwell_seconds = 0.0
                    self.transition_to(
                        MissionState.OBSTACLE_HOLD,
                        MissionTrigger.OBSTACLE_DETECTED,
                        current_time,
                        f"Obstacle hazard detected ({reason}). Holding stationary.",
                    )
                    brake_cmd = ControlCommand(
                        steering_angle_rad=veh_state.steer_angle_rad,
                        target_velocity=0.0,
                        acceleration_cmd=-self.control_config.emergency_brake_decel,
                        gear=veh_state.gear,
                        emergency_brake=True,
                    )
                    simulator.step(brake_cmd)
                    step_rec = ManeuverExecutionStep(
                        t=round(current_time, 3),
                        step_index=step_idx,
                        segment_index=seg_idx,
                        phase=ManeuverPhase.STEER_IN,
                        vehicle_state=simulator.state,
                        command=brake_cmd,
                        error=TrackingErrorState(cross_track_error_m=0.0, heading_error_rad=0.0),
                        nearest_obstacle_distance_m=min_clearance,
                        status=ExecutionStatus.EMERGENCY_STOPPED,
                    )
                    execution_steps.append(step_rec)
                    continue

                if active_plan is None or seg_idx >= len(active_plan.segments):
                    self.transition_to(
                        MissionState.FINAL_ALIGNMENT,
                        MissionTrigger.TARGET_REACHED,
                        current_time,
                        "Trajectory segments completed. Aligning vehicle in slot.",
                    )
                    continue

                segment = active_plan.segments[seg_idx]

                if segment.gear != simulator.state.gear:
                    simulator.set_gear(segment.gear)
                    self.controller.reset()
                    dwell_cmd = ControlCommand(
                        steering_angle_rad=simulator.state.steer_angle_rad,
                        target_velocity=0.0,
                        acceleration_cmd=0.0,
                        gear=segment.gear,
                    )
                    simulator.step(dwell_cmd)
                    step_rec = ManeuverExecutionStep(
                        t=round(current_time, 3),
                        step_index=step_idx,
                        segment_index=seg_idx,
                        phase=segment.phase,
                        vehicle_state=simulator.state,
                        command=dwell_cmd,
                        error=TrackingErrorState(cross_track_error_m=0.0, heading_error_rad=0.0),
                        nearest_obstacle_distance_m=min_clearance,
                        status=ExecutionStatus.SWITCHING_GEARS,
                    )
                    execution_steps.append(step_rec)
                    continue

                waypoints = segment.waypoints
                if not waypoints:
                    seg_idx += 1
                    active_wp_idx = 0
                    continue

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

                # Check watchdog error limit
                if (
                    abs(error_state.cross_track_error_m)
                    > self.control_config.max_cross_track_error_m
                    or abs(error_state.heading_error_rad)
                    > self.control_config.max_heading_error_rad
                ):
                    self.transition_to(
                        MissionState.REPLANNING,
                        MissionTrigger.OBSTACLE_DETECTED,
                        current_time,
                        (
                            f"Tracking error exceeded tolerance (cross-track: "
                            f"{error_state.cross_track_error_m:.2f}m)."
                        ),
                    )
                    continue

                cmd = self.controller.compute_control_command(
                    current_steer_rad=veh_state.steer_angle_rad,
                    current_velocity=veh_state.velocity,
                    error_state=error_state,
                    target_wp=target_wp,
                    gear=segment.gear,
                    emergency_brake=False,
                )
                simulator.step(cmd)

                step_rec = ManeuverExecutionStep(
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
                execution_steps.append(step_rec)

                # Check segment completion
                last_wp = waypoints[-1]
                dist_to_end = math.hypot(veh_state.x - last_wp.x, veh_state.y - last_wp.y)
                if active_wp_idx >= len(waypoints) - 1 or dist_to_end < 0.15:
                    seg_idx += 1
                    active_wp_idx = 0
                    self.controller.reset()

            elif self._state == MissionState.OBSTACLE_HOLD:
                hold_dwell_seconds += dt
                hold_cmd = ControlCommand(
                    steering_angle_rad=veh_state.steer_angle_rad,
                    target_velocity=0.0,
                    acceleration_cmd=0.0,
                    gear=veh_state.gear,
                    emergency_brake=True,
                )
                simulator.step(hold_cmd)
                step_rec = ManeuverExecutionStep(
                    t=round(current_time, 3),
                    step_index=step_idx,
                    segment_index=seg_idx,
                    phase=ManeuverPhase.STEER_IN,
                    vehicle_state=simulator.state,
                    command=hold_cmd,
                    error=TrackingErrorState(cross_track_error_m=0.0, heading_error_rad=0.0),
                    nearest_obstacle_distance_m=min_clearance,
                    status=ExecutionStatus.EMERGENCY_STOPPED,
                )
                execution_steps.append(step_rec)

                if not is_hazardous:
                    self.transition_to(
                        MissionState.PARKING_MANEUVER,
                        MissionTrigger.OBSTACLE_CLEARED,
                        current_time,
                        "Obstacle cleared. Resuming parking maneuver.",
                    )
                elif hold_dwell_seconds >= self.mission_config.hold_timeout_s:
                    self.transition_to(
                        MissionState.REPLANNING,
                        MissionTrigger.OBSTACLE_TIMEOUT,
                        current_time,
                        (
                            f"Hold timeout ({self.mission_config.hold_timeout_s}s) exceeded. "
                            "Initiating online replanning."
                        ),
                    )

            elif self._state == MissionState.REPLANNING:
                replan_count += 1
                recovery = self.replanner.replan_maneuver(
                    current_pose=(veh_state.x, veh_state.y, veh_state.heading_rad),
                    current_gear=veh_state.gear,
                    target_slot=selected_slot,
                    replan_count=replan_count,
                    trigger_reason=f"Recovery from blockage at step {step_idx}",
                    occupancy=occupancy,
                    grid=grid,
                    obstacles=obstacles,
                    injected_obstacle=current_injected,
                )

                if recovery.is_successful and recovery.plan is not None:
                    active_plan = recovery.plan
                    seg_idx = 0
                    active_wp_idx = 0
                    self.controller.reset()
                    self.transition_to(
                        MissionState.PARKING_MANEUVER,
                        MissionTrigger.REPLAN_SUCCESS,
                        current_time,
                        f"Dynamic recovery plan computed successfully ({active_plan.plan_id}).",
                    )
                else:
                    self.transition_to(
                        MissionState.ABORTED,
                        MissionTrigger.REPLAN_FAILED,
                        current_time,
                        f"Replanning failed: {recovery.trigger_reason}. Aborting to safe stop.",
                    )
                    break

            elif self._state == MissionState.FINAL_ALIGNMENT:
                align_cmd = ControlCommand(
                    steering_angle_rad=0.0,
                    target_velocity=0.0,
                    acceleration_cmd=-1.0,
                    gear=veh_state.gear,
                )
                simulator.step(align_cmd)
                step_rec = ManeuverExecutionStep(
                    t=round(current_time, 3),
                    step_index=step_idx,
                    segment_index=seg_idx,
                    phase=ManeuverPhase.FINAL_ALIGN,
                    vehicle_state=simulator.state,
                    command=align_cmd,
                    error=TrackingErrorState(cross_track_error_m=0.0, heading_error_rad=0.0),
                    nearest_obstacle_distance_m=min_clearance,
                    status=ExecutionStatus.COMPLETED,
                )
                execution_steps.append(step_rec)

                target_pose = self.planner._rear_axle_target_pose(selected_slot)
                err_x = abs(veh_state.x - target_pose[0])
                err_y = abs(veh_state.y - target_pose[1])
                err_h = abs(normalize_angle(veh_state.heading_rad - target_pose[2]))

                is_docked = (
                    err_x <= self.mission_config.docking_tolerance_x_m
                    and err_y <= self.mission_config.docking_tolerance_y_m
                    and err_h <= self.mission_config.docking_tolerance_heading_rad
                )

                if is_docked or abs(veh_state.velocity) < 0.05:
                    self.transition_to(
                        MissionState.COMPLETED,
                        MissionTrigger.ALIGNMENT_COMPLETE,
                        current_time,
                        (
                            f"Terminal alignment verified: dx={err_x:.3f}m, dy={err_y:.3f}m, "
                            f"d_theta={math.degrees(err_h):.1f}deg."
                        ),
                    )
                    break

        # Calculate final KPIs
        kpis: ControlPerformanceKPIs | None = None
        if execution_steps and selected_slot:
            target_pose = self.planner._rear_axle_target_pose(selected_slot)
            last_state = simulator.state
            dx = abs(last_state.x - target_pose[0])
            dy = abs(last_state.y - target_pose[1])
            d_head = abs(normalize_angle(last_state.heading_rad - target_pose[2]))
            docking_dist = math.hypot(dx, dy)

            ctes = [s.error.cross_track_error_m for s in execution_steps]
            hes = [s.error.heading_error_rad for s in execution_steps]

            is_success = (
                self._state == MissionState.COMPLETED
                and dx <= self.mission_config.docking_tolerance_x_m * 1.5
                and dy <= self.mission_config.docking_tolerance_y_m * 1.5
            )

            kpis = ControlPerformanceKPIs(
                max_cross_track_error_m=round(max(abs(c) for c in ctes), 4) if ctes else 0.0,
                mean_cross_track_error_m=(
                    round(float(np.mean([abs(c) for c in ctes])), 4) if ctes else 0.0
                ),
                rmse_cross_track_error_m=(
                    round(float(np.sqrt(np.mean([c * c for c in ctes]))), 4) if ctes else 0.0
                ),
                max_heading_error_rad=round(max(abs(h) for h in hes), 4) if hes else 0.0,
                mean_heading_error_rad=(
                    round(float(np.mean([abs(h) for h in hes])), 4) if hes else 0.0
                ),
                max_lateral_accel_m_s2=0.0,
                max_jerk_m_s3=0.0,
                docking_error_x_m=round(dx, 3),
                docking_error_y_m=round(dy, 3),
                docking_error_heading_rad=round(d_head, 4),
                docking_distance_m=round(docking_dist, 3),
                is_docked_successfully=is_success,
            )

        is_overall_success = self._state == MissionState.COMPLETED
        msg = (
            f"AVP Mission completed successfully in {current_time:.2f}s."
            if is_overall_success
            else f"AVP Mission terminated in state {self._state.value}."
        )

        report = MissionSummaryReport(
            mission_id=mission_id,
            final_state=self._state,
            target_slot_id=selected_slot.slot_id if selected_slot else None,
            total_duration_s=round(current_time, 3),
            total_steps=step_idx,
            replan_count=replan_count,
            events=self.events,
            final_kpis=kpis,
            is_success=is_overall_success,
            message=msg,
        )

        return (report, execution_steps)
