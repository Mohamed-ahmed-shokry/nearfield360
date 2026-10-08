"""Visual mission dashboard overlay and lifecycle timeline telemetry renderer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.control.models import ExecutionStatus, ManeuverExecutionStep
from nearfield360.geometry.bev import BevGrid
from nearfield360.mission.models import MissionEvent, MissionState, MissionSummaryReport
from nearfield360.planning.kinematics import AckermannVehicle
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.slots.models import ParkingSlot, SlotOccupancyStatus

# State color palette (BGR)
_STATE_COLORS: dict[MissionState, tuple[int, int, int]] = {
    MissionState.STANDBY: (160, 160, 160),
    MissionState.SEARCHING: (255, 200, 0),
    MissionState.SLOT_SELECTED: (255, 140, 0),
    MissionState.APPROACH: (200, 200, 0),
    MissionState.PARKING_MANEUVER: (50, 205, 50),
    MissionState.OBSTACLE_HOLD: (0, 165, 255),
    MissionState.REPLANNING: (255, 0, 255),
    MissionState.FINAL_ALIGNMENT: (180, 105, 255),
    MissionState.COMPLETED: (0, 255, 0),
    MissionState.ABORTED: (0, 0, 255),
}


def render_mission_dashboard_overlay(
    grid: BevGrid,
    report: MissionSummaryReport,
    steps: Sequence[ManeuverExecutionStep],
    *,
    slots: Sequence[ParkingSlot] | None = None,
    plan: ParkingTrajectoryPlan | None = None,
    base_canvas: NDArray[np.uint8] | None = None,
    vehicle: AckermannVehicle | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render comprehensive BEV mission execution dashboard overlay."""
    h, w = grid.shape
    canvas = (
        base_canvas.copy() if base_canvas is not None else np.full((h, w, 3), 32, dtype=np.uint8)
    )
    veh = vehicle or AckermannVehicle()

    # 1. Render parking slots
    if slots:
        slot_overlay = canvas.copy()
        for slot in slots:
            is_target = report.target_slot_id == slot.slot_id
            if is_target:
                scolor = (0, 255, 128)  # Bright emerald for target
            elif slot.status == SlotOccupancyStatus.VACANT:
                scolor = (0, 200, 0)
            else:
                scolor = (0, 0, 200)

            pts_px = [
                (
                    round((c.x - grid.x_min) / grid.resolution),
                    round((c.y - grid.y_min) / grid.resolution),
                )
                for c in slot.corners
            ]
            poly_arr = np.array(pts_px, dtype=np.int32)
            cv2.fillPoly(slot_overlay, [poly_arr], scolor)
            thickness = 3 if is_target else 1
            cv2.polylines(canvas, [poly_arr], isClosed=True, color=scolor, thickness=thickness)

            cx = round((slot.center[0] - grid.x_min) / grid.resolution)
            cy = round((slot.center[1] - grid.y_min) / grid.resolution)
            label = f"{slot.slot_id} [TARGET]" if is_target else slot.slot_id
            cv2.putText(
                canvas,
                label,
                (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.addWeighted(slot_overlay, 0.25, canvas, 0.75, 0, canvas)

    # 2. Render reference planned trajectory
    if plan and plan.segments:
        all_wps = plan.all_waypoints
        if len(all_wps) >= 2:
            ref_px = [
                (
                    round((wp.x - grid.x_min) / grid.resolution),
                    round((wp.y - grid.y_min) / grid.resolution),
                )
                for wp in all_wps
            ]
            cv2.polylines(
                canvas,
                [np.array(ref_px, dtype=np.int32)],
                isClosed=False,
                color=(180, 180, 180),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    # 3. Render actual executed vehicle path
    if len(steps) >= 2:
        for i in range(1, len(steps)):
            s_prev = steps[i - 1]
            s_curr = steps[i]
            p1 = (
                round((s_prev.vehicle_state.x - grid.x_min) / grid.resolution),
                round((s_prev.vehicle_state.y - grid.y_min) / grid.resolution),
            )
            p2 = (
                round((s_curr.vehicle_state.x - grid.x_min) / grid.resolution),
                round((s_curr.vehicle_state.y - grid.y_min) / grid.resolution),
            )
            if s_curr.status == ExecutionStatus.EMERGENCY_STOPPED:
                seg_col = (0, 0, 255)
            elif s_curr.status == ExecutionStatus.SWITCHING_GEARS:
                seg_col = (0, 215, 255)
            else:
                seg_col = (50, 205, 50)
            cv2.line(canvas, p1, p2, seg_col, 2, cv2.LINE_AA)

    # 4. Render key vehicle footprints (start and end)
    if steps:
        for idx in (0, len(steps) - 1):
            st = steps[idx].vehicle_state
            footprint = veh.compute_footprint_polygon(st.x, st.y, st.heading_rad)
            f_px = [
                (
                    round((pt[0] - grid.x_min) / grid.resolution),
                    round((pt[1] - grid.y_min) / grid.resolution),
                )
                for pt in footprint
            ]
            f_col = (255, 255, 255) if idx == 0 else (0, 255, 0)
            cv2.polylines(
                canvas,
                [np.array(f_px, dtype=np.int32)],
                isClosed=True,
                color=f_col,
                thickness=2,
            )

    # 5. Top HUD banner
    banner_h = 75
    banner_overlay = canvas[:banner_h, :].copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(banner_overlay, 0.85, canvas[:banner_h, :], 0.15, 0, canvas[:banner_h, :])

    # Status badge
    state_col = _STATE_COLORS.get(report.final_state, (200, 200, 200))
    cv2.rectangle(canvas, (10, 10), (160, 40), state_col, -1)
    cv2.putText(
        canvas,
        report.final_state.value.upper(),
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0) if report.is_success else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Telemetry HUD text
    t_text = (
        f"Mission: {report.mission_id} | Slot: {report.target_slot_id or 'None'} | "
        f"Dur: {report.total_duration_s:.1f}s | Steps: {report.total_steps} | "
        f"Replans: {report.replan_count}"
    )
    cv2.putText(
        canvas,
        t_text,
        (175, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if report.final_kpis:
        k = report.final_kpis
        kpi_text = (
            f"Docking Err: dx={k.docking_error_x_m:.2f}m dy={k.docking_error_y_m:.2f}m "
            f"dHead={math.degrees(k.docking_error_heading_rad):.1f}deg | "
            f"Max CTE: {k.max_cross_track_error_m:.2f}m | Docked: {k.is_docked_successfully}"
        )
        cv2.putText(
            canvas,
            kpi_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas


def render_mission_timeline_chart(
    report: MissionSummaryReport,
    steps: Sequence[ManeuverExecutionStep],
    *,
    width: int = 1000,
    height: int = 550,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render mission state lifecycle timeline Gantt chart and clearance profiles."""
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)

    # Title header
    cv2.putText(
        canvas,
        f"AVP Mission Lifecycle Timeline & Telemetry [{report.mission_id}]",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    if not steps and not report.events:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), canvas)
        return canvas

    total_t = max(1.0, report.total_duration_s)
    margin_l = 80
    margin_r = 40
    plot_w = width - margin_l - margin_r

    def t_to_x(t: float) -> int:
        frac = min(1.0, max(0.0, t / total_t))
        return int(margin_l + frac * plot_w)

    # Panel 1: State timeline (Gantt bars)
    p1_top = 55
    p1_h = 100
    cv2.rectangle(canvas, (margin_l, p1_top), (margin_l + plot_w, p1_top + p1_h), (45, 45, 45), -1)
    cv2.putText(
        canvas,
        "MISSION STATE PROGRESSION",
        (margin_l + 10, p1_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    events: list[MissionEvent] = report.events
    for idx, ev in enumerate(events):
        t_start = ev.t
        t_end = events[idx + 1].t if idx + 1 < len(events) else total_t
        x1 = t_to_x(t_start)
        x2 = max(x1 + 3, t_to_x(t_end))
        col = _STATE_COLORS.get(ev.target_state, (150, 150, 150))
        cv2.rectangle(canvas, (x1, p1_top + 35), (x2, p1_top + 80), col, -1)
        cv2.rectangle(canvas, (x1, p1_top + 35), (x2, p1_top + 80), (20, 20, 20), 1)

        # Label inside block if wide enough
        if (x2 - x1) > 40:
            lbl = ev.target_state.value[:8]
            cv2.putText(
                canvas,
                lbl,
                (x1 + 4, p1_top + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # Panel 2: Velocity & Obstacle Clearance curves
    p2_top = 185
    p2_h = 320
    cv2.rectangle(canvas, (margin_l, p2_top), (margin_l + plot_w, p2_top + p2_h), (45, 45, 45), -1)
    cv2.putText(
        canvas,
        "VELOCITY & OBSTACLE CLEARANCE PROFILE",
        (margin_l + 10, p2_top + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if steps:
        times = [s.t for s in steps]
        vels = [s.vehicle_state.velocity for s in steps]
        clears = [min(5.0, s.nearest_obstacle_distance_m) for s in steps]

        # Draw velocity curve (Cyan)
        max_v = max(1.5, max(abs(v) for v in vels))
        v_pts: list[tuple[int, int]] = []
        for t, v in zip(times, vels, strict=False):
            px = t_to_x(t)
            py = int(p2_top + 160 - (v / max_v) * 120)
            v_pts.append((px, py))
        if len(v_pts) >= 2:
            cv2.polylines(
                canvas,
                [np.array(v_pts, dtype=np.int32)],
                isClosed=False,
                color=(255, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Draw clearance curve (Yellow)
        c_pts: list[tuple[int, int]] = []
        for t, c in zip(times, clears, strict=False):
            px = t_to_x(t)
            py = int(p2_top + p2_h - 20 - (c / 5.0) * 120)
            c_pts.append((px, py))
        if len(c_pts) >= 2:
            cv2.polylines(
                canvas,
                [np.array(c_pts, dtype=np.int32)],
                isClosed=False,
                color=(0, 200, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Draw legend
        cv2.putText(
            canvas,
            "Velocity (m/s) [Cyan]",
            (margin_l + 350, p2_top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Clearance (m) [Yellow]",
            (margin_l + 550, p2_top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
