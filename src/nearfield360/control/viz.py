"""BEV execution overlay and time-series telemetry visualization for parking control."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.control.models import ExecutionStatus, ManeuverExecutionReport
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.kinematics import AckermannVehicle
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.slots.models import ParkingSlot, SlotOccupancyStatus

# BGR color definitions
_ACTUAL_PATH_COLOR = (50, 205, 50)  # Lime green
_PLANNED_PATH_COLOR = (200, 200, 200)  # Light grey
_EMERGENCY_COLOR = (0, 0, 255)  # Red
_SWITCHING_COLOR = (0, 215, 255)  # Gold


def render_control_execution_bev_overlay(
    grid: BevGrid,
    plan: ParkingTrajectoryPlan,
    report: ManeuverExecutionReport,
    *,
    slots: Sequence[ParkingSlot] | None = None,
    base_canvas: NDArray[np.uint8] | None = None,
    vehicle: AckermannVehicle | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render planned vs actual trajectories, footprints, and telemetry HUD on BEV grid."""
    h, w = grid.shape
    canvas = (
        base_canvas.copy() if base_canvas is not None else np.full((h, w, 3), 32, dtype=np.uint8)
    )
    veh = vehicle or AckermannVehicle()

    # 1. Render parking slots if provided
    if slots:
        slot_overlay = canvas.copy()
        for slot in slots:
            scolor = (0, 200, 0) if slot.status == SlotOccupancyStatus.VACANT else (0, 0, 200)
            pts_px = [
                (
                    round((c.x - grid.x_min) / grid.resolution),
                    round((c.y - grid.y_min) / grid.resolution),
                )
                for c in slot.corners
            ]
            poly_arr = np.array(pts_px, dtype=np.int32)
            cv2.fillPoly(slot_overlay, [poly_arr], scolor)
            cv2.polylines(canvas, [poly_arr], isClosed=True, color=scolor, thickness=2)
        cv2.addWeighted(slot_overlay, 0.25, canvas, 0.75, 0, canvas)

    # 2. Render planned trajectory path (reference line)
    all_ref_pts = plan.all_waypoints
    if len(all_ref_pts) >= 2:
        ref_px = [
            (
                round((wp.x - grid.x_min) / grid.resolution),
                round((wp.y - grid.y_min) / grid.resolution),
            )
            for wp in all_ref_pts
        ]
        cv2.polylines(
            canvas,
            [np.array(ref_px, dtype=np.int32)],
            isClosed=False,
            color=_PLANNED_PATH_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # 3. Render actual executed path taken by vehicle
    if len(report.steps) >= 2:
        act_px = [
            (
                round((step.vehicle_state.x - grid.x_min) / grid.resolution),
                round((step.vehicle_state.y - grid.y_min) / grid.resolution),
            )
            for step in report.steps
        ]
        path_color = (
            _EMERGENCY_COLOR
            if report.status == ExecutionStatus.EMERGENCY_STOPPED
            else _ACTUAL_PATH_COLOR
        )
        cv2.polylines(
            canvas,
            [np.array(act_px, dtype=np.int32)],
            isClosed=False,
            color=path_color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    # 4. Render key vehicle footprints: start pose, gear switch poses, and final pose
    footprint_poses: list[tuple[float, float, float, tuple[int, int, int]]] = [
        (*plan.start_pose, (0, 255, 255)),  # Yellow start
    ]
    for step in report.steps:
        if step.status == ExecutionStatus.SWITCHING_GEARS:
            footprint_poses.append((*step.vehicle_state.pose, _SWITCHING_COLOR))
            break

    if report.steps:
        final_veh = report.steps[-1].vehicle_state
        end_color = (0, 255, 0) if report.kpis.is_docked_successfully else (0, 0, 255)
        footprint_poses.append((*final_veh.pose, end_color))

    for fx, fy, fth, fcolor in footprint_poses:
        corners = veh.compute_footprint_polygon(fx, fy, fth)
        c_px = [
            (
                round((cx - grid.x_min) / grid.resolution),
                round((cy - grid.y_min) / grid.resolution),
            )
            for cx, cy in corners
        ]
        poly = np.array(c_px, dtype=np.int32)
        cv2.polylines(
            canvas, [poly], isClosed=True, color=fcolor, thickness=2, lineType=cv2.LINE_AA
        )

    # 5. Overlay text telemetry HUD
    max_cte_cm = report.kpis.max_cross_track_error_m * 100
    mean_cte_cm = report.kpis.mean_cross_track_error_m * 100
    dock_cm = report.kpis.docking_distance_m * 100
    dock_deg = math.degrees(report.kpis.docking_error_heading_rad)
    docked_str = "YES" if report.kpis.is_docked_successfully else "NO"

    hud_lines = [
        f"Control Status: {report.status.value.upper()}",
        f"Duration: {report.duration_s:.2f}s | Steps: {report.total_steps}",
        f"Max CTE: {max_cte_cm:.1f}cm | Mean: {mean_cte_cm:.1f}cm",
        f"Docking Err: {dock_cm:.1f}cm (dTh: {dock_deg:.1f}deg)",
        f"Docked: {docked_str}",
    ]
    for i, line in enumerate(hud_lines):
        y_pos = 25 + i * 20
        cv2.putText(
            canvas,
            line,
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas


def render_control_telemetry_chart(
    report: ManeuverExecutionReport,
    output_path: Path | None = None,
    width: int = 800,
    height: int = 600,
) -> NDArray[np.uint8]:
    """Render a 3-panel time-series telemetry chart of errors, velocity, and steering."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    steps = report.steps
    if not steps:
        cv2.putText(
            canvas,
            "No telemetry steps available",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), canvas)
        return canvas

    times = [s.t for s in steps]
    t_min = min(times)
    t_max = max(times)
    t_range = max(1e-3, t_max - t_min)

    # 3 subpanels: Panel 0 (Errors), Panel 1 (Velocity), Panel 2 (Steering)
    panel_h = (height - 60) // 3
    margin_l = 80
    margin_r = 40
    plot_w = width - margin_l - margin_r

    def t_to_x(t: float) -> int:
        return int(margin_l + ((t - t_min) / t_range) * plot_w)

    def draw_panel(
        y_top: int,
        title: str,
        series_list: list[tuple[list[float], tuple[int, int, int], str]],
        y_label: str,
    ) -> None:
        y_bottom = y_top + panel_h - 20
        cv2.rectangle(
            canvas,
            (margin_l, y_top),
            (margin_l + plot_w, y_bottom),
            (50, 50, 50),
            1,
        )
        cv2.putText(
            canvas,
            title,
            (margin_l + 10, y_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        all_vals: list[float] = []
        for vals, _, _ in series_list:
            all_vals.extend(vals)

        v_min = min(all_vals) if all_vals else -1.0
        v_max = max(all_vals) if all_vals else 1.0
        if abs(v_max - v_min) < 1e-4:
            v_max += 0.5
            v_min -= 0.5
        v_range = v_max - v_min

        def val_to_y(v: float) -> int:
            normalized = (v - v_min) / v_range
            return int(y_bottom - normalized * (y_bottom - y_top - 25))

        # Zero reference line if in range
        if v_min <= 0.0 <= v_max:
            zy = val_to_y(0.0)
            cv2.line(canvas, (margin_l, zy), (margin_l + plot_w, zy), (70, 70, 70), 1)

        # Plot series
        legend_x = margin_l + 200
        for vals, scolor, sname in series_list:
            pts: list[tuple[int, int]] = []
            for t, v in zip(times, vals, strict=False):
                px = t_to_x(t)
                py = val_to_y(v)
                pts.append((px, py))
            if len(pts) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(pts, dtype=np.int32)],
                    isClosed=False,
                    color=scolor,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            # Draw legend entry
            cv2.line(canvas, (legend_x, y_top + 14), (legend_x + 20, y_top + 14), scolor, 2)
            cv2.putText(
                canvas,
                sname,
                (legend_x + 25, y_top + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                scolor,
                1,
                cv2.LINE_AA,
            )
            legend_x += 160

        # Y axis min and max labels
        cv2.putText(
            canvas,
            f"{v_max:.2f}",
            (10, y_top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
        )
        cv2.putText(
            canvas,
            f"{v_min:.2f}",
            (10, y_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
        )

    # Panel 1: Cross-track & Heading Errors
    cte_vals = [s.error.cross_track_error_m for s in steps]
    he_vals = [s.error.heading_error_rad for s in steps]
    draw_panel(
        30,
        "Tracking Errors",
        [
            (cte_vals, (0, 255, 255), "Cross-Track (m)"),
            (he_vals, (255, 100, 100), "Heading (rad)"),
        ],
        "Error",
    )

    # Panel 2: Velocity
    v_ref = [s.command.target_velocity for s in steps]
    v_act = [s.vehicle_state.velocity for s in steps]
    draw_panel(
        30 + panel_h,
        "Velocity Profile",
        [
            (v_ref, (200, 200, 200), "Target (m/s)"),
            (v_act, (50, 205, 50), "Actual (m/s)"),
        ],
        "Speed",
    )

    # Panel 3: Steering
    delta_cmd = [s.command.steering_angle_rad for s in steps]
    delta_act = [s.vehicle_state.steer_angle_rad for s in steps]
    draw_panel(
        30 + 2 * panel_h,
        "Steering Profile",
        [
            (delta_cmd, (255, 165, 0), "Command (rad)"),
            (delta_act, (0, 215, 255), "Actual (rad)"),
        ],
        "Steer",
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
