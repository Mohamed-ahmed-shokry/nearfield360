"""Visual rendering engine for facility vector maps, routes, and localization dashboards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.mapping.localization import LocalizationStepRecord
from nearfield360.mapping.models import (
    FacilityMap,
    GlobalRoute,
    LocalizationReport,
    SlotReservationStatus,
    WaypointType,
)

# Color constants (BGR)
COLOR_BG = (28, 28, 28)
COLOR_OBSTACLE = (80, 80, 90)
COLOR_LANE = (70, 70, 75)
COLOR_LANE_CENTER = (140, 140, 150)
COLOR_VACANT = (40, 180, 60)
COLOR_OCCUPIED = (50, 50, 200)
COLOR_RESERVED = (40, 160, 220)
COLOR_ROUTE = (0, 215, 255)
COLOR_TRUE_PATH = (60, 220, 60)
COLOR_DR_PATH = (60, 60, 220)
COLOR_EKF_PATH = (255, 200, 0)
COLOR_LANDMARK_FIX = (0, 255, 255)


def render_facility_bev(
    facility: FacilityMap,
    route: GlobalRoute | None = None,
    estimated_poses: Sequence[tuple[float, float, float]] | None = None,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render top-down metric BEV layout of the parking facility, driving lanes, and slots."""
    canvas = np.full((canvas_size, canvas_size, 3), COLOR_BG, dtype=np.uint8)

    # Coordinate mapping from metric coordinates to pixels
    margin = 70
    span_x = max(1.0, facility.bounds.x_max - facility.bounds.x_min)
    span_y = max(1.0, facility.bounds.y_max - facility.bounds.y_min)
    max_span = max(span_x, span_y)
    scale = (canvas_size - 2 * margin) / max_span

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        px = round(margin + (x - facility.bounds.x_min) * scale)
        py = round(canvas_size - margin - (y - facility.bounds.y_min) * scale)
        return (px, py)

    # 1. Draw structural obstacles (walls, curbs, pillars)
    for obs in facility.obstacles:
        pts = np.array([to_pixel(p[0], p[1]) for p in obs.polygon], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(canvas, [pts], COLOR_OBSTACLE)
        cv2.polylines(canvas, [pts], isClosed=False, color=(120, 120, 130), thickness=3)

    # 2. Draw driving lanes
    for lane in facility.lanes:
        pixel_pts = [to_pixel(p[0], p[1]) for p in lane.centerline_points]
        pts_np = np.array(pixel_pts, dtype=np.int32)

        # Draw lane corridor bounds
        corridor_px = round((lane.width_m / 2.0) * scale)
        cv2.polylines(
            canvas, [pts_np], isClosed=False, color=COLOR_LANE, thickness=max(2, corridor_px * 2)
        )
        # Draw centerline
        cv2.polylines(
            canvas,
            [pts_np],
            isClosed=False,
            color=COLOR_LANE_CENTER,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Draw direction arrows along the lane
        for i in range(len(pixel_pts) - 1):
            p1 = pixel_pts[i]
            p2 = pixel_pts[i + 1]
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            tip_x = int(mx + 8 * math.cos(angle))
            tip_y = int(my + 8 * math.sin(angle))
            cv2.arrowedLine(canvas, (mx, my), (tip_x, tip_y), (180, 180, 190), 1, tipLength=0.5)

    # 3. Draw parking slots
    for slot in facility.slots:
        corner_pts = np.array([to_pixel(c[0], c[1]) for c in slot.corners], dtype=np.int32)

        is_target = route is not None and route.target_slot_id == slot.slot_id
        if is_target:
            slot_color = (0, 220, 255)
        elif slot.status == SlotReservationStatus.VACANT:
            slot_color = COLOR_VACANT
        elif slot.status == SlotReservationStatus.OCCUPIED:
            slot_color = COLOR_OCCUPIED
        else:
            slot_color = COLOR_RESERVED

        # Fill semi-transparent slot interior
        slot_mask = canvas.copy()
        cv2.fillPoly(slot_mask, [corner_pts], slot_color)
        alpha = 0.55 if is_target else 0.35
        cv2.addWeighted(slot_mask, alpha, canvas, 1.0 - alpha, 0, canvas)
        cv2.polylines(
            canvas,
            [corner_pts],
            isClosed=True,
            color=slot_color,
            thickness=2 if not is_target else 3,
        )

        # Draw slot ID text
        c_px = to_pixel(slot.center[0], slot.center[1])
        cv2.putText(
            canvas,
            slot.slot_id.replace("bay_", "").replace("surface_bay_", ""),
            (c_px[0] - 12, c_px[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    # 4. Draw waypoints
    for wp in facility.waypoints:
        pt = to_pixel(wp.x, wp.y)
        if wp.waypoint_type == WaypointType.ENTRY:
            color = (50, 220, 50)
            radius = 6
        elif wp.waypoint_type == WaypointType.EXIT:
            color = (50, 50, 220)
            radius = 6
        elif wp.waypoint_type == WaypointType.INTERSECTION:
            color = (220, 150, 50)
            radius = 4
        else:
            color = (160, 160, 160)
            radius = 3

        cv2.circle(canvas, pt, radius, color, -1)
        cv2.circle(canvas, pt, radius + 1, (20, 20, 20), 1)

    # 5. Draw planned global route if provided
    if route and route.waypoints:
        route_pts = np.array([to_pixel(w.x, w.y) for w in route.waypoints], dtype=np.int32)
        cv2.polylines(
            canvas,
            [route_pts],
            isClosed=False,
            color=COLOR_ROUTE,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        # Draw start pose arrow
        start_pt = to_pixel(route.start_pose[0], route.start_pose[1])
        cv2.circle(canvas, start_pt, 7, (0, 255, 0), -1)
        h_rad = route.start_pose[2]
        tip_x = int(start_pt[0] + 16 * math.cos(h_rad))
        tip_y = int(start_pt[1] - 16 * math.sin(h_rad))
        cv2.arrowedLine(canvas, start_pt, (tip_x, tip_y), (0, 255, 0), 2, tipLength=0.4)

    # 6. Draw estimated vehicle trajectory if provided
    if estimated_poses:
        est_pts = np.array([to_pixel(p[0], p[1]) for p in estimated_poses], dtype=np.int32)
        cv2.polylines(
            canvas,
            [est_pts],
            isClosed=False,
            color=COLOR_EKF_PATH,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # 7. Render HUD Title Header
    cv2.rectangle(canvas, (10, 10), (canvas_size - 10, 55), (40, 40, 45), -1)
    cv2.rectangle(canvas, (10, 10), (canvas_size - 10, 55), (80, 80, 90), 1)

    vacant_count = sum(1 for s in facility.slots if s.status == SlotReservationStatus.VACANT)
    cv2.putText(
        canvas,
        f"{facility.name} [{facility.facility_type}]",
        (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    info_text = (
        f"Slots: {len(facility.slots)} ({vacant_count} vacant) | Lanes: {len(facility.lanes)} | "
        f"Bounds: [{facility.bounds.x_min:.0f}, {facility.bounds.x_max:.0f}] x "
        f"[{facility.bounds.y_min:.0f}, {facility.bounds.y_max:.0f}] m"
    )
    cv2.putText(
        canvas,
        info_text,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas


def render_localization_dashboard(
    report: LocalizationReport,
    records: Sequence[LocalizationStepRecord],
    *,
    facility: FacilityMap | None = None,
    width: int = 1200,
    height: int = 700,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel localization dashboard with trajectory and telemetry."""
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # Panel split
    bev_w = 600
    chart_w = width - bev_w

    # Left Panel: BEV Trajectory comparison
    cv2.rectangle(canvas, (10, 10), (bev_w - 5, height - 10), (35, 35, 40), -1)
    cv2.rectangle(canvas, (10, 10), (bev_w - 5, height - 10), (70, 70, 80), 1)

    # Compute bounding box of records for auto-scaling
    all_x = (
        [r.true_pose[0] for r in records]
        + [r.dead_reckoning_pose[0] for r in records]
        + [r.estimated_pose[0] for r in records]
    )
    all_y = (
        [r.true_pose[1] for r in records]
        + [r.dead_reckoning_pose[1] for r in records]
        + [r.estimated_pose[1] for r in records]
    )
    min_x, max_x = min(all_x) - 3.0, max(all_x) + 3.0
    min_y, max_y = min(all_y) - 3.0, max(all_y) + 3.0

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    max_span = max(span_x, span_y)

    margin = 50
    bev_scale = (min(bev_w - 20, height - 20) - 2 * margin) / max_span

    def to_bev_px(x: float, y: float) -> tuple[int, int]:
        px = round(10 + margin + (x - min_x) * bev_scale)
        py = round(height - 10 - margin - (y - min_y) * bev_scale)
        return (px, py)

    # Draw BEV True Path, Dead-Reckoning, and EKF Estimated Path
    true_pts = np.array(
        [to_bev_px(r.true_pose[0], r.true_pose[1]) for r in records], dtype=np.int32
    )
    dr_pts = np.array(
        [to_bev_px(r.dead_reckoning_pose[0], r.dead_reckoning_pose[1]) for r in records],
        dtype=np.int32,
    )
    ekf_pts = np.array(
        [to_bev_px(r.estimated_pose[0], r.estimated_pose[1]) for r in records], dtype=np.int32
    )

    cv2.polylines(
        canvas, [true_pts], isClosed=False, color=COLOR_TRUE_PATH, thickness=2, lineType=cv2.LINE_AA
    )
    cv2.polylines(
        canvas, [dr_pts], isClosed=False, color=COLOR_DR_PATH, thickness=1, lineType=cv2.LINE_AA
    )
    cv2.polylines(
        canvas, [ekf_pts], isClosed=False, color=COLOR_EKF_PATH, thickness=2, lineType=cv2.LINE_AA
    )

    # Draw covariance ellipses and landmark fixes
    for i, r in enumerate(records):
        pt = to_bev_px(r.estimated_pose[0], r.estimated_pose[1])
        if r.observed_landmarks:
            cv2.circle(canvas, pt, 5, COLOR_LANDMARK_FIX, -1)
            cv2.circle(canvas, pt, 6, (0, 0, 0), 1)

        # Draw 2-sigma covariance ellipse every 10 steps
        if i % 10 == 0:
            sigma_x = math.sqrt(max(1e-4, r.covariance[0][0]))
            sigma_y = math.sqrt(max(1e-4, r.covariance[1][1]))
            rad_x = max(2, round(2.0 * sigma_x * bev_scale))
            rad_y = max(2, round(2.0 * sigma_y * bev_scale))
            cv2.ellipse(canvas, pt, (rad_x, rad_y), 0, 0, 360, (180, 180, 0), 1)

    # Left Panel HUD / Legend
    cv2.putText(
        canvas,
        "Vehicle Trajectory & Covariance",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "True [Green]",
        (25, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TRUE_PATH,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Odometry [Red]",
        (130, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_DR_PATH,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "EKF Filter [Cyan]",
        (250, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_EKF_PATH,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Fix [Yellow]",
        (385, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_LANDMARK_FIX,
        1,
        cv2.LINE_AA,
    )

    # Right Panel: Telemetry curves
    p_right_l = bev_w + 5
    p_right_w = chart_w - 15

    # Top chart: Position Error over time
    top_chart_top = 10
    top_chart_h = 320
    cv2.rectangle(
        canvas,
        (p_right_l, top_chart_top),
        (p_right_l + p_right_w, top_chart_top + top_chart_h),
        (35, 35, 40),
        -1,
    )
    cv2.rectangle(
        canvas,
        (p_right_l, top_chart_top),
        (p_right_l + p_right_w, top_chart_top + top_chart_h),
        (70, 70, 80),
        1,
    )

    err_title = (
        f"Position Error (m) | Mean: {report.mean_position_error_m:.3f}m, "
        f"Max: {report.max_position_error_m:.3f}m"
    )
    cv2.putText(
        canvas,
        err_title,
        (p_right_l + 15, top_chart_top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    pos_errors = [
        math.hypot(r.estimated_pose[0] - r.true_pose[0], r.estimated_pose[1] - r.true_pose[1])
        for r in records
    ]
    max_err_val = max(0.5, max(pos_errors) * 1.25)

    c_margin_l = p_right_l + 45
    c_margin_r = p_right_l + p_right_w - 20
    c_margin_t = top_chart_top + 45
    c_margin_b = top_chart_top + top_chart_h - 25
    c_w = c_margin_r - c_margin_l
    c_h = c_margin_b - c_margin_t

    cv2.line(canvas, (c_margin_l, c_margin_b), (c_margin_r, c_margin_b), (100, 100, 110), 1)
    cv2.line(canvas, (c_margin_l, c_margin_t), (c_margin_l, c_margin_b), (100, 100, 110), 1)

    pts_err: list[tuple[int, int]] = []
    for i, err in enumerate(pos_errors):
        frac_x = i / max(1, len(pos_errors) - 1)
        frac_y = min(1.0, err / max_err_val)
        px = int(c_margin_l + frac_x * c_w)
        py = int(c_margin_b - frac_y * c_h)
        pts_err.append((px, py))

    if len(pts_err) >= 2:
        cv2.polylines(
            canvas,
            [np.array(pts_err, dtype=np.int32)],
            isClosed=False,
            color=(50, 180, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Bottom chart: Position Uncertainty (1-sigma) over time
    bot_chart_top = top_chart_top + top_chart_h + 10
    bot_chart_h = height - bot_chart_top - 10
    cv2.rectangle(
        canvas,
        (p_right_l, bot_chart_top),
        (p_right_l + p_right_w, bot_chart_top + bot_chart_h),
        (35, 35, 40),
        -1,
    )
    cv2.rectangle(
        canvas,
        (p_right_l, bot_chart_top),
        (p_right_l + p_right_w, bot_chart_top + bot_chart_h),
        (70, 70, 80),
        1,
    )

    unc_title = (
        f"Filter Uncertainty (1-sigma m) | Max: {report.max_position_uncertainty_m:.3f}m | "
        f"Fixes: {report.total_landmark_updates}"
    )
    cv2.putText(
        canvas,
        unc_title,
        (p_right_l + 15, bot_chart_top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    uncertainties = [math.sqrt(max(1e-4, r.covariance[0][0] + r.covariance[1][1])) for r in records]
    max_unc_val = max(0.5, max(uncertainties) * 1.25)

    bc_margin_t = bot_chart_top + 45
    bc_margin_b = bot_chart_top + bot_chart_h - 25
    bc_h = bc_margin_b - bc_margin_t

    cv2.line(canvas, (c_margin_l, bc_margin_b), (c_margin_r, bc_margin_b), (100, 100, 110), 1)
    cv2.line(canvas, (c_margin_l, bc_margin_t), (c_margin_l, bc_margin_b), (100, 100, 110), 1)

    pts_unc: list[tuple[int, int]] = []
    for i, unc in enumerate(uncertainties):
        frac_x = i / max(1, len(uncertainties) - 1)
        frac_y = min(1.0, unc / max_unc_val)
        px = int(c_margin_l + frac_x * c_w)
        py = int(bc_margin_b - frac_y * bc_h)
        pts_unc.append((px, py))

    if len(pts_unc) >= 2:
        cv2.polylines(
            canvas,
            [np.array(pts_unc, dtype=np.int32)],
            isClosed=False,
            color=(255, 120, 50),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
