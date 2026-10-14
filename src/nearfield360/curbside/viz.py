"""Visual rendering engine for curbside staging bays, pedestrian safety circles, and queue HUD."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.curbside.models import (
    BayStatus,
    CurbsideReport,
    HandoverType,
    SafetyInterlockStatus,
)
from nearfield360.mapping.models import FacilityMap

# Color constants (BGR)
COLOR_BG = (24, 24, 28)
COLOR_PANEL_BG = (35, 35, 42)
COLOR_LANE = (65, 65, 75)
COLOR_LANE_CENTER = (120, 120, 135)
COLOR_OBSTACLE = (75, 75, 85)
COLOR_BAY_VACANT = (70, 70, 85)
COLOR_BAY_UNLOADING = (30, 165, 255)  # Amber
COLOR_BAY_BOARDING = (220, 180, 40)  # Cyan-blue
COLOR_BAY_READY = (60, 200, 90)  # Lime green
COLOR_HALO_INTERLOCKED = (30, 30, 230)  # Red warning
COLOR_HALO_CLEAR = (60, 200, 90)  # Green clear
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_TEXT_MUTED = (140, 140, 150)
COLOR_GAUGE_BG = (55, 55, 65)


def _get_bay_color(status: BayStatus) -> tuple[int, int, int]:
    if status == BayStatus.OCCUPIED_UNLOADING:
        return COLOR_BAY_UNLOADING
    if status == BayStatus.OCCUPIED_BOARDING:
        return COLOR_BAY_BOARDING
    if status == BayStatus.HANDOVER_READY:
        return COLOR_BAY_READY
    return COLOR_BAY_VACANT


def render_curbside_dashboard(
    facility: FacilityMap,
    report: CurbsideReport,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel BEV layout of curbside staging bays, pedestrian circles, and queue HUD."""
    panel_width = 440
    width = canvas_size + panel_width
    height = canvas_size
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    margin = 60
    span_x = max(1.0, facility.bounds.x_max - facility.bounds.x_min)
    span_y = max(1.0, facility.bounds.y_max - facility.bounds.y_min)
    max_span = max(span_x, span_y)
    scale = (canvas_size - 2 * margin) / max_span

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        px = round(margin + (x - facility.bounds.x_min) * scale)
        py = round(canvas_size - margin - (y - facility.bounds.y_min) * scale)
        return (px, py)

    # 1. Structural obstacles
    for obs in facility.obstacles:
        pts = np.array([to_pixel(x, y) for x, y in obs.polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_OBSTACLE, thickness=4)

    # 2. Driving lanes
    for lane in facility.lanes:
        lane_pts = np.array([to_pixel(x, y) for x, y in lane.centerline_points], dtype=np.int32)
        cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE, thickness=16)
        cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE_CENTER, thickness=2)

    # Map report bays to facility slots
    bay_map = {b.bay_id: b for b in report.bays}
    bay_indices = list(bay_map.keys())

    # 3. Draw parking slots and curbside bays
    for i, slot in enumerate(facility.slots):
        corners_px = np.array([to_pixel(x, y) for x, y in slot.corners], dtype=np.int32)
        cx, cy = to_pixel(slot.center[0], slot.center[1])

        # If slot index matches a curbside bay
        if i < len(bay_indices):
            bay = bay_map[bay_indices[i]]
            b_color = _get_bay_color(bay.status)

            # Fill bay
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [corners_px], b_color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            cv2.polylines(canvas, [corners_px], isClosed=True, color=b_color, thickness=2)

            # Draw safety interlock circle around occupied bay
            if bay.is_occupied:
                is_interlocked = bay.interlock_status != SafetyInterlockStatus.DISENGAGED_CLEAR
                h_color = COLOR_HALO_INTERLOCKED if is_interlocked else COLOR_HALO_CLEAR
                halo_radius = max(20, round(2.0 * scale))
                cv2.circle(canvas, (cx, cy), halo_radius, h_color, 2, cv2.LINE_AA)

            # Bay label
            label = f"{bay.bay_id[-4:]}"
            cv2.putText(
                canvas,
                label,
                (cx - 16, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.polylines(canvas, [corners_px], isClosed=True, color=COLOR_BAY_VACANT, thickness=1)

    # 4. Right HUD Panel
    panel_x = canvas_size
    cv2.rectangle(canvas, (panel_x, 0), (width, height), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, height), (60, 60, 72), 2)

    px_left = panel_x + 24
    py = 40

    cv2.putText(
        canvas,
        "CURBSIDE STAGING HUD",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_TEXT_TITLE,
        2,
        cv2.LINE_AA,
    )
    py += 24
    cv2.putText(
        canvas,
        f"Facility: {report.facility_id}  |  t={report.timestamp_s:.1f}s",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Global KPI Box
    py += 35
    cv2.rectangle(canvas, (px_left, py), (width - 24, py + 75), COLOR_BG, -1)
    cv2.rectangle(canvas, (px_left, py), (width - 24, py + 75), (55, 55, 65), 1)

    kpi_col1 = px_left + 15
    kpi_col2 = px_left + 200

    cv2.putText(
        canvas,
        "Vacant Staging Bays",
        (kpi_col1, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{report.vacant_bays_count}/{report.total_bays_count}",
        (kpi_col1, py + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 220, 120),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Active Interlocks",
        (kpi_col2, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    interlock_color = (
        COLOR_HALO_INTERLOCKED if report.active_interlocks_count > 0 else (80, 220, 120)
    )
    cv2.putText(
        canvas,
        f"{report.active_interlocks_count} ACTIVE",
        (kpi_col2, py + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        interlock_color,
        2,
        cv2.LINE_AA,
    )

    py += 100
    cv2.putText(
        canvas,
        "CURBSIDE STAGING BAYS",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    # Render each bay
    for b in report.bays:
        py += 30
        b_color = _get_bay_color(b.status)
        cv2.circle(canvas, (px_left + 4, py - 4), 4, b_color, -1)

        type_badge = "DROP-OFF" if b.bay_type == HandoverType.DROPOFF else "PICK-UP"
        status_txt = f"{b.bay_id} [{type_badge}]: {b.status.value.upper()}"
        cv2.putText(
            canvas,
            status_txt,
            (px_left + 16, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_BODY,
            1,
            cv2.LINE_AA,
        )

        py += 16
        dwell_txt = (
            f"  Vehicle: {b.occupied_vehicle_id or 'None'}  |  Dwell: {b.current_dwell_s:.1f}s"
        )
        cv2.putText(
            canvas,
            dwell_txt,
            (px_left + 16, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    # Queue and Dwell Summary
    py += 40
    cv2.putText(
        canvas,
        "HANDOVER QUEUE & AUDIT",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    py += 24
    cv2.putText(
        canvas,
        f"Active Queue Requests:   {len(report.active_missions)}",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )
    py += 20
    cv2.putText(
        canvas,
        f"Completed Missions:      {report.completed_missions_count}",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )
    py += 20
    cv2.putText(
        canvas,
        f"Average Dwell Time:      {report.average_dwell_duration_s:.1f} s",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )
    py += 20
    cv2.putText(
        canvas,
        f"Total Overstay Charges:  ${report.total_overstay_penalties_usd:.2f}",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
