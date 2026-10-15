"""Visual rendering engine for teleoperation cockpit, remote guidance paths, and MRM HUD."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.mapping.models import FacilityMap
from nearfield360.teleop.models import (
    MRMStage,
    NetworkQuality,
    TeleopMode,
    TeleopReport,
)

# Color constants (BGR)
COLOR_BG = (22, 22, 26)
COLOR_PANEL_BG = (32, 32, 38)
COLOR_LANE = (60, 60, 70)
COLOR_LANE_CENTER = (110, 110, 125)
COLOR_OBSTACLE = (70, 70, 80)
COLOR_SLOT_VACANT = (50, 50, 60)
COLOR_VEHICLE_BODY = (180, 120, 40)  # Blue-cyan
COLOR_VEHICLE_MRM = (40, 40, 220)  # Red alert
COLOR_GUIDANCE_PATH = (80, 220, 120)  # Green
COLOR_MRM_PATH = (40, 80, 240)  # Red-orange
COLOR_QUALITY_OPTIMAL = (60, 200, 90)  # Lime
COLOR_QUALITY_DEGRADED = (30, 165, 255)  # Amber
COLOR_QUALITY_CRITICAL = (40, 40, 230)  # Red
COLOR_QUALITY_SEVERED = (90, 90, 100)  # Gray
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_TEXT_MUTED = (130, 130, 140)
COLOR_GAUGE_BG = (50, 50, 60)


def _get_quality_color(quality: NetworkQuality) -> tuple[int, int, int]:
    if quality == NetworkQuality.OPTIMAL:
        return COLOR_QUALITY_OPTIMAL
    if quality == NetworkQuality.DEGRADED:
        return COLOR_QUALITY_DEGRADED
    if quality == NetworkQuality.CRITICAL:
        return COLOR_QUALITY_CRITICAL
    return COLOR_QUALITY_SEVERED


def render_teleop_dashboard(
    facility: FacilityMap,
    report: TeleopReport,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel BEV layout of teleoperation sessions, paths, and cockpit HUD."""
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

    # 1. Render lanes
    for lane in facility.lanes:
        if len(lane.centerline_points) >= 2:
            pts = np.array([to_pixel(x, y) for x, y in lane.centerline_points], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_LANE_CENTER, thickness=1)

    # 2. Render obstacles
    for obs in facility.obstacles:
        if len(obs.polygon) >= 3:
            pts = np.array([to_pixel(x, y) for x, y in obs.polygon], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], color=COLOR_OBSTACLE)

    # 3. Render parking slots
    for slot in facility.slots:
        pts = np.array([to_pixel(cx, cy) for cx, cy in slot.corners], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_SLOT_VACANT, thickness=1)

    # 4. Render active teleoperation sessions
    for idx, session in enumerate(report.sessions):
        # Synthetic display anchor along facility center
        center_x = 0.5 * (facility.bounds.x_min + facility.bounds.x_max) + (idx * 6.0)
        center_y = 0.5 * (facility.bounds.y_min + facility.bounds.y_max)
        px, py = to_pixel(center_x, center_y)

        is_mrm = (
            session.active_mode == TeleopMode.MRM_SAFE_STOP
            or session.mrm_stage != MRMStage.INACTIVE
        )
        veh_color = COLOR_VEHICLE_MRM if is_mrm else COLOR_VEHICLE_BODY

        # Vehicle bounding box
        vl = round(4.5 * scale)
        vw = round(2.0 * scale)
        cv2.rectangle(
            canvas,
            (px - vl // 2, py - vw // 2),
            (px + vl // 2, py + vw // 2),
            veh_color,
            thickness=-1,
        )

        # Pulse halo if MRM
        if is_mrm:
            cv2.circle(
                canvas,
                (px, py),
                radius=round(3.5 * scale),
                color=COLOR_VEHICLE_MRM,
                thickness=2,
            )
            cv2.putText(
                canvas,
                "MRM STOP",
                (px - 35, py - vw // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_VEHICLE_MRM,
                1,
                cv2.LINE_AA,
            )
        else:
            # Guidance path arrow
            target_px, target_py = to_pixel(center_x + 5.0, center_y)
            cv2.arrowedLine(
                canvas,
                (px, py),
                (target_px, target_py),
                COLOR_GUIDANCE_PATH,
                thickness=2,
                tipLength=0.3,
            )

        cv2.putText(
            canvas,
            session.vehicle_id,
            (px - 30, py + vw // 2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_TEXT_BODY,
            1,
            cv2.LINE_AA,
        )

    # 5. Right-hand Teleoperation Cockpit HUD
    p_x0 = canvas_size
    cv2.rectangle(canvas, (p_x0, 0), (width, height), COLOR_PANEL_BG, thickness=-1)
    cv2.line(canvas, (p_x0, 0), (p_x0, height), COLOR_LANE, thickness=2)

    pad_x = p_x0 + 25
    y_cursor = 45

    # Panel Title
    cv2.putText(
        canvas,
        "TELEOPERATION COCKPIT",
        (pad_x, y_cursor),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT_TITLE,
        2,
        cv2.LINE_AA,
    )
    y_cursor += 25
    cv2.putText(
        canvas,
        f"Facility: {facility.map_id} | ISO 23374 / SAE J3016",
        (pad_x, y_cursor),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    y_cursor += 40
    cv2.line(canvas, (pad_x, y_cursor), (pad_x + 390, y_cursor), COLOR_LANE, thickness=1)
    y_cursor += 30

    # Network latency and link quality
    cv2.putText(
        canvas,
        "LINK QUALITY & LATENCY (RTT)",
        (pad_x, y_cursor),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )
    y_cursor += 25

    # Quality badge
    rtt = report.average_rtt_ms
    no_sessions = report.active_sessions_count == 0 and report.completed_sessions_count == 0
    if no_sessions or rtt < 100.0:
        quality = NetworkQuality.OPTIMAL
    elif rtt < 250.0:
        quality = NetworkQuality.DEGRADED
    else:
        quality = NetworkQuality.CRITICAL

    q_color = _get_quality_color(quality)
    cv2.rectangle(canvas, (pad_x, y_cursor), (pad_x + 120, y_cursor + 24), q_color, thickness=-1)
    cv2.putText(
        canvas,
        quality.value.upper(),
        (pad_x + 12, y_cursor + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Average RTT: {rtt:.1f} ms",
        (pad_x + 140, y_cursor + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    y_cursor += 45
    # Latency gauge bar
    bar_w = 390
    bar_h = 16
    cv2.rectangle(
        canvas, (pad_x, y_cursor), (pad_x + bar_w, y_cursor + bar_h), COLOR_GAUGE_BG, thickness=-1
    )
    fill_ratio = min(1.0, max(0.05, rtt / 350.0))
    fill_w = round(bar_w * fill_ratio)
    cv2.rectangle(
        canvas, (pad_x, y_cursor), (pad_x + fill_w, y_cursor + bar_h), q_color, thickness=-1
    )

    y_cursor += 45
    cv2.line(canvas, (pad_x, y_cursor), (pad_x + 390, y_cursor), COLOR_LANE, thickness=1)
    y_cursor += 30

    # Operations & Intervention KPI cards
    kpi_labels = [
        ("Active Sessions", f"{report.active_sessions_count}"),
        ("Completed Sessions", f"{report.completed_sessions_count}"),
        ("Total Interventions", f"{report.total_interventions_count}"),
        ("MRM Executions", f"{report.mrm_executions_count}"),
    ]

    for label, val in kpi_labels:
        cv2.putText(
            canvas,
            f"{label}:",
            (pad_x, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        is_mrm_kpi = label == "MRM Executions" and int(val) > 0
        val_color = COLOR_VEHICLE_MRM if is_mrm_kpi else COLOR_TEXT_TITLE
        cv2.putText(
            canvas,
            val,
            (pad_x + 230, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            val_color,
            1,
            cv2.LINE_AA,
        )
        y_cursor += 28

    y_cursor += 20
    cv2.line(canvas, (pad_x, y_cursor), (pad_x + 390, y_cursor), COLOR_LANE, thickness=1)
    y_cursor += 30

    # Active Engagements list
    cv2.putText(
        canvas,
        "ACTIVE REMOTE ENGAGEMENTS",
        (pad_x, y_cursor),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )
    y_cursor += 25

    if not report.sessions:
        cv2.putText(
            canvas,
            "No active teleoperation sessions",
            (pad_x, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
    else:
        for sess in report.sessions[:4]:
            mode_badge = sess.active_mode.value.upper()
            cv2.putText(
                canvas,
                f"{sess.vehicle_id} ({sess.operator_id})",
                (pad_x, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOR_TEXT_BODY,
                1,
                cv2.LINE_AA,
            )
            mode_color = (
                COLOR_VEHICLE_MRM
                if sess.active_mode == TeleopMode.MRM_SAFE_STOP
                else COLOR_GUIDANCE_PATH
            )
            cv2.putText(
                canvas,
                f"[{mode_badge}] Cmds: {sess.commands_processed_count}",
                (pad_x + 180, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                mode_color,
                1,
                cv2.LINE_AA,
            )
            y_cursor += 24

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
