"""Visual rendering engine for multi-tenant partitions, VIP corridors, and SLA telemetry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.mapping.models import FacilityMap
from nearfield360.tenant.models import (
    TenantPartition,
    TenantPartitionReport,
    TenantTier,
)

# Color constants (BGR)
COLOR_BG = (24, 24, 28)
COLOR_PANEL_BG = (35, 35, 42)
COLOR_LANE = (65, 65, 75)
COLOR_LANE_CENTER = (120, 120, 135)
COLOR_CORRIDOR_VIP = (0, 190, 255)  # Bright amber gold
COLOR_CORRIDOR_PREEMPT = (30, 30, 230)  # Red warning
COLOR_OBSTACLE = (75, 75, 85)
COLOR_SLOT_PUBLIC = (90, 90, 105)
COLOR_SLOT_FLEET = (210, 160, 40)  # Cyan-blue
COLOR_SLOT_CARSHARE = (200, 100, 50)  # Cobalt
COLOR_SLOT_VIP = (50, 215, 255)  # Gold
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_TEXT_MUTED = (140, 140, 150)
COLOR_GAUGE_BG = (55, 55, 65)
COLOR_GAUGE_FILL = (60, 200, 90)  # Green
COLOR_GAUGE_WARN = (30, 70, 240)  # Orange-red


def _get_tier_color(tier: TenantTier) -> tuple[int, int, int]:
    if tier == TenantTier.VIP_EXECUTIVE:
        return COLOR_SLOT_VIP
    if tier == TenantTier.COMMERCIAL_FLEET:
        return COLOR_SLOT_FLEET
    if tier == TenantTier.CAR_SHARING:
        return COLOR_SLOT_CARSHARE
    return COLOR_SLOT_PUBLIC


def render_tenant_dashboard(
    facility: FacilityMap,
    report: TenantPartitionReport,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel BEV layout of tenant bay partitions, express corridors, and SLA HUD."""
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

    # 2. Identify VIP express lanes
    vip_lanes: dict[str, bool] = {}
    for c in report.express_corridors:
        for lid in c.lane_ids:
            vip_lanes[lid] = c.is_preempting

    # 3. Driving lanes with express corridor highlight
    for lane in facility.lanes:
        lane_pts = np.array([to_pixel(x, y) for x, y in lane.centerline_points], dtype=np.int32)
        if lane.lane_id in vip_lanes:
            is_preempt = vip_lanes[lane.lane_id]
            col = COLOR_CORRIDOR_PREEMPT if is_preempt else COLOR_CORRIDOR_VIP
            cv2.polylines(canvas, [lane_pts], isClosed=False, color=col, thickness=20)
            cv2.polylines(canvas, [lane_pts], isClosed=False, color=(255, 255, 255), thickness=2)
        else:
            cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE, thickness=16)
            cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE_CENTER, thickness=2)

    # Build slot -> partition lookup
    slot_to_part: dict[str, TenantPartition] = {}
    for p in report.partitions:
        for sid in p.assigned_slot_ids:
            slot_to_part[sid] = p

    # 4. Draw parking slots
    for slot in facility.slots:
        corners_px = np.array([to_pixel(x, y) for x, y in slot.corners], dtype=np.int32)
        tenant_part = slot_to_part.get(slot.slot_id)

        if tenant_part is not None:
            tier_color = _get_tier_color(tenant_part.tier)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [corners_px], tier_color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            cv2.polylines(canvas, [corners_px], isClosed=True, color=tier_color, thickness=2)

            cx, cy = to_pixel(slot.center[0], slot.center[1])
            cv2.putText(
                canvas,
                tenant_part.tenant_id[:5],
                (cx - 14, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.polylines(canvas, [corners_px], isClosed=True, color=COLOR_SLOT_PUBLIC, thickness=1)

    # 5. Right HUD Panel
    panel_x = canvas_size
    cv2.rectangle(canvas, (panel_x, 0), (width, height), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, height), (60, 60, 72), 2)

    px_left = panel_x + 24
    py = 40

    cv2.putText(
        canvas,
        "MULTI-TENANT SLA HUD",
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
        "SLA Compliance",
        (kpi_col1, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{report.overall_sla_compliance * 100:.1f}%",
        (kpi_col1, py + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 220, 120),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Total Penalties",
        (kpi_col2, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"${report.total_breach_penalties:.2f}",
        (kpi_col2, py + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (60, 100, 255),
        2,
        cv2.LINE_AA,
    )

    py += 100
    cv2.putText(
        canvas,
        "TENANT PARTITION QUOTAS",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    # Render each partition
    gauge_w = 220
    gauge_h = 10

    for part in report.partitions:
        py += 30
        tier_color = _get_tier_color(part.tier)
        cv2.circle(canvas, (px_left + 4, py - 4), 4, tier_color, -1)
        cv2.putText(
            canvas,
            f"{part.tenant_name} ({part.tenant_id})",
            (px_left + 16, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_BODY,
            1,
            cv2.LINE_AA,
        )

        py += 14
        # Gauge bar
        cv2.rectangle(
            canvas, (px_left + 16, py), (px_left + 16 + gauge_w, py + gauge_h), COLOR_GAUGE_BG, -1
        )
        util = min(1.0, part.utilization_ratio)
        fill_w = round(gauge_w * util)
        g_color = COLOR_GAUGE_WARN if part.is_quota_exceeded else COLOR_GAUGE_FILL
        if fill_w > 0:
            cv2.rectangle(
                canvas, (px_left + 16, py), (px_left + 16 + fill_w, py + gauge_h), g_color, -1
            )

        cv2.putText(
            canvas,
            f"{part.current_occupancy}/{part.dedicated_bay_quota} bays",
            (px_left + 26 + gauge_w, py + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    # Express Corridors section
    py += 35
    cv2.putText(
        canvas,
        "VIP EXPRESS CORRIDORS",
        (px_left, py),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    if not report.express_corridors:
        py += 24
        cv2.putText(
            canvas,
            "No active VIP corridors reserved.",
            (px_left, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
    else:
        for c in report.express_corridors:
            py += 26
            col = COLOR_CORRIDOR_PREEMPT if c.is_preempting else COLOR_CORRIDOR_VIP
            cv2.circle(canvas, (px_left + 4, py - 4), 4, col, -1)
            status_txt = f"{c.corridor_id}: {c.active_vehicle_id or 'IDLE'}"
            if c.is_preempting:
                status_txt += " [PREEMPTING]"
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

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
