"""Visual rendering engine for EV charging facility layout, SoC badges, and grid load telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargingDispatchReport,
    ChargingSession,
    EVBatteryState,
)
from nearfield360.mapping.models import FacilityMap

# Color constants (BGR)
COLOR_BG = (24, 24, 28)
COLOR_PANEL_BG = (35, 35, 42)
COLOR_LANE = (65, 65, 75)
COLOR_LANE_CENTER = (120, 120, 135)
COLOR_OBSTACLE = (75, 75, 85)
COLOR_SLOT_REGULAR = (90, 90, 105)
COLOR_CHG_AVAILABLE = (50, 205, 50)  # Lime green
COLOR_CHG_CHARGING = (30, 165, 255)  # Amber orange
COLOR_CHG_WAITING_RELOC = (255, 191, 0)  # Cyan
COLOR_CHG_OUT_OF_SERVICE = (0, 0, 220)  # Red
COLOR_CABLE = (0, 255, 255)  # Bright yellow
COLOR_VEHICLE_FILL = (230, 230, 240)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_GRID_BAR_BG = (55, 55, 65)
COLOR_GRID_BAR_FILL = (0, 180, 255)


def render_charging_dashboard(
    facility: FacilityMap,
    bays: Sequence[ChargerBay],
    sessions: Sequence[ChargingSession],
    batteries: Sequence[EVBatteryState] | None = None,
    report: ChargingDispatchReport | None = None,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel BEV facility layout with charging bays, vehicle SoC, and grid load HUD."""
    width = canvas_size + 420
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

    # 1. Draw structural obstacles
    for obs in facility.obstacles:
        pts = np.array([to_pixel(x, y) for x, y in obs.polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_OBSTACLE, thickness=4)

    # 2. Draw driving lanes
    for lane in facility.lanes:
        lane_pts = np.array([to_pixel(x, y) for x, y in lane.centerline_points], dtype=np.int32)
        cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE, thickness=16)
        cv2.polylines(canvas, [lane_pts], isClosed=False, color=COLOR_LANE_CENTER, thickness=2)

    # Index bays by slot_id
    bay_by_slot = {b.slot_id: b for b in bays}
    session_by_bay = {s.bay_id: s for s in sessions}

    # 3. Draw parking slots & highlight charging bays
    for slot in facility.slots:
        corners_px = np.array([to_pixel(x, y) for x, y in slot.corners], dtype=np.int32)
        is_charging_bay = slot.slot_id in bay_by_slot

        if is_charging_bay:
            bay = bay_by_slot[slot.slot_id]
            if bay.status == ChargerPortStatus.AVAILABLE:
                fill_color = COLOR_CHG_AVAILABLE
            elif bay.status == ChargerPortStatus.OCCUPIED_CHARGING:
                fill_color = COLOR_CHG_CHARGING
            elif bay.status == ChargerPortStatus.OCCUPIED_COMPLETE_WAITING_RELOCATION:
                fill_color = COLOR_CHG_WAITING_RELOC
            else:
                fill_color = COLOR_CHG_OUT_OF_SERVICE

            # Translucent fill for charging bays
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [corners_px], fill_color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            cv2.polylines(canvas, [corners_px], isClosed=True, color=fill_color, thickness=3)

            # Draw lightning bolt symbol / marker at bay center
            cx, cy = to_pixel(slot.center[0], slot.center[1])
            cv2.putText(
                canvas,
                "EV",
                (cx - 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # If occupied by vehicle, draw vehicle footprint and cable
            if bay.occupied_by_vehicle_id:
                # Cable connection line from slot edge to vehicle
                sess = session_by_bay.get(bay.bay_id)
                soc_text = f"{int(sess.current_soc * 100)}%" if sess else "EV"
                cv2.circle(canvas, (cx, cy), 14, (0, 0, 0), -1)
                cv2.circle(canvas, (cx, cy), 14, fill_color, 2)
                cv2.putText(
                    canvas,
                    soc_text,
                    (cx - 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            # Regular parking slot
            cv2.polylines(
                canvas, [corners_px], isClosed=True, color=COLOR_SLOT_REGULAR, thickness=1
            )

    # 4. Telemetry Side Panel (Right)
    panel_x = canvas_size + 15
    cv2.rectangle(
        canvas,
        (canvas_size, 0),
        (width, height),
        COLOR_PANEL_BG,
        -1,
    )
    cv2.line(canvas, (canvas_size, 0), (canvas_size, height), (70, 70, 80), 2)

    cv2.putText(
        canvas,
        "EV CHARGING & GRID TELEMETRY",
        (panel_x, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT_TITLE,
        2,
        cv2.LINE_AA,
    )

    y_pos = 80
    total_bays = len(bays)
    available_bays = sum(1 for b in bays if b.status == ChargerPortStatus.AVAILABLE)
    charging_bays = sum(1 for b in bays if b.status == ChargerPortStatus.OCCUPIED_CHARGING)
    reloc_bays = sum(
        1 for b in bays if b.status == ChargerPortStatus.OCCUPIED_COMPLETE_WAITING_RELOCATION
    )

    stats = [
        f"Total Charging Bays:    {total_bays}",
        f"Available Bays:         {available_bays}",
        f"Active Charging:        {charging_bays}",
        f"Waiting Relocation:     {reloc_bays}",
    ]
    for stat_line in stats:
        cv2.putText(
            canvas,
            stat_line,
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_BODY,
            1,
            cv2.LINE_AA,
        )
        y_pos += 26

    # 5. Grid Power Bar & Summary
    y_pos += 15
    cv2.line(canvas, (panel_x, y_pos), (width - 20, y_pos), (70, 70, 80), 1)
    y_pos += 30

    cv2.putText(
        canvas,
        "FACILITY GRID POWER LOAD",
        (panel_x, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )
    y_pos += 25

    if report:
        grid_cap = report.power_summary.grid_capacity_kw
        grid_alloc = report.power_summary.total_allocated_power_kw
        util = report.power_summary.grid_utilization_ratio
        headroom = report.power_summary.available_headroom_kw
        is_curtailed = report.power_summary.is_curtailed
    else:
        grid_cap = 250.0
        grid_alloc = sum(b.current_power_kw for b in bays)
        util = min(1.0, grid_alloc / max(1.0, grid_cap))
        headroom = max(0.0, grid_cap - grid_alloc)
        is_curtailed = False

    bar_w = width - panel_x - 30
    bar_h = 20
    cv2.rectangle(canvas, (panel_x, y_pos), (panel_x + bar_w, y_pos + bar_h), COLOR_GRID_BAR_BG, -1)
    fill_w = int(bar_w * util)
    fill_color = (0, 80, 255) if is_curtailed else COLOR_GRID_BAR_FILL
    cv2.rectangle(canvas, (panel_x, y_pos), (panel_x + fill_w, y_pos + bar_h), fill_color, -1)
    cv2.rectangle(canvas, (panel_x, y_pos), (panel_x + bar_w, y_pos + bar_h), (120, 120, 130), 1)

    y_pos += 40
    power_lines = [
        f"Capacity:      {grid_cap:.1f} kW",
        f"Allocated:     {grid_alloc:.1f} kW ({int(util * 100)}%)",
        f"Headroom:      {headroom:.1f} kW",
        f"Curtailment:   {'ACTIVE' if is_curtailed else 'NORMAL'}",
    ]
    for pl in power_lines:
        color = (0, 80, 255) if "ACTIVE" in pl else COLOR_TEXT_BODY
        cv2.putText(
            canvas,
            pl,
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        y_pos += 26

    # 6. Active Vehicle Sessions List
    y_pos += 15
    cv2.line(canvas, (panel_x, y_pos), (width - 20, y_pos), (70, 70, 80), 1)
    y_pos += 30

    cv2.putText(
        canvas,
        "ACTIVE VEHICLE SESSIONS",
        (panel_x, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )
    y_pos += 25

    if sessions:
        for s in sessions[:5]:
            soc_pct = int(s.current_soc * 100)
            target_pct = int(s.target_soc * 100)
            txt = (
                f"{s.vehicle_id} @ {s.bay_id}: "
                f"{soc_pct}% -> {target_pct}% ({s.allocated_power_kw:.0f}kW)"
            )
            cv2.putText(
                canvas,
                txt,
                (panel_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLOR_TEXT_BODY,
                1,
                cv2.LINE_AA,
            )
            y_pos += 22
    else:
        cv2.putText(
            canvas,
            "No active sessions",
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (140, 140, 150),
            1,
            cv2.LINE_AA,
        )

    # 7. Legend at bottom left
    leg_y = height - 40
    cv2.circle(canvas, (30, leg_y), 6, COLOR_CHG_AVAILABLE, -1)
    cv2.putText(
        canvas, "Available Bay", (45, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_BODY, 1
    )

    cv2.circle(canvas, (170, leg_y), 6, COLOR_CHG_CHARGING, -1)
    cv2.putText(
        canvas,
        "Active Charging",
        (185, leg_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        COLOR_TEXT_BODY,
        1,
    )

    cv2.circle(canvas, (320, leg_y), 6, COLOR_CHG_WAITING_RELOC, -1)
    cv2.putText(
        canvas,
        "Relocation Pending",
        (335, leg_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        COLOR_TEXT_BODY,
        1,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
