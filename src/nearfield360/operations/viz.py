"""Tactical facility operations control tower dashboard visualizer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.operations.models import DailyKPIReport, HourlySnapshot

# High-contrast tactical color palette (BGR)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_GRID = (45, 50, 60)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

# Metric colors
COLOR_ARRIVALS = (240, 160, 40)  # Blue
COLOR_DEPARTURES = (60, 210, 95)  # Green
COLOR_BLOCKED = (50, 50, 235)  # Red
COLOR_OCCUPANCY = (40, 180, 240)  # Amber
COLOR_ENERGY = (215, 115, 225)  # Purple/Magenta
COLOR_REVENUE = (50, 215, 235)  # Gold / Solar yellow


def _draw_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_operations_dashboard(
    kpis: DailyKPIReport,
    snapshots: Sequence[HourlySnapshot],
    slot_capacity: int,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a 1280x720 tactical operations dashboard showing day activity."""
    width, height = 1280, 720
    canvas: NDArray[np.uint8] = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Banner
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "NEARFIELD360 // FACILITY OPERATIONS CONTROL TOWER",
        (35, 48),
        0.75,
        COLOR_TEXT_TITLE,
        2,
    )
    sla_pct = kpis.sla_compliance_ratio * 100.0
    _draw_text(
        canvas,
        f"CAPACITY: {slot_capacity} SLOTS | SLA: {sla_pct:.1f}%",
        (width - 430, 48),
        0.55,
        COLOR_DEPARTURES if kpis.sla_compliance_ratio >= 0.85 else COLOR_OCCUPANCY,
        1,
    )

    # 2. Top Summary KPI Cards (4 cards)
    card_w = (width - 40 - 3 * 15) // 4
    cards_data = [
        (
            "THROUGHPUT",
            f"{kpis.completed_journeys} veh",
            f"{kpis.throughput_per_hour:.1f} / hr",
            COLOR_DEPARTURES,
        ),
        (
            "PEAK OCCUPANCY",
            f"{kpis.peak_occupancy} / {slot_capacity}",
            f"{kpis.peak_occupancy_ratio * 100.0:.1f}% util",
            COLOR_OCCUPANCY,
        ),
        (
            "ENERGY NET",
            f"{kpis.net_grid_energy_kwh:.1f} kWh",
            f"+{kpis.energy_charged_kwh:.1f} / -{kpis.energy_exported_kwh:.1f}",
            COLOR_ENERGY,
        ),
        (
            "TOTAL REVENUE",
            f"${kpis.total_revenue_usd:.2f}",
            f"P: ${kpis.parking_revenue_usd:.2f} | E: ${kpis.energy_revenue_usd:.2f}",
            COLOR_REVENUE,
        ),
    ]

    for i, (title, val, sub, col) in enumerate(cards_data):
        x1 = 20 + i * (card_w + 15)
        x2 = x1 + card_w
        cv2.rectangle(canvas, (x1, 80), (x2, 160), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (x1, 80), (x2, 160), COLOR_CARD_BORDER, 1)
        cv2.line(canvas, (x1 + 10, 80), (x1 + 35, 80), col, 3)
        _draw_text(canvas, title, (x1 + 15, 105), 0.45, COLOR_TEXT_MUTED, 1)
        _draw_text(canvas, val, (x1 + 15, 132), 0.7, col, 2)
        _draw_text(canvas, sub, (x1 + 15, 150), 0.4, COLOR_TEXT_BODY, 1)

    # 3. Chart 1: Hourly Occupancy vs Capacity (Left Panel)
    c1_x1, c1_y1, c1_x2, c1_y2 = 20, 180, 820, 430
    cv2.rectangle(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "HOURLY OCCUPANCY & VEHICLE THROUGHPUT",
        (c1_x1 + 15, c1_y1 + 25),
        0.55,
        COLOR_TEXT_TITLE,
        1,
    )

    # Plot timeline points
    num_snaps = max(len(snapshots), 1)
    plot_w = c1_x2 - c1_x1 - 60
    plot_h = c1_y2 - c1_y1 - 60
    ox, oy = c1_x1 + 40, c1_y2 - 25

    # Capacity line
    cap_max = max(slot_capacity, 1)
    cap_y = int(oy - (plot_h * 0.9))
    cv2.line(canvas, (ox, cap_y), (ox + plot_w, cap_y), (80, 80, 120), 1, cv2.LINE_AA)
    _draw_text(
        canvas, f"Cap: {slot_capacity}", (ox + plot_w - 70, cap_y - 5), 0.35, (120, 120, 160), 1
    )

    occ_points: list[tuple[int, int]] = []
    arr_points: list[tuple[int, int]] = []
    for s in snapshots:
        px = int(ox + (s.hour / max(num_snaps - 1, 1)) * plot_w)
        occ_y = int(oy - (s.occupied_slots / cap_max) * (plot_h * 0.85))
        occ_points.append((px, occ_y))
        arr_y = int(oy - min(s.arrivals / (cap_max * 0.5), 1.0) * (plot_h * 0.85))
        arr_points.append((px, arr_y))

    if len(occ_points) > 1:
        cv2.polylines(
            canvas, [np.array(occ_points, dtype=np.int32)], False, COLOR_OCCUPANCY, 2, cv2.LINE_AA
        )
        cv2.polylines(
            canvas, [np.array(arr_points, dtype=np.int32)], False, COLOR_ARRIVALS, 1, cv2.LINE_AA
        )

    for px, py in occ_points:
        cv2.circle(canvas, (px, py), 3, COLOR_OCCUPANCY, -1)

    # Legend for Chart 1
    cv2.circle(canvas, (c1_x1 + 450, c1_y1 + 22), 4, COLOR_OCCUPANCY, -1)
    _draw_text(canvas, "Occupancy", (c1_x1 + 460, c1_y1 + 26), 0.4, COLOR_TEXT_BODY, 1)
    cv2.circle(canvas, (c1_x1 + 550, c1_y1 + 22), 4, COLOR_ARRIVALS, -1)
    _draw_text(canvas, "Arrivals", (c1_x1 + 560, c1_y1 + 26), 0.4, COLOR_TEXT_BODY, 1)

    # 4. Chart 2: Hourly Facility Load & V2G Export (Bottom Left Panel)
    c2_x1, c2_y1, c2_x2, c2_y2 = 20, 450, 820, 700
    cv2.rectangle(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "FACILITY ELECTRICAL POWER PROFILE (LOAD & V2G EXPORT)",
        (c2_x1 + 15, c2_y1 + 25),
        0.55,
        COLOR_TEXT_TITLE,
        1,
    )

    max_load = max((s.facility_load_kw for s in snapshots), default=100.0)
    max_load = max(max_load, 80.0)
    oy2 = c2_y2 - 25
    load_points: list[tuple[int, int]] = []
    exp_points: list[tuple[int, int]] = []
    for s in snapshots:
        px = int(ox + (s.hour / max(num_snaps - 1, 1)) * plot_w)
        ly = int(oy2 - (s.facility_load_kw / max_load) * (plot_h * 0.85))
        load_points.append((px, ly))
        ey = int(oy2 - (s.export_kw / max_load) * (plot_h * 0.85))
        exp_points.append((px, ey))

    if len(load_points) > 1:
        cv2.polylines(
            canvas, [np.array(load_points, dtype=np.int32)], False, COLOR_ENERGY, 2, cv2.LINE_AA
        )
        cv2.polylines(
            canvas, [np.array(exp_points, dtype=np.int32)], False, COLOR_REVENUE, 2, cv2.LINE_AA
        )

    cv2.circle(canvas, (c2_x1 + 550, c2_y1 + 22), 4, COLOR_ENERGY, -1)
    _draw_text(canvas, "Load kW", (c2_x1 + 560, c2_y1 + 26), 0.4, COLOR_TEXT_BODY, 1)
    cv2.circle(canvas, (c2_x1 + 650, c2_y1 + 22), 4, COLOR_REVENUE, -1)
    _draw_text(canvas, "V2G Export kW", (c2_x1 + 660, c2_y1 + 26), 0.4, COLOR_TEXT_BODY, 1)

    # 5. Right Sidebar: Operational Status & Service Level Breakdown
    s_x1, s_y1, s_x2, s_y2 = 840, 180, width - 20, 700
    cv2.rectangle(canvas, (s_x1, s_y1), (s_x2, s_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (s_x1, s_y1), (s_x2, s_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas, "SERVICE LEVEL COMPLIANCE", (s_x1 + 15, s_y1 + 30), 0.55, COLOR_TEXT_TITLE, 1
    )

    # Service level breakdown bars
    levels = [
        ("Nominal Dispatch", kpis.service_level_counts.get("nominal", 0), COLOR_DEPARTURES),
        ("Curbside Queue", kpis.service_level_counts.get("curbside_handover", 0), COLOR_OCCUPANCY),
        ("Degraded Review", kpis.service_level_counts.get("degraded_dispatch", 0), (80, 175, 230)),
        ("Blocked / Balked", kpis.service_level_counts.get("blocked", 0), COLOR_BLOCKED),
    ]

    total_req = max(kpis.total_arrivals, 1)
    bar_y = s_y1 + 70
    bar_max_w = s_x2 - s_x1 - 40
    for label, count, col in levels:
        pct = (count / total_req) * 100.0
        bw = int((count / total_req) * bar_max_w)
        _draw_text(
            canvas, f"{label}: {count} ({pct:.1f}%)", (s_x1 + 20, bar_y), 0.45, COLOR_TEXT_BODY, 1
        )
        cv2.rectangle(
            canvas, (s_x1 + 20, bar_y + 8), (s_x1 + 20 + bar_max_w, bar_y + 22), COLOR_GRID, -1
        )
        if bw > 0:
            cv2.rectangle(canvas, (s_x1 + 20, bar_y + 8), (s_x1 + 20 + bw, bar_y + 22), col, -1)
        bar_y += 45

    # Dispatch latencies and queue metrics
    _draw_text(
        canvas, "DISPATCH TIMINGS & QUEUES", (s_x1 + 15, bar_y + 25), 0.55, COLOR_TEXT_TITLE, 1
    )
    metrics = [
        f"Mean Dispatch Wait : {kpis.mean_dispatch_wait_s:.1f} s",
        f"95th %tile Wait    : {kpis.p95_dispatch_wait_s:.1f} s",
        f"Blocked Vehicles   : {kpis.blocked_vehicles}",
        f"Energy Net Balance : {kpis.net_grid_energy_kwh:.1f} kWh",
        f"Total Daily Revenue: ${kpis.total_revenue_usd:.2f}",
    ]
    for idx, m in enumerate(metrics):
        _draw_text(canvas, m, (s_x1 + 20, bar_y + 60 + idx * 30), 0.45, COLOR_TEXT_BODY, 1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
