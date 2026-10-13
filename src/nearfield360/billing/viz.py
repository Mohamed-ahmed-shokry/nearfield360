"""Visual rendering engine for facility billing status, revenue breakdown, and financial KPIs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.billing.models import (
    FinancialSettlementReport,
    ParkingReceipt,
    PaymentStatus,
)
from nearfield360.mapping.models import FacilityMap

# Color constants (BGR)
COLOR_BG = (24, 24, 28)
COLOR_PANEL_BG = (35, 35, 42)
COLOR_LANE = (65, 65, 75)
COLOR_LANE_CENTER = (120, 120, 135)
COLOR_OBSTACLE = (75, 75, 85)
COLOR_SLOT_REGULAR = (90, 90, 105)
COLOR_SLOT_SETTLED = (50, 205, 50)  # Lime green
COLOR_SLOT_PENDING = (30, 165, 255)  # Amber orange
COLOR_SLOT_WAIVED = (200, 200, 200)  # Grey
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_TEXT_MUTED = (140, 140, 150)
COLOR_BAR_BG = (55, 55, 65)
COLOR_BAR_PARKING = (100, 180, 255)
COLOR_BAR_ENERGY = (0, 230, 180)
COLOR_BAR_PENALTY = (80, 80, 240)


def render_billing_dashboard(
    facility: FacilityMap,
    report: FinancialSettlementReport,
    active_receipts: Sequence[ParkingReceipt] | None = None,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel BEV facility layout with billing badges and financial KPIs."""
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

    receipt_list = active_receipts or report.receipts
    # Map vehicle / receipts to bay positions if available
    receipt_by_idx = dict(enumerate(receipt_list))

    # 3. Draw parking slots
    for i, slot in enumerate(facility.slots):
        corners_px = np.array([to_pixel(x, y) for x, y in slot.corners], dtype=np.int32)
        r = receipt_by_idx.get(i)

        if r is not None:
            if r.payment_status == PaymentStatus.SETTLED:
                fill_color = COLOR_SLOT_SETTLED
            elif r.payment_status == PaymentStatus.PENDING:
                fill_color = COLOR_SLOT_PENDING
            else:
                fill_color = COLOR_SLOT_WAIVED

            overlay = canvas.copy()
            cv2.fillPoly(overlay, [corners_px], fill_color)
            cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)
            cv2.polylines(canvas, [corners_px], isClosed=True, color=fill_color, thickness=2)

            # Badge showing fee
            cx, cy = to_pixel(slot.center[0], slot.center[1])
            cv2.putText(
                canvas,
                f"${r.total_charged:.0f}",
                (cx - 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.polylines(
                canvas, [corners_px], isClosed=True, color=COLOR_SLOT_REGULAR, thickness=1
            )

    # 4. Telemetry Side Panel (Right)
    panel_x = canvas_size + 15
    cv2.rectangle(
        canvas,
        (panel_x, 20),
        (width - 20, height - 20),
        COLOR_PANEL_BG,
        thickness=-1,
    )
    cv2.rectangle(
        canvas,
        (panel_x, 20),
        (width - 20, height - 20),
        (55, 55, 65),
        thickness=1,
    )

    y = 55
    cv2.putText(
        canvas,
        "AVP FINANCIAL TELEMETRY",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COLOR_TEXT_TITLE,
        2,
        cv2.LINE_AA,
    )

    y += 35
    cv2.putText(
        canvas,
        f"Facility ID: {report.facility_id}",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    y += 24
    cv2.putText(
        canvas,
        f"Total Revenue: ${report.total_revenue:.2f}",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 230, 180),
        2,
        cv2.LINE_AA,
    )

    y += 22
    cv2.putText(
        canvas,
        f"Transactions: {report.total_receipts_count}  (Settled: {report.settled_receipts_count})",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    y += 35
    cv2.putText(
        canvas,
        "REVENUE BREAKDOWN",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    # Breakdown bars
    tot = max(1.0, report.total_revenue)
    categories = [
        ("Parking Dwell", report.parking_revenue, COLOR_BAR_PARKING),
        ("EV Charging", report.energy_revenue, COLOR_BAR_ENERGY),
        ("Overstay Penalty", report.penalty_revenue, COLOR_BAR_PENALTY),
    ]

    for label, val, bar_color in categories:
        y += 26
        cv2.putText(
            canvas,
            f"{label}: ${val:.2f}",
            (panel_x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_BODY,
            1,
            cv2.LINE_AA,
        )
        y += 10
        cv2.rectangle(
            canvas, (panel_x + 20, y), (panel_x + 360, y + 10), COLOR_BAR_BG, thickness=-1
        )
        bar_len = int(340 * (val / tot))
        if bar_len > 0:
            cv2.rectangle(
                canvas, (panel_x + 20, y), (panel_x + 20 + bar_len, y + 10), bar_color, thickness=-1
            )
        y += 10

    y += 35
    cv2.putText(
        canvas,
        "FINANCIAL AUDIT & TAX",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_TITLE,
        1,
        cv2.LINE_AA,
    )

    y += 24
    cv2.putText(
        canvas,
        f"Tax Collected: ${report.tax_collected:.2f}",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    y += 20
    cv2.putText(
        canvas,
        f"Valet Fees: ${report.valet_service_revenue:.2f}",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    y += 20
    cv2.putText(
        canvas,
        f"Avg Session Dwell: {report.average_dwell_minutes:.1f} min",
        (panel_x + 20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_BODY,
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
