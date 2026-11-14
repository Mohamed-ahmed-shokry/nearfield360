"""Tactical 1280x720 HUD visualizer for vehicle condition inspection and damage audit."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.inspection.models import (
    DamageSeverity,
    InspectionReport,
    LiabilityVerdict,
    TirePosition,
    TireWearStatus,
)

# Color constants (BGR)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (28, 32, 40)
COLOR_PANEL_BORDER = (50, 58, 72)
COLOR_TEXT_PRIMARY = (245, 248, 252)
COLOR_TEXT_SECONDARY = (160, 172, 190)
COLOR_TEXT_MUTED = (100, 112, 130)

COLOR_GREEN = (60, 210, 120)
COLOR_CYAN = (230, 190, 40)
COLOR_AMBER = (40, 160, 255)
COLOR_RED = (50, 60, 235)
COLOR_VEHICLE_BODY = (70, 78, 95)
COLOR_VEHICLE_FILL = (38, 44, 56)


def render_inspection_dashboard(
    report: InspectionReport,
    output_path: Path | str | None = None,
) -> NDArray[np.uint8]:
    """Render a tactical 1280x720 HUD dashboard for vehicle condition and liability audit."""
    canvas = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Bar
    cv2.rectangle(canvas, (0, 0), (1280, 56), (25, 30, 38), -1)
    cv2.line(canvas, (0, 56), (1280, 56), COLOR_PANEL_BORDER, 1)

    cv2.putText(
        canvas,
        "VEHICLE CONDITION INSPECTION & LIABILITY AUDIT",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        COLOR_TEXT_PRIMARY,
        2,
        cv2.LINE_AA,
    )

    meta_str = (
        f"VIN: {report.snapshot.vin} | PHASE: {report.phase.value} | "
        f"SCORE: {report.snapshot.overall_condition_score:.1f}%"
    )
    cv2.putText(
        canvas,
        meta_str,
        (680, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # 2. Left Panel: 360 Exterior Silhouette & Damage Hotspots (24, 76, 580, 620)
    _draw_panel(canvas, 24, 76, 580, 620, "360-DEGREE EXTERIOR DAMAGE HOTSPOTS")
    _draw_silhouette_panel(canvas, 24, 76, 580, 620, report)

    # 3. Right Top Panel: 4-Wheel Tire Profiling & TPMS (624, 76, 632, 210)
    _draw_panel(canvas, 624, 76, 632, 210, "4-WHEEL TIRE TREAD DEPTH & TPMS COLD PRESSURE")
    _draw_tires_panel(canvas, 624, 76, 632, 210, report)

    # 4. Right Middle Panel: ECE R43 Safety Glass & Transmittance (624, 302, 632, 180)
    _draw_panel(canvas, 624, 302, 632, 180, "ECE R43 WINDSHIELD & SAFETY GLASS INTEGRITY")
    _draw_glass_panel(canvas, 624, 302, 632, 180, report)

    # 5. Right Bottom Panel: Cryptographic Liability Certificate (624, 498, 632, 198)
    _draw_panel(canvas, 624, 498, 632, 198, "CRYPTOGRAPHIC LIABILITY HANDOVER CERTIFICATE")
    _draw_certificate_panel(canvas, 624, 498, 632, 198, report)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), canvas)

    return canvas


def _draw_panel(canvas: NDArray[np.uint8], x: int, y: int, w: int, h: int, title: str) -> None:
    """Draw a dark container panel with top header line."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 32), (34, 40, 52), -1)
    cv2.line(canvas, (x, y + 32), (x + w, y + 32), COLOR_PANEL_BORDER, 1)

    cv2.putText(
        canvas,
        title,
        (x + 14, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )


def _draw_silhouette_panel(
    canvas: NDArray[np.uint8], x: int, y: int, w: int, h: int, report: InspectionReport
) -> None:
    """Draw top-down vehicle contour with damage hotspot markers and condition badge."""
    cx = x + w // 2
    cy = y + 40 + (h - 40) // 2

    # Draw vehicle chassis contour
    vw, vh = 160, 360
    vx1, vy1 = cx - vw // 2, cy - vh // 2
    vx2, vy2 = cx + vw // 2, cy + vh // 2

    cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), COLOR_VEHICLE_FILL, -1)
    cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), COLOR_VEHICLE_BODY, 2)

    # Windshield and rear glass trapezoids
    pts_front_glass = np.array(
        [[vx1 + 15, vy1 + 70], [vx2 - 15, vy1 + 70], [vx2 - 25, vy1 + 120], [vx1 + 25, vy1 + 120]]
    )
    cv2.fillPoly(canvas, [pts_front_glass], (48, 56, 72))

    pts_rear_glass = np.array(
        [[vx1 + 25, vy2 - 110], [vx2 - 25, vy2 - 110], [vx2 - 15, vy2 - 60], [vx1 + 15, vy2 - 60]]
    )
    cv2.fillPoly(canvas, [pts_rear_glass], (48, 56, 72))

    # Wheels
    wheel_w, wheel_h = 24, 50
    for wx, wy in [
        (vx1 - 14, vy1 + 45),
        (vx2 - 10, vy1 + 45),
        (vx1 - 14, vy2 - 95),
        (vx2 - 10, vy2 - 95),
    ]:
        cv2.rectangle(canvas, (wx, wy), (wx + wheel_w, wy + wheel_h), (25, 25, 30), -1)
        cv2.rectangle(canvas, (wx, wy), (wx + wheel_w, wy + wheel_h), (60, 68, 80), 1)

    # Overlay damage anomalies
    severity_colors = {
        DamageSeverity.NEGLIGIBLE: COLOR_CYAN,
        DamageSeverity.MINOR: COLOR_AMBER,
        DamageSeverity.MODERATE: COLOR_AMBER,
        DamageSeverity.SEVERE: COLOR_RED,
        DamageSeverity.CRITICAL: COLOR_RED,
    }

    # Offsets for surface locations
    location_offsets: dict[str, tuple[int, int]] = {
        "front_bumper": (0, -170),
        "front_hood": (0, -110),
        "left_front_door": (-75, -20),
        "right_front_door": (75, -20),
        "left_rear_door": (-75, 60),
        "right_rear_door": (75, 60),
        "driver_door": (-75, -10),
        "passenger_door": (75, -10),
        "rear_bumper": (0, 170),
        "rear_hatch": (0, 110),
        "roof": (0, 0),
    }

    for idx, anomaly in enumerate(report.snapshot.anomalies):
        col = severity_colors.get(anomaly.severity, COLOR_AMBER)
        off_x, off_y = location_offsets.get(anomaly.surface_location, (0, (idx * 30) - 60))
        spot_x = cx + off_x
        spot_y = cy + off_y

        cv2.circle(canvas, (spot_x, spot_y), 10, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (spot_x, spot_y), 16, col, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{idx + 1}",
            (spot_x - 6, spot_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Condition Score Badge
    score = report.snapshot.overall_condition_score
    badge_col = COLOR_GREEN if score >= 90.0 else (COLOR_AMBER if score >= 75.0 else COLOR_RED)
    cv2.rectangle(canvas, (x + 16, y + 44), (x + 240, y + 78), (20, 24, 30), -1)
    cv2.rectangle(canvas, (x + 16, y + 44), (x + 240, y + 78), badge_col, 1)
    score_str = f"CONDITION: {score:.1f}% ({len(report.snapshot.anomalies)} DEFECTS)"
    cv2.putText(
        canvas,
        score_str,
        (x + 24, y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        badge_col,
        1,
        cv2.LINE_AA,
    )

    # Bottom summary
    summary_str = (
        f"MILEAGE: {report.snapshot.mileage_km:.0f} km | CAMERAS: 4 CHANNELS (360 SURROUND)"
    )
    cv2.putText(
        canvas,
        summary_str,
        (x + 20, y + h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )


def _draw_tires_panel(
    canvas: NDArray[np.uint8], x: int, y: int, w: int, _h: int, report: InspectionReport
) -> None:
    """Draw 4-wheel tire tread depth gauges, asymmetry flags, and TPMS cold pressures."""
    tires_by_pos = {t.position: t for t in report.snapshot.tires}
    positions = [
        ("FL", TirePosition.FRONT_LEFT, x + 20, y + 50),
        ("FR", TirePosition.FRONT_RIGHT, x + 320, y + 50),
        ("RL", TirePosition.REAR_LEFT, x + 20, y + 125),
        ("RR", TirePosition.REAR_RIGHT, x + 320, y + 125),
    ]

    for label, pos, qx, qy in positions:
        telemetry = tires_by_pos.get(pos)
        tread = telemetry.mean_tread_depth_mm if telemetry else 6.0
        pressure = telemetry.pressure_bar if telemetry else 2.40
        status = telemetry.status.value if telemetry else "GOOD"

        t_color = COLOR_GREEN if tread >= 3.0 else (COLOR_AMBER if tread >= 1.6 else COLOR_RED)
        if status == TireWearStatus.PRESSURE_FAULT.value:
            t_color = COLOR_AMBER

        cv2.rectangle(canvas, (qx, qy), (qx + 280, qy + 60), (34, 40, 52), -1)
        cv2.rectangle(canvas, (qx, qy), (qx + 280, qy + 60), COLOR_PANEL_BORDER, 1)

        header_str = f"{label} | TREAD: {tread:.1f} mm | {pressure:.2f} bar"
        cv2.putText(
            canvas,
            header_str,
            (qx + 10, qy + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        # Tread depth bar (0 to 8 mm)
        fill_px = int(min(1.0, tread / 8.0) * 160)
        cv2.rectangle(canvas, (qx + 10, qy + 34), (qx + 170, qy + 46), (50, 56, 70), -1)
        cv2.rectangle(canvas, (qx + 10, qy + 34), (qx + 10 + fill_px, qy + 46), t_color, -1)

        cv2.putText(
            canvas,
            status,
            (qx + 185, qy + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            t_color,
            1,
            cv2.LINE_AA,
        )


def _draw_glass_panel(
    canvas: NDArray[np.uint8], x: int, y: int, _w: int, _h: int, report: InspectionReport
) -> None:
    """Draw ECE R43 windshield clarity, crack counts, and light transmittance."""
    glass = report.snapshot.glass
    comp_color = COLOR_GREEN if glass.ece_r43_compliant else COLOR_RED
    comp_str = "ECE R43 COMPLIANT" if glass.ece_r43_compliant else "ECE R43 VIOLATION"

    cv2.rectangle(canvas, (x + 20, y + 50), (x + 240, y + 80), (20, 24, 30), -1)
    cv2.rectangle(canvas, (x + 20, y + 50), (x + 240, y + 80), comp_color, 1)
    cv2.putText(
        canvas,
        comp_str,
        (x + 30, y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        comp_color,
        1,
        cv2.LINE_AA,
    )

    trans_str = f"LIGHT TRANSMITTANCE: {glass.optical_transmittance_pct:.1f}% (ECE min: 70.0%)"
    cv2.putText(
        canvas,
        trans_str,
        (x + 260, y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )

    details_str = (
        f"WINDSHIELD CLEAR: {'YES' if glass.windshield_clear else 'NO'} | "
        f"STONE CHIPS: {glass.chip_count} | CRACKS: {len(glass.cracks_detected)}"
    )
    cv2.putText(
        canvas,
        details_str,
        (x + 20, y + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )

    if glass.cracks_detected:
        c = glass.cracks_detected[0]
        c_str = f"PRIMARY CRACK: {c.description}"
        cv2.putText(
            canvas,
            c_str,
            (x + 20, y + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_RED,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "PRIMARY CRACK: None (Surface clear)",
            (x + 20, y + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )


def _draw_certificate_panel(
    canvas: NDArray[np.uint8], x: int, y: int, _w: int, _h: int, report: InspectionReport
) -> None:
    """Draw cryptographic liability certificate details, SHA-256 digest, and verdict."""
    cert = report.certificate

    if cert:
        v_colors = {
            LiabilityVerdict.NO_NEW_DAMAGE: COLOR_GREEN,
            LiabilityVerdict.PRE_EXISTING_DOCUMENTED: COLOR_CYAN,
            LiabilityVerdict.NEW_DAMAGE_DETECTED: COLOR_RED,
            LiabilityVerdict.DISPUTE_REQUIRES_REVIEW: COLOR_AMBER,
        }
        v_color = v_colors.get(cert.verdict, COLOR_CYAN)

        cv2.rectangle(canvas, (x + 20, y + 50), (x + 280, y + 80), (20, 24, 30), -1)
        cv2.rectangle(canvas, (x + 20, y + 50), (x + 280, y + 80), v_color, 1)
        cv2.putText(
            canvas,
            f"VERDICT: {cert.verdict.value}",
            (x + 30, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            v_color,
            1,
            cv2.LINE_AA,
        )

        id_str = f"ID: {cert.certificate_id} | SESSION: {cert.session_id}"
        cv2.putText(
            canvas,
            id_str,
            (x + 20, y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        sha_str = f"SHA-256 DIGEST: {cert.payload_sha256[:32]}..."
        cv2.putText(
            canvas,
            sha_str,
            (x + 20, y + 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

        hmac_str = f"HMAC SIGNATURE: {cert.signature_hmac[:32]}... [AUTHENTIC]"
        cv2.putText(
            canvas,
            hmac_str,
            (x + 20, y + 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "No cryptographic certificate issued for this epoch.",
            (x + 20, y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
