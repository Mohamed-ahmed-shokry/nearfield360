"""Tactical 1280x720 HUD visualizer for SAE J2954 WPT alignment, FOD, and efficiency."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.wpt.models import (
    WPTAlignmentStatus,
    WPTInterlockState,
    WPTReport,
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
COLOR_COIL_PRIMARY = (80, 80, 120)
COLOR_COIL_SECONDARY = (255, 140, 0)


def render_wpt_dashboard(
    report: WPTReport,
    output_path: Path | str | None = None,
) -> NDArray[np.uint8]:
    """Render a tactical 1280x720 HUD dashboard for SAE J2954 WPT status and telemetry."""
    canvas = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Bar
    cv2.rectangle(canvas, (0, 0), (1280, 56), (25, 30, 38), -1)
    cv2.line(canvas, (0, 56), (1280, 56), COLOR_PANEL_BORDER, 1)

    cv2.putText(
        canvas,
        "SAE J2954 WIRELESS POWER TRANSFER & INDUCTIVE DOCKING",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        COLOR_TEXT_PRIMARY,
        2,
        cv2.LINE_AA,
    )

    meta_str = (
        f"VEH: {report.vehicle_id} | PAD: {report.pad_id} | "
        f"PWR: {report.power_class.value} | Z: {report.z_class.value}"
    )
    cv2.putText(
        canvas,
        meta_str,
        (650, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # 2. Left Panel: Magnetic Coil Alignment & Flux Heatmap (24, 76, 580, 620)
    _draw_panel(canvas, 24, 76, 580, 620, "MAGNETIC RESONANCE COIL ALIGNMENT")
    _draw_alignment_view(canvas, 24, 76, 580, 620, report)

    # 3. Right Top Panel: Resonance & Transfer Efficiency (624, 76, 632, 220)
    _draw_panel(canvas, 624, 76, 632, 220, "ELECTRICAL TRANSFER EFFICIENCY & POWER GOVERNOR")
    _draw_efficiency_panel(canvas, 624, 76, 632, 220, report)

    # 4. Right Middle Panel: Foreign Object Detection & LOD (624, 312, 632, 175)
    _draw_panel(canvas, 624, 312, 632, 175, "FOREIGN OBJECT DETECTION (FOD) & LIVING OBJECT (LOD)")
    _draw_fod_panel(canvas, 624, 312, 632, 175, report)

    # 5. Right Bottom Panel: Autonomous Inching Trajectory & Telemetry (624, 503, 632, 193)
    _draw_panel(canvas, 624, 503, 632, 193, "AUTONOMOUS DOCKING INCHING TRAJECTORY & TELEMETRY")
    _draw_docking_panel(canvas, 624, 503, 632, 193, report)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), canvas)

    return canvas


def _draw_panel(canvas: NDArray[np.uint8], x: int, y: int, w: int, h: int, title: str) -> None:
    """Draw a dark panel container with header and subtle borders."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 32), (34, 40, 52), -1)
    cv2.line(canvas, (x, y + 32), (x + w, y + 32), COLOR_PANEL_BORDER, 1)

    cv2.putText(
        canvas,
        title,
        (x + 14, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )


def _draw_alignment_view(
    canvas: NDArray[np.uint8], x: int, y: int, w: int, h: int, report: WPTReport
) -> None:
    """Draw top-down 2D magnetic flux field, tolerance envelope, and coil positions."""
    cx = x + w // 2
    cy = y + 40 + (h - 40) // 2

    # Scale: 1 mm -> 1.5 pixels
    scale = 1.3

    # Draw polar grid concentric rings
    for r_mm, col in [(50, (40, 46, 58)), (100, (48, 56, 70)), (150, (40, 46, 58))]:
        r_px = int(r_mm * scale)
        cv2.circle(canvas, (cx, cy), r_px, col, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{r_mm}mm",
            (cx + 6, cy - r_px + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    # Grid axes
    cv2.line(canvas, (cx - 240, cy), (cx + 240, cy), (48, 56, 70), 1)
    cv2.line(canvas, (cx, cy - 220), (cx, cy + 220), (48, 56, 70), 1)

    # SAE J2954 Tolerance box (+/- 100 mm longitudinal x, +/- 75 mm lateral y)
    box_w_px = int(75.0 * scale * 2)
    box_h_px = int(100.0 * scale * 2)
    cv2.rectangle(
        canvas,
        (cx - box_w_px // 2, cy - box_h_px // 2),
        (cx + box_w_px // 2, cy + box_h_px // 2),
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # Docked radius threshold circle (30 mm)
    cv2.circle(canvas, (cx, cy), int(30.0 * scale), COLOR_GREEN, 1, cv2.LINE_AA)

    # Primary Ground Assembly coil outline (pad at origin)
    cv2.circle(canvas, (cx, cy), int(120.0 * scale * 0.4), COLOR_COIL_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "GA COIL (GROUND)",
        (cx - 55, cy + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Secondary Vehicle Assembly coil position
    dx_px = int(report.offset.delta_x_mm * scale)
    dy_px = int(report.offset.delta_y_mm * scale)
    va_x = cx + dy_px  # Lateral is horizontal
    va_y = cy - dx_px  # Longitudinal is vertical (forward is up)

    # Draw secondary coil
    coil_color = COLOR_GREEN if report.offset.is_within_tolerance else COLOR_AMBER
    if report.offset.alignment_status == WPTAlignmentStatus.DOCKED:
        coil_color = COLOR_GREEN
    elif report.offset.alignment_status == WPTAlignmentStatus.MISALIGNED:
        coil_color = COLOR_RED

    cv2.circle(canvas, (va_x, va_y), int(110.0 * scale * 0.4), coil_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (va_x, va_y), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Heading indicator arrow
    heading_rad = math.radians(report.offset.delta_theta_deg)
    arrow_len = 35
    arrow_tip_x = int(va_x + arrow_len * math.sin(heading_rad))
    arrow_tip_y = int(va_y - arrow_len * math.cos(heading_rad))
    cv2.arrowedLine(
        canvas,
        (va_x, va_y),
        (arrow_tip_x, arrow_tip_y),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.3,
    )

    # Status Pill Overlay
    status_str = f"STATUS: {report.offset.alignment_status.value}"
    cv2.rectangle(canvas, (x + 16, y + 44), (x + 220, y + 74), (20, 24, 30), -1)
    cv2.rectangle(canvas, (x + 16, y + 44), (x + 220, y + 74), coil_color, 1)
    cv2.putText(
        canvas,
        status_str,
        (x + 24, y + 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        coil_color,
        1,
        cv2.LINE_AA,
    )

    # Offset metrics box bottom
    offset_str = (
        f"dX: {report.offset.delta_x_mm:+.1f}mm  |  dY: {report.offset.delta_y_mm:+.1f}mm  |  "
        f"dZ: {report.offset.delta_z_mm:.1f}mm  |  dTh: {report.offset.delta_theta_deg:+.2f}deg"
    )
    cv2.putText(
        canvas,
        offset_str,
        (x + 20, y + h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )


def _draw_efficiency_panel(
    canvas: NDArray[np.uint8], x: int, y: int, w: int, _h: int, report: WPTReport
) -> None:
    """Draw electrical coupling factor, transfer efficiency gauge, and power governor."""
    eff = report.efficiency

    # Coupling factor k
    cv2.putText(
        canvas,
        "COUPLING FACTOR (k):",
        (x + 20, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )
    k_str = f"{eff.coupling_factor_k:.3f} (nominal: 0.260)"
    cv2.putText(
        canvas,
        k_str,
        (x + 220, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )

    # Progress bar for k (0.0 to 0.40)
    k_fill = int(min(1.0, eff.coupling_factor_k / 0.40) * 360)
    cv2.rectangle(canvas, (x + 220, y + 70), (x + 580, y + 84), (45, 52, 65), -1)
    cv2.rectangle(canvas, (x + 220, y + 70), (x + 220 + k_fill, y + 84), COLOR_CYAN, -1)

    # DC-to-DC Transfer Efficiency
    cv2.putText(
        canvas,
        "DC-DC EFFICIENCY:",
        (x + 20, y + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )
    eff_pct = eff.transfer_efficiency * 100.0
    eff_color = COLOR_GREEN if eff_pct >= 85.0 else (COLOR_AMBER if eff_pct >= 70.0 else COLOR_RED)
    eff_str = f"{eff_pct:.1f}% (SAE min: 85.0%)"
    cv2.putText(
        canvas,
        eff_str,
        (x + 220, y + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        eff_color,
        1,
        cv2.LINE_AA,
    )

    eff_fill = int(min(1.0, eff.transfer_efficiency) * 360)
    cv2.rectangle(canvas, (x + 220, y + 125), (x + 580, y + 139), (45, 52, 65), -1)
    cv2.rectangle(canvas, (x + 220, y + 125), (x + 220 + eff_fill, y + 139), eff_color, -1)

    # Power Governor Command & Delivery
    p_str = (
        f"CMD: {eff.commanded_power_kw:.2f}kW | "
        f"DELIV: {eff.delivered_power_kw:.2f}kW | "
        f"LOSS: {eff.thermal_loss_kw:.2f}kW"
    )
    cv2.putText(
        canvas,
        p_str,
        (x + 20, y + 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )


def _draw_fod_panel(
    canvas: NDArray[np.uint8], x: int, y: int, _w: int, _h: int, report: WPTReport
) -> None:
    """Draw Foreign Object Detection and safety interlock state."""
    fod = report.fod

    interlock_color = COLOR_GREEN if fod.interlock_state == WPTInterlockState.NORMAL else COLOR_RED
    interlock_str = f"INTERLOCK: {fod.interlock_state.value}"
    cv2.rectangle(canvas, (x + 20, y + 48), (x + 260, y + 78), (20, 24, 30), -1)
    cv2.rectangle(canvas, (x + 20, y + 48), (x + 260, y + 78), interlock_color, 1)
    cv2.putText(
        canvas,
        interlock_str,
        (x + 30, y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        interlock_color,
        1,
        cv2.LINE_AA,
    )

    fod_flag_color = COLOR_RED if fod.metal_object_detected else COLOR_GREEN
    fod_flag = "METAL OBJECT: DETECTED" if fod.metal_object_detected else "METAL OBJECT: CLEAR"
    cv2.putText(
        canvas,
        fod_flag,
        (x + 290, y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        fod_flag_color,
        1,
        cv2.LINE_AA,
    )

    lod_flag_color = COLOR_RED if fod.living_object_detected else COLOR_GREEN
    lod_flag = "LIVING OBJECT: DETECTED" if fod.living_object_detected else "LIVING OBJECT: CLEAR"
    cv2.putText(
        canvas,
        lod_flag,
        (x + 290, y + 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        lod_flag_color,
        1,
        cv2.LINE_AA,
    )

    metrics_str = (
        f"FREQ SHIFT: {fod.frequency_shift_khz:+.2f} kHz (limit: 1.50 kHz)  |  "
        f"Q DROP: {fod.q_drop_percent:.1f}% (limit: 15.0%)"
    )
    cv2.putText(
        canvas,
        metrics_str,
        (x + 20, y + 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"DIAGNOSTIC: {fod.reason}",
        (x + 20, y + 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )


def _draw_docking_panel(
    canvas: NDArray[np.uint8], x: int, y: int, _w: int, _h: int, report: WPTReport
) -> None:
    """Draw autonomous inching docking summary, steps, and convergence status."""
    total_steps = len(report.docking_history)
    docked = any(s.docked for s in report.docking_history) if report.docking_history else False

    summary_str = (
        f"TRAJECTORY ITERATIONS: {total_steps}  |  DOCKING CONVERGED: {'YES' if docked else 'NO'}"
    )
    cv2.putText(
        canvas,
        summary_str,
        (x + 20, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )

    if report.docking_history:
        first = report.docking_history[0]
        last = report.docking_history[-1]
        start_str = (
            f"INIT: x={first.vehicle_x_m:+.3f}m, "
            f"y={first.vehicle_y_m:+.3f}m, th={first.heading_deg:+.1f}deg"
        )
        final_str = (
            f"FINAL: x={last.vehicle_x_m:+.3f}m, "
            f"y={last.vehicle_y_m:+.3f}m, r={last.coil_offset.radial_offset_mm:.1f}mm"
        )
        cv2.putText(
            canvas,
            start_str,
            (x + 20, y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            final_str,
            (x + 20, y + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_GREEN if docked else COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "No discrete inching steps recorded in telemetry frame.",
            (x + 20, y + 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    ts_str = f"TELEMETRY CAPTURE TIMESTAMP: {report.generated_at_utc}"
    cv2.putText(
        canvas,
        ts_str,
        (x + 20, y + 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
