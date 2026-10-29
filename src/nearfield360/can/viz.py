"""Tactical HIL oscilloscope and CAN bus telemetry visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.can.models import HILSimulationReport

# Tactical color palette (BGR)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_GRID = (45, 50, 60)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_SUCCESS = (60, 210, 95)  # Green
COLOR_WARNING = (40, 180, 240)  # Amber
COLOR_ALERT = (50, 50, 235)  # Red
COLOR_OSCILLOSCOPE = (240, 215, 50)  # Cyan/Yellow
COLOR_TRACE_CTRL = (240, 140, 40)  # Blue
COLOR_TRACE_FB = (60, 210, 95)  # Green


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


def render_hil_dashboard(
    report: HILSimulationReport,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a 1280x720 tactical CAN bus oscilloscope and HIL diagnostics dashboard."""
    width, height = 1280, 720
    canvas: NDArray[np.uint8] = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Banner
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "NEARFIELD360 // HARDWARE-IN-THE-LOOP (HIL) CAN BUS GATEWAY",
        (35, 48),
        0.70,
        COLOR_TEXT_TITLE,
        2,
    )
    status_text = "BUS STATUS: NOMINAL" if not report.telemetry.is_overloaded else "BUS OVERLOADED"
    status_col = COLOR_SUCCESS if not report.telemetry.is_overloaded else COLOR_ALERT
    _draw_text(canvas, status_text, (width - 320, 48), 0.55, status_col, 2)

    # 2. Metric Cards (4 cards across top)
    card_w = (width - 40 - 3 * 15) // 4
    t = report.telemetry
    cards = [
        (
            "FRAMES RX / TX",
            f"{t.frames_received} / {t.frames_sent}",
            f"Dropped: {t.dropped_frames}",
            COLOR_SUCCESS,
        ),
        (
            "BUS LOAD UTIL",
            f"{t.bus_load_percent:.1f}%",
            "Cap: 500 kbps",
            COLOR_WARNING if t.bus_load_percent > 70 else COLOR_SUCCESS,
        ),
        (
            "MEAN CLOCK JITTER",
            f"{t.mean_jitter_us:.1f} us",
            f"Peak: {t.peak_jitter_us:.1f} us",
            COLOR_OSCILLOSCOPE,
        ),
        (
            "CORRUPTED FRAMES",
            f"{t.corrupted_frames}",
            f"Faults: {len(report.fault_injection_log)}",
            COLOR_ALERT if t.corrupted_frames > 0 else COLOR_SUCCESS,
        ),
    ]

    for i, (title, val, sub, col) in enumerate(cards):
        x1 = 20 + i * (card_w + 15)
        x2 = x1 + card_w
        cv2.rectangle(canvas, (x1, 80), (x2, 160), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (x1, 80), (x2, 160), COLOR_CARD_BORDER, 1)
        cv2.line(canvas, (x1 + 10, 80), (x1 + 35, 80), col, 3)
        _draw_text(canvas, title, (x1 + 15, 105), 0.45, COLOR_TEXT_MUTED, 1)
        _draw_text(canvas, val, (x1 + 15, 132), 0.65, col, 2)
        _draw_text(canvas, sub, (x1 + 15, 150), 0.40, COLOR_TEXT_BODY, 1)

    # 3. Oscilloscope / Physical Signal Timing (Left Panel)
    o_x1, o_y1, o_x2, o_y2 = 20, 180, 800, 430
    cv2.rectangle(canvas, (o_x1, o_y1), (o_x2, o_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (o_x1, o_y1), (o_x2, o_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "CAN BUS OSCILLOSCOPE (CYCLIC MESSAGE TIMING & JITTER)",
        (o_x1 + 15, o_y1 + 25),
        0.52,
        COLOR_TEXT_TITLE,
        1,
    )

    # Draw grid lines for oscilloscope
    for gx in range(o_x1 + 40, o_x2 - 20, 60):
        cv2.line(canvas, (gx, o_y1 + 40), (gx, o_y2 - 20), COLOR_GRID, 1, cv2.LINE_AA)
    for gy in range(o_y1 + 50, o_y2 - 20, 40):
        cv2.line(canvas, (o_x1 + 40, gy), (o_x2 - 20, gy), COLOR_GRID, 1, cv2.LINE_AA)

    # Draw simulated square-wave / pulse signal frames across oscilloscope
    ox_start = o_x1 + 50
    signal_y1 = o_y1 + 110  # Control stream (0x120)
    signal_y2 = o_y1 + 180  # Feedback stream (0x220)

    _draw_text(canvas, "ID: 0x120 (CTRL)", (o_x1 + 15, signal_y1 - 10), 0.40, COLOR_TRACE_CTRL, 1)
    _draw_text(canvas, "ID: 0x220 (FB)", (o_x1 + 15, signal_y2 - 10), 0.40, COLOR_TRACE_FB, 1)

    # Generate synthetic pulse representations
    for pulse_x in range(ox_start, o_x2 - 40, 50):
        # 0x120 Control pulses
        cv2.line(canvas, (pulse_x, signal_y1), (pulse_x + 15, signal_y1), COLOR_TRACE_CTRL, 2)
        cv2.line(
            canvas, (pulse_x + 15, signal_y1), (pulse_x + 15, signal_y1 - 30), COLOR_TRACE_CTRL, 2
        )
        cv2.line(
            canvas,
            (pulse_x + 15, signal_y1 - 30),
            (pulse_x + 35, signal_y1 - 30),
            COLOR_TRACE_CTRL,
            2,
        )
        cv2.line(
            canvas, (pulse_x + 35, signal_y1 - 30), (pulse_x + 35, signal_y1), COLOR_TRACE_CTRL, 2
        )

    for pulse_x in range(ox_start, o_x2 - 40, 30):
        # 0x220 Feedback pulses
        cv2.line(canvas, (pulse_x, signal_y2), (pulse_x + 10, signal_y2), COLOR_TRACE_FB, 2)
        cv2.line(
            canvas, (pulse_x + 10, signal_y2), (pulse_x + 10, signal_y2 - 25), COLOR_TRACE_FB, 2
        )
        cv2.line(
            canvas,
            (pulse_x + 10, signal_y2 - 25),
            (pulse_x + 22, signal_y2 - 25),
            COLOR_TRACE_FB,
            2,
        )
        cv2.line(
            canvas, (pulse_x + 22, signal_y2 - 25), (pulse_x + 22, signal_y2), COLOR_TRACE_FB, 2
        )

    # 4. Bus Load History / Spectrum (Bottom Left Panel)
    b_x1, b_y1, b_x2, b_y2 = 20, 450, 800, 700
    cv2.rectangle(canvas, (b_x1, b_y1), (b_x2, b_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (b_x1, b_y1), (b_x2, b_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "BUS LOAD PROFILE & BANDWIDTH SATURATION",
        (b_x1 + 15, b_y1 + 25),
        0.52,
        COLOR_TEXT_TITLE,
        1,
    )

    load_h = int((t.bus_load_percent / 100.0) * (b_y2 - b_y1 - 80))
    bar_y_top = b_y2 - 30 - load_h
    cv2.rectangle(
        canvas,
        (b_x1 + 60, bar_y_top),
        (b_x1 + 180, b_y2 - 30),
        COLOR_WARNING if t.bus_load_percent > 70 else COLOR_SUCCESS,
        -1,
    )
    _draw_text(
        canvas, f"{t.bus_load_percent:.1f}%", (b_x1 + 80, bar_y_top - 10), 0.55, COLOR_TEXT_BODY, 1
    )
    _draw_text(canvas, "Current Load", (b_x1 + 65, b_y2 - 12), 0.40, COLOR_TEXT_MUTED, 1)

    # Threshold mark
    thresh_y = b_y2 - 30 - int(0.85 * (b_y2 - b_y1 - 80))
    cv2.line(canvas, (b_x1 + 40, thresh_y), (b_x2 - 40, thresh_y), COLOR_ALERT, 1, cv2.LINE_AA)
    _draw_text(canvas, "Warning Threshold (85%)", (b_x2 - 230, thresh_y - 5), 0.38, COLOR_ALERT, 1)

    # 5. Right Sidebar: Target ECU Health & Fault Injection Log
    s_x1, s_y1, s_x2, s_y2 = 820, 180, width - 20, 700
    cv2.rectangle(canvas, (s_x1, s_y1), (s_x2, s_y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (s_x1, s_y1), (s_x2, s_y2), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas, "TARGET ECU HEALTH & FAULT LOG", (s_x1 + 15, s_y1 + 28), 0.52, COLOR_TEXT_TITLE, 1
    )

    curr_y = s_y1 + 65
    for ecu, health_val in report.ecu_health_status.items():
        col = (
            COLOR_SUCCESS
            if health_val == "nominal"
            else (COLOR_WARNING if health_val == "degraded" else COLOR_ALERT)
        )
        cv2.circle(canvas, (s_x1 + 25, curr_y - 5), 6, col, -1)
        _draw_text(
            canvas,
            f"ECU: {ecu:12s} [{health_val.upper()}]",
            (s_x1 + 45, curr_y),
            0.45,
            COLOR_TEXT_BODY,
            1,
        )
        curr_y += 35

    cv2.line(canvas, (s_x1 + 15, curr_y), (s_x2 - 15, curr_y), COLOR_CARD_BORDER, 1)
    curr_y += 25
    _draw_text(canvas, "FAULT INJECTION LOG:", (s_x1 + 15, curr_y), 0.48, COLOR_TEXT_MUTED, 1)
    curr_y += 25

    log_entries = report.fault_injection_log[-6:]  # Show last 6 entries
    if not log_entries:
        _draw_text(
            canvas,
            "No injected faults. Bus clear.",
            (s_x1 + 20, curr_y + 10),
            0.42,
            COLOR_SUCCESS,
            1,
        )
    else:
        for entry in log_entries:
            t_s = float(entry.get("time_s", 0.0))
            ecu_s = str(entry.get("ecu", "ECU"))
            fault_s = str(entry.get("fault", "none"))
            f_str = f"t={t_s:.2f}s | {ecu_s} | {fault_s}"
            _draw_text(canvas, f_str, (s_x1 + 20, curr_y), 0.38, COLOR_WARNING, 1)
            curr_y += 24

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
