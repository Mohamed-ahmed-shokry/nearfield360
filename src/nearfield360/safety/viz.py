"""Tactical ASIL-D safety supervisor HUD and cyclic watchdog telemetry visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.safety.models import (
    HeartbeatStatus,
    SafetyState,
    SafetySupervisorReport,
    ViolationSeverity,
)

# Tactical color palette (BGR format for OpenCV)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_GRID = (45, 50, 60)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_NORMAL = (60, 210, 95)  # Green
COLOR_BOOT = (240, 160, 40)  # Blue
COLOR_DEGRADED = (40, 180, 240)  # Amber
COLOR_EMERGENCY = (50, 50, 235)  # Bright Red
COLOR_LATCH = (40, 40, 140)  # Dark Red


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


def _get_state_color(state: SafetyState) -> tuple[int, int, int]:
    if state == SafetyState.NORMAL_OPERATION:
        return COLOR_NORMAL
    if state == SafetyState.BOOT_SELF_TEST:
        return COLOR_BOOT
    if state == SafetyState.DEGRADED_PERCEPTION:
        return COLOR_DEGRADED
    if state == SafetyState.EMERGENCY_SAFE_STOP:
        return COLOR_EMERGENCY
    return COLOR_LATCH


def _get_status_color(status: HeartbeatStatus) -> tuple[int, int, int]:
    if status == HeartbeatStatus.HEALTHY:
        return COLOR_NORMAL
    if status == HeartbeatStatus.WARN_JITTER:
        return COLOR_DEGRADED
    if status == HeartbeatStatus.MISSED:
        return (40, 140, 240)
    return COLOR_EMERGENCY


def _get_severity_color(severity: ViolationSeverity) -> tuple[int, int, int]:
    if severity == ViolationSeverity.INFO:
        return COLOR_BOOT
    if severity == ViolationSeverity.WARNING:
        return COLOR_DEGRADED
    return COLOR_EMERGENCY


def render_safety_dashboard(
    report: SafetySupervisorReport,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a 1280x720 tactical ASIL-D functional safety HUD and watchdog dashboard."""
    width, height = 1280, 720
    canvas: NDArray[np.uint8] = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Banner
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_CARD_BORDER, 1)

    _draw_text(
        canvas,
        "NEARFIELD360 // ISO 26262 ASIL-D FUNCTIONAL SAFETY SUPERVISOR",
        (35, 46),
        0.65,
        COLOR_TEXT_TITLE,
        2,
    )

    state_color = _get_state_color(report.active_state)
    state_label = f"[{report.active_state.value.upper()}]"
    cv2.rectangle(canvas, (880, 24), (1120, 56), state_color, -1)
    _draw_text(canvas, state_label, (895, 46), 0.50, (0, 0, 0), 2)

    _draw_text(
        canvas,
        f"T={report.timestamp_s:.3f}s",
        (1140, 46),
        0.55,
        COLOR_TEXT_BODY,
        1,
    )

    # 2. Left Panel: Cyclic ECU Heartbeat Watchdog Matrix
    p1_x, p1_y, p1_w, p1_h = 20, 80, 560, 620
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "CYCLIC ECU WATCHDOG STATUS (ASIL-D DEADLINES)",
        (p1_x + 15, p1_y + 30),
        0.55,
        COLOR_TEXT_TITLE,
        2,
    )

    row_y = p1_y + 60
    for node_name, health in report.node_healths.items():
        cv2.rectangle(
            canvas,
            (p1_x + 15, row_y),
            (p1_x + p1_w - 15, row_y + 65),
            COLOR_CARD_BG,
            -1,
        )
        cv2.rectangle(
            canvas,
            (p1_x + 15, row_y),
            (p1_x + p1_w - 15, row_y + 65),
            COLOR_CARD_BORDER,
            1,
        )

        status_col = _get_status_color(health.status)
        cv2.circle(canvas, (p1_x + 35, row_y + 32), 8, status_col, -1)

        _draw_text(
            canvas,
            node_name.upper(),
            (p1_x + 55, row_y + 26),
            0.50,
            COLOR_TEXT_BODY,
            2,
        )
        metrics_str = (
            f"Latency: {health.latency_ms:.1f}ms  |  "
            f"Jitter: {health.jitter_ms:.1f}ms  |  "
            f"Misses: {health.consecutive_misses}"
        )
        _draw_text(
            canvas,
            metrics_str,
            (p1_x + 55, row_y + 50),
            0.40,
            COLOR_TEXT_MUTED,
            1,
        )

        badge_txt = health.status.value.upper()
        cv2.rectangle(
            canvas,
            (p1_x + p1_w - 130, row_y + 15),
            (p1_x + p1_w - 25, row_y + 45),
            status_col,
            1,
        )
        _draw_text(
            canvas,
            badge_txt,
            (p1_x + p1_w - 120, row_y + 35),
            0.40,
            status_col,
            1,
        )

        row_y += 75

    # 3. Top Right Panel: Actuator Safety Override & Kinematic Guard
    p2_x, p2_y, p2_w, p2_h = 600, 80, 660, 260
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "ACTUATOR SAFETY OVERRIDES & FAIL-SAFE LIMITS",
        (p2_x + 15, p2_y + 30),
        0.55,
        COLOR_TEXT_TITLE,
        2,
    )

    override = report.actuator_override
    card_items = [
        (
            "OVERRIDE ENGAGED",
            "ACTIVE" if override.is_override_active else "INACTIVE",
            COLOR_EMERGENCY if override.is_override_active else COLOR_NORMAL,
        ),
        (
            "MAX SPEED LIMIT",
            f"{override.max_permitted_speed_mps:.1f} m/s",
            COLOR_TEXT_BODY,
        ),
        (
            "TARGET DECELERATION",
            f"{override.target_deceleration_mps2:.2f} m/s²",
            COLOR_EMERGENCY if override.target_deceleration_mps2 > 0 else COLOR_TEXT_BODY,
        ),
        (
            "STEERING RACK CLAMP",
            "LOCKED" if override.steering_clamped else "NOMINAL",
            COLOR_EMERGENCY if override.steering_clamped else COLOR_NORMAL,
        ),
        (
            "HAZARD LIGHTS",
            "FLASHING" if override.emergency_hazard_lights else "OFF",
            COLOR_DEGRADED if override.emergency_hazard_lights else COLOR_TEXT_MUTED,
        ),
    ]

    grid_x, grid_y = p2_x + 15, p2_y + 50
    for idx, (label, val_str, val_col) in enumerate(card_items):
        cx = grid_x + (idx % 3) * 210
        cy = grid_y + (idx // 3) * 95
        cv2.rectangle(canvas, (cx, cy), (cx + 200, cy + 85), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + 200, cy + 85), COLOR_CARD_BORDER, 1)
        _draw_text(canvas, label, (cx + 12, cy + 28), 0.38, COLOR_TEXT_MUTED, 1)
        _draw_text(canvas, val_str, (cx + 12, cy + 62), 0.60, val_col, 2)

    # 4. Bottom Right Panel: Safety Transition Audit Log & Violations
    p3_x, p3_y, p3_w, p3_h = 600, 360, 660, 340
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "STATE TRANSITION AUDIT & PLAUSIBILITY VIOLATIONS",
        (p3_x + 15, p3_y + 30),
        0.55,
        COLOR_TEXT_TITLE,
        2,
    )

    # Render recent transitions
    log_y = p3_y + 60
    _draw_text(canvas, "RECENT TRANSITIONS:", (p3_x + 15, log_y), 0.45, COLOR_TEXT_MUTED, 1)
    log_y += 22

    for tr in report.transition_history[-3:]:
        tr_line = (
            f"[{tr.timestamp_s:.2f}s] {tr.from_state.value.upper()} -> "
            f"{tr.to_state.value.upper()}: {tr.reason[:48]}"
        )
        _draw_text(canvas, tr_line, (p3_x + 25, log_y), 0.40, COLOR_TEXT_BODY, 1)
        log_y += 24

    log_y += 10
    _draw_text(canvas, "ACTIVE VIOLATIONS:", (p3_x + 15, log_y), 0.45, COLOR_TEXT_MUTED, 1)
    log_y += 22

    if not report.active_violations:
        _draw_text(
            canvas,
            "No active kinematic or sensor violations.",
            (p3_x + 25, log_y),
            0.40,
            COLOR_NORMAL,
            1,
        )
    else:
        for viol in report.active_violations[-3:]:
            v_col = _get_severity_color(viol.severity)
            v_line = (
                f"[{viol.severity.value.upper()}] {viol.violation_type}: "
                f"{viol.description[:45]} (meas={viol.measured_value:.2f})"
            )
            _draw_text(canvas, v_line, (p3_x + 25, log_y), 0.40, v_col, 1)
            log_y += 24

    # Save to disk if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
