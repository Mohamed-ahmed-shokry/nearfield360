"""Tactical 1280x720 ODD HUD dashboard visualizer for ISO 21448 / ISO 23374."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.odd.models import (
    ODDReport,
    ODDState,
    SensorHealthStatus,
)

# Tactical BGR color palette
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_ACCENT_CYAN = (230, 210, 40)
COLOR_STATE_GREEN = (60, 210, 95)
COLOR_STATE_AMBER = (40, 180, 240)
COLOR_STATE_ORANGE = (30, 120, 245)
COLOR_STATE_RED = (50, 50, 240)


def _draw_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _get_odd_color(state: ODDState) -> tuple[int, int, int]:
    if state == ODDState.ODD_BREACH_MRM:
        return COLOR_STATE_RED
    if state == ODDState.CRITICAL_RESTRICTED:
        return COLOR_STATE_ORANGE
    if state == ODDState.ADVISORY_DEGRADED:
        return COLOR_STATE_AMBER
    return COLOR_STATE_GREEN


def _draw_panel(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    title: str,
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_CARD_BORDER, 1)
    cv2.rectangle(img, (x1, y1), (x2, y1 + 30), COLOR_CARD_BG, -1)
    cv2.line(img, (x1, y1 + 30), (x2, y1 + 30), COLOR_CARD_BORDER, 1)
    _draw_text(img, title, (x1 + 12, y1 + 20), 0.52, COLOR_ACCENT_CYAN, 1)


def _render_radar_chart(
    img: NDArray[np.uint8],
    center: tuple[int, int],
    radius: int,
    scores: list[float],
    labels: list[str],
) -> None:
    cx, cy = center
    num_axes = len(scores)
    angles = [i * 2.0 * math.pi / num_axes - math.pi / 2.0 for i in range(num_axes)]

    # Grid rings
    for ring_ratio in (0.25, 0.50, 0.75, 1.0):
        r = int(radius * ring_ratio)
        cv2.circle(img, (cx, cy), r, (45, 50, 60), 1)

    # Axis spokes and labels
    for ang, lbl in zip(angles, labels, strict=False):
        ex = int(cx + radius * math.cos(ang))
        ey = int(cy + radius * math.sin(ang))
        cv2.line(img, (cx, cy), (ex, ey), (55, 62, 75), 1)

        # Label offset
        lx = int(cx + (radius + 22) * math.cos(ang)) - 25
        ly = int(cy + (radius + 15) * math.sin(ang)) + 4
        _draw_text(img, lbl, (lx, ly), 0.38, COLOR_TEXT_MUTED, 1)

    # Filled data polygon
    pts: list[list[int]] = []
    for ang, score in zip(angles, scores, strict=False):
        clamped_score = max(0.05, min(1.0, score))
        px = int(cx + (radius * clamped_score) * math.cos(ang))
        py = int(cy + (radius * clamped_score) * math.sin(ang))
        pts.append([px, py])

    poly_pts = np.array(pts, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly_pts], (40, 180, 220))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [poly_pts], isClosed=True, color=COLOR_ACCENT_CYAN, thickness=2)


def render_odd_dashboard(
    report: ODDReport,
    output_path: str | Path = Path("artifacts/odd_hud.png"),
) -> Path:
    """Render a comprehensive 1280x720 tactical dark-mode ODD HUD dashboard."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    canvas: NDArray[np.uint8] = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header (y: 0 to 65)
    cv2.rectangle(canvas, (0, 0), (1280, 65), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (0, 65), (1280, 65), COLOR_CARD_BORDER, 1)

    _draw_text(
        canvas,
        "NEARFIELD360 // ISO 21448 SOTIF / ISO 23374 DYNAMIC ODD MONITOR",
        (25, 28),
        0.58,
        COLOR_TEXT_TITLE,
        2,
    )
    time_str = report.timestamp_iso[:19]
    _draw_text(
        canvas,
        f"VEHICLE: {report.vehicle_id}  |  REPORT: {report.report_id}  |  TIME: {time_str}",
        (25, 50),
        0.42,
        COLOR_TEXT_MUTED,
        1,
    )

    # ODD State Header Badge
    latest = report.latest_evaluation
    badge_color = _get_odd_color(latest.current_state)
    badge_text = latest.current_state.value.upper()
    cv2.rectangle(canvas, (1030, 15), (1255, 50), badge_color, -1)
    _draw_text(canvas, badge_text, (1045, 38), 0.55, (10, 15, 20), 2)

    # 2. Panel 1: Top-Left (x: 20 to 620, y: 75 to 380) - 5-Axis ODD Radar Chart
    _draw_panel(canvas, 20, 75, 620, 380, "5-AXIS ODD ENVELOPE RADAR")
    metrics = latest.metrics
    lux_score = min(1.0, metrics.ambient_lux / 500.0)
    vis_score = max(0.0, 1.0 - (metrics.precipitation_rate_mmph / 25.0))
    fric_score = max(0.0, min(1.0, metrics.friction_coeff / 0.85))
    conn_score = max(0.0, 1.0 - (metrics.v2x_latency_ms / 300.0))
    sensor_score = max(0.0, 1.0 - metrics.max_camera_occlusion)

    radar_scores = [lux_score, vis_score, fric_score, conn_score, sensor_score]
    radar_labels = ["LUX", "VISIBILITY", "FRICTION", "V2X LATENCY", "SENSORS"]
    _render_radar_chart(canvas, (320, 235), 105, radar_scores, radar_labels)

    # 3. Panel 2: Top-Right (x: 640 to 1260, y: 75 to 380) - Operational Governance & Limiters
    _draw_panel(canvas, 640, 75, 1260, 380, "ACTUATION GOVERNOR & SOTIF SAFETY STATUS")
    gx, gy = 660, 125
    _draw_text(canvas, "GOVERNED SPEED LIMIT:", (gx, gy), 0.48, COLOR_TEXT_MUTED, 1)
    speed_kmh = latest.governor_speed_limit_mps * 3.6
    speed_str = f"{latest.governor_speed_limit_mps:.1f} m/s ({speed_kmh:.1f} km/h)"
    _draw_text(canvas, speed_str, (gx + 220, gy), 0.52, COLOR_TEXT_TITLE, 2)

    # Speed gauge bar
    bar_w = int(280 * (latest.governor_speed_limit_mps / 4.5))
    cv2.rectangle(canvas, (gx + 220, gy + 10), (gx + 500, gy + 22), (40, 45, 55), -1)
    cv2.rectangle(canvas, (gx + 220, gy + 10), (gx + 220 + bar_w, gy + 22), badge_color, -1)

    gy += 45
    _draw_text(canvas, "SAFETY HEADWAY MULTIPLIER:", (gx, gy), 0.48, COLOR_TEXT_MUTED, 1)
    headway_str = f"{latest.safety_headway_multiplier:.2f}x"
    _draw_text(canvas, headway_str, (gx + 250, gy), 0.52, COLOR_ACCENT_CYAN, 2)

    gy += 35
    _draw_text(canvas, "ASIL-D MRM INTERLOCK:", (gx, gy), 0.48, COLOR_TEXT_MUTED, 1)
    mrm_str = "ACTIVE (SAFE STOP)" if latest.mrm_required else "NOMINAL (CLEAR)"
    mrm_col = COLOR_STATE_RED if latest.mrm_required else COLOR_STATE_GREEN
    _draw_text(canvas, mrm_str, (gx + 220, gy), 0.52, mrm_col, 2)

    gy += 40
    _draw_text(canvas, "ACTIVE LIMITING FACTORS:", (gx, gy), 0.48, COLOR_TEXT_MUTED, 1)
    lim_text = ", ".join(latest.active_limiters) if latest.active_limiters else "None (Nominal)"
    _draw_text(canvas, lim_text[:50], (gx + 220, gy), 0.45, COLOR_STATE_AMBER, 1)

    gy += 35
    _draw_text(canvas, "RECOMMENDATIONS:", (gx, gy), 0.48, COLOR_TEXT_MUTED, 1)
    for i, rec in enumerate(latest.recommendations[:2]):
        _draw_text(canvas, f"- {rec}", (gx + 20, gy + 22 + i * 22), 0.42, COLOR_TEXT_BODY, 1)

    # 4. Panel 3: Bottom-Left (x: 20 to 620, y: 395 to 705) - 4-Camera Health Matrix
    _draw_panel(canvas, 20, 395, 620, 705, "SURROUND CAMERA HEALTH & LENS CLEANLINESS")
    default_cams = ["FV", "RV", "MVL", "MVR"]
    for idx, cam_id in enumerate(default_cams):
        cy_base = 445 + idx * 60
        status = report.sensor_statuses.get(
            cam_id,
            SensorHealthStatus(sensor_id=cam_id, is_healthy=True, occlusion_ratio=0.05),
        )
        is_clean = status.is_healthy and status.occlusion_ratio < 0.3
        cam_col = COLOR_STATE_GREEN if is_clean else COLOR_STATE_ORANGE
        if status.occlusion_ratio >= 0.65 or not status.is_healthy:
            cam_col = COLOR_STATE_RED

        cv2.rectangle(canvas, (40, cy_base), (100, cy_base + 45), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (40, cy_base), (100, cy_base + 45), COLOR_CARD_BORDER, 1)
        _draw_text(canvas, cam_id, (52, cy_base + 28), 0.55, COLOR_TEXT_TITLE, 2)

        stat_label = f"STATUS: {'OK' if status.is_healthy else 'FAULT'}"
        _draw_text(canvas, stat_label, (120, cy_base + 18), 0.42, cam_col, 1)
        snr_label = f"SNR: {status.snr_db:.1f} dB"
        _draw_text(canvas, snr_label, (120, cy_base + 38), 0.40, COLOR_TEXT_MUTED, 1)

        # Occlusion progress bar
        occ_label = f"OCCLUSION: {status.occlusion_ratio * 100.0:.1f}%"
        _draw_text(canvas, occ_label, (260, cy_base + 18), 0.42, COLOR_TEXT_BODY, 1)
        occ_bar = int(180 * status.occlusion_ratio)
        cv2.rectangle(canvas, (260, cy_base + 26), (440, cy_base + 38), (40, 45, 55), -1)
        cv2.rectangle(canvas, (260, cy_base + 26), (260 + occ_bar, cy_base + 38), cam_col, -1)

    # 5. Panel 4: Bottom-Right (x: 640 to 1260, y: 395 to 705) - Temporal State History & KPIs
    _draw_panel(canvas, 640, 395, 1260, 705, "TEMPORAL ODD STATE TRANSITIONS & AUDIT LOG")
    hx, hy = 665, 445
    samples_str = f"TOTAL SAMPLES EVALUATED: {report.total_samples}"
    _draw_text(canvas, samples_str, (hx, hy), 0.48, COLOR_TEXT_TITLE, 1)

    mrm_col = COLOR_STATE_RED if report.mrm_triggered_count > 0 else COLOR_STATE_GREEN
    _draw_text(canvas, f"MRM STOPS: {report.mrm_triggered_count}", (hx + 320, hy), 0.48, mrm_col, 1)

    # History timeline bar
    hy += 40
    _draw_text(canvas, "STATE TRANSITION TIMELINE:", (hx, hy), 0.42, COLOR_TEXT_MUTED, 1)
    bar_x = hx
    bar_y = hy + 15
    bar_total_w = 540
    hist = report.history[-40:] if report.history else [latest]
    seg_w = max(2, bar_total_w // max(1, len(hist)))

    for i, h in enumerate(hist):
        c = _get_odd_color(h.current_state)
        x_start = bar_x + i * seg_w
        x_end = bar_x + (i + 1) * seg_w
        cv2.rectangle(canvas, (x_start, bar_y), (x_end, bar_y + 35), c, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_total_w, bar_y + 35), COLOR_CARD_BORDER, 1)

    # Legend
    hy += 80
    legend_items = [
        ("NOMINAL", COLOR_STATE_GREEN),
        ("ADVISORY", COLOR_STATE_AMBER),
        ("RESTRICTED", COLOR_STATE_ORANGE),
        ("BREACH MRM", COLOR_STATE_RED),
    ]
    for idx, (st_name, col) in enumerate(legend_items):
        lx = hx + idx * 135
        cv2.rectangle(canvas, (lx, hy), (lx + 15, hy + 15), col, -1)
        _draw_text(canvas, st_name, (lx + 22, hy + 12), 0.40, COLOR_TEXT_BODY, 1)

    cv2.imwrite(str(path), canvas)
    return path
