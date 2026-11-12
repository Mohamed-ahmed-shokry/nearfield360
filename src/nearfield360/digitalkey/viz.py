"""Tactical 1280x720 HUD dashboard visualizer for CCC Digital Key 3.0 & UWB proximity interlock."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.digitalkey.models import (
    DevicePosition3D,
    DigitalKeyReport,
    HandoverState,
    ProximityZone,
    UWBAnchor,
)
from nearfield360.digitalkey.solver import get_default_uwb_anchors

# Tactical BGR color palette
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_GRID_LINE = (42, 48, 58)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_ACCENT_CYAN = (230, 210, 40)
COLOR_STATE_GREEN = (60, 210, 95)
COLOR_STATE_AMBER = (40, 180, 240)
COLOR_STATE_ORANGE = (30, 120, 245)
COLOR_STATE_RED = (50, 50, 240)
COLOR_ACCENT_MAGENTA = (210, 80, 210)


def _draw_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


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
    cv2.rectangle(img, (x1, y1), (x2, y1 + 28), COLOR_CARD_BG, -1)
    cv2.line(img, (x1, y1 + 28), (x2, y1 + 28), COLOR_CARD_BORDER, 1)
    _draw_text(img, title, (x1 + 10, y1 + 19), 0.48, COLOR_ACCENT_CYAN, 1)


def _draw_spatial_view(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pos: DevicePosition3D,
    anchors: Sequence[UWBAnchor],
) -> None:
    """Render top-down vehicle layout, concentric proximity zones, and device track."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2 + 28) // 2
    scale = 32.0  # pixels per meter

    # Draw grid
    for r_m in (1.5, 3.0, 5.0, 8.0):
        r_px = int(r_m * scale)
        color = (
            COLOR_STATE_RED
            if r_m == 1.5
            else (
                COLOR_STATE_ORANGE
                if r_m == 3.0
                else (COLOR_STATE_AMBER if r_m == 5.0 else COLOR_GRID_LINE)
            )
        )
        cv2.circle(img, (cx, cy), r_px, color, 1, cv2.LINE_AA)
        _draw_text(img, f"{r_m:.1f}m", (cx + r_px + 2, cy - 2), 0.32, color, 1)

    # Draw vehicle footprint (length 4.0m = 128px, width 1.8m = 58px)
    vw, vh = int(1.8 * scale), int(4.0 * scale)
    vx1, vy1 = cx - vw // 2, cy - vh // 2
    vx2, vy2 = cx + vw // 2, cy + vh // 2
    cv2.rectangle(img, (vx1, vy1), (vx2, vy2), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (vx1, vy1), (vx2, vy2), COLOR_ACCENT_CYAN, 2)
    # Windshield orientation line
    cv2.line(img, (vx1, vy1 + 25), (vx2, vy1 + 25), COLOR_ACCENT_CYAN, 1)
    _draw_text(img, "AVP VEHICLE", (cx - 36, cy + 5), 0.38, COLOR_TEXT_MUTED, 1)

    # Draw UWB anchors
    for a in anchors:
        ax_px = int(cx + (a.y_m * scale))  # y is lateral in vehicle frame
        ay_px = int(cy - (a.x_m * scale))  # x is longitudinal forward
        cv2.circle(img, (ax_px, ay_px), 4, COLOR_ACCENT_MAGENTA, -1)
        _draw_text(img, a.anchor_id, (ax_px + 6, ay_px - 2), 0.35, COLOR_ACCENT_MAGENTA, 1)

    # Draw Device Position
    dx_px = int(cx + (pos.y_m * scale))
    dy_px = int(cy - (pos.x_m * scale))
    # Distance vector
    cv2.line(img, (cx, cy), (dx_px, dy_px), COLOR_STATE_GREEN, 1, cv2.LINE_AA)
    # Device beacon
    dev_color = COLOR_STATE_RED if pos.zone == ProximityZone.TOUCH else COLOR_STATE_GREEN
    cv2.circle(img, (dx_px, dy_px), 7, dev_color, -1, cv2.LINE_AA)
    cv2.circle(img, (dx_px, dy_px), 12, dev_color, 1, cv2.LINE_AA)
    label = f"KEY FOB ({pos.distance_m:.2f}m) [{pos.zone.value}]"
    _draw_text(img, label, (dx_px + 14, dy_px + 4), 0.40, COLOR_TEXT_TITLE, 1)


def _draw_uwb_status(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    report: DigitalKeyReport,
) -> None:
    """Render UWB physical layer telemetry and anti-relay status card."""
    del y2
    # Relay Defense Banner
    banner_y = y1 + 38
    if report.relay_attack_detected:
        cv2.rectangle(img, (x1 + 12, banner_y), (x2 - 12, banner_y + 36), COLOR_STATE_RED, -1)
        _draw_text(
            img,
            "ALERT: UWB RELAY ATTACK DETECTED // DISTANCE BOUNDING VIOLATION",
            (x1 + 22, banner_y + 24),
            0.48,
            (255, 255, 255),
            2,
        )
    else:
        cv2.rectangle(img, (x1 + 12, banner_y), (x2 - 12, banner_y + 36), (20, 60, 30), -1)
        cv2.rectangle(img, (x1 + 12, banner_y), (x2 - 12, banner_y + 36), COLOR_STATE_GREEN, 1)
        _draw_text(
            img,
            "STATUS: IEEE 802.15.4z STS VERIFIED // ANTI-RELAY DEFENSE ACTIVE",
            (x1 + 22, banner_y + 24),
            0.48,
            COLOR_STATE_GREEN,
            1,
        )

    # Telemetry Table Header
    ty = banner_y + 60
    _draw_text(img, "ANCHOR", (x1 + 15, ty), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(img, "RANGE (M)", (x1 + 110, ty), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(img, "ToF (NS)", (x1 + 210, ty), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(img, "SNR (DB)", (x1 + 310, ty), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(img, "STS VALID", (x1 + 410, ty), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(img, "AoA AZIMUTH", (x1 + 520, ty), 0.40, COLOR_TEXT_MUTED, 1)
    cv2.line(img, (x1 + 12, ty + 8), (x2 - 12, ty + 8), COLOR_GRID_LINE, 1)

    row_y = ty + 30
    for m in report.uwb_measurements[:6]:
        _draw_text(img, m.anchor_id, (x1 + 15, row_y), 0.42, COLOR_TEXT_TITLE, 1)
        _draw_text(img, f"{m.distance_m:.2f} m", (x1 + 110, row_y), 0.42, COLOR_TEXT_BODY, 1)
        _draw_text(img, f"{m.tof_ns:.2f} ns", (x1 + 210, row_y), 0.42, COLOR_TEXT_BODY, 1)
        _draw_text(img, f"{m.snr_db:.1f} dB", (x1 + 310, row_y), 0.42, COLOR_TEXT_BODY, 1)
        sts_str = "PASS" if m.sts_valid else "FAIL"
        sts_col = COLOR_STATE_GREEN if m.sts_valid else COLOR_STATE_RED
        _draw_text(img, sts_str, (x1 + 410, row_y), 0.42, sts_col, 1)
        aoa_str = f"{m.aoa_azimuth_deg:+.1f} deg" if m.aoa_azimuth_deg is not None else "--"
        _draw_text(img, aoa_str, (x1 + 520, row_y), 0.42, COLOR_ACCENT_CYAN, 1)
        row_y += 24


def _draw_interlock_status(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    report: DigitalKeyReport,
) -> None:
    """Render CCC Digital Key delegation info and safety standstill interlock status."""
    del y2
    # Standstill Interlock Card
    iy = y1 + 38
    interlock_col = COLOR_STATE_RED if report.interlock.standstill_command else COLOR_STATE_GREEN
    cmd_text = (
        "STANDSTILL COMMAND: ACTIVE (HALT)"
        if report.interlock.standstill_command
        else "STANDSTILL COMMAND: CLEAR (AUTONOMOUS NAV OK)"
    )
    cv2.rectangle(img, (x1 + 12, iy), (x2 - 12, iy + 34), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (x1 + 12, iy), (x2 - 12, iy + 34), interlock_col, 1)
    _draw_text(img, cmd_text, (x1 + 20, iy + 22), 0.46, interlock_col, 1)

    # Details
    dy = iy + 55
    _draw_text(img, f"REASON: {report.interlock.reason}", (x1 + 15, dy), 0.38, COLOR_TEXT_BODY, 1)
    dist_str = f"{report.interlock.detected_distance_m:.2f} m"
    _draw_text(
        img,
        f"ZONE: {report.proximity_zone.value}  |  DIST: {dist_str}",
        (x1 + 15, dy + 22),
        0.38,
        COLOR_TEXT_MUTED,
        1,
    )

    # Delegation Token Info
    token_y = dy + 52
    cv2.rectangle(img, (x1 + 12, token_y), (x2 - 12, token_y + 110), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (x1 + 12, token_y), (x2 - 12, token_y + 110), COLOR_CARD_BORDER, 1)
    _draw_text(
        img,
        "CCC DIGITAL KEY RELEASE 3.0 TOKEN",
        (x1 + 20, token_y + 20),
        0.42,
        COLOR_ACCENT_CYAN,
        1,
    )

    if report.delegation_token:
        tok = report.delegation_token
        scopes_str = ", ".join(s.value for s in tok.scopes)
        _draw_text(
            img,
            f"TOKEN ID: {tok.token_id}  |  USER: {tok.user_id}  |  DEV: {tok.device_id}",
            (x1 + 20, token_y + 44),
            0.36,
            COLOR_TEXT_BODY,
            1,
        )
        _draw_text(
            img, f"SCOPES: {scopes_str}", (x1 + 20, token_y + 66), 0.36, COLOR_STATE_GREEN, 1
        )
        _draw_text(
            img,
            f"EXPIRES: {tok.expires_at_utc[:19]} UTC",
            (x1 + 20, token_y + 88),
            0.36,
            COLOR_TEXT_MUTED,
            1,
        )
    else:
        _draw_text(
            img,
            "NO ACTIVE DELEGATION TOKEN (UNPAIRED / MANUAL DRIVER)",
            (x1 + 20, token_y + 55),
            0.40,
            COLOR_TEXT_MUTED,
            1,
        )


def render_digitalkey_dashboard(
    report: DigitalKeyReport,
    output_path: str | Path | None = None,
) -> NDArray[np.uint8]:
    """Render a tactical 1280x720 HUD dashboard for CCC Digital Key 3.0 and proximity interlock."""
    w, h = 1280, 720
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = COLOR_BG

    # 1. Header Bar
    cv2.rectangle(canvas, (0, 0), (w, 54), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (0, 54), (w, 54), COLOR_CARD_BORDER, 1)

    _draw_text(
        canvas,
        "NEARFIELD360 CCC DIGITAL KEY 3.0 // UWB RANGING & PROXIMITY INTERLOCK",
        (16, 34),
        0.58,
        COLOR_TEXT_TITLE,
        2,
    )

    # Header badges (State & VIN)
    state_col = (
        COLOR_STATE_GREEN
        if report.handover_state in (HandoverState.AVP_DELEGATED, HandoverState.MANUAL_CONTROL)
        else COLOR_STATE_AMBER
    )
    _draw_text(
        canvas,
        f"STATE: {report.handover_state.value}",
        (880, 34),
        0.50,
        state_col,
        2,
    )
    _draw_text(canvas, f"VIN: {report.vin}", (1090, 34), 0.42, COLOR_TEXT_MUTED, 1)

    # 2. Left Panel: Spatial Top-Down View
    _draw_panel(
        canvas,
        15,
        64,
        570,
        705,
        "UWB SPATIAL POSITIONING & PROXIMITY ZONES (TOP-DOWN)",
    )
    anchors = get_default_uwb_anchors()
    _draw_spatial_view(canvas, 15, 64, 570, 705, report.device_position, anchors)

    # 3. Top-Right Panel: UWB & Anti-Relay Status
    _draw_panel(
        canvas,
        585,
        64,
        1265,
        380,
        "UWB PHYSICAL LAYER & ANTI-RELAY DEFENSE STATUS",
    )
    _draw_uwb_status(canvas, 585, 64, 1265, 380, report)

    # 4. Bottom-Right Panel: Delegation Token & Interlock Decision
    _draw_panel(
        canvas,
        585,
        392,
        1265,
        705,
        "CCC DIGITAL KEY DELEGATION & SAFETY INTERLOCK",
    )
    _draw_interlock_status(canvas, 585, 392, 1265, 705, report)

    # Save to disk if requested
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), canvas)

    return canvas
