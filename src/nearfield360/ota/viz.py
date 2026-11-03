"""Tactical UN R156 OTA software update and dual-slot A/B partition visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.ota.models import (
    OTACampaignReport,
    PartitionSlot,
    PreflightStatus,
    SlotMetadata,
    UpdateState,
)

# Tactical color palette (BGR format for OpenCV)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_COMMITTED = (60, 210, 95)  # Green
COLOR_IN_PROGRESS = (240, 160, 40)  # Blue
COLOR_PENDING = (40, 180, 240)  # Amber
COLOR_FAILED = (50, 50, 235)  # Bright Red
COLOR_ROLLED_BACK = (40, 40, 160)  # Crimson


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


def _get_state_color(state: UpdateState) -> tuple[int, int, int]:
    if state == UpdateState.COMMITTED:
        return COLOR_COMMITTED
    if state in (UpdateState.APPLYING, UpdateState.TESTING, UpdateState.DOWNLOADING):
        return COLOR_IN_PROGRESS
    if state in (UpdateState.STAGED, UpdateState.PREFLIGHT, UpdateState.VERIFYING):
        return COLOR_PENDING
    if state == UpdateState.ROLLED_BACK:
        return COLOR_ROLLED_BACK
    return COLOR_FAILED


def _render_slot_card(
    canvas: NDArray[np.uint8],
    slot_label: str,
    slot_meta: SlotMetadata | None,
    x: int,
    y: int,
    w: int,
) -> None:
    is_active = slot_meta.is_active if slot_meta else False
    border_col = COLOR_COMMITTED if is_active else COLOR_CARD_BORDER

    cv2.rectangle(canvas, (x, y), (x + w, y + 215), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 215), border_col, 2)

    _draw_text(canvas, slot_label, (x + 20, y + 32), 0.52, COLOR_TEXT_TITLE, 2)
    badge_x = x + w - 140
    if is_active:
        cv2.rectangle(canvas, (badge_x, y + 15), (x + w - 20, y + 42), COLOR_COMMITTED, -1)
        _draw_text(canvas, "ACTIVE (BOOT)", (badge_x + 8, y + 34), 0.36, (0, 0, 0), 2)
    else:
        cv2.rectangle(canvas, (badge_x, y + 15), (x + w - 20, y + 42), COLOR_CARD_BORDER, -1)
        _draw_text(canvas, "STANDBY", (badge_x + 18, y + 34), 0.36, COLOR_TEXT_MUTED, 1)

    if slot_meta:
        v_str = f"Firmware Version : {slot_meta.version}"
        _draw_text(canvas, v_str, (x + 20, y + 70), 0.42, COLOR_TEXT_BODY, 1)

        rev_str = f"Monotonic Rev    : {slot_meta.monotonic_revision}"
        _draw_text(canvas, rev_str, (x + 20, y + 100), 0.42, COLOR_TEXT_BODY, 1)

        rx_col = COLOR_COMMITTED if is_active else COLOR_TEXT_MUTED
        rx_str = f"RXSWIN Identifier: {slot_meta.rxswin}"
        _draw_text(canvas, rx_str, (x + 20, y + 130), 0.42, rx_col, 1)

        h_str = f"SHA-256 Digest   : {slot_meta.sha256_hash[:28]}..."
        _draw_text(canvas, h_str, (x + 20, y + 160), 0.38, COLOR_TEXT_MUTED, 1)

        b_status = "YES" if slot_meta.is_bootable else "NO"
        st_str = f"Bootable: {b_status}  |  Boots: {slot_meta.boot_count}"
        _draw_text(canvas, st_str, (x + 20, y + 190), 0.40, COLOR_TEXT_MUTED, 1)


def render_ota_dashboard(
    report: OTACampaignReport,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a 1280x720 tactical UN R156 SUMS and A/B partition dashboard PNG artifact."""
    width, height = 1280, 720
    canvas: NDArray[np.uint8] = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Banner
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_CARD_BORDER, 1)

    _draw_text(
        canvas,
        "NEARFIELD360 // UN R156 SOFTWARE UPDATE MANAGEMENT SYSTEM",
        (35, 46),
        0.60,
        COLOR_TEXT_TITLE,
        2,
    )

    state_col = _get_state_color(report.active_state)
    state_label = f"[{report.active_state.value.upper()}]"
    cv2.rectangle(canvas, (860, 24), (1070, 56), state_col, -1)
    _draw_text(canvas, state_label, (875, 46), 0.48, (0, 0, 0), 2)

    _draw_text(canvas, f"DUR={report.duration_s:.1f}s", (1095, 46), 0.48, COLOR_TEXT_BODY, 1)

    # 2. Left Panel: Dual-Slot A/B Partition Topology & RXSWIN Mapping
    p1_x, p1_y, p1_w, p1_h = 20, 80, 580, 620
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_CARD_BORDER, 1)
    header_left = f"A/B PARTITIONS (TARGET ECU: {report.target_ecu.upper()})"
    _draw_text(canvas, header_left, (p1_x + 15, p1_y + 30), 0.50, COLOR_TEXT_TITLE, 2)

    slot_a = report.partition_state.slots.get(PartitionSlot.SLOT_A.value)
    slot_b = report.partition_state.slots.get(PartitionSlot.SLOT_B.value)

    _render_slot_card(canvas, "PARTITION SLOT A", slot_a, p1_x + 15, p1_y + 55, p1_w - 30)
    _render_slot_card(canvas, "PARTITION SLOT B", slot_b, p1_x + 15, p1_y + 295, p1_w - 30)

    # Rollback summary pill
    target_boot = report.partition_state.next_boot_slot.value.upper()
    rb_txt = f"Rollbacks: {report.partition_state.rollback_count} | Target Boot: {target_boot}"
    _draw_text(canvas, rb_txt, (p1_x + 25, p1_y + 550), 0.42, COLOR_TEXT_MUTED, 1)

    # 3. Top Right Panel: Safety Preflight & Manifest Cryptography
    p2_x, p2_y, p2_w, p2_h = 620, 80, 640, 260
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "PREFLIGHT SAFETY INTERLOCKS & CRYPTOGRAPHY",
        (p2_x + 15, p2_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    preflight = report.preflight
    pre_col = COLOR_COMMITTED if preflight.status == PreflightStatus.PASSED else COLOR_FAILED
    gear_col = COLOR_COMMITTED if preflight.gear_is_park else COLOR_FAILED
    spd_col = COLOR_COMMITTED if preflight.speed_mps < 0.05 else COLOR_FAILED
    soc_col = COLOR_COMMITTED if preflight.battery_soc_pct >= 30.0 else COLOR_FAILED
    sfty_col = COLOR_COMMITTED if preflight.safety_supervisor_ready else COLOR_FAILED

    card_items = [
        ("PREFLIGHT STATUS", preflight.status.value.upper(), pre_col),
        ("GEAR SELECTOR", "PARK (P)" if preflight.gear_is_park else "NON-PARK", gear_col),
        ("VEHICLE SPEED", f"{preflight.speed_mps:.2f} m/s", spd_col),
        ("BATTERY SOC", f"{preflight.battery_soc_pct:.1f}%", soc_col),
        (
            "SAFETY SUPERVISOR",
            "NOMINAL" if preflight.safety_supervisor_ready else "FAULT",
            sfty_col,
        ),
        ("PAYLOAD SIZE", f"{report.manifest.payload_size_bytes / 1024:.1f} KB", COLOR_TEXT_BODY),
    ]

    grid_x, grid_y = p2_x + 15, p2_y + 50
    for idx, (label, val_str, val_col) in enumerate(card_items):
        cx = grid_x + (idx % 3) * 205
        cy = grid_y + (idx // 3) * 95
        cv2.rectangle(canvas, (cx, cy), (cx + 195, cy + 85), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + 195, cy + 85), COLOR_CARD_BORDER, 1)
        _draw_text(canvas, label, (cx + 12, cy + 28), 0.36, COLOR_TEXT_MUTED, 1)
        _draw_text(canvas, val_str, (cx + 12, cy + 62), 0.55, val_col, 2)

    # 4. Bottom Right Panel: Campaign Audit Trail Log
    p3_x, p3_y, p3_w, p3_h = 620, 360, 640, 340
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "UN R156 CAMPAIGN AUDIT TRAIL LOG",
        (p3_x + 15, p3_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    log_y = p3_y + 65
    for record in report.audit_trail[-7:]:
        st_color = _get_state_color(record.to_state)
        f_s = record.from_state.value.upper()
        t_s = record.to_state.value.upper()
        line_header = f"[{record.timestamp_s:.1f}s] {f_s} -> {t_s}:"
        _draw_text(canvas, line_header, (p3_x + 20, log_y), 0.38, st_color, 1)
        _draw_text(canvas, record.message[:64], (p3_x + 220, log_y), 0.38, COLOR_TEXT_BODY, 1)
        log_y += 36

    # Save to disk if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
