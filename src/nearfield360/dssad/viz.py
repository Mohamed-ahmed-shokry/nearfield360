"""Tactical 1280x720 forensic HUD dashboard visualizer for UN R157 DSSAD & UN R160 EDR."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.dssad.exporter import DSSADExporter
from nearfield360.dssad.models import DSSADReport, EDRIncidentRecord, EventSeverity

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


def _render_waveforms(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    incident: EDRIncidentRecord,
) -> None:
    """Render synchronized pre/post crash telemetry curves with trigger point."""
    samples = incident.pre_trigger_telemetry + incident.post_trigger_telemetry
    if not samples:
        _draw_text(
            img, "NO TELEMETRY SAMPLES CAPTURED", (x1 + 40, y1 + 150), 0.6, COLOR_TEXT_MUTED, 1
        )
        return

    # Coordinate boundaries for plot area
    px1, py1, px2, py2 = x1 + 15, y1 + 35, x2 - 15, y2 - 15
    cv2.rectangle(img, (px1, py1), (px2, py2), (20, 23, 28), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), COLOR_CARD_BORDER, 1)

    # Grid lines
    for gy in range(py1 + 30, py2, 35):
        cv2.line(img, (px1, gy), (px2, gy), COLOR_GRID_LINE, 1)
    for gx in range(px1 + 60, px2, 70):
        cv2.line(img, (gx, py1), (gx, py2), COLOR_GRID_LINE, 1)

    # Trigger marker line at trigger index
    trigger_ns = incident.trigger_event.timestamp_ns
    timestamps = [s.timestamp_ns for s in samples]
    min_t, max_t = timestamps[0], timestamps[-1]
    span_t = max(max_t - min_t, 1)

    trigger_x = int(px1 + ((trigger_ns - min_t) / span_t) * (px2 - px1))
    trigger_x = max(px1, min(px2, trigger_x))
    cv2.line(img, (trigger_x, py1), (trigger_x, py2), COLOR_STATE_RED, 2)
    _draw_text(img, "T=0 (IMPACT/TRIGGER)", (trigger_x - 65, py1 + 15), 0.38, COLOR_STATE_RED, 1)

    # Plot velocity curve (scale 0-10 m/s)
    speeds = [s.speed_mps for s in samples]
    max_speed = max(max(speeds), 1.0)
    pts_speed = []
    for s in samples:
        sx = int(px1 + ((s.timestamp_ns - min_t) / span_t) * (px2 - px1))
        sy = int(py2 - (s.speed_mps / max_speed) * (py2 - py1 - 25) - 10)
        pts_speed.append((sx, sy))

    if len(pts_speed) > 1:
        cv2.polylines(img, [np.array(pts_speed, dtype=np.int32)], False, COLOR_ACCENT_CYAN, 2)

    # Plot longitudinal accel curve
    accels = [s.longitudinal_accel_mps2 for s in samples]
    max_accel_mag = max(max(abs(a) for a in accels), 5.0)
    mid_y = (py1 + py2) // 2
    pts_accel = []
    for s in samples:
        ax = int(px1 + ((s.timestamp_ns - min_t) / span_t) * (px2 - px1))
        # Positive accel up, negative down
        ay = int(mid_y - (s.longitudinal_accel_mps2 / max_accel_mag) * ((py2 - py1) // 2 - 20))
        pts_accel.append((ax, ay))

    if len(pts_accel) > 1:
        cv2.polylines(img, [np.array(pts_accel, dtype=np.int32)], False, COLOR_STATE_ORANGE, 2)

    # Legend
    cv2.line(img, (px1 + 10, py2 - 8), (px1 + 30, py2 - 8), COLOR_ACCENT_CYAN, 2)
    _draw_text(
        img, f"Speed (Max: {max_speed:.1f} m/s)", (px1 + 35, py2 - 5), 0.38, COLOR_TEXT_BODY, 1
    )

    cv2.line(img, (px1 + 180, py2 - 8), (px1 + 200, py2 - 8), COLOR_STATE_ORANGE, 2)
    _draw_text(
        img,
        f"Long Accel (Peak: {incident.peak_accel_g:.2f}g)",
        (px1 + 205, py2 - 5),
        0.38,
        COLOR_TEXT_BODY,
        1,
    )


def _render_event_log(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    report: DSSADReport,
) -> None:
    """Render recent discrete DSSAD event timeline."""
    events = report.events[-5:]  # last 5 events
    y_cursor = y1 + 45

    if not events:
        _draw_text(img, "NO DISCRETE EVENTS IN LOG", (x1 + 30, y_cursor), 0.45, COLOR_TEXT_MUTED, 1)
        return

    for evt in reversed(events):
        # Severity color tag
        col = COLOR_STATE_GREEN
        if evt.severity == EventSeverity.FATAL:
            col = COLOR_STATE_RED
        elif evt.severity == EventSeverity.CRITICAL:
            col = COLOR_STATE_ORANGE
        elif evt.severity == EventSeverity.WARNING:
            col = COLOR_STATE_AMBER

        cv2.circle(img, (x1 + 20, y_cursor - 4), 5, col, -1)
        _draw_text(img, f"[{evt.sequence_id:03d}]", (x1 + 32, y_cursor), 0.42, COLOR_TEXT_MUTED, 1)
        _draw_text(img, evt.event_type.value, (x1 + 80, y_cursor), 0.45, col, 1)
        _draw_text(
            img, f"({evt.trigger_subsystem})", (x1 + 320, y_cursor), 0.40, COLOR_TEXT_MUTED, 1
        )
        y_cursor += 16
        _draw_text(img, evt.description[:75], (x1 + 32, y_cursor), 0.38, COLOR_TEXT_BODY, 1)
        y_cursor += 22


def _render_incident_card(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    incident: EDRIncidentRecord | None,
) -> None:
    """Render incident impact metrics card."""
    if incident is None:
        _draw_text(img, "NO FROZEN INCIDENT RECORDED", (x1 + 20, y1 + 80), 0.5, COLOR_TEXT_MUTED, 1)
        return

    y = y1 + 48
    _draw_text(img, f"Incident ID: {incident.incident_id}", (x1 + 15, y), 0.50, COLOR_TEXT_TITLE, 1)
    y += 24
    _draw_text(
        img,
        f"Trigger Event: {incident.trigger_event.event_type.value}",
        (x1 + 15, y),
        0.45,
        COLOR_STATE_ORANGE,
        1,
    )
    y += 24
    _draw_text(
        img,
        f"Severity: {incident.trigger_event.severity.value}",
        (x1 + 15, y),
        0.45,
        COLOR_STATE_RED,
        1,
    )
    y += 28

    # Metric boxes: Delta-V and Peak-G
    cv2.rectangle(img, (x1 + 15, y), (x1 + 195, y + 55), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (x1 + 15, y), (x1 + 195, y + 55), COLOR_CARD_BORDER, 1)
    _draw_text(img, "DELTA-V (IMPACT)", (x1 + 25, y + 20), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(
        img, f"{incident.delta_v_mps:.2f} m/s", (x1 + 25, y + 46), 0.65, COLOR_ACCENT_CYAN, 2
    )

    cv2.rectangle(img, (x1 + 215, y), (x1 + 395, y + 55), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (x1 + 215, y), (x1 + 395, y + 55), COLOR_CARD_BORDER, 1)
    _draw_text(img, "PEAK ACCELERATION", (x1 + 225, y + 20), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(img, f"{incident.peak_accel_g:.2f} g", (x1 + 225, y + 46), 0.65, COLOR_STATE_RED, 2)

    y += 75
    snap = incident.trigger_event.snapshot
    _draw_text(img, f"ODD State: {snap.odd_state}", (x1 + 15, y), 0.42, COLOR_TEXT_BODY, 1)
    y += 20
    _draw_text(
        img,
        f"AEB Active: {'YES (EMERGENCY)' if snap.aeb_active else 'NO'}",
        (x1 + 15, y),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    y += 20
    dtc_str = ", ".join(snap.active_dtcs) if snap.active_dtcs else "NONE"
    _draw_text(img, f"Active DTCs: {dtc_str}", (x1 + 15, y), 0.42, COLOR_TEXT_BODY, 1)
    y += 22
    lock_badge = "WRITE-ONCE NON-VOLATILE LOCKED [RO]" if incident.is_locked else "UNLOCKED"
    _draw_text(img, lock_badge, (x1 + 15, y), 0.44, COLOR_STATE_GREEN, 1)


def _render_audit_card(
    img: NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    report: DSSADReport,
    incident: EDRIncidentRecord | None,
) -> None:
    """Render cryptographic audit and UDS diagnostic frame readout."""
    y = y1 + 45
    status_color = COLOR_STATE_GREEN if report.chain_verified else COLOR_STATE_RED
    status_text = "VERIFIED / INTACT (UN R157 COMPLIANT)" if report.chain_verified else "CORRUPTED"
    _draw_text(img, f"Chain Status: {status_text}", (x1 + 15, y), 0.48, status_color, 1)
    y += 22
    _draw_text(
        img,
        f"Total Events Chained: {report.total_events_recorded}",
        (x1 + 15, y),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    y += 20
    _draw_text(
        img,
        f"Total Incidents Frozen: {report.total_incidents_frozen}",
        (x1 + 15, y),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    y += 25

    _draw_text(
        img,
        "ISO 14229 UDS DID (0x19ED) POSITIVE RESPONSE:",
        (x1 + 15, y),
        0.42,
        COLOR_ACCENT_CYAN,
        1,
    )
    y += 20

    if incident is not None:
        uds_data = DSSADExporter.export_uds_did_frame(incident)
        raw_hex = uds_data["raw_hex"]
        # Format in groups of 4 bytes
        chunks = [raw_hex[i : i + 8] for i in range(0, min(len(raw_hex), 48), 8)]
        formatted_hex = " ".join(chunks)
        cv2.rectangle(img, (x1 + 15, y), (x2 - 15, y + 40), (15, 18, 22), -1)
        cv2.rectangle(img, (x1 + 15, y), (x2 - 15, y + 40), COLOR_CARD_BORDER, 1)
        _draw_text(img, formatted_hex, (x1 + 25, y + 25), 0.45, COLOR_STATE_GREEN, 1)
        y += 55
        digest_preview = incident.tamper_chain_digest[:32] + "..."
        _draw_text(
            img, f"Root Chain Digest: {digest_preview}", (x1 + 15, y), 0.38, COLOR_TEXT_MUTED, 1
        )
    else:
        _draw_text(
            img, "NO INCIDENT DATA FOR UDS READOUT", (x1 + 15, y + 20), 0.42, COLOR_TEXT_MUTED, 1
        )


def render_dssad_dashboard(
    report: DSSADReport, output_path: str | Path | None = None
) -> NDArray[np.uint8]:
    """Render full 1280x720 tactical forensic dashboard for DSSAD & EDR regulatory inspection."""
    img = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # Top Header
    cv2.rectangle(img, (0, 0), (1280, 55), COLOR_PANEL_BG, -1)
    cv2.line(img, (0, 55), (1280, 55), COLOR_CARD_BORDER, 1)

    _draw_text(
        img,
        "NEARFIELD360 DSSAD & EDR FORENSIC INVESTIGATION HUD",
        (20, 26),
        0.62,
        COLOR_TEXT_TITLE,
        2,
    )
    _draw_text(
        img,
        "UN R157 AUTOMATED DRIVING STORAGE & UN R160 EVENT DATA RECORDER | REGULATORY AUDIT",
        (20, 46),
        0.38,
        COLOR_TEXT_MUTED,
        1,
    )

    # Header Badges
    chain_badge_col = COLOR_STATE_GREEN if report.chain_verified else COLOR_STATE_RED
    chain_badge_txt = "AUDIT CHAIN: VERIFIED" if report.chain_verified else "AUDIT CHAIN: BREACHED"
    cv2.rectangle(img, (960, 14), (1140, 42), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (960, 14), (1140, 42), chain_badge_col, 1)
    _draw_text(img, chain_badge_txt, (970, 32), 0.40, chain_badge_col, 1)

    cv2.rectangle(img, (1150, 14), (1260, 42), COLOR_CARD_BG, -1)
    cv2.rectangle(img, (1150, 14), (1260, 42), COLOR_ACCENT_CYAN, 1)
    _draw_text(img, report.vehicle_id, (1160, 32), 0.42, COLOR_TEXT_TITLE, 1)

    incident = report.incidents[-1] if report.incidents else None

    # Panel 1: Pre/Post-Crash Telemetry Waveforms (top-left)
    _draw_panel(img, 20, 70, 820, 450, "PRE/POST-CRASH TELEMETRY WAVEFORMS (-5.0s to +2.0s)")
    if incident is not None:
        _render_waveforms(img, 20, 70, 820, 450, incident)
    else:
        _draw_text(img, "NO INCIDENT TELEMETRY TO RENDER", (250, 250), 0.6, COLOR_TEXT_MUTED, 1)

    # Panel 2: UN R157 Event Timeline & Cryptographic Log (bottom-left)
    _draw_panel(img, 20, 465, 820, 700, "UN R157 DISCRETE EVENT LOG & CRYPTOGRAPHIC TIMELINE")
    _render_event_log(img, 20, 465, 820, 700, report)

    # Panel 3: Incident Impact Metrics & Chassis Telemetry (top-right)
    _draw_panel(img, 835, 70, 1260, 390, "INCIDENT IMPACT METRICS & CHASSIS TELEMETRY")
    _render_incident_card(img, 835, 70, 1260, 390, incident)

    # Panel 4: Cryptographic Audit Trail & UDS Readout (bottom-right)
    _draw_panel(img, 835, 405, 1260, 700, "CRYPTOGRAPHIC AUDIT & ISO 14229 UDS DID (0x19ED)")
    _render_audit_card(img, 835, 405, 1260, 700, report, incident)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), img)

    return img
