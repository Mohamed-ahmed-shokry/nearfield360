"""Tactical UN R155 CSMS, in-vehicle intrusion detection, and vSOC dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    VSOCReport,
)

# Tactical color palette (BGR format for OpenCV)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

# Threat / Severity Colors
COLOR_SEV_INFO = (200, 200, 200)  # Light Grey
COLOR_SEV_LOW = (60, 210, 95)  # Green
COLOR_SEV_MEDIUM = (40, 180, 240)  # Amber
COLOR_SEV_HIGH = (30, 120, 245)  # Orange
COLOR_SEV_CRITICAL = (50, 50, 240)  # Red


def _draw_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _get_severity_color(sev: IncidentSeverity) -> tuple[int, int, int]:
    if sev == IncidentSeverity.CRITICAL:
        return COLOR_SEV_CRITICAL
    if sev == IncidentSeverity.HIGH:
        return COLOR_SEV_HIGH
    if sev == IncidentSeverity.MEDIUM:
        return COLOR_SEV_MEDIUM
    if sev == IncidentSeverity.LOW:
        return COLOR_SEV_LOW
    return COLOR_SEV_INFO


def render_csms_dashboard(
    report: VSOCReport,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a tactical 1280x720 UN R155 CSMS / In-Vehicle IDS HUD dashboard to PNG."""
    canvas: NDArray[np.uint8] = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # 1. Top Header Banner
    cv2.rectangle(canvas, (0, 0), (1280, 55), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (0, 55), (1280, 55), COLOR_CARD_BORDER, 2)
    _draw_text(
        canvas,
        "UN R155 / ISO/SAE 21434 CYBERSECURITY MANAGEMENT SYSTEM (CSMS) & vSOC HUD",
        (25, 36),
        0.65,
        COLOR_TEXT_TITLE,
        2,
    )
    status_text = (
        "ACTIVE CONTAINMENT" if report.threat_assessment.mitigation_active else "MONITORING ONLY"
    )
    _draw_text(canvas, f"STATUS: [{status_text}]", (940, 36), 0.55, COLOR_SEV_LOW, 2)

    # 2. Panel 1 (Top Left): Threat Assessment & Security Posture
    p1_x, p1_y, p1_w, p1_h = 20, 70, 580, 270
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "VEHICLE CYBER THREAT LEVEL & POSTURE",
        (p1_x + 15, p1_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    assessment = report.threat_assessment
    max_col = _get_severity_color(assessment.max_severity)
    cv2.rectangle(canvas, (p1_x + 15, p1_y + 45), (p1_x + 280, p1_y + 115), COLOR_CARD_BG, -1)
    _draw_text(canvas, "MAX SEVERITY LEVEL", (p1_x + 25, p1_y + 68), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas,
        assessment.max_severity.value.upper(),
        (p1_x + 25, p1_y + 102),
        0.80,
        max_col,
        2,
    )

    cv2.rectangle(canvas, (p1_x + 295, p1_y + 45), (p1_x + 560, p1_y + 115), COLOR_CARD_BG, -1)
    _draw_text(canvas, "MEAN ANOMALY SCORE", (p1_x + 305, p1_y + 68), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas,
        f"{assessment.mean_anomaly_score:.2f} / 1.00",
        (p1_x + 305, p1_y + 102),
        0.80,
        max_col,
        2,
    )

    # Frame metrics
    m_y = p1_y + 150
    _draw_text(
        canvas,
        f"Total CAN Frames Evaluated : {assessment.total_frames_evaluated:,}",
        (p1_x + 20, m_y),
        0.45,
        COLOR_TEXT_BODY,
        1,
    )
    _draw_text(
        canvas,
        f"Anomalous Frames Detected  : {assessment.anomalous_frames_detected:,}",
        (p1_x + 20, m_y + 32),
        0.45,
        COLOR_TEXT_BODY,
        1,
    )
    _draw_text(
        canvas,
        f"Active Node Quarantines    : {assessment.active_quarantines_count}",
        (p1_x + 20, m_y + 64),
        0.45,
        COLOR_TEXT_BODY,
        1,
    )
    _draw_text(
        canvas,
        f"Compliance Standard        : {report.standard}",
        (p1_x + 20, m_y + 96),
        0.40,
        COLOR_TEXT_MUTED,
        1,
    )

    # 3. Panel 2 (Top Right): Active Quarantines & Defense Actions
    p2_x, p2_y, p2_w, p2_h = 620, 70, 640, 270
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "ACTIVE NODE QUARANTINE & CONTAINMENT MATRIX",
        (p2_x + 15, p2_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    q_y = p2_y + 65
    if not report.quarantined_entities:
        _draw_text(
            canvas,
            "No active node quarantines. Network perimeter nominal.",
            (p2_x + 20, q_y + 40),
            0.45,
            COLOR_SEV_LOW,
            1,
        )
    else:
        for q in report.quarantined_entities[:5]:
            cv2.rectangle(canvas, (p2_x + 15, q_y), (p2_x + 625, q_y + 34), COLOR_CARD_BG, -1)
            cv2.rectangle(canvas, (p2_x + 15, q_y), (p2_x + 625, q_y + 34), COLOR_CARD_BORDER, 1)
            _draw_text(
                canvas,
                f"BLACKLISTED: {q.entity_id}",
                (p2_x + 25, q_y + 22),
                0.40,
                COLOR_SEV_CRITICAL,
                2,
            )
            _draw_text(
                canvas,
                f"Expires: {q.expires_at_s:.1f}s | {q.reason[:40]}",
                (p2_x + 230, q_y + 22),
                0.36,
                COLOR_TEXT_BODY,
                1,
            )
            q_y += 38

    # 4. Panel 3 (Bottom Left): Threat Categories Breakdown
    p3_x, p3_y, p3_w, p3_h = 20, 360, 580, 340
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "CYBER THREAT CLASSIFICATION BREAKDOWN",
        (p3_x + 15, p3_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    # Compute category frequency
    cat_counts: dict[str, int] = {cat.value: 0 for cat in CyberThreatCategory}
    for e in report.events:
        cat_counts[e.category.value] += 1

    bar_y = p3_y + 65
    for cat_name, count in list(cat_counts.items())[:6]:
        _draw_text(canvas, cat_name.upper()[:22], (p3_x + 20, bar_y + 14), 0.38, COLOR_TEXT_BODY, 1)
        # Progress bar
        bar_w = min(300, count * 30)
        cv2.rectangle(
            canvas, (p3_x + 220, bar_y), (p3_x + 220 + 300, bar_y + 18), COLOR_CARD_BG, -1
        )
        if bar_w > 0:
            cv2.rectangle(
                canvas, (p3_x + 220, bar_y), (p3_x + 220 + bar_w, bar_y + 18), COLOR_SEV_HIGH, -1
            )
        _draw_text(canvas, str(count), (p3_x + 530, bar_y + 14), 0.38, COLOR_TEXT_MUTED, 1)
        bar_y += 38

    # 5. Panel 4 (Bottom Right): Real-Time Incident Audit Log
    p4_x, p4_y, p4_w, p4_h = 620, 360, 640, 340
    cv2.rectangle(canvas, (p4_x, p4_y), (p4_x + p4_w, p4_y + p4_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (p4_x, p4_y), (p4_x + p4_w, p4_y + p4_h), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "vSOC REAL-TIME SECURITY EVENT AUDIT TRAIL",
        (p4_x + 15, p4_y + 30),
        0.50,
        COLOR_TEXT_TITLE,
        2,
    )

    log_y = p4_y + 65
    recent_events = report.events[-6:]
    if not recent_events:
        _draw_text(
            canvas, "No security events logged.", (p4_x + 20, log_y + 30), 0.45, COLOR_TEXT_MUTED, 1
        )
    else:
        for evt in recent_events:
            sev_col = _get_severity_color(evt.severity)
            action_tag = (
                "[BLOCK]"
                if evt.defense_action in (DefenseAction.DROP_FRAME, DefenseAction.QUARANTINE_NODE)
                else "[ALERT]"
            )
            hdr = f"{action_tag} {evt.severity.value.upper()} [{evt.source_node}]:"
            _draw_text(canvas, hdr, (p4_x + 20, log_y), 0.38, sev_col, 1)
            _draw_text(canvas, evt.message[:55], (p4_x + 230, log_y), 0.38, COLOR_TEXT_BODY, 1)
            log_y += 42

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
