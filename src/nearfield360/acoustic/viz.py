"""Tactical 1280x720 acoustic HUD dashboard and polar DoA visualizer."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.acoustic.models import (
    AcousticReport,
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    SirenType,
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


def _get_yield_color(state: EmergencyYieldState) -> tuple[int, int, int]:
    if state == EmergencyYieldState.EMERGENCY_STOP_ACTIVE:
        return COLOR_STATE_RED
    if state == EmergencyYieldState.YIELD_PREPARED:
        return COLOR_STATE_ORANGE
    if state == EmergencyYieldState.MONITORING:
        return COLOR_STATE_AMBER
    return COLOR_STATE_GREEN


def render_acoustic_dashboard(
    report: AcousticReport,
    output_path: str | Path = Path("artifacts/acoustic_hud.png"),
) -> Path:
    """Render a tactical 1280x720 dark-mode acoustic sensing HUD dashboard PNG."""
    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    canvas: NDArray[np.uint8] = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Bar
    cv2.rectangle(canvas, (0, 0), (1280, 65), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (0, 65), (1280, 65), COLOR_CARD_BORDER, 1)

    _draw_text(
        canvas, "NEARFIELD360 ACOUSTIC SIREN & DOA RADAR HUD", (24, 28), 0.72, COLOR_TEXT_TITLE, 2
    )
    _draw_text(
        canvas,
        "Multi-Microphone Acoustic Triangulation & Audio-Visual Emergency Yield Interlock",
        (24, 50),
        0.42,
        COLOR_TEXT_MUTED,
        1,
    )

    state_col = _get_yield_color(report.yield_state)
    state_badge = f"[{report.yield_state.value.upper()}]"
    _draw_text(canvas, "YIELD INTERLOCK:", (920, 38), 0.50, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, state_badge, (1060, 40), 0.65, state_col, 2)

    # 2. Panel 1: Polar Acoustic Radar Compass (x: 20 to 450, y: 75 to 440)
    cv2.rectangle(canvas, (20, 75), (450, 440), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 75), (450, 440), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas, "POLAR DIRECTION-OF-ARRIVAL (DoA) RADAR", (35, 102), 0.52, COLOR_TEXT_TITLE, 2
    )

    center_x, center_y = 235, 260
    radius = 135

    # Concentric range rings
    for r_step in (45, 90, 135):
        cv2.circle(canvas, (center_x, center_y), r_step, COLOR_CARD_BORDER, 1)
    cv2.line(canvas, (center_x - radius, center_y), (center_x + radius, center_y), (45, 52, 62), 1)
    cv2.line(canvas, (center_x, center_y - radius), (center_x, center_y + radius), (45, 52, 62), 1)

    # Labels for camera sectors
    _draw_text(canvas, "FV [0°]", (center_x - 22, center_y - radius - 8), 0.42, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas, "MVR [90°]", (center_x + radius + 6, center_y + 4), 0.40, COLOR_TEXT_MUTED, 1
    )
    _draw_text(
        canvas, "RV [180°]", (center_x - 24, center_y + radius + 16), 0.40, COLOR_TEXT_MUTED, 1
    )
    _draw_text(
        canvas, "MVL [-90°]", (center_x - radius - 64, center_y + 4), 0.40, COLOR_TEXT_MUTED, 1
    )

    # Vehicle footprint glyph at center
    veh_w, veh_h = 16, 28
    cv2.rectangle(
        canvas,
        (center_x - veh_w // 2, center_y - veh_h // 2),
        (center_x + veh_w // 2, center_y + veh_h // 2),
        (80, 90, 105),
        -1,
    )
    cv2.rectangle(
        canvas,
        (center_x - veh_w // 2, center_y - veh_h // 2),
        (center_x + veh_w // 2, center_y + veh_h // 2),
        (130, 140, 160),
        1,
    )

    # Draw acoustic DoA ray if available
    if report.doa is not None:
        az_deg = report.doa.azimuth_deg
        # Screen angle: 0 deg points up (-Y), +90 deg points right (+X)
        rad = math.radians(az_deg - 90.0)
        end_x = int(center_x + radius * math.cos(rad))
        end_y = int(center_y + radius * math.sin(rad))

        beam_col = COLOR_STATE_RED if report.is_emergency_active() else COLOR_ACCENT_CYAN
        cv2.line(canvas, (center_x, center_y), (end_x, end_y), beam_col, 2)
        cv2.circle(canvas, (end_x, end_y), 7, beam_col, -1)
        cv2.circle(canvas, (end_x, end_y), 11, beam_col, 1)

        doa_lbl = f"Azimuth: {az_deg:+.1f}° | Conf: {report.doa.confidence * 100:.0f}%"
        spl_lbl = f"Sound Level: {report.doa.sound_pressure_level_db:.1f} dB SPL"
        _draw_text(canvas, doa_lbl, (35, 415), 0.44, COLOR_TEXT_BODY, 1)
        _draw_text(canvas, spl_lbl, (250, 415), 0.44, COLOR_ACCENT_CYAN, 1)
    else:
        _draw_text(
            canvas,
            "No directional siren source triangulated.",
            (80, 415),
            0.42,
            COLOR_TEXT_MUTED,
            1,
        )

    # 3. Panel 2: Spectrogram & Modulation Analysis (x: 470 to 1260, y: 75 to 440)
    cv2.rectangle(canvas, (470, 75), (1260, 440), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (470, 75), (1260, 440), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "SPECTRAL SIREN DETECTION & MODULATION PROFILE",
        (485, 102),
        0.52,
        COLOR_TEXT_TITLE,
        2,
    )

    # Spectrogram window box
    spec_x, spec_y, spec_w, spec_h = 490, 120, 745, 180
    cv2.rectangle(canvas, (spec_x, spec_y), (spec_x + spec_w, spec_y + spec_h), (12, 14, 18), -1)
    cv2.rectangle(
        canvas, (spec_x, spec_y), (spec_x + spec_w, spec_y + spec_h), COLOR_CARD_BORDER, 1
    )

    # Draw synthetic spectral sweep trajectory for visualization
    s_det = report.siren_detection
    pts = []
    num_pts = 60
    for idx in range(num_pts):
        px = spec_x + int(idx * (spec_w / num_pts))
        if s_det.detected and s_det.siren_type == SirenType.WAIL:
            # Swept wave
            norm_val = 0.5 + 0.35 * math.sin(idx * 0.25)
        elif s_det.detected and s_det.siren_type == SirenType.YELP:
            norm_val = 0.5 + 0.35 * math.sin(idx * 1.5)
        elif s_det.detected and s_det.siren_type == SirenType.HI_LO:
            norm_val = 0.65 if (idx // 6) % 2 == 0 else 0.35
        elif s_det.detected and s_det.siren_type == SirenType.HORN:
            norm_val = 0.50
        else:
            norm_val = 0.15 + (math.sin(idx * 1.7) * 0.05)
        py = spec_y + int((1.0 - norm_val) * spec_h)
        pts.append((px, py))

    for k in range(len(pts) - 1):
        line_col = COLOR_ACCENT_CYAN if s_det.detected else (50, 55, 65)
        cv2.line(canvas, pts[k], pts[k + 1], line_col, 2)

    _draw_text(canvas, "1800 Hz", (spec_x + 8, spec_y + 18), 0.36, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, "400 Hz", (spec_x + 8, spec_y + spec_h - 8), 0.36, COLOR_TEXT_MUTED, 1)

    # Detection parameter badges below spectrogram
    card_y = 320
    # Card 1: Siren Type
    cv2.rectangle(canvas, (490, card_y), (720, card_y + 100), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (490, card_y), (720, card_y + 100), COLOR_CARD_BORDER, 1)
    _draw_text(canvas, "SIREN SIGNATURE", (505, card_y + 24), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas, s_det.siren_type.value.upper(), (505, card_y + 55), 0.70, COLOR_TEXT_TITLE, 2
    )
    _draw_text(
        canvas,
        f"Conf: {s_det.confidence * 100:.1f}%",
        (505, card_y + 85),
        0.42,
        COLOR_STATE_GREEN,
        1,
    )

    # Card 2: Fundamental & SNR
    cv2.rectangle(canvas, (735, card_y), (965, card_y + 100), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (735, card_y), (965, card_y + 100), COLOR_CARD_BORDER, 1)
    _draw_text(canvas, "PITCH & SNR", (750, card_y + 24), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas, f"{s_det.fundamental_freq_hz:.0f} Hz", (750, card_y + 55), 0.65, COLOR_TEXT_TITLE, 2
    )
    _draw_text(
        canvas, f"SNR: {s_det.snr_db:.1f} dB", (750, card_y + 85), 0.42, COLOR_ACCENT_CYAN, 1
    )

    # Card 3: Modulation Cycle Rate
    cv2.rectangle(canvas, (980, card_y), (1240, card_y + 100), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (980, card_y), (1240, card_y + 100), COLOR_CARD_BORDER, 1)
    _draw_text(canvas, "SWEEP CYCLE RATE", (995, card_y + 24), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(
        canvas, f"{s_det.cycle_rate_hz:.2f} Hz", (995, card_y + 55), 0.65, COLOR_TEXT_TITLE, 2
    )
    _draw_text(
        canvas,
        f"Harmonic Ratio: {s_det.harmonic_energy_ratio * 100:.0f}%",
        (995, card_y + 85),
        0.40,
        COLOR_TEXT_MUTED,
        1,
    )

    # 4. Panel 3: Audio-Visual Sector Correlation (x: 20 to 620, y: 460 to 700)
    cv2.rectangle(canvas, (20, 460), (620, 700), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 460), (620, 700), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas, "AUDIO-VISUAL CAMERA SECTOR CORRELATION", (35, 485), 0.50, COLOR_TEXT_TITLE, 2
    )

    sectors = [CameraSector.FRONT, CameraSector.RIGHT, CameraSector.REAR, CameraSector.LEFT]
    matched_sec = report.sector_match.camera_sector if report.sector_match else None

    for s_idx, sec in enumerate(sectors):
        bx = 35 + s_idx * 140
        by = 505
        is_matched = sec == matched_sec
        b_col = (
            COLOR_STATE_RED
            if (is_matched and report.is_emergency_active())
            else (COLOR_ACCENT_CYAN if is_matched else COLOR_CARD_BG)
        )
        cv2.rectangle(canvas, (bx, by), (bx + 125, by + 90), b_col, -1 if is_matched else 1)
        if not is_matched:
            cv2.rectangle(canvas, (bx, by), (bx + 125, by + 90), COLOR_CARD_BORDER, 1)

        t_col = (10, 10, 10) if is_matched else COLOR_TEXT_TITLE
        _draw_text(canvas, sec.value, (bx + 40, by + 35), 0.60, t_col, 2)
        status_txt = "MATCH" if is_matched else "STANDBY"
        _draw_text(canvas, status_txt, (bx + 30, by + 65), 0.38, t_col, 1)

    # Sector correlation summary text
    if report.sector_match:
        vis_tag = "CONFIRMED" if report.sector_match.visual_confirmation else "UNCONFIRMED"
        c_txt = (
            f"Sector: {report.sector_match.camera_sector.value} | Visual: {vis_tag} | "
            f"Fused Conf: {report.sector_match.fused_confidence * 100:.1f}%"
        )
        _draw_text(canvas, c_txt, (35, 630), 0.42, COLOR_TEXT_BODY, 1)
        _draw_text(
            canvas,
            f"Yield Action Recommended: {report.sector_match.yield_action_recommended}",
            (35, 655),
            0.42,
            COLOR_STATE_ORANGE
            if report.sector_match.yield_action_recommended
            else COLOR_STATE_GREEN,
            1,
        )

    # 5. Panel 4: Emergency Vehicle Yield Interlock & Telemetry (x: 640 to 1260, y: 460 to 700)
    cv2.rectangle(canvas, (640, 460), (1260, 700), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (640, 460), (1260, 700), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas, "DOPPLER VELOCITY & YIELD INTERLOCK AUDIT", (655, 485), 0.50, COLOR_TEXT_TITLE, 2
    )

    # Doppler state card
    d_col = (
        COLOR_STATE_RED
        if report.doppler == DopplerState.APPROACHING
        else (COLOR_STATE_GREEN if report.doppler == DopplerState.RECEDING else COLOR_STATE_AMBER)
    )
    cv2.rectangle(canvas, (655, 505), (935, 595), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (655, 505), (935, 595), COLOR_CARD_BORDER, 1)
    _draw_text(canvas, "DOPPLER STATUS", (670, 528), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, report.doppler.value.upper(), (670, 558), 0.60, d_col, 2)
    _draw_text(
        canvas,
        f"Radial Speed: {report.estimated_radial_speed_mps:+.1f} m/s",
        (670, 582),
        0.40,
        COLOR_TEXT_BODY,
        1,
    )

    # Safety supervisor interlock card
    cv2.rectangle(canvas, (955, 505), (1240, 595), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (955, 505), (1240, 595), COLOR_CARD_BORDER, 1)
    _draw_text(canvas, "ASIL-D SUPERVISOR INTERLOCK", (970, 528), 0.38, COLOR_TEXT_MUTED, 1)
    interlock_txt = "COMMAND SAFE STOP" if report.is_emergency_active() else "NOMINAL MONITORING"
    interlock_col = COLOR_STATE_RED if report.is_emergency_active() else COLOR_STATE_GREEN
    _draw_text(canvas, interlock_txt, (970, 558), 0.52, interlock_col, 2)
    _draw_text(
        canvas, f"Timestamp: t={report.timestamp_s:.2f} s", (970, 582), 0.40, COLOR_TEXT_MUTED, 1
    )

    # Operational guidance message
    _draw_text(canvas, "Operational Telemetry Message:", (655, 625), 0.40, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, report.summary_message[:80], (655, 650), 0.42, COLOR_TEXT_BODY, 1)
    if len(report.summary_message) > 80:
        _draw_text(canvas, report.summary_message[80:160], (655, 672), 0.42, COLOR_TEXT_BODY, 1)

    cv2.imwrite(str(target_path), canvas)
    return target_path
