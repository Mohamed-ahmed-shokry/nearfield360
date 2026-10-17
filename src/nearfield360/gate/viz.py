"""Visual rendering engine for tactical access gateway BEV maps and situational HUD."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.gate.models import (
    BarrierArmState,
    BarrierKinematicsState,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    InductiveLoopId,
    InductiveLoopState,
)

# Color constants (BGR)
COLOR_BG = (22, 24, 28)
COLOR_PANEL_BG = (30, 32, 38)
COLOR_GRID = (45, 50, 60)
COLOR_ROADWAY = (38, 42, 50)
COLOR_ROAD_MARKING = (90, 95, 105)
COLOR_LANE_DIVIDER = (200, 190, 80)
COLOR_PORTAL_POST = (110, 115, 130)
COLOR_LOOP_VACANT = (60, 65, 75)
COLOR_LOOP_OCCUPIED = (50, 180, 240)  # Bright amber/yellow
COLOR_LOOP_SAFETY_OCCUPIED = (40, 60, 240)  # Red warning
COLOR_BARRIER_POST = (50, 130, 220)  # Orange cabinet
COLOR_BARRIER_ARM_OPEN = (70, 200, 90)  # Green
COLOR_BARRIER_ARM_CLOSED = (40, 50, 230)  # Red
COLOR_BARRIER_STRIPE = (240, 240, 240)
COLOR_ALPR_BEAM = (220, 100, 40)  # Blue/cyan laser cone
COLOR_VEHICLE_FILL = (45, 90, 160)
COLOR_VEHICLE_OUTLINE = (70, 140, 240)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (140, 145, 155)
COLOR_CARD_BG = (38, 42, 50)
COLOR_OK = (70, 200, 90)
COLOR_ALERT = (40, 50, 240)


def _put_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _world_to_canvas(
    x: float, y: float, origin: tuple[int, int], scale: float, height: int
) -> tuple[int, int]:
    cx = int(origin[0] + x * scale)
    cy = int(height - (origin[1] + y * scale))
    return cx, cy


def render_gate_dashboard(
    gate_id: str,
    barrier: BarrierKinematicsState,
    loops: dict[InductiveLoopId, InductiveLoopState],
    *,
    decision: GateAccessDecision | None = None,
    request: GateAccessRequest | None = None,
    audit: GateAuditReport | None = None,
    tailgating_active: bool = False,
    output_path: Path | str | None = None,
    width: int = 1280,
    height: int = 720,
) -> NDArray[np.uint8]:
    """Render a dual-panel tactical BEV gateway map and access authorization HUD."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = COLOR_BG

    left_w = 780
    right_x = left_w

    # ================= LEFT PANEL: BEV GATEWAY MAP =================
    cv2.rectangle(canvas, (0, 0), (left_w, height), COLOR_BG, -1)

    # Coordinate mapping: center gateway at (390, 360), scale: 25 px/m
    origin = (390, 360)
    scale = 25.0

    # Draw roadway corridor
    road_half_w = 2.5
    top_pt = _world_to_canvas(-road_half_w, 12.0, origin, scale, height)
    bot_pt = _world_to_canvas(road_half_w, -12.0, origin, scale, height)
    cv2.rectangle(canvas, (top_pt[0], top_pt[1]), (bot_pt[0], bot_pt[1]), COLOR_ROADWAY, -1)

    # Lane boundaries
    cv2.line(
        canvas,
        _world_to_canvas(-road_half_w, -12.0, origin, scale, height),
        _world_to_canvas(-road_half_w, 12.0, origin, scale, height),
        COLOR_ROAD_MARKING,
        2,
    )
    cv2.line(
        canvas,
        _world_to_canvas(road_half_w, -12.0, origin, scale, height),
        _world_to_canvas(road_half_w, 12.0, origin, scale, height),
        COLOR_ROAD_MARKING,
        2,
    )

    # Dashed centerline
    for y_m in range(-11, 12, 2):
        p1 = _world_to_canvas(0.0, float(y_m), origin, scale, height)
        p2 = _world_to_canvas(0.0, float(y_m) + 1.0, origin, scale, height)
        cv2.line(canvas, p1, p2, COLOR_LANE_DIVIDER, 1)

    # Inductive Loop sensors
    loop_specs = [
        (InductiveLoopId.LOOP_A_APPROACH, -8.0, "LOOP A: APPROACH"),
        (InductiveLoopId.LOOP_B_PRESENCE, -4.0, "LOOP B: SCAN"),
        (InductiveLoopId.LOOP_C_SAFETY, 0.0, "LOOP C: SAFETY"),
        (InductiveLoopId.LOOP_D_DEPARTURE, 4.0, "LOOP D: DEPART"),
    ]

    for loop_id, y_center, label in loop_specs:
        loop_state = loops.get(loop_id)
        is_occ = loop_state.occupied if loop_state else False
        if is_occ:
            col = (
                COLOR_LOOP_SAFETY_OCCUPIED
                if loop_id == InductiveLoopId.LOOP_C_SAFETY
                else COLOR_LOOP_OCCUPIED
            )
            thick = -1
        else:
            col = COLOR_LOOP_VACANT
            thick = 2

        lp_top = _world_to_canvas(-1.2, y_center + 1.0, origin, scale, height)
        lp_bot = _world_to_canvas(1.2, y_center - 1.0, origin, scale, height)
        cv2.rectangle(canvas, (lp_top[0], lp_top[1]), (lp_bot[0], lp_bot[1]), col, thick)
        _put_text(canvas, label, (lp_top[0] - 140, lp_top[1] + 25), 0.38, COLOR_TEXT_MUTED, 1)

    # Structural gateway portal posts
    portal_post_left = _world_to_canvas(-2.2, 0.0, origin, scale, height)
    portal_post_right = _world_to_canvas(2.2, 0.0, origin, scale, height)
    cv2.circle(canvas, portal_post_left, 12, COLOR_PORTAL_POST, -1)
    cv2.circle(canvas, portal_post_right, 12, COLOR_PORTAL_POST, -1)
    _put_text(
        canvas,
        "PORTAL POST",
        (portal_post_left[0] - 80, portal_post_left[1] - 15),
        0.35,
        COLOR_TEXT_MUTED,
        1,
    )

    # ALPR optical camera scanning cone toward Loop B
    cam_pos = _world_to_canvas(-2.8, -6.0, origin, scale, height)
    target_pos = _world_to_canvas(0.0, -4.0, origin, scale, height)
    cv2.circle(canvas, cam_pos, 7, (240, 180, 50), -1)
    pts_beam = np.array(
        [
            cam_pos,
            (target_pos[0] - 30, target_pos[1] - 20),
            (target_pos[0] + 30, target_pos[1] + 20),
        ],
        np.int32,
    )
    cv2.polylines(canvas, [pts_beam], True, COLOR_ALPR_BEAM, 1)
    _put_text(canvas, "ALPR CAM", (cam_pos[0] - 65, cam_pos[1] + 5), 0.35, (240, 180, 50), 1)

    # Barrier cabinet and boom arm
    cabinet_pt = _world_to_canvas(-1.8, 0.0, origin, scale, height)
    cv2.rectangle(
        canvas,
        (cabinet_pt[0] - 8, cabinet_pt[1] - 10),
        (cabinet_pt[0] + 8, cabinet_pt[1] + 10),
        COLOR_BARRIER_POST,
        -1,
    )

    # Calculate boom arm endpoint based on barrier_angle_deg
    arm_len_px = 3.6 * scale
    angle_rad = math.radians(barrier.barrier_angle_deg)
    # 0 deg = horizontal across lane (+X), 90 deg = vertical (+Y on screen, up)
    arm_dx = arm_len_px * math.cos(angle_rad)
    arm_dy = -arm_len_px * math.sin(angle_rad)  # canvas Y is inverted
    arm_end = (int(cabinet_pt[0] + arm_dx), int(cabinet_pt[1] + arm_dy))

    arm_col = (
        COLOR_BARRIER_ARM_OPEN
        if barrier.state == BarrierArmState.RAISED_OPEN
        else COLOR_BARRIER_ARM_CLOSED
    )
    cv2.line(canvas, cabinet_pt, arm_end, arm_col, 5)

    # Vehicle bounding box (if request envelope is provided)
    if request is not None:
        veh_y = -4.0  # positioned at Loop B presence scanner
        env = request.envelope
        v_top = _world_to_canvas(
            -env.width_m / 2.0, veh_y + env.length_m / 2.0, origin, scale, height
        )
        v_bot = _world_to_canvas(
            env.width_m / 2.0, veh_y - env.length_m / 2.0, origin, scale, height
        )
        cv2.rectangle(canvas, (v_top[0], v_top[1]), (v_bot[0], v_bot[1]), COLOR_VEHICLE_FILL, -1)
        cv2.rectangle(canvas, (v_top[0], v_top[1]), (v_bot[0], v_bot[1]), COLOR_VEHICLE_OUTLINE, 2)
        plate_str = request.detected_plate or request.vehicle_id
        _put_text(
            canvas, f"[{plate_str}]", (v_top[0] - 5, v_top[1] + 25), 0.45, COLOR_TEXT_TITLE, 1
        )

    # Left panel header
    _put_text(
        canvas, f"TACTICAL ACCESS GATEWAY BEV: {gate_id}", (25, 35), 0.65, COLOR_TEXT_TITLE, 2
    )
    _put_text(
        canvas, "ROADWAY PROFILE & INDUCTIVE SAFETY INTERLOCKS", (25, 58), 0.40, COLOR_TEXT_MUTED, 1
    )

    # ================= RIGHT PANEL: ACCESS HUD =================
    cv2.rectangle(canvas, (right_x, 0), (width, height), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (right_x, 0), (right_x, height), COLOR_GRID, 2)

    hud_x = right_x + 25
    _put_text(canvas, "GATEWAY CONTROL & AUDIT HUD", (hud_x, 38), 0.65, COLOR_TEXT_TITLE, 2)
    _put_text(canvas, f"GATE IDENTIFIER: {gate_id}", (hud_x, 62), 0.42, COLOR_TEXT_MUTED, 1)

    # Card 1: Barrier Mechanical State
    c1_y = 80
    cv2.rectangle(canvas, (hud_x, c1_y), (width - 25, c1_y + 115), COLOR_CARD_BG, -1)
    _put_text(canvas, "BARRIER BOOM KINEMATICS", (hud_x + 15, c1_y + 24), 0.48, COLOR_TEXT_TITLE, 1)

    state_col = (
        COLOR_OK
        if barrier.state == BarrierArmState.RAISED_OPEN
        else (COLOR_ALERT if barrier.safety_interlock_active else (240, 180, 50))
    )
    _put_text(
        canvas,
        f"ARM STATE:     {barrier.state.value.upper()}",
        (hud_x + 15, c1_y + 50),
        0.45,
        state_col,
        1,
    )
    _put_text(
        canvas,
        f"BOOM ANGLE:    {barrier.barrier_angle_deg:.1f} deg",
        (hud_x + 15, c1_y + 72),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    _put_text(
        canvas,
        f"ANGULAR VEL:   {barrier.angular_velocity_deg_s:+.1f} deg/s",
        (hud_x + 15, c1_y + 92),
        0.40,
        COLOR_TEXT_MUTED,
        1,
    )
    if barrier.safety_interlock_active:
        _put_text(
            canvas, "[SAFETY REVERSAL ACTIVE]", (hud_x + 240, c1_y + 92), 0.40, COLOR_ALERT, 1
        )

    # Card 2: Access Decision & Credentials
    c2_y = 210
    cv2.rectangle(canvas, (hud_x, c2_y), (width - 25, c2_y + 130), COLOR_CARD_BG, -1)
    _put_text(
        canvas, "ACCESS AUTHORIZATION STATUS", (hud_x + 15, c2_y + 24), 0.48, COLOR_TEXT_TITLE, 1
    )

    if decision is not None:
        dec_col = COLOR_OK if decision.granted else COLOR_ALERT
        status_txt = (
            "ACCESS GRANTED" if decision.granted else f"REJECTED: {decision.status.value.upper()}"
        )
        _put_text(canvas, status_txt, (hud_x + 15, c2_y + 52), 0.50, dec_col, 2)
        margin_txt = (
            f"HEAD: {decision.headroom_margin_m:+.2f}m | WIDTH: {decision.width_margin_m:+.2f}m"
        )
        _put_text(
            canvas,
            margin_txt,
            (hud_x + 15, c2_y + 78),
            0.40,
            COLOR_TEXT_BODY,
            1,
        )
        _put_text(
            canvas,
            f"MSG: {decision.message[:52]}",
            (hud_x + 15, c2_y + 100),
            0.38,
            COLOR_TEXT_MUTED,
            1,
        )
        if decision.issued_pass_id:
            _put_text(
                canvas,
                f"PASS: {decision.issued_pass_id[:35]}",
                (hud_x + 15, c2_y + 120),
                0.38,
                COLOR_OK,
                1,
            )
    else:
        _put_text(
            canvas, "STATUS: STANDBY / IDLE", (hud_x + 15, c2_y + 55), 0.45, COLOR_TEXT_MUTED, 1
        )

    # Card 3: Anti-Tailgating Interlock Status
    c3_y = 355
    cv2.rectangle(canvas, (hud_x, c3_y), (width - 25, c3_y + 75), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "ANTI-TAILGATING PERIMETER INTERLOCK",
        (hud_x + 15, c3_y + 24),
        0.48,
        COLOR_TEXT_TITLE,
        1,
    )
    if tailgating_active:
        cv2.rectangle(canvas, (hud_x + 10, c3_y + 35), (width - 35, c3_y + 65), (20, 20, 160), -1)
        _put_text(
            canvas,
            "ALARM: UNAUTHORIZED INGRESS TAILGATER DETECTED",
            (hud_x + 18, c3_y + 55),
            0.44,
            (50, 60, 255),
            2,
        )
    else:
        _put_text(
            canvas,
            "NORMAL: SINGLE VEHICLE SPACING ENFORCED",
            (hud_x + 15, c3_y + 52),
            0.44,
            COLOR_OK,
            1,
        )

    # Card 4: Loop Detectors Summary
    c4_y = 445
    cv2.rectangle(canvas, (hud_x, c4_y), (width - 25, c4_y + 115), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "ROADWAY INDUCTIVE LOOP DETECTORS",
        (hud_x + 15, c4_y + 22),
        0.48,
        COLOR_TEXT_TITLE,
        1,
    )

    lp_y = c4_y + 44
    for lid in InductiveLoopId:
        st = loops.get(lid)
        is_det = st.occupied if st else False
        dwell = st.dwell_duration_s if st else 0.0
        lid_short = lid.value.replace("loop_", "").upper()
        col = COLOR_OK if is_det else COLOR_TEXT_MUTED
        txt = f"[{'DETECTED' if is_det else 'CLEAR'}] {lid_short:<14} dwell={dwell:.1f}s"
        _put_text(canvas, txt, (hud_x + 15, lp_y), 0.38, col, 1)
        lp_y += 18

    # Card 5: Operational Audit Report
    c5_y = 575
    cv2.rectangle(canvas, (hud_x, c5_y), (width - 25, c5_y + 115), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "GATEWAY OPERATIONAL AUDIT METRICS",
        (hud_x + 15, c5_y + 22),
        0.48,
        COLOR_TEXT_TITLE,
        1,
    )
    if audit is not None:
        pass_pct = (audit.granted_count / max(1, audit.total_transits)) * 100.0
        line1 = f"TRANSITS: {audit.total_transits} | PASS: {audit.granted_count} ({pass_pct:.0f}%)"
        line2 = f"REJECTS: {audit.rejected_count} | CLEARANCE FAILS: {audit.clearance_violations}"
        line3 = (
            f"TAILGATE: {audit.tailgating_incidents} | "
            f"AVG DURATION: {audit.average_transit_duration_s:.1f}s"
        )
        _put_text(canvas, line1, (hud_x + 15, c5_y + 48), 0.40, COLOR_TEXT_BODY, 1)
        _put_text(canvas, line2, (hud_x + 15, c5_y + 70), 0.40, COLOR_TEXT_BODY, 1)
        _put_text(canvas, line3, (hud_x + 15, c5_y + 92), 0.40, COLOR_TEXT_MUTED, 1)
    else:
        _put_text(
            canvas,
            "AUDIT REPORT: NO CYCLES LOGGED",
            (hud_x + 15, c5_y + 55),
            0.42,
            COLOR_TEXT_MUTED,
            1,
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)

    return canvas
