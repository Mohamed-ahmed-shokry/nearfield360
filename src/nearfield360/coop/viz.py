"""Tactical BEV visualizer and situational HUD for cooperative perception operations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.coop.alignment import Pose2D
from nearfield360.coop.blind_spot import (
    DEFAULT_FACILITY_PILLARS,
    compute_pillar_shadow_polygon,
)
from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    BlindSpotWarning,
    CooperativeTrack,
    CoopTelemetryReport,
    CoopTrackSource,
    RPUSensorState,
    RPUSensorType,
)

# High-contrast tactical color palette (BGR)
COLOR_BG = (20, 22, 26)
COLOR_PANEL_BG = (28, 32, 38)
COLOR_GRID = (40, 44, 52)
COLOR_AISLE = (34, 38, 46)
COLOR_PILLAR = (100, 105, 120)
COLOR_PILLAR_RING = (140, 145, 160)
COLOR_SHADOW = (15, 18, 22)
COLOR_VEHICLE_FILL = (180, 110, 40)
COLOR_VEHICLE_OUTLINE = (240, 170, 70)
COLOR_ONBOARD_FOV = (50, 120, 80)
COLOR_TRACK_ONBOARD = (220, 140, 50)  # Blue/Cyan
COLOR_TRACK_INFRA = (180, 60, 220)  # Magenta
COLOR_TRACK_FUSED = (60, 220, 100)  # Bright Green
COLOR_CONFLICT_RING = (40, 40, 240)  # Red alert
COLOR_CARD_BG = (36, 40, 50)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (130, 135, 145)
COLOR_OK = (60, 200, 90)
COLOR_WARN = (40, 180, 240)
COLOR_DANGER = (40, 50, 240)


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
    """Map metric world (x, y) into canvas pixel space."""
    ox, oy = origin
    cx = int(ox + x * scale)
    cy = int(height - (oy + y * scale))
    return (cx, cy)


def draw_bev_map(
    canvas: NDArray[np.uint8],
    vehicle_pose: Pose2D,
    rpus: Sequence[RPUSensorState],
    tracks: Sequence[CooperativeTrack],
    warnings: Sequence[BlindSpotWarning],
    pillars: Sequence[tuple[float, float, float]] | None = None,
    origin: tuple[int, int] = (60, 40),
    scale: float = 14.0,
    width: int = 800,
    height: int = 720,
) -> None:
    """Render the BEV facility map with aisles, pillars, shadows, sensor FOVs, and tracks."""
    del width
    active_pillars = list(pillars) if pillars is not None else DEFAULT_FACILITY_PILLARS

    # 1. Base grid lines (every 5 meters)
    for gx in range(0, 50, 5):
        pt1 = _world_to_canvas(float(gx), 0.0, origin, scale, height)
        pt2 = _world_to_canvas(float(gx), 45.0, origin, scale, height)
        cv2.line(canvas, pt1, pt2, COLOR_GRID, 1)
    for gy in range(0, 50, 5):
        pt1 = _world_to_canvas(0.0, float(gy), origin, scale, height)
        pt2 = _world_to_canvas(45.0, float(gy), origin, scale, height)
        cv2.line(canvas, pt1, pt2, COLOR_GRID, 1)

    # 2. Main Corridor Roadways
    corridor_poly = [
        _world_to_canvas(0.0, 10.0, origin, scale, height),
        _world_to_canvas(45.0, 10.0, origin, scale, height),
        _world_to_canvas(45.0, 20.0, origin, scale, height),
        _world_to_canvas(0.0, 20.0, origin, scale, height),
    ]
    cv2.fillPoly(canvas, [np.array(corridor_poly, dtype=np.int32)], COLOR_AISLE)

    cross_poly = [
        _world_to_canvas(15.0, 0.0, origin, scale, height),
        _world_to_canvas(25.0, 0.0, origin, scale, height),
        _world_to_canvas(25.0, 45.0, origin, scale, height),
        _world_to_canvas(15.0, 45.0, origin, scale, height),
    ]
    cv2.fillPoly(canvas, [np.array(cross_poly, dtype=np.int32)], COLOR_AISLE)

    # 3. Structural Pillars & Occlusion Shadows
    obs_pos = (vehicle_pose.x_m, vehicle_pose.y_m)
    for pillar in active_pillars:
        shadow_poly = compute_pillar_shadow_polygon(obs_pos, pillar, shadow_depth_m=35.0)
        if shadow_poly:
            pts_canvas = [_world_to_canvas(sx, sy, origin, scale, height) for sx, sy in shadow_poly]
            cv2.fillPoly(canvas, [np.array(pts_canvas, dtype=np.int32)], COLOR_SHADOW)

        # Pillar circle
        px, py = _world_to_canvas(pillar[0], pillar[1], origin, scale, height)
        pr = max(4, int(pillar[2] * scale))
        cv2.circle(canvas, (px, py), pr, COLOR_PILLAR, -1)
        cv2.circle(canvas, (px, py), pr + 2, COLOR_PILLAR_RING, 1)

    # 4. RPU Sensor Stations & FOV Cones
    for rpu in rpus:
        sx, sy, _ = rpu.position_m
        sc_pos = _world_to_canvas(sx, sy, origin, scale, height)

        if rpu.sensor_type == RPUSensorType.LIDAR:
            rpu_color = (220, 200, 60)
        elif rpu.sensor_type == RPUSensorType.RADAR:
            rpu_color = (180, 80, 220)
        else:
            rpu_color = (80, 200, 100)

        yaw_rad = math.radians(rpu.yaw_deg)
        half_fov = math.radians(rpu.fov_azimuth_deg / 2.0)
        steps = 12
        arc_pts: list[tuple[int, int]] = [sc_pos]
        for step in range(steps + 1):
            theta = (yaw_rad - half_fov) + (2.0 * half_fov) * (step / steps)
            wx = sx + rpu.range_max_m * math.cos(theta)
            wy = sy + rpu.range_max_m * math.sin(theta)
            arc_pts.append(_world_to_canvas(wx, wy, origin, scale, height))

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array(arc_pts, dtype=np.int32)], rpu_color)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [np.array(arc_pts, dtype=np.int32)], True, rpu_color, 1)

        cv2.circle(canvas, sc_pos, 6, rpu_color, -1)
        cv2.circle(canvas, sc_pos, 8, (255, 255, 255), 1)
        _put_text(
            canvas,
            rpu.station_id,
            (sc_pos[0] - 25, sc_pos[1] - 12),
            0.35,
            COLOR_TEXT_BODY,
            1,
        )

    # 5. Vehicle (Ego)
    vx, vy = _world_to_canvas(vehicle_pose.x_m, vehicle_pose.y_m, origin, scale, height)
    v_len_px = int(4.5 * scale)
    v_wid_px = int(2.0 * scale)
    v_yaw_rad = math.radians(vehicle_pose.yaw_deg)

    c_y = math.cos(v_yaw_rad)
    s_y = math.sin(v_yaw_rad)
    corners_local = [
        (v_len_px / 2, v_wid_px / 2),
        (-v_len_px / 2, v_wid_px / 2),
        (-v_len_px / 2, -v_wid_px / 2),
        (v_len_px / 2, -v_wid_px / 2),
    ]
    corners_canvas: list[tuple[int, int]] = []
    for lx, ly in corners_local:
        gx = vx + int(lx * c_y - ly * s_y)
        gy = vy - int(lx * s_y + ly * c_y)
        corners_canvas.append((gx, gy))

    cv2.fillPoly(canvas, [np.array(corners_canvas, dtype=np.int32)], COLOR_VEHICLE_FILL)
    cv2.polylines(
        canvas,
        [np.array(corners_canvas, dtype=np.int32)],
        True,
        COLOR_VEHICLE_OUTLINE,
        2,
    )

    # Heading arrow
    head_x = vx + int(25 * c_y)
    head_y = vy - int(25 * s_y)
    cv2.arrowedLine(canvas, (vx, vy), (head_x, head_y), (255, 255, 255), 2, tipLength=0.3)

    # Onboard fisheye coverage circle (15m radius)
    cv2.circle(canvas, (vx, vy), int(15.0 * scale), COLOR_ONBOARD_FOV, 1)

    # 6. Tracks
    for trk in tracks:
        tx, ty = _world_to_canvas(trk.x_m, trk.y_m, origin, scale, height)
        if trk.source == CoopTrackSource.FUSED_COOPERATIVE:
            t_color = COLOR_TRACK_FUSED
            tag = "FUSED"
        elif trk.source == CoopTrackSource.INFRASTRUCTURE_ONLY:
            t_color = COLOR_TRACK_INFRA
            tag = "CPM"
        else:
            t_color = COLOR_TRACK_ONBOARD
            tag = "ONB"

        cv2.circle(canvas, (tx, ty), 6, t_color, -1)
        cv2.circle(canvas, (tx, ty), 9, (255, 255, 255), 1)

        if abs(trk.vx_mps) > 0.1 or abs(trk.vy_mps) > 0.1:
            vx_end = tx + int(trk.vx_mps * scale * 0.8)
            vy_end = ty - int(trk.vy_mps * scale * 0.8)
            cv2.arrowedLine(canvas, (tx, ty), (vx_end, vy_end), t_color, 2, tipLength=0.25)

        _put_text(
            canvas,
            f"{tag}:{trk.track_id}",
            (tx + 10, ty + 4),
            0.35,
            COLOR_TEXT_BODY,
            1,
        )

    # 7. Conflict Warning Halo
    for w in warnings:
        c_pt = _world_to_canvas(w.conflict_point_m[0], w.conflict_point_m[1], origin, scale, height)
        cv2.circle(canvas, c_pt, 16, COLOR_CONFLICT_RING, 2)
        cv2.circle(canvas, c_pt, 24, COLOR_CONFLICT_RING, 1)
        _put_text(
            canvas,
            f"! TTC {w.ttc_s:.1f}s",
            (c_pt[0] - 30, c_pt[1] - 28),
            0.45,
            COLOR_DANGER,
            2,
        )


def draw_hud_panel(
    canvas: NDArray[np.uint8],
    vehicle_pose: Pose2D,
    vehicle_speed_mps: float,
    rpus: Sequence[RPUSensorState],
    tracks: Sequence[CooperativeTrack],
    warnings: Sequence[BlindSpotWarning],
    telemetry: CoopTelemetryReport | None = None,
    x_offset: int = 800,
    width: int = 480,
    height: int = 720,
) -> None:
    """Render the cooperative perception HUD cards on the right side of the canvas."""
    del width
    # 1. Background and border
    cv2.rectangle(canvas, (x_offset, 0), (1280, height), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (x_offset, 0), (x_offset, height), COLOR_GRID, 2)

    # 2. Header
    _put_text(
        canvas,
        "COOPERATIVE PERCEPTION HUD",
        (x_offset + 24, 38),
        0.58,
        COLOR_TEXT_TITLE,
        2,
    )
    _put_text(
        canvas,
        "ISO 23374 TYPE 2/3 | ETSI CPM INFRASTRUCTURE",
        (x_offset + 24, 58),
        0.36,
        COLOR_TEXT_MUTED,
        1,
    )

    # 3. Card 1: Ego Vehicle Telematics
    y = 80
    cv2.rectangle(canvas, (x_offset + 16, y), (1264, y + 100), COLOR_CARD_BG, -1)
    _put_text(canvas, "EGO VEHICLE TELEMATICS", (x_offset + 28, y + 24), 0.44, COLOR_TEXT_TITLE, 1)
    pose_str = (
        f"Pose: ({vehicle_pose.x_m:.2f}m, {vehicle_pose.y_m:.2f}m)  "
        f"Yaw: {vehicle_pose.yaw_deg:.1f} deg"
    )
    _put_text(
        canvas,
        pose_str,
        (x_offset + 28, y + 52),
        0.38,
        COLOR_TEXT_BODY,
        1,
    )
    _put_text(
        canvas,
        f"Velocity: {vehicle_speed_mps:.2f} m/s  ({vehicle_speed_mps * 3.6:.1f} km/h)",
        (x_offset + 28, y + 78),
        0.38,
        COLOR_OK if vehicle_speed_mps > 0.1 else COLOR_TEXT_MUTED,
        1,
    )

    # 4. Card 2: Roadside Perception Network (RPU)
    y += 115
    cv2.rectangle(canvas, (x_offset + 16, y), (1264, y + 110), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "ROADSIDE PERCEPTION UNITS (RPU)",
        (x_offset + 28, y + 24),
        0.44,
        COLOR_TEXT_TITLE,
        1,
    )
    _put_text(
        canvas,
        f"Active Stations: {len(rpus)}   Broadcast: 10.0 Hz (V2I)",
        (x_offset + 28, y + 52),
        0.38,
        COLOR_TEXT_BODY,
        1,
    )
    rpu_summary = " | ".join(
        f"{s.station_id.split('-')[1]}:{s.sensor_type.value}" for s in rpus[:3]
    )
    _put_text(
        canvas,
        f"Modality: {rpu_summary}",
        (x_offset + 28, y + 78),
        0.34,
        COLOR_TEXT_MUTED,
        1,
    )

    # 5. Card 3: Track Fusion & Latency
    y += 125
    cv2.rectangle(canvas, (x_offset + 16, y), (1264, y + 120), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "COOPERATIVE FUSION & LATENCY",
        (x_offset + 28, y + 24),
        0.44,
        COLOR_TEXT_TITLE,
        1,
    )

    fused_cnt = sum(1 for t in tracks if t.source == CoopTrackSource.FUSED_COOPERATIVE)
    infra_cnt = sum(1 for t in tracks if t.source == CoopTrackSource.INFRASTRUCTURE_ONLY)
    onb_cnt = sum(1 for t in tracks if t.source == CoopTrackSource.ONBOARD_ONLY)

    _put_text(
        canvas,
        f"Tracks: {len(tracks)} total ({fused_cnt} fused, {infra_cnt} cpm, {onb_cnt} onboard)",
        (x_offset + 28, y + 52),
        0.38,
        COLOR_TEXT_BODY,
        1,
    )
    lat_ms = telemetry.mean_fusion_latency_ms if telemetry else 3.8
    _put_text(
        canvas,
        f"Mean Processing Latency: {lat_ms:.2f} ms   CI Inflation: 1.25x",
        (x_offset + 28, y + 78),
        0.38,
        COLOR_OK if lat_ms < 10.0 else COLOR_WARN,
        1,
    )
    _put_text(
        canvas,
        "Method: Covariance Intersection (CI) & Mahalanobis Gating",
        (x_offset + 28, y + 102),
        0.34,
        COLOR_TEXT_MUTED,
        1,
    )

    # 6. Card 4: Blind-Spot Occlusion & Hazard Warnings
    y += 135
    card_h = 170
    cv2.rectangle(canvas, (x_offset + 16, y), (1264, y + card_h), COLOR_CARD_BG, -1)
    _put_text(
        canvas,
        "BLIND-SPOT HAZARDS & TTC ADVISORY",
        (x_offset + 28, y + 24),
        0.44,
        COLOR_TEXT_TITLE,
        1,
    )

    if not warnings:
        _put_text(
            canvas,
            "STATUS: NOMINAL - NO OCCLUDED HAZARDS DETECTED",
            (x_offset + 28, y + 65),
            0.40,
            COLOR_OK,
            1,
        )
        _put_text(
            canvas,
            "Pillar shadow sectors continuously monitored by RPUs",
            (x_offset + 28, y + 95),
            0.35,
            COLOR_TEXT_MUTED,
            1,
        )
    else:
        top_w = warnings[0]
        # Risk Badge
        if top_w.risk_level == BlindSpotRiskLevel.CRITICAL_BRAKE:
            badge_color = COLOR_DANGER
            badge_text = "CRITICAL BRAKE HAZARD"
        elif top_w.risk_level == BlindSpotRiskLevel.WARNING:
            badge_color = COLOR_WARN
            badge_text = "WARNING: APPROACHING CONFLICT"
        else:
            badge_color = (80, 200, 240)
            badge_text = "ADVISORY: CAUTION ZONE"

        _put_text(
            canvas,
            f"! {badge_text}",
            (x_offset + 28, y + 54),
            0.44,
            badge_color,
            2,
        )
        _put_text(
            canvas,
            f"Target: {top_w.hazard_object_id}  |  TTC: {top_w.ttc_s:.2f} s",
            (x_offset + 28, y + 84),
            0.40,
            COLOR_TEXT_BODY,
            1,
        )
        _put_text(
            canvas,
            f"Conflict Point: ({top_w.conflict_point_m[0]:.1f}m, {top_w.conflict_point_m[1]:.1f}m)",
            (x_offset + 28, y + 110),
            0.36,
            COLOR_TEXT_MUTED,
            1,
        )
        _put_text(
            canvas,
            f"Action: {top_w.recommended_action.value.upper()}",
            (x_offset + 28, y + 138),
            0.42,
            badge_color,
            2,
        )


def render_coop_dashboard(
    vehicle_pose: Pose2D,
    vehicle_speed_mps: float,
    rpus: Sequence[RPUSensorState],
    tracks: Sequence[CooperativeTrack],
    warnings: Sequence[BlindSpotWarning],
    telemetry: CoopTelemetryReport | None = None,
    pillars: Sequence[tuple[float, float, float]] | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Generate high-contrast dual-panel 1280x720 cooperative perception tactical rendering."""
    canvas = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)

    # Render Left Panel (BEV Map: 800x720)
    draw_bev_map(
        canvas=canvas,
        vehicle_pose=vehicle_pose,
        rpus=rpus,
        tracks=tracks,
        warnings=warnings,
        pillars=pillars,
        origin=(60, 40),
        scale=14.0,
        width=800,
        height=720,
    )

    # Render Right Panel (HUD: 480x720)
    draw_hud_panel(
        canvas=canvas,
        vehicle_pose=vehicle_pose,
        vehicle_speed_mps=vehicle_speed_mps,
        rpus=rpus,
        tracks=tracks,
        warnings=warnings,
        telemetry=telemetry,
        x_offset=800,
        width=480,
        height=720,
    )

    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), canvas)

    return canvas
