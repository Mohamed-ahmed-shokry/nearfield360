"""Tactical multi-modal FMCW radar Range-Doppler and ultrasonic BEV dashboard visualizer."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.sensor.models import RangeDopplerMap, SensorSimulationReport

# Tactical automotive color palette (BGR)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_GRID = (45, 50, 60)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

COLOR_RADAR = (255, 190, 40)  # Cyan
COLOR_USS_CLEAR = (60, 210, 95)  # Green
COLOR_USS_WARN = (40, 180, 240)  # Amber
COLOR_USS_ALERT = (50, 50, 235)  # Red
COLOR_VEHICLE = (180, 140, 100)  # Blue-grey
COLOR_FUSED_OBS = (0, 215, 255)  # Gold/Yellow


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


def render_sensor_dashboard(
    report: SensorSimulationReport,
    rd_map: RangeDopplerMap | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render a 1280x720 tactical FMCW Range-Doppler and ultrasonic BEV dashboard PNG."""
    width, height = 1280, 720
    canvas: NDArray[np.uint8] = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # 1. Header Banner
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (20, 15), (width - 20, 65), COLOR_CARD_BORDER, 1)
    _draw_text(
        canvas,
        "NEARFIELD360 // MULTI-MODAL RADAR & ULTRASONIC SENSOR SUITE",
        (35, 46),
        0.65,
        COLOR_TEXT_TITLE,
        2,
    )

    header_status = (
        f"TIME: {report.timestamp_s:.2f}s | "
        f"RADAR: {report.radar_points_count} pts | "
        f"USS ACTIVE: {report.ultrasonic_echoes_count} | "
        f"FUSED: {report.fused_obstacles_count}"
    )
    _draw_text(canvas, header_status, (700, 46), 0.50, COLOR_RADAR, 1)

    # Panel geometry: Left (Range-Doppler), Right (Vehicle BEV)
    panel_y, panel_h = 80, 615
    panel_w = 600

    # 2. Left Panel: 2D Range-Doppler Heatmap
    left_x = 20
    cv2.rectangle(
        canvas, (left_x, panel_y), (left_x + panel_w, panel_y + panel_h), COLOR_PANEL_BG, -1
    )
    cv2.rectangle(
        canvas, (left_x, panel_y), (left_x + panel_w, panel_y + panel_h), COLOR_CARD_BORDER, 1
    )
    _draw_text(
        canvas,
        "FMCW 77 GHz RANGE-DOPPLER SPECTRUM (2D FFT)",
        (left_x + 15, panel_y + 28),
        0.52,
        COLOR_TEXT_TITLE,
        1,
    )

    plot_x, plot_y = left_x + 60, panel_y + 50
    plot_w, plot_h = 510, 500
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (10, 12, 16), -1)

    if rd_map is not None and len(rd_map.matrix_db) > 0:
        mat_arr = np.array(rd_map.matrix_db, dtype=np.float32)
        # Normalize to 0-255
        p_min, p_max = rd_map.min_power_db, rd_map.max_power_db
        p_range = max(1.0, p_max - p_min)
        norm_mat = np.clip((mat_arr - p_min) / p_range * 255.0, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(norm_mat, cv2.COLORMAP_TURBO)
        resized_heatmap = cv2.resize(
            color_heatmap, (plot_w, plot_h), interpolation=cv2.INTER_LINEAR
        )
        canvas[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = resized_heatmap

    # Axes grid lines and labels
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), COLOR_GRID, 1)
    # Zero velocity center line
    mid_y = plot_y + plot_h // 2
    cv2.line(canvas, (plot_x, mid_y), (plot_x + plot_w, mid_y), (100, 100, 120), 1, cv2.LINE_AA)

    _draw_text(
        canvas, "Range (0m -> 50m)", (plot_x + 180, plot_y + plot_h + 25), 0.45, COLOR_TEXT_MUTED, 1
    )
    _draw_text(canvas, "+15 m/s", (left_x + 5, plot_y + 15), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, "  0 m/s", (left_x + 5, mid_y + 5), 0.38, COLOR_TEXT_MUTED, 1)
    _draw_text(canvas, "-15 m/s", (left_x + 5, plot_y + plot_h - 5), 0.38, COLOR_TEXT_MUTED, 1)

    # Overlay detected radar targets on Range-Doppler plot
    for target in report.radar_point_cloud.points:
        # map range [0, 50m] to [0, plot_w], velocity [-15, +15m/s] to [plot_h, 0]
        tx = int(plot_x + (target.range_m / 50.0) * plot_w)
        ty = int(mid_y - (target.radial_velocity_mps / 15.0) * (plot_h / 2.0))
        tx = int(np.clip(tx, plot_x, plot_x + plot_w))
        ty = int(np.clip(ty, plot_y, plot_y + plot_h))

        cv2.circle(canvas, (tx, ty), 6, COLOR_USS_ALERT, 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 2, (255, 255, 255), -1)
        lbl = f"R:{target.range_m:.1f}m v:{target.radial_velocity_mps:.1f}"
        _draw_text(canvas, lbl, (tx + 8, ty - 5), 0.36, COLOR_TEXT_TITLE, 1)

    # 3. Right Panel: Vehicle BEV & Ultrasonic Acoustic Beam Map
    right_x = left_x + panel_w + 20
    cv2.rectangle(
        canvas, (right_x, panel_y), (right_x + panel_w, panel_y + panel_h), COLOR_PANEL_BG, -1
    )
    cv2.rectangle(
        canvas, (right_x, panel_y), (right_x + panel_w, panel_y + panel_h), COLOR_CARD_BORDER, 1
    )
    _draw_text(
        canvas,
        "VEHICLE BEV ACOUSTIC LOBES & CROSS-MODAL FUSION",
        (right_x + 15, panel_y + 28),
        0.52,
        COLOR_TEXT_TITLE,
        1,
    )

    bev_x, bev_y = right_x + 15, panel_y + 50
    bev_w, bev_h = 570, 545
    cv2.rectangle(canvas, (bev_x, bev_y), (bev_x + bev_w, bev_y + bev_h), (12, 14, 18), -1)

    # BEV Origin (center vehicle footprint): origin at center of display
    # World frame: X forward (upward on screen), Y left (leftward on screen)
    cx, cy = bev_x + bev_w // 2, bev_y + bev_h // 2 + 30
    scale_px_per_m = 32.0  # 32 pixels per meter

    def world_to_bev(wx: float, wy: float) -> tuple[int, int]:
        # X is forward (negative Y in pixel space), Y is left (negative X in pixel space)
        bx = int(cx - wy * scale_px_per_m)
        by = int(cy - wx * scale_px_per_m)
        return bx, by

    # Draw range rings (2m, 4m, 6m, 8m)
    for radius_m in [2.0, 4.0, 6.0, 8.0]:
        r_px = int(radius_m * scale_px_per_m)
        cv2.circle(canvas, (cx, cy), r_px, COLOR_GRID, 1, cv2.LINE_AA)
        _draw_text(canvas, f"{radius_m:.0f}m", (cx + r_px + 2, cy - 2), 0.35, COLOR_TEXT_MUTED, 1)

    # Draw Vehicle Body Polygon (Length 4.8m, Width 1.9m, rear axle at (0, 0))
    # Front bumper at X = 3.8m, Rear bumper at X = -1.0m, Width +/- 0.95m
    vf_fl = world_to_bev(3.8, 0.95)
    vf_fr = world_to_bev(3.8, -0.95)
    vf_rr = world_to_bev(-1.0, -0.95)
    vf_rl = world_to_bev(-1.0, 0.95)
    vehicle_poly = np.array([vf_fl, vf_fr, vf_rr, vf_rl], dtype=np.int32)
    cv2.fillPoly(canvas, [vehicle_poly], (45, 55, 70))
    cv2.polylines(canvas, [vehicle_poly], True, COLOR_VEHICLE, 2, cv2.LINE_AA)

    # Front orientation arrow
    arrow_tip = world_to_bev(4.2, 0.0)
    arrow_base = world_to_bev(3.2, 0.0)
    cv2.arrowedLine(canvas, arrow_base, arrow_tip, COLOR_RADAR, 2, tipLength=0.3)

    # Render Ultrasonic Transducer Beams
    for echo in report.ultrasonic_frame.echoes:
        pose = echo.sensor_pose
        sx, sy = world_to_bev(pose.x_m, pose.y_m)
        # Small circle for transducer
        cv2.circle(canvas, (sx, sy), 3, (200, 200, 220), -1)

        # Beam cone sector
        beam_half_deg = 30.0
        # Convert yaw_rad to screen degrees (0 rad is X forward = -90 deg on screen)
        screen_yaw_deg = -math.degrees(pose.yaw_rad) - 90.0
        cone_radius_px = int(echo.range_m * scale_px_per_m)

        color_cone = COLOR_USS_ALERT if echo.in_detection_cone else COLOR_USS_CLEAR
        thickness = 2 if echo.in_detection_cone else 1

        cv2.ellipse(
            canvas,
            (sx, sy),
            (cone_radius_px, cone_radius_px),
            screen_yaw_deg,
            -beam_half_deg,
            beam_half_deg,
            color_cone,
            thickness,
            cv2.LINE_AA,
        )

        if echo.in_detection_cone:
            _draw_text(canvas, f"{echo.range_m:.2f}m", (sx + 5, sy - 5), 0.32, color_cone, 1)

    # Render Fused Obstacles
    for obs in report.fused_obstacles:
        ox, oy = world_to_bev(obs.x_m, obs.y_m)
        cv2.rectangle(canvas, (ox - 10, oy - 10), (ox + 10, oy + 10), COLOR_FUSED_OBS, 2)
        cv2.circle(canvas, (ox, oy), 3, (255, 255, 255), -1)

        # Radial velocity vector
        if abs(obs.radial_velocity_mps) > 0.1:
            vx_end = int(ox)
            vy_end = int(oy - obs.radial_velocity_mps * scale_px_per_m * 0.4)
            cv2.arrowedLine(canvas, (ox, oy), (vx_end, vy_end), COLOR_RADAR, 2, tipLength=0.3)

        info_lbl = f"{obs.obstacle_id} ({'/'.join(obs.source_modalities)})"
        _draw_text(canvas, info_lbl, (ox + 12, oy - 2), 0.35, COLOR_TEXT_TITLE, 1)
        _draw_text(
            canvas,
            f"R:{obs.min_range_m:.2f}m c:{obs.confidence:.2f}",
            (ox + 12, oy + 12),
            0.32,
            COLOR_RADAR,
            1,
        )

    # Save output if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
