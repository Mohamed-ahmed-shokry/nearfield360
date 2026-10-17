"""Visual rendering engine for life safety tactical BEV maps and situational HUD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.emergency.models import (
    EmergencyVehicle,
    EvacuationRoute,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    LifeSafetyReport,
    VehicleEvacuationStatus,
)

# Color constants (BGR)
COLOR_BG = (20, 22, 25)
COLOR_PANEL_BG = (28, 30, 36)
COLOR_GRID = (40, 44, 52)
COLOR_PORTAL_SAFE = (70, 200, 90)  # Green
COLOR_HAZARD_CORE = (40, 40, 230)  # Bright red
COLOR_HAZARD_HALO = (30, 80, 220)  # Red-orange
COLOR_DOOR_OPEN = (70, 200, 90)
COLOR_DOOR_CLOSING = (30, 180, 255)  # Amber
COLOR_DOOR_LOCKED = (40, 40, 240)  # Red
COLOR_RESPONDER = (0, 215, 255)  # Amber-gold
COLOR_RESPONDER_SIREN = (255, 100, 255)  # Magenta siren wave
COLOR_VEHICLE_TRAPPED = (30, 30, 240)  # Red
COLOR_VEHICLE_EVACUATING = (220, 160, 40)  # Cyan
COLOR_VEHICLE_YIELDING = (40, 200, 240)  # Yellow
COLOR_VEHICLE_SAFE = (80, 180, 80)  # Green
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (210, 215, 225)
COLOR_TEXT_MUTED = (135, 140, 150)
COLOR_CARD_BG = (36, 40, 48)


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


def render_emergency_dashboard(
    report: LifeSafetyReport,
    *,
    hazards: list[HazardZone] | None = None,
    fire_doors: list[FireDoorInterlock] | None = None,
    responders: list[EmergencyVehicle] | None = None,
    routes: Mapping[str, EvacuationRoute] | None = None,
    vehicle_poses: Mapping[str, tuple[float, float]] | None = None,
    vehicle_statuses: Mapping[str, VehicleEvacuationStatus] | None = None,
    exit_portals: Mapping[str, tuple[float, float]] | None = None,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel tactical BEV incident map and life safety command HUD."""
    panel_width = 440
    width = canvas_size + panel_width
    height = canvas_size

    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    canvas[:, canvas_size:] = COLOR_PANEL_BG

    cv2.line(canvas, (canvas_size, 0), (canvas_size, height), (55, 60, 72), 2)

    # Grid parameters: mapping facility [-20, 120] to canvas
    scale = canvas_size / 140.0
    origin = (int(30 * scale), int(30 * scale))

    # Draw tactical grid lines
    for gx in range(-20, 120, 20):
        p1 = _world_to_canvas(float(gx), -20.0, origin, scale, height)
        p2 = _world_to_canvas(float(gx), 100.0, origin, scale, height)
        cv2.line(canvas, p1, p2, COLOR_GRID, 1)

    for gy in range(-20, 120, 20):
        p1 = _world_to_canvas(-20.0, float(gy), origin, scale, height)
        p2 = _world_to_canvas(100.0, float(gy), origin, scale, height)
        cv2.line(canvas, p1, p2, COLOR_GRID, 1)

    # Render Exit Portals
    portals = exit_portals or {"Exit-North": (0.0, 90.0), "Exit-South": (0.0, -10.0)}
    for pname, (px, py) in portals.items():
        cx, cy = _world_to_canvas(px, py, origin, scale, height)
        cv2.rectangle(canvas, (cx - 16, cy - 8), (cx + 16, cy + 8), COLOR_PORTAL_SAFE, -1)
        _put_text(canvas, pname, (cx - 30, cy - 12), 0.38, COLOR_PORTAL_SAFE, 1)

    # Render Hazard Zones
    active_hz = hazards if hazards is not None else report.active_hazards
    for hz in active_hz:
        if not hz.active:
            continue
        cx, cy = _world_to_canvas(hz.center_x, hz.center_y, origin, scale, height)
        pix_r = max(5, int(hz.radius_m * scale))

        # Concentric danger rings
        cv2.circle(canvas, (cx, cy), pix_r, COLOR_HAZARD_CORE, 2)
        cv2.circle(canvas, (cx, cy), int(pix_r * 1.3), COLOR_HAZARD_HALO, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, COLOR_HAZARD_CORE, -1)
        _put_text(
            canvas, f"HAZARD: {hz.hazard_id}", (cx - 40, cy - pix_r - 6), 0.40, COLOR_HAZARD_CORE, 1
        )

    # Render Fire Doors
    active_doors = fire_doors if fire_doors is not None else report.fire_doors
    for door in active_doors:
        cx, cy = _world_to_canvas(door.location_x, door.location_y, origin, scale, height)
        door_color = COLOR_DOOR_OPEN
        if door.status == FireDoorStatus.CLOSING:
            door_color = COLOR_DOOR_CLOSING
        elif door.status == FireDoorStatus.LOCKED_SHUT:
            door_color = COLOR_DOOR_LOCKED

        cv2.line(canvas, (cx - 15, cy), (cx + 15, cy), door_color, 4)
        _put_text(
            canvas, f"{door.door_id} ({door.status.value})", (cx - 30, cy + 18), 0.35, door_color, 1
        )

    # Render Evacuation Routes
    if routes:
        for route in routes.values():
            if len(route.waypoints) >= 2:
                pts = [
                    _world_to_canvas(wp.x_m, wp.y_m, origin, scale, height)
                    for wp in route.waypoints
                ]
                route_color = COLOR_PORTAL_SAFE if route.is_cleared else COLOR_HAZARD_CORE
                for idx in range(len(pts) - 1):
                    cv2.line(canvas, pts[idx], pts[idx + 1], route_color, 2, cv2.LINE_AA)

    # Render Vehicles
    poses = vehicle_poses or {}
    statuses = vehicle_statuses or {}
    for vid, (vx, vy) in poses.items():
        cx, cy = _world_to_canvas(vx, vy, origin, scale, height)
        vstatus = statuses.get(vid, VehicleEvacuationStatus.STATIONARY_SAFE)

        vcolor = COLOR_VEHICLE_SAFE
        if vstatus == VehicleEvacuationStatus.TRAPPED_IN_HAZARD:
            vcolor = COLOR_VEHICLE_TRAPPED
        elif vstatus == VehicleEvacuationStatus.EVACUATING:
            vcolor = COLOR_VEHICLE_EVACUATING
        elif vstatus == VehicleEvacuationStatus.HOLD_YIELDING:
            vcolor = COLOR_VEHICLE_YIELDING

        cv2.circle(canvas, (cx, cy), 8, vcolor, -1)
        cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 1)
        _put_text(canvas, vid, (cx - 15, cy - 12), 0.38, (230, 230, 230), 1)

    # Render Emergency Responders
    active_resp = responders if responders is not None else report.emergency_responders
    for resp in active_resp:
        rx, ry = _world_to_canvas(resp.x, resp.y, origin, scale, height)
        cv2.circle(canvas, (rx, ry), 12, COLOR_RESPONDER, -1)
        if resp.active_siren:
            cv2.circle(canvas, (rx, ry), 22, COLOR_RESPONDER_SIREN, 2, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 32, COLOR_RESPONDER_SIREN, 1, cv2.LINE_AA)

        # Direction vector
        arrow_dx = int(18 * math.cos(resp.yaw_rad))
        arrow_dy = int(-18 * math.sin(resp.yaw_rad))
        cv2.arrowedLine(canvas, (rx, ry), (rx + arrow_dx, ry + arrow_dy), (255, 255, 255), 2)

        _put_text(
            canvas,
            f"{resp.responder_type.value.upper()} [{resp.vehicle_id}]",
            (rx - 45, ry - 38),
            0.42,
            COLOR_RESPONDER,
            1,
        )

    # ---------------- RIGHT PANEL: LIFE SAFETY COMMAND HUD ----------------
    px = canvas_size + 20
    py = 35

    _put_text(canvas, "LIFE SAFETY TACTICAL HUD", (px, py), 0.70, COLOR_TEXT_TITLE, 2)
    py += 24
    _put_text(canvas, f"Facility: {report.facility_name}", (px, py), 0.42, COLOR_TEXT_MUTED, 1)

    py += 35
    # Severity & Alert Badge
    badge_color = COLOR_HAZARD_CORE if report.emergency_active else COLOR_PORTAL_SAFE
    alert_txt = (
        f"ALERT: {report.severity.value.upper()}"
        if report.emergency_active
        else "NORMAL OPERATIONS"
    )
    cv2.rectangle(canvas, (px, py), (px + 400, py + 38), badge_color, -1)
    _put_text(canvas, alert_txt, (px + 20, py + 26), 0.58, (255, 255, 255), 2)

    py += 55
    # Card 1: Incident Summary
    cv2.rectangle(canvas, (px, py), (px + 400, py + 95), COLOR_CARD_BG, -1)
    _put_text(canvas, "INCIDENT OVERVIEW", (px + 12, py + 24), 0.48, COLOR_TEXT_TITLE, 1)
    _put_text(
        canvas, f"Type: {report.incident_type.value}", (px + 12, py + 48), 0.42, COLOR_TEXT_BODY, 1
    )
    _put_text(
        canvas, f"Active Hazards: {len(active_hz)}", (px + 12, py + 70), 0.42, COLOR_TEXT_BODY, 1
    )
    evp_status = "ACTIVE (PREEMPTION ENGAGED)" if report.evp_active else "INACTIVE"
    _put_text(canvas, f"EVP Preemption: {evp_status}", (px + 12, py + 88), 0.40, COLOR_RESPONDER, 1)

    py += 112
    # Card 2: Evacuation Progress & Fleet Status
    cv2.rectangle(canvas, (px, py), (px + 400, py + 120), COLOR_CARD_BG, -1)
    _put_text(canvas, "EVACUATION COUNTERS", (px + 12, py + 24), 0.48, COLOR_TEXT_TITLE, 1)
    _put_text(
        canvas,
        f"Total Registered Vehicles: {report.total_vehicles}",
        (px + 12, py + 48),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    _put_text(
        canvas,
        f"Evacuated: {report.evacuated_count}",
        (px + 12, py + 68),
        0.42,
        COLOR_PORTAL_SAFE,
        1,
    )
    _put_text(
        canvas,
        f"Evacuating: {report.evacuating_count}",
        (px + 12, py + 86),
        0.42,
        COLOR_VEHICLE_EVACUATING,
        1,
    )
    _put_text(
        canvas,
        f"Yielding (EVP): {report.yielding_count}",
        (px + 12, py + 104),
        0.42,
        COLOR_VEHICLE_YIELDING,
        1,
    )
    trapped_color = COLOR_HAZARD_CORE if report.trapped_count > 0 else COLOR_PORTAL_SAFE
    _put_text(
        canvas,
        f"Trapped in Hazard: {report.trapped_count}",
        (px + 220, py + 104),
        0.42,
        trapped_color,
        1,
    )

    py += 135
    # Card 3: First Responders & Fire Barriers
    cv2.rectangle(canvas, (px, py), (px + 400, py + 105), COLOR_CARD_BG, -1)
    _put_text(canvas, "LIFE SAFETY INFRASTRUCTURE", (px + 12, py + 24), 0.48, COLOR_TEXT_TITLE, 1)
    _put_text(
        canvas,
        f"First Responders on Scene: {len(active_resp)}",
        (px + 12, py + 50),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    _put_text(
        canvas,
        f"Fire Doors / Shutters: {len(active_doors)}",
        (px + 12, py + 72),
        0.42,
        COLOR_TEXT_BODY,
        1,
    )
    _put_text(
        canvas,
        "ISO 23374 / NFPA 88A Life Safety Validated",
        (px + 12, py + 94),
        0.38,
        COLOR_PORTAL_SAFE,
        1,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
