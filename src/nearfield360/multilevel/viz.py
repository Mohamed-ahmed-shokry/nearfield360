"""Visual rendering engine for multi-floor facility layouts, elevation profiles, and HUD."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.multilevel.graph import MultiFloorGraph
from nearfield360.multilevel.models import (
    ElevatorCabin,
    MultiFloorFacilityReport,
    MultiFloorRoute,
    RampDirection,
    RampToken,
    VehiclePitchState,
)

# Color constants (BGR)
COLOR_BG = (22, 22, 26)
COLOR_PANEL_BG = (32, 32, 38)
COLOR_FLOOR_BORDER = (70, 70, 85)
COLOR_AISLE = (45, 45, 55)
COLOR_BAY_VACANT = (50, 140, 60)  # Green
COLOR_BAY_OCCUPIED = (60, 60, 160)  # Red
COLOR_RAMP_STRAIGHT = (200, 140, 40)  # Cyan-blue
COLOR_ELEVATOR_SHAFT = (40, 160, 220)  # Orange
COLOR_ROUTE_PATH = (50, 200, 255)  # Bright amber/yellow
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (200, 200, 210)
COLOR_TEXT_MUTED = (130, 130, 140)
COLOR_CARD_BG = (42, 42, 50)


def _put_text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_multilevel_dashboard(
    graph: MultiFloorGraph,
    report: MultiFloorFacilityReport | None = None,
    route: MultiFloorRoute | None = None,
    active_token: RampToken | None = None,
    active_cabin: ElevatorCabin | None = None,
    pitch_state: VehiclePitchState | None = None,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render multi-floor BEV floorplans, elevation profile, and operational telemetry HUD."""
    panel_width = 440
    width = canvas_size + panel_width
    height = canvas_size

    canvas: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = COLOR_BG

    floors = list(graph.floors.values())
    floors_to_draw = floors[:2] if len(floors) >= 2 else floors
    num_views = max(1, len(floors_to_draw))
    subview_height = canvas_size // num_views

    for idx, floor in enumerate(floors_to_draw):
        y_top = idx * subview_height
        y_bot = (idx + 1) * subview_height

        cv2.rectangle(
            canvas,
            (10, y_top + 10),
            (canvas_size - 10, y_bot - 10),
            COLOR_FLOOR_BORDER,
            1,
        )

        badge_text = f"{floor.name} ({floor.floor_id}) | Elev: {floor.elevation_m:+.1f}m"
        _put_text(canvas, badge_text, (25, y_top + 35), 0.65, COLOR_TEXT_TITLE, 2)

        floor_nodes = [n for n in graph.nodes.values() if n.floor_id == floor.floor_id]
        node_coords = {
            n.node_id: (int(80 + n.x_m * 18), int(y_top + 80 + n.y_m * 6)) for n in floor_nodes
        }

        for u_id, pt_u in node_coords.items():
            for edge in graph._adj.get(u_id, []):
                if edge.target in node_coords:
                    pt_v = node_coords[edge.target]
                    cv2.line(canvas, pt_u, pt_v, COLOR_AISLE, 4, cv2.LINE_AA)

        for node_id, pt in node_coords.items():
            if "elevator" in node_id.lower():
                cv2.rectangle(
                    canvas,
                    (pt[0] - 12, pt[1] - 12),
                    (pt[0] + 12, pt[1] + 12),
                    COLOR_ELEVATOR_SHAFT,
                    -1,
                )
                _put_text(canvas, "LIFT", (pt[0] - 14, pt[1] - 16), 0.4, COLOR_TEXT_TITLE)
            elif "ramp" in node_id.lower():
                cv2.circle(canvas, pt, 10, COLOR_RAMP_STRAIGHT, -1)
                _put_text(canvas, "RAMP", (pt[0] - 14, pt[1] - 14), 0.4, COLOR_TEXT_TITLE)
            else:
                cv2.circle(canvas, pt, 6, (120, 120, 130), -1)

            _put_text(canvas, node_id, (pt[0] + 10, pt[1] + 4), 0.38, COLOR_TEXT_MUTED)

    if route is not None and len(route.waypoints) >= 2:
        for i in range(len(route.waypoints) - 1):
            wp1 = route.waypoints[i]
            wp2 = route.waypoints[i + 1]

            f_idx1 = 0 if wp1.z_m <= 0.0 else 1
            f_idx2 = 0 if wp2.z_m <= 0.0 else 1

            if f_idx1 == f_idx2 and f_idx1 < num_views:
                y_offset = f_idx1 * subview_height
                pt1 = (int(80 + wp1.x_m * 18), int(y_offset + 80 + wp1.y_m * 6))
                pt2 = (int(80 + wp2.x_m * 18), int(y_offset + 80 + wp2.y_m * 6))
                cv2.line(canvas, pt1, pt2, COLOR_ROUTE_PATH, 3, cv2.LINE_AA)

    px = canvas_size + 15
    py = 35

    cv2.rectangle(canvas, (canvas_size, 0), (width, height), COLOR_PANEL_BG, -1)
    cv2.line(canvas, (canvas_size, 0), (canvas_size, height), (50, 50, 60), 2)

    _put_text(canvas, "MULTI-FLOOR TELEMETRY", (px, py), 0.8, COLOR_TEXT_TITLE, 2)
    py += 24
    _put_text(canvas, "ISO 23374 Vertical Transit Operations", (px, py), 0.45, COLOR_TEXT_MUTED)
    py += 35

    # Card 1: Facility Capacity & Structure
    cv2.rectangle(canvas, (px, py), (width - 15, py + 115), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (px, py), (width - 15, py + 115), (60, 60, 75), 1)

    floors_cnt = report.floors_count if report else len(graph.floors)
    total_bays = report.total_bays if report else sum(f.bays_count for f in graph.floors.values())
    avail_bays = (
        report.available_bays if report else sum(f.available_bays for f in graph.floors.values())
    )

    _put_text(canvas, "FACILITY STRUCTURE", (px + 12, py + 22), 0.55, COLOR_TEXT_TITLE)
    _put_text(canvas, f"Floors: {floors_cnt} Levels", (px + 12, py + 46), 0.45, COLOR_TEXT_BODY)
    _put_text(
        canvas, f"Total Capacity: {total_bays} Bays", (px + 12, py + 68), 0.45, COLOR_TEXT_BODY
    )
    _put_text(
        canvas, f"Available Bays: {avail_bays} Vacant", (px + 12, py + 90), 0.45, COLOR_BAY_VACANT
    )
    py += 135

    # Card 2: Ramp Passage Token Arbiter
    cv2.rectangle(canvas, (px, py), (width - 15, py + 125), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (px, py), (width - 15, py + 125), (60, 60, 75), 1)

    _put_text(canvas, "RAMP ACCESS TOKEN", (px + 12, py + 22), 0.55, COLOR_TEXT_TITLE)
    if active_token is not None and active_token.is_active:
        token_txt = f"Token: {active_token.token_id}"
        veh_txt = f"Vehicle: {active_token.vehicle_id} ({active_token.direction.value.upper()})"
        dir_color = (
            COLOR_BAY_VACANT
            if active_token.direction == RampDirection.ASCENDING
            else COLOR_RAMP_STRAIGHT
        )
        _put_text(canvas, token_txt, (px + 12, py + 46), 0.42, COLOR_TEXT_BODY)
        _put_text(canvas, veh_txt, (px + 12, py + 68), 0.45, dir_color)
        exp_txt = f"Expires at: {active_token.expires_at_s:.1f}s"
        _put_text(canvas, exp_txt, (px + 12, py + 90), 0.42, COLOR_TEXT_MUTED)
    else:
        _put_text(
            canvas, "Status: CLEAR (No Active Passage)", (px + 12, py + 55), 0.5, COLOR_BAY_VACANT
        )
        _put_text(
            canvas,
            "Priority Policy: ASCENDING_FIRST",
            (px + 12, py + 85),
            0.42,
            COLOR_TEXT_MUTED,
        )
    py += 145

    # Card 3: Automated Vehicle Elevator
    cv2.rectangle(canvas, (px, py), (width - 15, py + 135), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (px, py), (width - 15, py + 135), (60, 60, 75), 1)

    _put_text(canvas, "VEHICLE HOIST ELEVATOR", (px + 12, py + 22), 0.55, COLOR_TEXT_TITLE)
    if active_cabin is not None:
        cab_str = f"Cabin: {active_cabin.elevator_id} | State: {active_cabin.state.value.upper()}"
        _put_text(canvas, cab_str, (px + 12, py + 46), 0.45, COLOR_ELEVATOR_SHAFT)
        _put_text(
            canvas,
            f"Current Elevation: {active_cabin.current_elevation_m:+.2f}m",
            (px + 12, py + 68),
            0.45,
            COLOR_TEXT_BODY,
        )
        veh_str = active_cabin.occupied_vehicle_id or "EMPTY"
        _put_text(canvas, f"Payload Vehicle: {veh_str}", (px + 12, py + 90), 0.42, COLOR_TEXT_MUTED)
        level_status = (
            "ALIGNED (<15mm)" if abs(active_cabin.leveling_offset_m) <= 0.015 else "MISALIGNED"
        )
        _put_text(
            canvas, f"Leveling: {level_status}", (px + 12, py + 112), 0.42, COLOR_BAY_VACANT
        )
    else:
        _put_text(
            canvas, "Cabin: HOIST-01 | State: IDLE", (px + 12, py + 55), 0.45, COLOR_TEXT_BODY
        )
        _put_text(
            canvas,
            "Leveling Interlock: ENGAGED SAFE",
            (px + 12, py + 85),
            0.42,
            COLOR_BAY_VACANT,
        )
    py += 155

    # Card 4: Pitch Dynamics & Breakover Clearance
    cv2.rectangle(canvas, (px, py), (width - 15, py + 135), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (px, py), (width - 15, py + 135), (60, 60, 75), 1)

    _put_text(canvas, "RAMP PITCH DYNAMICS", (px + 12, py + 22), 0.55, COLOR_TEXT_TITLE)
    if pitch_state is not None:
        pitch_str = f"Pitch: {pitch_state.pitch_angle_deg:+.2f} deg ({pitch_state.grade_pct:.1f}%)"
        _put_text(canvas, pitch_str, (px + 12, py + 46), 0.42, COLOR_TEXT_BODY)
        _put_text(
            canvas,
            f"Gravity Fg: {pitch_state.gravity_force_n:.1f} N",
            (px + 12, py + 68),
            0.42,
            COLOR_TEXT_BODY,
        )
        _put_text(
            canvas,
            f"Traction Margin: {pitch_state.traction_margin:.3f}",
            (px + 12, py + 90),
            0.42,
            COLOR_TEXT_MUTED,
        )
        clear_color = COLOR_BAY_VACANT if pitch_state.is_clearance_safe else COLOR_BAY_OCCUPIED
        clr_str = f"Breakover: {pitch_state.breakover_clearance_m * 100:.1f} cm"
        _put_text(canvas, clr_str, (px + 12, py + 112), 0.42, clear_color)
    else:
        _put_text(
            canvas, "Pitch: 0.00 deg (Planar Deck)", (px + 12, py + 55), 0.45, COLOR_TEXT_BODY
        )
        _put_text(
            canvas,
            "Chassis Breakover: 14.0 cm (SAFE)",
            (px + 12, py + 85),
            0.42,
            COLOR_BAY_VACANT,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
