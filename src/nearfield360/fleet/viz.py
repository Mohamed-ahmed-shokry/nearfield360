"""Visual rendering engine for multi-vehicle fleet coordination and conflict telemetry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.fleet.coordinator import SimulatedVehicle
from nearfield360.fleet.models import ConflictPoint, FleetDispatchReport
from nearfield360.mapping.models import FacilityMap, SlotReservationStatus

# Color constants (BGR)
COLOR_BG = (28, 28, 28)
COLOR_LANE = (70, 70, 75)
COLOR_LANE_CENTER = (130, 130, 140)
COLOR_OBSTACLE = (80, 80, 90)
COLOR_VACANT = (40, 180, 60)
COLOR_RESERVED = (40, 160, 220)
COLOR_OCCUPIED = (50, 50, 200)
COLOR_CONFLICT_ZONE = (0, 0, 255)
COLOR_VEHICLE_FILL = (0, 215, 255)
COLOR_VEHICLE_YIELD = (255, 140, 0)
COLOR_TEXT = (240, 240, 240)


def render_fleet_coordination_dashboard(
    facility: FacilityMap,
    vehicles: Sequence[SimulatedVehicle],
    conflicts: Sequence[ConflictPoint] | None = None,
    report: FleetDispatchReport | None = None,
    *,
    canvas_size: int = 900,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Render dual-panel multi-vehicle BEV layout and fleet coordination telemetry dashboard."""
    width = canvas_size + 380
    height = canvas_size
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    margin = 60
    span_x = max(1.0, facility.bounds.x_max - facility.bounds.x_min)
    span_y = max(1.0, facility.bounds.y_max - facility.bounds.y_min)
    max_span = max(span_x, span_y)
    scale = (canvas_size - 2 * margin) / max_span

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        px = round(margin + (x - facility.bounds.x_min) * scale)
        py = round(canvas_size - margin - (y - facility.bounds.y_min) * scale)
        return (px, py)

    # 1. Draw structural obstacles
    for obs in facility.obstacles:
        pts = np.array([to_pixel(x, y) for x, y in obs.polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_OBSTACLE, thickness=4)

    # 2. Draw lanes and centerlines
    for lane in facility.lanes:
        pts = np.array([to_pixel(x, y) for x, y in lane.centerline_points], dtype=np.int32)
        lw_px = max(4, round(lane.width_m * scale))
        cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_LANE, thickness=lw_px)
        cv2.polylines(canvas, [pts], isClosed=False, color=COLOR_LANE_CENTER, thickness=1)

    # 3. Draw parking slots
    for slot in facility.slots:
        corners_px = np.array([to_pixel(x, y) for x, y in slot.corners], dtype=np.int32)
        status_col = (
            COLOR_VACANT
            if slot.status == SlotReservationStatus.VACANT
            else COLOR_RESERVED
            if slot.status == SlotReservationStatus.RESERVED
            else COLOR_OCCUPIED
        )
        cv2.fillPoly(
            canvas, [corners_px], color=(status_col[0] // 3, status_col[1] // 3, status_col[2] // 3)
        )
        cv2.polylines(canvas, [corners_px], isClosed=True, color=status_col, thickness=2)

    # 4. Draw conflict points
    if conflicts:
        for c in conflicts:
            cpx, cpy = to_pixel(c.conflict_location[0], c.conflict_location[1])
            # Draw pulsing conflict halo
            cv2.circle(canvas, (cpx, cpy), 24, (0, 0, 180), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cpx, cpy), 12, COLOR_CONFLICT_ZONE, thickness=-1)
            cv2.putText(
                canvas,
                c.conflict_id,
                (cpx + 14, cpy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_CONFLICT_ZONE,
                1,
                cv2.LINE_AA,
            )

    # 5. Draw vehicle trajectories and footprints
    for veh in vehicles:
        if veh.trajectory:
            traj_pts = np.array(
                [to_pixel(wp.x, wp.y) for wp in veh.trajectory.waypoints], dtype=np.int32
            )
            cv2.polylines(canvas, [traj_pts], isClosed=False, color=(200, 180, 0), thickness=2)

        # Vehicle oriented bounding box
        vx, vy, v_theta = veh.pose
        v_px, v_py = to_pixel(vx, vy)
        v_length = 4.5
        v_width = 1.9

        # 4 corners in vehicle body frame
        hl, hw = v_length / 2.0, v_width / 2.0
        cos_t, sin_t = math.cos(v_theta), math.sin(v_theta)
        body_corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        box_corners_px: list[tuple[int, int]] = []
        for bx, by in body_corners:
            gx = vx + (bx * cos_t - by * sin_t)
            gy = vy + (bx * sin_t + by * cos_t)
            box_corners_px.append(to_pixel(gx, gy))

        v_color = COLOR_VEHICLE_YIELD if veh.speed_mps < 0.1 else COLOR_VEHICLE_FILL
        poly_pts = np.array(box_corners_px, dtype=np.int32)
        cv2.fillPoly(canvas, [poly_pts], color=v_color)
        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(255, 255, 255), thickness=1)

        # Heading indicator
        head_x = vx + (hl + 0.8) * cos_t
        head_y = vy + (hl + 0.8) * sin_t
        h_px, h_py = to_pixel(head_x, head_y)
        cv2.arrowedLine(
            canvas, (v_px, v_py), (h_px, h_py), color=(255, 255, 255), thickness=2, tipLength=0.3
        )

        # Vehicle ID badge
        cv2.putText(
            canvas,
            veh.vehicle_id,
            (v_px - 20, v_py - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    # 6. Telemetry HUD Sidebar (Right Panel)
    panel_x = canvas_size + 15
    cv2.rectangle(canvas, (canvas_size, 0), (width, height), (38, 38, 42), thickness=-1)
    cv2.line(canvas, (canvas_size, 0), (canvas_size, height), (80, 80, 90), thickness=2)

    cv2.putText(
        canvas,
        "FLEET COORDINATION HUD",
        (panel_x, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (panel_x, 52), (width - 20, 52), (100, 100, 110), thickness=1)

    y_pos = 85
    cv2.putText(
        canvas,
        f"Facility: {facility.name}",
        (panel_x, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_TEXT,
        1,
    )
    y_pos += 28
    cv2.putText(
        canvas,
        f"Active Vehicles: {len(vehicles)}",
        (panel_x, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_TEXT,
        1,
    )
    y_pos += 28

    if report:
        cv2.putText(
            canvas,
            f"Time: {report.timestamp_s:.1f} s",
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
        )
        y_pos += 28
        vacant_cnt = report.occupancy_summary.vacant_slots
        total_cnt = report.occupancy_summary.total_slots
        cv2.putText(
            canvas,
            f"Vacant Bays: {vacant_cnt}/{total_cnt}",
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
        )
        y_pos += 28
        cv2.putText(
            canvas,
            f"Conflicts Resolved: {report.resolved_conflicts_count}",
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 220, 60),
            1,
        )
        y_pos += 28
        cv2.putText(
            canvas,
            f"Avg Wait Time: {report.average_wait_time_s:.2f} s",
            (panel_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
        )
        y_pos += 35

    cv2.line(canvas, (panel_x, y_pos - 10), (width - 20, y_pos - 10), (100, 100, 110), thickness=1)
    cv2.putText(
        canvas,
        "FLEET VEHICLE STATUS",
        (panel_x, y_pos + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 0),
        1,
    )
    y_pos += 35

    for veh in vehicles:
        status_str = f"{veh.vehicle_id}: {veh.status.value.upper()} (v={veh.speed_mps:.1f}m/s)"
        col = COLOR_VEHICLE_FILL if veh.speed_mps > 0.1 else (150, 150, 150)
        cv2.putText(canvas, status_str, (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        y_pos += 22

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas
