"""BEV visualization engine for tracked dynamic obstacles and predicted trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from nearfield360.geometry.bev import BevGrid
    from nearfield360.occupancy.risk import RiskZone
    from nearfield360.tracking.models import (
        TrackedObstacle,
        TrajectoryForecast,
    )

# Visual colors in BGR format
_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 165, 255),    # Orange for vehicles
    1: (255, 255, 0),    # Cyan for pedestrians
    2: (255, 0, 255),    # Magenta for bicycles
    3: (0, 255, 255),    # Yellow for traffic lights
    4: (0, 255, 0),      # Green for traffic signs
}

_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "forward_corridor": (0, 215, 255),
    "rear_corridor": (0, 128, 255),
    "left_clearance": (255, 191, 0),
    "right_clearance": (255, 144, 30),
    "near_circle": (255, 255, 255),
    "warning_circle": (200, 100, 255),
}


def render_tracking_bev_overlay(
    grid: BevGrid,
    obstacles: Sequence[TrackedObstacle],
    *,
    forecasts: Sequence[TrajectoryForecast] | None = None,
    zones: Sequence[RiskZone] | None = None,
    base_canvas: np.ndarray | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """Render dynamic obstacles, motion vectors, and predicted paths on a BEV canvas.

    Args:
        grid: BevGrid defining coordinates and cell dimensions.
        obstacles: Active TrackedObstacle instances to render.
        forecasts: Optional trajectory forecasts to visualize future motion paths.
        zones: Optional RiskZone masks to render as boundary overlays.
        base_canvas: Optional pre-existing BGR BEV image (e.g. fused occupancy map).
        output_path: Optional destination Path to save the rendered PNG image.

    Returns:
        BGR image array of shape (*grid.shape, 3) and dtype uint8.
    """
    h, w = grid.shape
    if base_canvas is not None:
        canvas = base_canvas.copy()
    else:
        # Dark charcoal background for clean visualization
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)

    # Render risk zones if supplied and not already on base_canvas
    if zones is not None and base_canvas is None:
        for zone in zones:
            color = _ZONE_COLORS.get(zone.name, (100, 100, 100))
            # Find boundary pixels
            eroded = cv2.erode(zone.mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            border = (zone.mask.astype(np.uint8) - eroded).astype(bool)
            canvas[border] = color

    forecast_map = {f.track_id: f for f in (forecasts or ())}

    # Render tracked obstacles
    for obs in obstacles:
        x, y = obs.position
        vx, vy = obs.velocity
        color = _CLASS_COLORS.get(obs.class_id, (0, 255, 255))

        indexed = grid.world_to_grid(np.array([[x, y]], dtype=np.float64))
        if not bool(indexed.valid[0]):
            continue

        r, c = int(indexed.indices[0, 0]), int(indexed.indices[0, 1])

        # Draw past history trail
        if len(obs.history) > 1:
            hist_pts = []
            for hx, hy in obs.history:
                h_idx = grid.world_to_grid(np.array([[hx, hy]], dtype=np.float64))
                if bool(h_idx.valid[0]):
                    hist_pts.append((int(h_idx.indices[0, 1]), int(h_idx.indices[0, 0])))
            if len(hist_pts) > 1:
                cv2.polylines(
                    canvas,
                    [np.array(hist_pts, dtype=np.int32)],
                    isClosed=False,
                    color=(160, 160, 160),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        # Draw future predicted path
        forecast = forecast_map.get(obs.track_id)
        path_color = (0, 0, 255) if (forecast and forecast.min_ttc_s is not None) else (80, 220, 80)
        if forecast and len(forecast.predicted_positions) > 1:
            future_pts = []
            for _, fx, fy in forecast.predicted_positions:
                f_idx = grid.world_to_grid(np.array([[fx, fy]], dtype=np.float64))
                if bool(f_idx.valid[0]):
                    future_pts.append((int(f_idx.indices[0, 1]), int(f_idx.indices[0, 0])))
            if len(future_pts) > 1:
                cv2.polylines(
                    canvas,
                    [np.array(future_pts, dtype=np.int32)],
                    isClosed=False,
                    color=path_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        # Draw obstacle bounding footprint
        half_w = max(2, int(0.4 / grid.resolution))
        half_l = max(2, int(0.4 / grid.resolution))
        r0 = max(0, r - half_l)
        r1 = min(h - 1, r + half_l)
        c0 = max(0, c - half_w)
        c1 = min(w - 1, c + half_w)
        cv2.rectangle(canvas, (c0, r0), (c1, r1), color, 2)

        # Draw velocity arrow (1 second lookahead)
        if obs.speed > 0.1:
            v_end = grid.world_to_grid(np.array([[x + vx, y + vy]], dtype=np.float64))
            if bool(v_end.valid[0]):
                end_c, end_r = int(v_end.indices[0, 1]), int(v_end.indices[0, 0])
                cv2.arrowedLine(
                    canvas,
                    (c, r),
                    (end_c, end_r),
                    color=(0, 255, 255),
                    thickness=2,
                    tipLength=0.3,
                )

        # Render label text
        label = f"#{obs.track_id} {obs.class_name} {obs.speed:.1f}m/s"
        if forecast and forecast.min_ttc_s is not None:
            label += f" TTC:{forecast.min_ttc_s:.1f}s"
        cv2.putText(
            canvas,
            label,
            (c + 4, max(12, r - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas


__all__ = ["render_tracking_bev_overlay"]
