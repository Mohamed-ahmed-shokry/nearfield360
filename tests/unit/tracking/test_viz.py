"""Unit tests for BEV tracking overlay visualization engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.risk import RiskZone, surround_parking_zones
from nearfield360.tracking.models import (
    TrackedObstacle,
    TrackState,
    TrajectoryForecast,
)
from nearfield360.tracking.viz import render_tracking_bev_overlay


def _setup_test_data() -> tuple[
    BevGrid, list[RiskZone], list[TrackedObstacle], list[TrajectoryForecast]
]:
    grid = BevGrid(x_min=-4.0, x_max=8.0, y_min=-4.0, y_max=4.0, resolution=0.1)
    zones = list(
        surround_parking_zones(
            grid,
            front_length=2.0,
            rear_length=2.0,
            half_width=0.8,
            start_x=0.0,
            rear_start_x=0.0,
            lateral_width=0.6,
            vehicle_x_min=-1.5,
            vehicle_x_max=1.5,
            near_radius=0.5,
            warning_radius=1.0,
        )
    )

    obs = [
        TrackedObstacle(
            track_id=1,
            class_id=0,
            class_name="vehicles",
            state=TrackState.CONFIRMED,
            position=(4.0, 0.0),
            velocity=(-1.0, 0.0),
            speed=1.0,
            hits=5,
            age=5,
            time_since_update=0,
            history=((4.5, 0.0), (4.3, 0.0), (4.0, 0.0)),
        ),
        TrackedObstacle(
            track_id=2,
            class_id=1,
            class_name="person",
            state=TrackState.CONFIRMED,
            position=(2.0, 2.0),
            velocity=(0.0, 0.5),
            speed=0.5,
            hits=3,
            age=3,
            time_since_update=0,
            history=((2.0, 1.8), (2.0, 2.0)),
        ),
    ]

    forecasts = [
        TrajectoryForecast(
            track_id=1,
            class_name="vehicles",
            predicted_positions=((0.0, 4.0, 0.0), (1.0, 3.0, 0.0), (2.0, 2.0, 0.0)),
            zone_intrusions=(("forward_corridor", 2.0),),
            min_ttc_s=2.0,
        )
    ]

    return grid, zones, obs, forecasts


def test_render_tracking_bev_overlay_creates_valid_image() -> None:
    grid, zones, obs, forecasts = _setup_test_data()

    canvas = render_tracking_bev_overlay(
        grid,
        obs,
        forecasts=forecasts,
        zones=zones,
    )

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (*grid.shape, 3)
    assert canvas.dtype == np.uint8
    # Canvas should not be uniformly empty
    assert np.any(canvas != 32)


def test_render_tracking_bev_overlay_with_base_canvas() -> None:
    grid, _, obs, _ = _setup_test_data()
    base = np.zeros((*grid.shape, 3), dtype=np.uint8)
    base[:, :] = (50, 50, 50)

    canvas = render_tracking_bev_overlay(grid, obs, base_canvas=base)
    assert canvas.shape == base.shape
    assert np.any(canvas != 50)


def test_render_tracking_bev_overlay_writes_file(tmp_path: Path) -> None:
    grid, zones, obs, forecasts = _setup_test_data()
    out_file = tmp_path / "bev_tracking.png"

    canvas = render_tracking_bev_overlay(
        grid,
        obs,
        forecasts=forecasts,
        zones=zones,
        output_path=out_file,
    )

    assert out_file.exists()
    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == canvas.shape
