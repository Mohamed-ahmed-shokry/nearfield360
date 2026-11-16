"""Unit tests for tactical 1280x720 HD map and localization HUD dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    LocalizationConvergence,
    LocalizationState,
    MapTile,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
    PoseHypothesis,
)
from nearfield360.hdmap.viz import render_hdmap_dashboard


def _sample_hdmap_and_state() -> tuple[HDVectorMap, LocalizationState, list[PoseHypothesis]]:
    left = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=40.0, y=2.0)])
    right = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=40.0, y=-2.0)])
    center = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=40.0, y=0.0)])
    lane = Lanelet(
        lanelet_id="L-TEST-01",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
    )

    slot_poly = Polyline2D(
        points=[
            Point2D(x=20.0, y=5.0),
            Point2D(x=22.5, y=5.0),
            Point2D(x=22.5, y=10.0),
            Point2D(x=20.0, y=10.0),
        ],
        is_closed=True,
    )
    space = ParkingSpaceFeature(
        slot_id="BAY-42",
        boundary=slot_poly,
        center=Point2D(x=21.25, y=7.5),
    )

    tile = MapTile(
        tile_id="TILE_0_0",
        x_min=0.0,
        y_min=-10.0,
        x_max=50.0,
        y_max=40.0,
        lanelets=[lane],
        parking_spaces=[space],
    )
    hd_map = HDVectorMap(map_id="MAP-VIZ-01", tiles=[tile])

    loc = LocalizationState(
        timestamp_s=12.5,
        estimated_x=20.0,
        estimated_y=0.0,
        estimated_yaw_rad=0.05,
        std_x=0.06,
        std_y=0.04,
        std_yaw_rad=0.012,
        status=LocalizationConvergence.CONVERGED,
        associated_features_count=6,
        active_tile_id="TILE_0_0",
    )

    particles = [
        PoseHypothesis(x=20.0 + 0.05 * i, y=0.0 + 0.02 * (i % 3), yaw_rad=0.05, weight=0.1)
        for i in range(10)
    ]
    return hd_map, loc, particles


def test_render_hdmap_dashboard_dimensions_and_dtype() -> None:
    hd_map, loc, particles = _sample_hdmap_and_state()
    obs = [(0.0, 2.0), (0.0, -2.0), (5.0, 0.0)]

    canvas = render_hdmap_dashboard(
        vector_map=hd_map,
        localization=loc,
        particles=particles,
        observed_points_vehicle=obs,
    )

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8


def test_render_hdmap_dashboard_file_saving(tmp_path: Path) -> None:
    hd_map, loc, particles = _sample_hdmap_and_state()
    out_file = tmp_path / "hdmap_hud.png"

    canvas = render_hdmap_dashboard(
        vector_map=hd_map,
        localization=loc,
        particles=particles,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000
    assert canvas.shape == (720, 1280, 3)


def test_render_hdmap_dashboard_status_variations() -> None:
    hd_map, loc, particles = _sample_hdmap_and_state()

    for st in (
        LocalizationConvergence.CONVERGED,
        LocalizationConvergence.TRACKING,
        LocalizationConvergence.CONVERGING,
        LocalizationConvergence.DEGRADED,
        LocalizationConvergence.LOST,
    ):
        alt_loc = loc.model_copy(update={"status": st})
        canvas = render_hdmap_dashboard(
            vector_map=hd_map,
            localization=alt_loc,
            particles=particles,
        )
        assert canvas.shape == (720, 1280, 3)
