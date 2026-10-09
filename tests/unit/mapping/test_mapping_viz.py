"""Unit tests for facility BEV and localization visualizer."""

from __future__ import annotations

from pathlib import Path

from nearfield360.mapping.builder import build_benchmark_garage, build_surface_lot
from nearfield360.mapping.localization import LocalizationSimulator
from nearfield360.mapping.router import GlobalRouter
from nearfield360.mapping.viz import render_facility_bev, render_localization_dashboard


def test_render_facility_bev_nominal(tmp_path: Path) -> None:
    garage = build_benchmark_garage()
    png_path = tmp_path / "garage_bev.png"

    canvas = render_facility_bev(garage, canvas_size=800, output_path=png_path)

    assert canvas.shape == (800, 800, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 1000


def test_render_facility_bev_with_route(tmp_path: Path) -> None:
    garage = build_benchmark_garage()
    router = GlobalRouter(garage)
    route = router.plan(
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="bay_a0_s0",
    )

    png_path = tmp_path / "route_bev.png"
    canvas = render_facility_bev(
        garage,
        route=route,
        estimated_poses=[(w.x, w.y, w.heading_rad) for w in route.waypoints[:5]],
        canvas_size=900,
        output_path=png_path,
    )

    assert canvas.shape == (900, 900, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 1000


def test_render_surface_lot_bev(tmp_path: Path) -> None:
    lot = build_surface_lot()
    png_path = tmp_path / "surface_bev.png"

    canvas = render_facility_bev(lot, canvas_size=700, output_path=png_path)
    assert canvas.shape == (700, 700, 3)
    assert png_path.is_file()


def test_render_localization_dashboard(tmp_path: Path) -> None:
    garage = build_benchmark_garage()
    router = GlobalRouter(garage)
    route = router.plan(
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="bay_a0_s0",
    )

    sim = LocalizationSimulator(garage, seed=42)
    wp_coords = [(w.x, w.y, w.heading_rad) for w in route.waypoints]
    report, records = sim.simulate(wp_coords)

    png_path = tmp_path / "localization_dashboard.png"
    canvas = render_localization_dashboard(
        report=report,
        records=records,
        facility=garage,
        width=1200,
        height=700,
        output_path=png_path,
    )

    assert canvas.shape == (700, 1200, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 5000
