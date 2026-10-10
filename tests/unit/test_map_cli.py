"""Unit and integration tests for nearfield360 map CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_map_info_default_benchmark_garage() -> None:
    result = runner.invoke(app, ["map", "info"])
    assert result.exit_code == 0
    assert "Benchmark Multi-Aisle Indoor Parking Garage" in result.stdout
    assert "Slots:          8 total" in result.stdout
    assert "Integrity:      VALID" in result.stdout


def test_map_info_json_output() -> None:
    result = runner.invoke(app, ["map", "info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["map_id"] == "benchmark_garage_avp"
    assert data["integrity"]["is_valid"] is True
    assert data["slot_counts"]["total"] == 8


def test_map_build_garage_and_lot(tmp_path: Path) -> None:
    garage_json = tmp_path / "garage.json"
    garage_png = tmp_path / "garage.png"
    result = runner.invoke(
        app,
        [
            "map",
            "build",
            "--type",
            "garage",
            "--aisles",
            "2",
            "--output",
            str(garage_json),
            "--png",
            str(garage_png),
        ],
    )
    assert result.exit_code == 0
    assert garage_json.is_file()
    assert garage_png.is_file()

    lot_json = tmp_path / "lot.json"
    lot_png = tmp_path / "lot.png"
    lot_result = runner.invoke(
        app,
        ["map", "build", "--type", "lot", "--output", str(lot_json), "--png", str(lot_png)],
    )
    assert lot_result.exit_code == 0
    assert lot_json.is_file()
    assert lot_png.is_file()


def test_map_route_nominal(tmp_path: Path) -> None:
    route_json = tmp_path / "route.json"
    route_png = tmp_path / "route.png"
    result = runner.invoke(
        app,
        [
            "map",
            "route",
            "--start-pose",
            "0.0,0.0,0.0",
            "--target-slot",
            "bay_a0_s0",
            "--output",
            str(route_json),
            "--png",
            str(route_png),
        ],
    )
    assert result.exit_code == 0
    assert "Planned route" in result.stdout
    assert route_json.is_file()
    assert route_png.is_file()

    data = json.loads(route_json.read_text(encoding="utf-8"))
    assert data["target_slot_id"] == "bay_a0_s0"
    assert len(data["waypoints"]) > 5


def test_map_route_invalid_start_pose() -> None:
    result = runner.invoke(app, ["map", "route", "--start-pose", "invalid_pose"])
    assert result.exit_code == 2
    assert "Invalid --start-pose" in result.stderr


def test_map_localize_simulation(tmp_path: Path) -> None:
    report_json = tmp_path / "loc_report.json"
    dashboard_png = tmp_path / "dashboard.png"
    bev_png = tmp_path / "bev_track.png"

    result = runner.invoke(
        app,
        [
            "map",
            "localize",
            "--target-slot",
            "bay_a0_s0",
            "--output",
            str(report_json),
            "--dashboard-png",
            str(dashboard_png),
            "--bev-png",
            str(bev_png),
        ],
    )
    assert result.exit_code == 0
    assert "Localization Simulation Complete" in result.stdout
    assert report_json.is_file()
    assert dashboard_png.is_file()
    assert bev_png.is_file()

    data = json.loads(report_json.read_text(encoding="utf-8"))
    assert data["trajectory_length_m"] > 5.0
    assert data["total_landmark_updates"] > 0
    assert data["mean_position_error_m"] < 0.35
