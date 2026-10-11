"""Unit and integration tests for nearfield360 fleet CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app
from nearfield360.mapping.builder import build_benchmark_garage

runner = CliRunner()


def test_fleet_cli_info_plain(tmp_path: Path) -> None:
    map_json = tmp_path / "facility.json"
    facility = build_benchmark_garage(
        aisle_count=2, slots_per_aisle=4, occupied_slot_indices=(1, 5)
    )
    map_json.write_text(facility.to_json(), encoding="utf-8")

    result = runner.invoke(app, ["fleet", "info", "--map-path", str(map_json)])
    assert result.exit_code == 0
    assert "Fleet Dispatch & Capacity Info" in result.stdout
    assert "Total Slots:  8" in result.stdout
    assert "Vacant:       6" in result.stdout


def test_fleet_cli_info_json(tmp_path: Path) -> None:
    map_json = tmp_path / "facility.json"
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2, occupied_slot_indices=())
    map_json.write_text(facility.to_json(), encoding="utf-8")

    result = runner.invoke(app, ["fleet", "info", "--map-path", str(map_json), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_slots"] == 2
    assert payload["vacant_slots"] == 2


def test_fleet_cli_dispatch(tmp_path: Path) -> None:
    out_json = tmp_path / "dispatch.json"
    result = runner.invoke(
        app,
        ["fleet", "dispatch", "--vehicle-id", "veh_alpha", "--output", str(out_json), "--json"],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["vehicle_id"] == "veh_alpha"
    assert "assigned_slot_id" in payload


def test_fleet_cli_simulate(tmp_path: Path) -> None:
    out_json = tmp_path / "sim_report.json"
    out_png = tmp_path / "sim_bev.png"

    result = runner.invoke(
        app,
        [
            "fleet",
            "simulate",
            "--vehicles",
            "2",
            "--steps",
            "20",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    assert out_png.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_vehicles_active"] == 2
    assert "occupancy_summary" in payload


def test_fleet_cli_deconflict(tmp_path: Path) -> None:
    out_json = tmp_path / "conflicts.json"
    result = runner.invoke(app, ["fleet", "deconflict", "--output", str(out_json), "--json"])
    assert result.exit_code == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
