"""Integration tests for nearfield360 charging CLI subcommands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.utils.artifacts import read_json, write_json

runner = CliRunner()


def test_charging_info_command(tmp_path: Path) -> None:
    out_json = tmp_path / "charging_info.json"
    result = runner.invoke(
        app,
        ["charging", "info", "--output", str(out_json), "--json"],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    data = read_json(out_json)
    assert "grid_capacity_kw" in data
    assert data["total_charging_bays"] == 2


def test_charging_grid_command(tmp_path: Path) -> None:
    out_json = tmp_path / "grid_status.json"
    result = runner.invoke(
        app,
        ["charging", "grid", "--active-power", "180.0", "--output", str(out_json)],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    data = read_json(out_json)
    assert data["total_requested_power_kw"] == 180.0
    assert not data["is_curtailed"]


def test_charging_schedule_command_with_png(tmp_path: Path) -> None:
    garage = build_benchmark_garage()
    map_json = tmp_path / "map.json"
    write_json(map_json, garage.model_dump())

    out_json = tmp_path / "schedule.json"
    out_png = tmp_path / "schedule.png"

    result = runner.invoke(
        app,
        [
            "charging",
            "schedule",
            "--map-path",
            str(map_json),
            "--vehicles",
            "3",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    assert out_png.exists()

    data = read_json(out_json)
    assert data["total_bays"] == 2
    assert len(data["active_sessions"]) == 2
    assert data["pending_queue_count"] == 1


def test_charging_simulate_command_with_png(tmp_path: Path) -> None:
    out_json = tmp_path / "sim.json"
    out_png = tmp_path / "sim.png"

    result = runner.invoke(
        app,
        [
            "charging",
            "simulate",
            "--vehicles",
            "2",
            "--steps",
            "10",
            "--dt",
            "60.0",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    assert out_png.exists()

    data = read_json(out_json)
    assert "simulation_metadata" in data
    assert data["simulation_metadata"]["steps"] == 10
    assert data["simulation_metadata"]["total_energy_delivered_kwh"] > 0.0
