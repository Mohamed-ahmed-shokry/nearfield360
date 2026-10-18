"""Unit tests for nearfield360 coop CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_coop_info(tmp_path: Path) -> None:
    # 1. Plain text
    result = runner.invoke(app, ["coop", "info"])
    assert result.exit_code == 0
    assert "NearField360 Infrastructure Cooperative Perception" in result.stdout
    assert "Max Latency Compensation:" in result.stdout
    assert "Demonstrator RPU Stations:" in result.stdout

    # 2. JSON output
    json_result = runner.invoke(app, ["coop", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert len(data["standards"]) == 3
    assert data["parameters"]["max_latency_compensation_s"] == 0.50
    assert data["default_pillars_count"] == 4

    # 3. Output file write
    out_file = tmp_path / "coop_info.json"
    file_result = runner.invoke(app, ["coop", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["parameters"]["association_gate_mahalanobis"] == 3.0


def test_cli_coop_rpu(tmp_path: Path) -> None:
    # 1. Default all stations
    result = runner.invoke(app, ["coop", "rpu"])
    assert result.exit_code == 0
    assert "Roadside Perception Units" in result.stdout
    assert "RPU-BLIND-CORNER-1" in result.stdout
    assert "RPU-RAMP-CREST-1" in result.stdout
    assert "Generated 3 CPM messages" in result.stdout

    # 2. Filter by specific station
    filt_res = runner.invoke(app, ["coop", "rpu", "--station-id", "RPU-BLIND-CORNER-1"])
    assert filt_res.exit_code == 0
    assert "RPU-BLIND-CORNER-1" in filt_res.stdout
    assert "RPU-RAMP-CREST-1" not in filt_res.stdout

    # 3. Non-existent station error
    err_res = runner.invoke(app, ["coop", "rpu", "--station-id", "NON-EXISTENT"])
    assert err_res.exit_code == 1

    # 4. JSON & Output file
    out_file = tmp_path / "rpu_dispatch.json"
    json_res = runner.invoke(app, ["coop", "rpu", "--json", "--output", str(out_file)])
    assert json_res.exit_code == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["station_count"] == 3
    assert len(data["cpm_messages"]) == 3


def test_cli_coop_fuse(tmp_path: Path) -> None:
    out_file = tmp_path / "fusion.json"
    result = runner.invoke(
        app,
        [
            "coop",
            "fuse",
            "--timestamp",
            "12.5",
            "--latency-s",
            "0.10",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Cooperative Track Fusion" in result.stdout
    assert "Input Onboard Tracks:" in result.stdout
    assert "Output Fused Tracks:" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["timestamp_s"] == 12.5
    assert data["wireless_latency_s"] == 0.10
    assert len(data["fused_tracks"]) == 2

    # JSON stdout check
    json_result = runner.invoke(app, ["coop", "fuse", "--json"])
    assert json_result.exit_code == 0
    json_data = json.loads(json_result.stdout)
    assert len(json_data["fused_tracks"]) == 2


def test_cli_coop_render(tmp_path: Path) -> None:
    out_png = tmp_path / "coop" / "hud.png"
    out_json = tmp_path / "coop" / "render.json"

    result = runner.invoke(
        app,
        [
            "coop",
            "render",
            "--vehicle-x",
            "14.0",
            "--vehicle-y",
            "15.0",
            "--vehicle-speed",
            "3.0",
            "--png",
            str(out_png),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert "Tactical BEV Cooperative Map successfully rendered to:" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 15000
    assert out_json.is_file()

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["tracks_rendered"] == 2
    assert summary["vehicle_pose"]["x_m"] == 14.0


def test_cli_coop_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    result = runner.invoke(app, ["coop", "audit", "--output", str(out_file)])
    assert result.exit_code == 0
    assert "Cooperative Perception (ISO 23374 / ETSI CPM) Audit Report:" in result.stdout
    assert "Compliance Status:           COMPLIANT" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["status"] == "COMPLIANT"
    assert data["iso_23374_type"] == "TYPE_2_AND_TYPE_3"
    assert data["all_stations_operational"] is True
    assert data["telemetry"]["active_rpus"] == 3
