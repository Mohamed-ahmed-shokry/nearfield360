"""Unit tests for nearfield360 v2g CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_v2g_info(tmp_path: Path) -> None:
    # 1. Plain text output
    result = runner.invoke(app, ["v2g", "info"])
    assert result.exit_code == 0
    assert "Vehicle-to-Grid (V2G) Configuration:" in result.stdout
    assert "Facility Transformer Limit" in result.stdout
    assert "Standard Time-of-Use Tariff Windows:" in result.stdout

    # 2. JSON output
    json_result = runner.invoke(app, ["v2g", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["transformer_limit_kw"] == 500.0
    assert data["peak_shaving_target_kw"] == 350.0
    assert len(data["tariff_windows"]) == 5

    # 3. File output
    out_file = tmp_path / "v2g_info.json"
    file_result = runner.invoke(app, ["v2g", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["grid_frequency_nominal_hz"] == 50.0


def test_cli_v2g_schedule(tmp_path: Path) -> None:
    # 1. Default schedule
    result = runner.invoke(app, ["v2g", "schedule"])
    assert result.exit_code == 0
    assert "V2G Arbitrage Dispatch Schedule for EV-DEMO-01:" in result.stdout
    assert "Slots Scheduled" in result.stdout
    assert "Departure SoC Guarantee  : PASSED" in result.stdout

    # 2. Customized parameters
    out_file = tmp_path / "sched.json"
    cust_res = runner.invoke(
        app,
        [
            "v2g",
            "schedule",
            "--vehicle-id",
            "EV-CUSTOM-99",
            "--battery-kwh",
            "100.0",
            "--current-soc",
            "0.80",
            "--min-departure-soc",
            "0.65",
            "--arrival-hour",
            "14.0",
            "--departure-hour",
            "20.0",
            "--output",
            str(out_file),
            "--json",
        ],
    )
    assert cust_res.exit_code == 0
    data = json.loads(cust_res.stdout)
    assert data["vehicle_id"] == "EV-CUSTOM-99"
    assert data["min_departure_soc_met"] is True
    assert out_file.is_file()


def test_cli_v2g_arbitrage(tmp_path: Path) -> None:
    out_png = tmp_path / "arb_dash.png"
    out_json = tmp_path / "arb_report.json"

    result = runner.invoke(
        app,
        [
            "v2g",
            "arbitrage",
            "--hour",
            "15.0",
            "--png",
            str(out_png),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert "Facility V2G Microgrid Arbitrage Summary:" in result.stdout
    assert "Connected Fleet Bays     : 10 bays" in result.stdout
    assert out_png.is_file()
    assert out_json.is_file()

    # JSON output
    json_result = runner.invoke(app, ["v2g", "arbitrage", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["total_bays_connected"] == 10
    assert data["total_battery_capacity_kwh"] > 700.0


def test_cli_v2g_render(tmp_path: Path) -> None:
    out_png = tmp_path / "render_dash.png"
    result = runner.invoke(app, ["v2g", "render", "-o", str(out_png), "--hour", "16.0"])
    assert result.exit_code == 0
    assert "Rendered V2G tactical energy dashboard" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 50_000


def test_cli_v2g_audit(tmp_path: Path) -> None:
    # 1. Plain text audit
    result = runner.invoke(app, ["v2g", "audit"])
    assert result.exit_code == 0
    assert "[PASS] status" in result.stdout
    assert "[PASS] inverter_droop_response" in result.stdout
    assert "[PASS] peak_shaving_target_compliance" in result.stdout
    assert "[PASS] transformer_headroom_safety" in result.stdout
    assert "[PASS] departure_soc_guarantees" in result.stdout

    # 2. JSON audit and file output
    out_file = tmp_path / "audit.json"
    json_res = runner.invoke(app, ["v2g", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["status"] == "PASS"
    assert out_file.is_file()
