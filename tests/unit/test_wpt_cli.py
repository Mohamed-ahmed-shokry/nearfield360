"""Unit tests for nearfield360 wpt CLI command group (SAE J2954)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_wpt_info(tmp_path: Path) -> None:
    # 1. Plain text output
    res = runner.invoke(app, ["wpt", "info"])
    assert res.exit_code == 0
    assert "SAE J2954 Wireless Power Transfer Configuration" in res.stdout
    assert "85.0 kHz" in res.stdout

    # 2. JSON output
    json_res = runner.invoke(app, ["wpt", "info", "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["nominal_frequency_khz"] == 85.0
    assert data["max_power_kw"] == 11.0

    # 3. File output
    out_file = tmp_path / "wpt_info.json"
    file_res = runner.invoke(app, ["wpt", "info", "--output", str(out_file)])
    assert file_res.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["target_air_gap_mm"] == 170.0


def test_cli_wpt_ping(tmp_path: Path) -> None:
    # 1. Normal in-band ping
    res = runner.invoke(app, ["wpt", "ping", "--freq", "85.0"])
    assert res.exit_code == 0
    assert "LINK_ESTABLISHED" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "ping.json"
    json_res = runner.invoke(app, ["wpt", "ping", "--freq", "85.0", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["beacon_link_active"] is True
    assert out_file.is_file()

    # 3. Out-of-band ping
    oob_res = runner.invoke(app, ["wpt", "ping", "--freq", "75.0"])
    assert oob_res.exit_code == 0
    assert "RESONANCE_OUT_OF_BAND" in oob_res.stdout


def test_cli_wpt_align(tmp_path: Path) -> None:
    # In tolerance alignment
    res = runner.invoke(app, ["wpt", "align", "--dx", "20.0", "--dy", "15.0", "--dz", "170.0"])
    assert res.exit_code == 0
    assert "Coil Status:" in res.stdout

    # JSON output with file
    out_file = tmp_path / "align.json"
    json_res = runner.invoke(
        app,
        ["wpt", "align", "--dx", "10.0", "--dy", "5.0", "--json", "-o", str(out_file)],
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["is_within_tolerance"] is True
    assert data["radial_offset_mm"] > 0.0
    assert out_file.is_file()


def test_cli_wpt_fod(tmp_path: Path) -> None:
    # Normal resonance
    res = runner.invoke(app, ["wpt", "fod", "--freq", "85.0", "--q", "280.0"])
    assert res.exit_code == 0
    assert "NORMAL" in res.stdout

    # Metal object trip
    metal_res = runner.invoke(app, ["wpt", "fod", "--freq", "87.5", "--q", "180.0"])
    assert metal_res.exit_code == 0
    assert "FOD_FAULT" in metal_res.stdout

    # Emergency trip JSON
    out_file = tmp_path / "fod.json"
    emerg_res = runner.invoke(app, ["wpt", "fod", "--emergency", "--json", "-o", str(out_file)])
    assert emerg_res.exit_code == 0
    data = json.loads(emerg_res.stdout)
    assert data["interlock_state"] == "EMERGENCY_SHUTDOWN"
    assert data["power_inhibited"] is True


def test_cli_wpt_dock(tmp_path: Path) -> None:
    # Docking execution
    res = runner.invoke(
        app, ["wpt", "dock", "--init-x", "0.20", "--init-y", "0.04", "--max-steps", "150"]
    )
    assert res.exit_code == 0
    assert "SUCCESS" in res.stdout

    # JSON output
    out_file = tmp_path / "dock.json"
    json_res = runner.invoke(app, ["wpt", "dock", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["docked"] is True
    assert data["iterations"] > 0
    assert out_file.is_file()


def test_cli_wpt_efficiency(tmp_path: Path) -> None:
    # Nominal efficiency
    res = runner.invoke(
        app, ["wpt", "efficiency", "--dr", "15.0", "--dz", "170.0", "--class", "WPT3"]
    )
    assert res.exit_code == 0
    assert "Coupling Factor" in res.stdout
    assert "Transfer Efficiency" in res.stdout

    # JSON export
    out_file = tmp_path / "eff.json"
    json_res = runner.invoke(app, ["wpt", "efficiency", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["coupling_factor_k"] > 0.15
    assert data["transfer_efficiency"] >= 0.85


def test_cli_wpt_render(tmp_path: Path) -> None:
    png_path = tmp_path / "wpt_hud.png"
    res = runner.invoke(app, ["wpt", "render", "--output", str(png_path)])
    assert res.exit_code == 0
    assert png_path.is_file()
    assert png_path.stat().st_size > 5000

    # JSON summary
    json_res = runner.invoke(
        app, ["wpt", "render", "--output", str(tmp_path / "hud2.png"), "--json"]
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["canvas_shape"] == [720, 1280, 3]


def test_cli_wpt_audit(tmp_path: Path) -> None:
    res = runner.invoke(app, ["wpt", "audit"])
    assert res.exit_code == 0
    assert "COMPLIANT" in res.stdout

    out_file = tmp_path / "audit.json"
    json_res = runner.invoke(app, ["wpt", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["compliant"] is True
    assert data["total_checks"] == 7
    assert data["passed_checks"] == 7
