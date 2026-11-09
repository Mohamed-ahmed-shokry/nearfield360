"""Unit tests for nearfield360 acoustic CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_acoustic_info(tmp_path: Path) -> None:
    """Verify acoustic info command prints configuration and exports JSON."""
    result = runner.invoke(app, ["acoustic", "info"])
    assert result.exit_code == 0
    assert "Automotive Acoustic Sensing & DoA Configuration:" in result.stdout
    assert "Array Radius" in result.stdout
    assert "Sampling Rate" in result.stdout

    # Test JSON output
    json_result = runner.invoke(app, ["acoustic", "info", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["sampling_rate_hz"] == 16000
    assert payload["num_microphones"] == 4
    assert len(payload["microphones"]) == 4

    # Test file output
    out_file = tmp_path / "acoustic_info.json"
    file_result = runner.invoke(app, ["acoustic", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()
    saved_payload = json.loads(out_file.read_text())
    assert saved_payload["sampling_rate_hz"] == 16000


def test_cli_acoustic_detect(tmp_path: Path) -> None:
    """Verify acoustic detect analyzes wail siren and ambient silence."""
    # Wail detection
    out_file = tmp_path / "detect_wail.json"
    result = runner.invoke(
        app,
        [
            "acoustic",
            "detect",
            "--siren-type",
            "wail",
            "--duration-s",
            "1.5",
            "--snr-db",
            "30.0",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "SIREN DETECTED" in result.stdout
    assert "WAIL" in result.stdout
    assert out_file.is_file()
    saved = json.loads(out_file.read_text())
    assert saved["detected"] is True
    assert saved["siren_type"] == "wail"

    # Ambient silence / none
    none_result = runner.invoke(
        app,
        ["acoustic", "detect", "--siren-type", "none", "--duration-s", "1.0", "--json"],
    )
    assert none_result.exit_code == 0
    none_payload = json.loads(none_result.stdout)
    assert none_payload["detected"] is False
    assert none_payload["siren_type"] == "none"


def test_cli_acoustic_doa(tmp_path: Path) -> None:
    """Verify multichannel DoA triangulation and sector correlation."""
    out_file = tmp_path / "doa_res.json"
    result = runner.invoke(
        app,
        [
            "acoustic",
            "doa",
            "--azimuth-deg",
            "60.0",
            "--siren-type",
            "wail",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Acoustic Direction-of-Arrival (DoA) Triangulation:" in result.stdout
    assert "Estimated Azimuth" in result.stdout
    assert out_file.is_file()

    # JSON mode
    json_result = runner.invoke(
        app,
        ["acoustic", "doa", "--azimuth-deg", "-45.0", "--json"],
    )
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert abs(data["azimuth_deg"] - (-45.0)) <= 8.0
    assert data["camera_sector"] in ("FV", "MVL")


def test_cli_acoustic_simulate(tmp_path: Path) -> None:
    """Verify multi-frame acoustic Doppler simulation and HUD export."""
    out_json = tmp_path / "sim_report.json"
    out_png = tmp_path / "sim_hud.png"

    result = runner.invoke(
        app,
        [
            "acoustic",
            "simulate",
            "--siren-type",
            "wail",
            "--azimuth-deg",
            "30.0",
            "--speed-mps",
            "15.0",
            "--distance-m",
            "50.0",
            "--frames",
            "4",
            "--visual",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "Acoustic Emergency Simulation Summary:" in result.stdout
    assert "Doppler State" in result.stdout
    assert out_json.is_file()
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000

    report_data = json.loads(out_json.read_text())
    assert report_data["yield_state"] in (
        "monitoring",
        "yield_prepared",
        "active_yield",
        "emergency_stop",
    )


def test_cli_acoustic_render(tmp_path: Path) -> None:
    """Verify tactical 1280x720 acoustic HUD dashboard rendering CLI."""
    hud_path = tmp_path / "test_hud.png"
    result = runner.invoke(
        app,
        [
            "acoustic",
            "render",
            "--png",
            str(hud_path),
            "--siren-type",
            "yelp",
            "--azimuth-deg",
            "-30.0",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert hud_path.is_file()
    assert hud_path.stat().st_size > 1000
    meta = json.loads(result.stdout)
    assert meta["status"] == "rendered"
    assert meta["siren_type"] == "yelp"


def test_cli_acoustic_audit(tmp_path: Path) -> None:
    """Verify acoustic regulatory compliance audit CLI."""
    out_file = tmp_path / "audit_report.json"
    result = runner.invoke(
        app,
        ["acoustic", "audit", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    assert "Acoustic Emergency Sensing Regulatory Audit: [COMPLIANT]" in result.stdout
    assert "[PASS]" in result.stdout
    assert out_file.is_file()

    json_result = runner.invoke(app, ["acoustic", "audit", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["compliant"] is True
    assert data["checks_total"] == 5
    assert data["checks_passed"] == 5
