"""Unit and integration tests for nearfield360 dssad CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app

runner = CliRunner()


def test_dssad_help() -> None:
    result = runner.invoke(app, ["dssad", "--help"])
    assert result.exit_code == 0
    assert "info" in result.stdout
    assert "record" in result.stdout
    assert "trigger" in result.stdout
    assert "query" in result.stdout
    assert "verify" in result.stdout
    assert "export" in result.stdout
    assert "render" in result.stdout
    assert "audit" in result.stdout


def test_dssad_info_json(tmp_path: Path) -> None:
    out_file = tmp_path / "dssad_info.json"
    result = runner.invoke(app, ["dssad", "info", "--json", "--output", str(out_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["sampling_rate_hz"] == 50
    assert data["pre_trigger_duration_s"] == 5.0
    assert data["post_trigger_duration_s"] == 2.0
    assert data["impact_accel_threshold_g"] == 2.5
    assert data["enable_tamper_proof_chain"] is True
    assert out_file.is_file()


def test_dssad_record(tmp_path: Path) -> None:
    out_file = tmp_path / "record.json"
    result = runner.invoke(
        app,
        [
            "dssad",
            "record",
            "--samples",
            "20",
            "--speed",
            "4.5",
            "--json",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "RECORDING_ACTIVE"
    assert data["ingested_samples"] == 20
    assert data["buffer_occupancy"] == 20
    assert out_file.is_file()


def test_dssad_trigger_and_render(tmp_path: Path) -> None:
    out_file = tmp_path / "incident.json"
    png_file = tmp_path / "incident.png"
    result = runner.invoke(
        app,
        [
            "dssad",
            "trigger",
            "--event",
            "COLLISION_IMPACT_DETECTED",
            "--severity",
            "FATAL",
            "--delta-v",
            "4.8",
            "--peak-g",
            "2.9",
            "--json",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "INCIDENT_FROZEN"
    assert data["event_type"] == "COLLISION_IMPACT_DETECTED"
    assert data["chain_verified"] is True
    assert out_file.is_file()
    assert png_file.is_file()
    assert png_file.stat().st_size > 15_000


def test_dssad_query(tmp_path: Path) -> None:
    out_file = tmp_path / "query.json"
    result = runner.invoke(
        app,
        ["dssad", "query", "--min-severity", "INFO", "--json", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["total_events_matched"] >= 1
    assert out_file.is_file()


def test_dssad_verify(tmp_path: Path) -> None:
    out_file = tmp_path / "verify.json"
    result = runner.invoke(app, ["dssad", "verify", "--json", "--output", str(out_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["valid"] is True
    assert data["total_blocks"] >= 2
    assert out_file.is_file()


def test_dssad_export_uds(tmp_path: Path) -> None:
    out_file = tmp_path / "uds.json"
    result = runner.invoke(
        app,
        ["dssad", "export", "--format", "uds", "--json", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["service_id"] == "0x62"
    assert data["did"] == "0x19ED"
    assert len(data["raw_hex"]) > 10
    assert out_file.is_file()


def test_dssad_render(tmp_path: Path) -> None:
    png_file = tmp_path / "render.png"
    result = runner.invoke(app, ["dssad", "render", "--output", str(png_file)])
    assert result.exit_code == 0
    assert "[PASS] Rendered forensic HUD dashboard" in result.stdout
    assert png_file.is_file()
    assert png_file.stat().st_size > 15_000


def test_dssad_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    result = runner.invoke(app, ["dssad", "audit", "--json", "--output", str(out_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "AUDIT_PASSED"
    assert data["total_checks"] == 5
    assert out_file.is_file()
