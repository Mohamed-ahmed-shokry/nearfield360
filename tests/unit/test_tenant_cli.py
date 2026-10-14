"""Unit tests for nearfield360 tenant CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_tenant_info() -> None:
    result = runner.invoke(app, ["tenant", "info"])
    assert result.exit_code == 0
    assert "Multi-Tenant Facility Partitioning & SLA Configuration:" in result.stdout
    assert "Guaranteed SLA Latency:" in result.stdout

    json_result = runner.invoke(app, ["tenant", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "max_tenant_partitions" in data
    assert "sla_max_dispatch_wait_s" in data
    assert "sla_breach_penalty_per_min" in data


def test_cli_tenant_partition(tmp_path: Path) -> None:
    out_file = tmp_path / "partition.json"
    png_file = tmp_path / "partition.png"
    result = runner.invoke(
        app,
        [
            "tenant",
            "partition",
            "--tenant-id",
            "alpha-fleet",
            "--name",
            "Alpha Mobility",
            "--quota",
            "2",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Tenant Partition Allocated:" in result.stdout
    assert "alpha-fleet" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["tenant_id"] == "alpha-fleet"
    assert data["dedicated_bay_quota"] == 2


def test_cli_tenant_corridor(tmp_path: Path) -> None:
    out_file = tmp_path / "corridor.json"
    png_file = tmp_path / "corridor.png"
    result = runner.invoke(
        app,
        [
            "tenant",
            "corridor",
            "--corridor-id",
            "express-main",
            "--vehicle-id",
            "vip-jet-01",
            "--duration",
            "45.0",
            "--priority",
            "3",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "VIP Express Corridor Reserved:" in result.stdout
    assert "express-main" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["corridor_id"] == "express-main"
    assert data["active_vehicle_id"] == "vip-jet-01"
    assert data["priority_level"] == 3


def test_cli_tenant_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    png_file = tmp_path / "audit.png"
    result = runner.invoke(
        app,
        [
            "tenant",
            "audit",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Multi-Tenant Facility SLA & Partition Audit:" in result.stdout
    assert "SLA Compliance:" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    json_result = runner.invoke(app, ["tenant", "audit", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "overall_sla_compliance" in data
    assert "partitions" in data
    assert len(data["partitions"]) == 2
