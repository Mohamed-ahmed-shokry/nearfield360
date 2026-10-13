"""Unit tests for nearfield360 billing CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.billing.calculator import BillingCalculator
from nearfield360.cli.app import app
from nearfield360.config import BillingConfig

runner = CliRunner()


def test_cli_billing_info() -> None:
    result = runner.invoke(app, ["billing", "info"])
    assert result.exit_code == 0
    assert "Base Hourly Parking Rate:" in result.stdout
    assert "EV Energy Consumption Rate:" in result.stdout

    json_result = runner.invoke(app, ["billing", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "base_hourly_rate" in data
    assert "kwh_tariff" in data


def test_cli_billing_calculate(tmp_path: Path) -> None:
    out_file = tmp_path / "receipt.json"
    result = runner.invoke(
        app,
        [
            "billing",
            "calculate",
            "--vehicle-id",
            "VEH_TEST",
            "--duration-minutes",
            "120.0",
            "--kwh",
            "20.0",
            "--overstay-minutes",
            "10.0",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Calculated Parking Receipt:" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["vehicle_id"] == "VEH_TEST"
    assert data["duration_minutes"] == 120.0
    assert data["total_charged"] > 0.0


def test_cli_billing_receipt_verification(tmp_path: Path) -> None:
    cfg = BillingConfig()
    calc = BillingCalculator(cfg)
    receipt = calc.calculate_session_charges(
        vehicle_id="VEH_VERIFY",
        entry_timestamp=0.0,
        exit_timestamp=3600.0,
        kwh_consumed=10.0,
    )
    receipt_file = tmp_path / "receipt_valid.json"
    receipt_file.write_text(json.dumps(receipt.model_dump()), encoding="utf-8")

    # 1. Valid signature check
    result = runner.invoke(app, ["billing", "receipt", str(receipt_file)])
    assert result.exit_code == 0
    assert "[PASS]" in result.stdout

    # 2. Tampered signature check
    tampered_data = receipt.model_dump()
    tampered_data["total_charged"] = 0.50
    tampered_file = tmp_path / "receipt_tampered.json"
    tampered_file.write_text(json.dumps(tampered_data), encoding="utf-8")

    tamper_result = runner.invoke(app, ["billing", "receipt", str(tampered_file)])
    assert tamper_result.exit_code == 1
    assert "FAILED" in tamper_result.stderr or "FAILED" in tamper_result.stdout


def test_cli_billing_settle(tmp_path: Path) -> None:
    out_json = tmp_path / "settlement.json"
    out_png = tmp_path / "settlement.png"

    result = runner.invoke(
        app,
        [
            "billing",
            "settle",
            "--sessions-count",
            "4",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "Financial Settlement Summary" in result.stdout
    assert out_json.is_file()
    assert out_png.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["total_receipts_count"] == 4
    assert data["total_revenue"] > 0.0
