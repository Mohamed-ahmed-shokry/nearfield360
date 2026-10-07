"""Unit and integration tests for nearfield360 plan CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


def _make_slots_json(path: Path) -> None:
    slot_payload = {
        "summary": {
            "total_slots": 1,
            "vacant_slots": 1,
            "occupied_slots": 0,
            "uncertain_slots": 0,
            "parallel_slots": 1,
            "perpendicular_slots": 0,
            "slanted_slots": 0,
            "feasible_approaches": 1,
        },
        "slots": [
            {
                "slot_id": "slot_01",
                "slot_type": "parallel",
                "corners": [
                    {"x": -4.0, "y": -1.2, "z": 0.0},
                    {"x": -4.0, "y": -3.2, "z": 0.0},
                    {"x": 2.0, "y": -3.2, "z": 0.0},
                    {"x": 2.0, "y": -1.2, "z": 0.0},
                ],
                "center": [-1.0, -2.2],
                "heading_rad": 0.0,
                "width_m": 2.0,
                "length_m": 6.0,
                "status": "vacant",
                "occupancy_ratio": 0.02,
                "uncertainty_ratio": 0.01,
                "confidence": 0.95,
                "approach_path": {
                    "entry_point": [-3.0, -1.0],
                    "target_point": [-1.0, -2.2],
                    "entry_heading_rad": 0.0,
                    "maneuver_length_m": 2.5,
                    "clearance_margin_m": 0.35,
                    "is_feasible": True,
                },
            }
        ],
    }
    path.write_text(json.dumps(slot_payload), encoding="utf-8")


def test_plan_help() -> None:
    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0
    assert "parking" in result.output


def test_plan_parking_help() -> None:
    result = runner.invoke(app, ["plan", "parking", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--slots-json" in result.output
    assert "--slot-id" in result.output
    assert "--png" in result.output


def test_plan_parking_from_slots_json(tmp_path: Path) -> None:
    slots_file = tmp_path / "slots.json"
    _make_slots_json(slots_file)

    out_file = tmp_path / "plan.json"
    png_file = tmp_path / "plan.png"

    result = runner.invoke(
        app,
        [
            "plan",
            "parking",
            "--slots-json",
            str(slots_file),
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    assert png_file.is_file()

    payload = read_json(out_file)
    assert payload["report"]["status"] == "success"
    assert payload["report"]["is_executable"] is True
    assert payload["report"]["selected_slot_id"] == "slot_01"
    assert payload["report"]["plan"]["gear_switches"] >= 1
    assert payload["report"]["plan"]["total_length_m"] > 0


def test_plan_parking_overwrite_guard(tmp_path: Path) -> None:
    slots_file = tmp_path / "slots.json"
    _make_slots_json(slots_file)
    out_file = tmp_path / "plan.json"
    out_file.write_text("existing", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "plan",
            "parking",
            "--slots-json",
            str(slots_file),
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code != 0
    assert "Artifact error" in result.output

    # With --overwrite it should succeed
    result_ow = runner.invoke(
        app,
        [
            "plan",
            "parking",
            "--slots-json",
            str(slots_file),
            "--output",
            str(out_file),
            "--overwrite",
        ],
    )
    assert result_ow.exit_code == 0


def test_plan_parking_no_vacant_slots(tmp_path: Path) -> None:
    slots_file = tmp_path / "empty_slots.json"
    slots_file.write_text(json.dumps({"slots": []}), encoding="utf-8")
    out_file = tmp_path / "plan_empty.json"

    result = runner.invoke(
        app,
        [
            "plan",
            "parking",
            "--slots-json",
            str(slots_file),
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    payload = read_json(out_file)
    assert payload["report"]["status"] == "no_vacant_slot"
    assert payload["report"]["is_executable"] is False
