"""Integration tests for the nearfield360 control CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import ParkingSlotType

runner = CliRunner()


def _write_sample_plan_json(path: Path) -> None:
    waypoints = [
        TrajectoryWaypoint(
            x=round(float(i) * 0.1, 2),
            y=0.0,
            heading_rad=0.0,
            velocity=0.6,
            gear=ManeuverGear.FORWARD,
            t=round(float(i) * 0.15, 2),
            distance_m=round(float(i) * 0.1, 2),
        )
        for i in range(15)
    ]
    segment = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.APPROACH,
        gear=ManeuverGear.FORWARD,
        length_m=1.4,
        duration_s=2.1,
        waypoints=waypoints,
    )
    plan = ParkingTrajectoryPlan(
        plan_id="cli-test-plan-01",
        slot_id="slot-01",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.SUCCESS,
        total_length_m=1.4,
        total_duration_s=2.1,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(1.4, 0.0, 0.0),
        is_executable=True,
        segments=[segment],
    )
    payload = {"plan": plan.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_control_help() -> None:
    res = runner.invoke(app, ["control", "--help"])
    assert res.exit_code == 0
    assert "Closed-loop trajectory tracking control" in res.output
    assert "execute" in res.output


def test_control_execute_help() -> None:
    res = runner.invoke(app, ["control", "execute", "--help"])
    assert res.exit_code == 0
    assert "--plan-json" in res.output
    assert "--telemetry-png" in res.output
    assert "--inject-obstacle-x" in res.output


def test_control_execute_with_plan_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_sample_plan_json(plan_path)

    out_report = tmp_path / "control_report.json"
    out_bev = tmp_path / "bev_control.png"
    out_telemetry = tmp_path / "telemetry.png"

    res = runner.invoke(
        app,
        [
            "control",
            "execute",
            "--plan-json",
            str(plan_path),
            "--output",
            str(out_report),
            "--png",
            str(out_bev),
            "--telemetry-png",
            str(out_telemetry),
        ],
    )
    assert res.exit_code == 0
    assert "Control execution completed" in res.output
    assert out_report.is_file()
    assert out_bev.is_file()
    assert out_telemetry.is_file()

    data = json.loads(out_report.read_text(encoding="utf-8"))
    assert "report" in data
    assert data["report"]["status"] == "completed"
    assert data["report"]["kpis"]["is_docked_successfully"] is True


def test_control_execute_with_obstacle_injection(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_sample_plan_json(plan_path)

    out_report = tmp_path / "control_report_hazard.json"

    res = runner.invoke(
        app,
        [
            "control",
            "execute",
            "--plan-json",
            str(plan_path),
            "--output",
            str(out_report),
            "--inject-obstacle-x",
            "0.5",
            "--inject-obstacle-y",
            "0.0",
            "--inject-obstacle-radius",
            "0.2",
            "--inject-obstacle-step",
            "3",
        ],
    )
    assert res.exit_code == 0
    assert "emergency_stopped" in res.output

    data = json.loads(out_report.read_text(encoding="utf-8"))
    assert data["report"]["status"] == "emergency_stopped"


def test_control_execute_missing_inputs(tmp_path: Path) -> None:
    out_report = tmp_path / "report.json"
    res = runner.invoke(app, ["control", "execute", "--output", str(out_report)])
    assert res.exit_code != 0
    assert "Must provide either --plan-json or --slots-json" in res.output
