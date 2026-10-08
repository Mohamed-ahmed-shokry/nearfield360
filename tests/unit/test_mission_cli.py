"""Integration tests for the nearfield360 mission CLI command group."""

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
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)

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
        plan_id="mission-test-plan-01",
        slot_id="slot_01",
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


def _write_sample_slots_json(path: Path) -> None:
    slot = ParkingSlot(
        slot_id="slot_sample_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=(
            ParkingSlotCorner(x=2.8, y=1.0, z=0.0),
            ParkingSlotCorner(x=2.8, y=6.0, z=0.0),
            ParkingSlotCorner(x=5.2, y=6.0, z=0.0),
            ParkingSlotCorner(x=5.2, y=1.0, z=0.0),
        ),
        center=(4.0, 3.5),
        heading_rad=1.5708,
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.95,
    )
    payload = {"slots": [slot.model_dump(mode="json")]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_mission_help() -> None:
    res = runner.invoke(app, ["mission", "--help"])
    assert res.exit_code == 0
    assert "Autonomous Valet Parking (AVP) mission lifecycle" in res.output
    assert "run" in res.output


def test_mission_run_help() -> None:
    res = runner.invoke(app, ["mission", "run", "--help"])
    assert res.exit_code == 0
    assert "--plan-json" in res.output
    assert "--slots-json" in res.output
    assert "--scenario" in res.output
    assert "--timeline-png" in res.output
    assert "--max-replans" in res.output
    assert "--hold-timeout" in res.output


def test_mission_run_nominal(tmp_path: Path) -> None:
    out_json = tmp_path / "mission_nominal.json"
    out_png = tmp_path / "mission_dashboard.png"
    out_timeline = tmp_path / "mission_timeline.png"

    res = runner.invoke(
        app,
        [
            "mission",
            "run",
            "--output",
            str(out_json),
            "--scenario",
            "nominal",
            "--png",
            str(out_png),
            "--timeline-png",
            str(out_timeline),
        ],
    )
    assert res.exit_code == 0
    assert "COMPLETED" in res.output
    assert out_json.is_file()
    assert out_png.is_file()
    assert out_png.stat().st_size > 0
    assert out_timeline.is_file()
    assert out_timeline.stat().st_size > 0

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "mission_report" in data
    assert data["mission_report"]["final_state"] == "completed"
    assert data["is_success"] is True
    assert data["steps_count"] > 0


def test_mission_run_transient_obstacle(tmp_path: Path) -> None:
    out_json = tmp_path / "mission_transient.json"
    res = runner.invoke(
        app,
        [
            "mission",
            "run",
            "--output",
            str(out_json),
            "--scenario",
            "transient_obstacle",
            "--hold-timeout",
            "15.0",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    report = data["mission_report"]
    # Check that events contain hold
    event_names = [e["target_state"] for e in report["events"]]
    assert "obstacle_hold" in event_names
    assert report["final_state"] == "completed"


def test_mission_run_blocked_replan(tmp_path: Path) -> None:
    out_json = tmp_path / "mission_replan.json"
    res = runner.invoke(
        app,
        [
            "mission",
            "run",
            "--output",
            str(out_json),
            "--scenario",
            "blocked_replan",
            "--max-replans",
            "3",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    report = data["mission_report"]
    assert report["replan_count"] >= 1
    event_names = [e["target_state"] for e in report["events"]]
    assert "replanning" in event_names


def test_mission_run_with_slots_json(tmp_path: Path) -> None:
    slots_path = tmp_path / "slots.json"
    _write_sample_slots_json(slots_path)
    out_json = tmp_path / "mission_slots.json"

    res = runner.invoke(
        app,
        [
            "mission",
            "run",
            "--slots-json",
            str(slots_path),
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["target_slot"] == "slot_sample_01"


def test_mission_run_with_plan_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_sample_plan_json(plan_path)
    out_json = tmp_path / "mission_plan.json"

    res = runner.invoke(
        app,
        [
            "mission",
            "run",
            "--plan-json",
            str(plan_path),
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["is_success"] is True
