"""Unit tests for closed-loop control visualization and telemetry charts."""

from __future__ import annotations

from pathlib import Path

from nearfield360.control.executor import ManeuverExecutor
from nearfield360.control.models import ManeuverExecutionReport
from nearfield360.control.viz import (
    render_control_execution_bev_overlay,
    render_control_telemetry_chart,
)
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import ParkingSlotType


def _make_dummy_plan_and_report() -> tuple[ParkingTrajectoryPlan, ManeuverExecutionReport]:
    waypoints = [
        TrajectoryWaypoint(
            x=float(i) * 0.1,
            y=0.0,
            heading_rad=0.0,
            velocity=0.5,
            gear=ManeuverGear.FORWARD,
            t=float(i) * 0.2,
        )
        for i in range(10)
    ]
    segment = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.APPROACH,
        gear=ManeuverGear.FORWARD,
        length_m=1.0,
        duration_s=2.0,
        waypoints=waypoints,
    )
    plan = ParkingTrajectoryPlan(
        plan_id="viz-plan-1",
        slot_id="slot-01",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.SUCCESS,
        total_length_m=1.0,
        total_duration_s=2.0,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(1.0, 0.0, 0.0),
        is_executable=True,
        segments=[segment],
    )
    executor = ManeuverExecutor()
    report = executor.execute_plan(plan)
    return plan, report


def test_render_control_execution_bev_overlay(tmp_path: Path) -> None:
    grid = BevGrid(x_min=-2.0, x_max=4.0, y_min=-3.0, y_max=3.0, resolution=0.1)
    plan, report = _make_dummy_plan_and_report()

    out_file = tmp_path / "bev_control.png"
    canvas = render_control_execution_bev_overlay(
        grid=grid,
        plan=plan,
        report=report,
        output_path=out_file,
    )
    assert canvas.shape == (grid.shape[0], grid.shape[1], 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 500


def test_render_control_telemetry_chart(tmp_path: Path) -> None:
    _, report = _make_dummy_plan_and_report()
    out_file = tmp_path / "telemetry.png"

    canvas = render_control_telemetry_chart(
        report=report,
        output_path=out_file,
    )
    assert canvas.shape == (600, 800, 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
