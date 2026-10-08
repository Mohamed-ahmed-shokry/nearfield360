"""CLI commands for Autonomous Valet Parking (AVP) mission execution and lifecycle orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from nearfield360.cli.state import get_state
from nearfield360.config import MissionConfig
from nearfield360.control.simulator import SimulatorNoiseConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.mission.executive import MissionExecutive
from nearfield360.mission.viz import (
    render_mission_dashboard_overlay,
    render_mission_timeline_chart,
)
from nearfield360.perception.evaluation import environment_metadata
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
from nearfield360.utils.artifacts import ArtifactError, write_json

mission_app = typer.Typer(
    help="Autonomous Valet Parking (AVP) mission executive, dynamic replanning, and lifecycle.",
    no_args_is_help=True,
)

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="JSON path to write AVP mission execution summary report.",
    ),
]

SlotsJsonOption = Annotated[
    Path | None,
    typer.Option(
        "--slots-json",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Precomputed parking slots JSON report.",
    ),
]

PlanJsonOption = Annotated[
    Path | None,
    typer.Option(
        "--plan-json",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Precomputed parking trajectory plan JSON to execute under mission manager.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Optional destination PNG path for BEV mission dashboard overlay.",
    ),
]

TimelinePngOption = Annotated[
    Path | None,
    typer.Option(
        "--timeline-png",
        resolve_path=True,
        help="Optional destination PNG path for mission state timeline and telemetry chart.",
    ),
]

ScenarioOption = Annotated[
    str,
    typer.Option(
        "--scenario",
        help="Simulation test scenario: 'nominal', 'transient_obstacle', or 'blocked_replan'.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace existing output artifacts."),
]


def _create_fallback_slot() -> ParkingSlot:
    corners = (
        ParkingSlotCorner(x=2.8, y=1.0, z=0.0),
        ParkingSlotCorner(x=2.8, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=1.0, z=0.0),
    )
    return ParkingSlot(
        slot_id="slot_demo_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=(4.0, 3.5),
        heading_rad=1.5708,
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.95,
    )


def _create_fallback_plan(slot: ParkingSlot) -> ParkingTrajectoryPlan:
    start_pose = (4.0, 0.5, 1.5708)
    target_pose = (4.0, 3.5, 1.5708)
    waypoints = [
        TrajectoryWaypoint(
            x=round(start_pose[0], 4),
            y=round(start_pose[1] + i * 0.15, 4),
            heading_rad=1.5708,
            curvature=0.0,
            velocity=0.4,
            acceleration=0.0,
            gear=ManeuverGear.FORWARD,
            t=round(i * 0.35, 3),
            distance_m=round(i * 0.15, 4),
        )
        for i in range(21)
    ]
    seg = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.DOCK,
        gear=ManeuverGear.FORWARD,
        length_m=3.0,
        duration_s=7.0,
        waypoints=waypoints,
    )
    return ParkingTrajectoryPlan(
        plan_id="demo_fallback_plan",
        slot_id=slot.slot_id,
        slot_type=slot.slot_type,
        status=PlanStatus.SUCCESS,
        start_pose=start_pose,
        target_pose=target_pose,
        total_length_m=3.0,
        total_duration_s=7.0,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        is_executable=True,
        segments=[seg],
    )


def _load_plan_from_json(path: Path) -> tuple[ParkingTrajectoryPlan, list[ParkingSlot]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "plan" in data and data["plan"] is not None:
        plan = ParkingTrajectoryPlan.model_validate(data["plan"])
    else:
        plan = ParkingTrajectoryPlan.model_validate(data)

    slots: list[ParkingSlot] = []
    if "slots" in data and isinstance(data["slots"], list):
        slots.extend(ParkingSlot.model_validate(s) for s in data["slots"])
    return plan, slots


@mission_app.command(name="run")
def run_mission_command(
    context: typer.Context,
    output: OutputOption,
    slots_json: SlotsJsonOption = None,
    plan_json: PlanJsonOption = None,
    scenario: ScenarioOption = "nominal",
    png: PngOption = None,
    timeline_png: TimelinePngOption = None,
    max_replans: Annotated[int | None, typer.Option(help="Max replan attempts.")] = None,
    hold_timeout: Annotated[float | None, typer.Option(help="Hold timeout in seconds.")] = None,
    overwrite: OverwriteOption = False,
    noise_pos: Annotated[float, typer.Option(help="Position noise std in metres.")] = 0.0,
    noise_heading: Annotated[float, typer.Option(help="Heading noise std in radians.")] = 0.0,
) -> None:
    """Run full Autonomous Valet Parking (AVP) mission lifecycle simulation."""
    config = get_state(context).config
    mission_cfg = config.mission
    if max_replans is not None or hold_timeout is not None:
        mission_cfg = MissionConfig(
            hold_timeout_s=hold_timeout if hold_timeout is not None else mission_cfg.hold_timeout_s,
            max_replans=max_replans if max_replans is not None else mission_cfg.max_replans,
            replan_pull_out_dist_m=mission_cfg.replan_pull_out_dist_m,
            approach_speed_m_s=mission_cfg.approach_speed_m_s,
            slot_tracking_distance_gate_m=mission_cfg.slot_tracking_distance_gate_m,
            slot_confirm_frames=mission_cfg.slot_confirm_frames,
            slot_max_miss_frames=mission_cfg.slot_max_miss_frames,
            docking_tolerance_x_m=mission_cfg.docking_tolerance_x_m,
            docking_tolerance_y_m=mission_cfg.docking_tolerance_y_m,
            docking_tolerance_heading_rad=mission_cfg.docking_tolerance_heading_rad,
            safety_dwell_steps=mission_cfg.safety_dwell_steps,
        )

    grid = BevGrid(
        x_min=config.bev.x_min,
        x_max=config.bev.x_max,
        y_min=config.bev.y_min,
        y_max=config.bev.y_max,
        resolution=config.bev.resolution,
    )
    fused_occ: NDArray[np.float64] = np.zeros(grid.shape, dtype=np.float64)

    evaluated_slots: list[ParkingSlot] = []
    loaded_plan: ParkingTrajectoryPlan | None = None

    if plan_json is not None:
        try:
            loaded_plan, evaluated_slots = _load_plan_from_json(plan_json)
        except Exception as exc:
            typer.secho(
                f"Failed to load plan from {plan_json}: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1) from exc
    elif slots_json is not None:
        try:
            doc = json.loads(slots_json.read_text(encoding="utf-8"))
            slots_raw = doc.get("slots", [])
            evaluated_slots = [ParkingSlot.model_validate(s) for s in slots_raw]
        except Exception as exc:
            typer.secho(
                f"Failed to load slots from {slots_json}: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1) from exc
    else:
        # Fallback default slot and approach plan for quick CLI exploration
        fallback_slot = _create_fallback_slot()
        evaluated_slots = [fallback_slot]
        loaded_plan = _create_fallback_plan(fallback_slot)

    # Scenario threat setup
    transient_obs: tuple[float, float, float] | None = None
    transient_window: tuple[int, int] | None = None
    persistent_obs: tuple[float, float, float] | None = None
    persistent_step: int | None = None

    norm_scenario = scenario.strip().lower()
    if norm_scenario == "transient_obstacle":
        transient_obs = (4.0, 1.2, 0.4)
        transient_window = (10, 22)
    elif norm_scenario == "blocked_replan":
        persistent_obs = (4.0, 1.0, 0.25)
        persistent_step = 10
        if hold_timeout is None:
            mission_cfg = mission_cfg.model_copy(update={"hold_timeout_s": 0.5})

    noise_cfg = (
        SimulatorNoiseConfig(pos_std_m=noise_pos, heading_std_rad=noise_heading)
        if (noise_pos > 0.0 or noise_heading > 0.0)
        else None
    )

    executive = MissionExecutive(
        mission_config=mission_cfg,
        planner_config=config.planner,
        control_config=config.control,
    )

    report, steps = executive.execute_mission(
        slots=evaluated_slots,
        initial_plan=loaded_plan,
        occupancy=fused_occ,
        grid=grid,
        transient_obstacle=transient_obs,
        transient_obstacle_window=transient_window,
        persistent_obstacle=persistent_obs,
        persistent_obstacle_step=persistent_step,
        noise_config=noise_cfg,
        mission_id=f"mission_{norm_scenario}",
    )

    # Export output JSON
    payload = {
        "environment": environment_metadata(),
        "mission_report": report.model_dump(),
        "steps_count": len(steps),
        "target_slot": report.target_slot_id,
        "is_success": report.is_success,
    }

    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    # Optional PNG artifacts
    if png is not None:
        render_mission_dashboard_overlay(
            grid=grid,
            report=report,
            steps=steps,
            slots=evaluated_slots,
            plan=loaded_plan,
            output_path=png,
        )
        typer.echo(f"Wrote BEV mission overlay: {png}")

    if timeline_png is not None:
        render_mission_timeline_chart(
            report=report,
            steps=steps,
            output_path=timeline_png,
        )
        typer.echo(f"Wrote mission timeline chart: {timeline_png}")

    # Terminal summary
    status_color = typer.colors.GREEN if report.is_success else typer.colors.RED
    typer.secho(
        f"Mission {report.mission_id} [{report.final_state.value.upper()}]: "
        f"{report.message} (Duration: {report.total_duration_s:.1f}s, "
        f"Steps: {report.total_steps}, Replans: {report.replan_count})",
        fg=status_color,
    )
