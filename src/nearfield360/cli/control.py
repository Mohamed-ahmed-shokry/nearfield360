"""CLI commands for closed-loop parking trajectory tracking and vehicle execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from numpy.typing import NDArray

from nearfield360.cli.state import get_state
from nearfield360.control.executor import ManeuverExecutor
from nearfield360.control.models import ExecutionStatus
from nearfield360.control.simulator import SimulatorNoiseConfig
from nearfield360.control.viz import (
    render_control_execution_bev_overlay,
    render_control_telemetry_chart,
)
from nearfield360.geometry.bev import BevGrid
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.planning.planner import ParkingTrajectoryPlanner
from nearfield360.slots.models import ParkingSlot
from nearfield360.utils.artifacts import ArtifactError, write_json

control_app = typer.Typer(
    help="Closed-loop trajectory tracking control, vehicle simulation, and dynamic safety.",
    no_args_is_help=True,
)

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="JSON path to write closed-loop maneuver execution report.",
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
        help="Precomputed parking trajectory plan JSON to execute.",
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
        help="Precomputed parking slots JSON report to plan and execute.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Optional destination PNG path for BEV closed-loop execution overlay.",
    ),
]

TelemetryPngOption = Annotated[
    Path | None,
    typer.Option(
        "--telemetry-png",
        resolve_path=True,
        help="Optional destination PNG path for time-series telemetry plots.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace existing output artifacts."),
]

NoisePosOption = Annotated[
    float,
    typer.Option("--noise-pos", help="Gaussian position noise standard deviation in metres."),
]

NoiseHeadingOption = Annotated[
    float,
    typer.Option("--noise-heading", help="Gaussian heading noise standard deviation in radians."),
]

InjectObstacleXOption = Annotated[
    float | None,
    typer.Option("--inject-obstacle-x", help="X coordinate for dynamic obstacle injection test."),
]

InjectObstacleYOption = Annotated[
    float | None,
    typer.Option("--inject-obstacle-y", help="Y coordinate for dynamic obstacle injection test."),
]

InjectObstacleRadiusOption = Annotated[
    float,
    typer.Option(
        "--inject-obstacle-radius",
        help="Radius for dynamic obstacle injection test in metres.",
    ),
]

InjectObstacleStepOption = Annotated[
    int | None,
    typer.Option("--inject-obstacle-step", help="Step index to trigger injected obstacle threat."),
]


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


@control_app.command(name="execute")
def execute_control_command(
    context: typer.Context,
    output: OutputOption,
    plan_json: PlanJsonOption = None,
    slots_json: SlotsJsonOption = None,
    png: PngOption = None,
    telemetry_png: TelemetryPngOption = None,
    overwrite: OverwriteOption = False,
    noise_pos: NoisePosOption = 0.0,
    noise_heading: NoiseHeadingOption = 0.0,
    inject_obstacle_x: InjectObstacleXOption = None,
    inject_obstacle_y: InjectObstacleYOption = None,
    inject_obstacle_radius: InjectObstacleRadiusOption = 0.25,
    inject_obstacle_step: InjectObstacleStepOption = None,
) -> None:
    """Execute a parking trajectory plan under closed-loop Stanley and longitudinal control."""
    config = get_state(context).config
    grid = BevGrid(
        x_min=config.bev.x_min,
        x_max=config.bev.x_max,
        y_min=config.bev.y_min,
        y_max=config.bev.y_max,
        resolution=config.bev.resolution,
    )

    plan: ParkingTrajectoryPlan | None = None
    evaluated_slots: list[ParkingSlot] = []
    fused_occ: NDArray[np.float64] = np.zeros(grid.shape, dtype=np.float64)

    if plan_json is not None:
        try:
            plan, evaluated_slots = _load_plan_from_json(plan_json)
        except Exception as exc:
            typer.secho(
                f"Failed to load plan from {plan_json}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from exc
    elif slots_json is not None:
        try:
            doc = json.loads(slots_json.read_text(encoding="utf-8"))
            slots_raw = doc.get("slots", [])
            evaluated_slots = [ParkingSlot.model_validate(s) for s in slots_raw]
            planner = ParkingTrajectoryPlanner(config.planner)
            plan_report = planner.plan_parking(
                slots=evaluated_slots,
                occupancy=fused_occ,
                grid=grid,
                start_pose=(0.0, 0.0, 0.0),
            )
            plan = plan_report.plan
        except Exception as exc:
            typer.secho(
                f"Failed to plan from slots JSON {slots_json}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from exc
    else:
        typer.secho(
            "Must provide either --plan-json or --slots-json to execute control.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if plan is None:
        typer.secho(
            "No executable parking trajectory plan available.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Configure noise and dynamic threats
    noise_cfg = (
        SimulatorNoiseConfig(pos_std_m=noise_pos, heading_std_rad=noise_heading)
        if (noise_pos > 0.0 or noise_heading > 0.0)
        else None
    )

    injected_threat = (
        (inject_obstacle_x, inject_obstacle_y, inject_obstacle_radius)
        if (inject_obstacle_x is not None and inject_obstacle_y is not None)
        else None
    )

    executor = ManeuverExecutor(config.control, config.planner)
    report = executor.execute_plan(
        plan=plan,
        occupancy=fused_occ,
        grid=grid,
        noise_config=noise_cfg,
        injected_obstacle=injected_threat,
        inject_obstacle_at_step=inject_obstacle_step,
    )

    # Save output artifacts
    payload: dict[str, Any] = {
        "environment": environment_metadata(),
        "config": config.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }

    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if png is not None:
        try:
            render_control_execution_bev_overlay(
                grid=grid,
                plan=plan,
                report=report,
                slots=evaluated_slots,
                output_path=png,
            )
        except (OSError, ValueError) as exc:
            typer.secho(f"BEV PNG rendering error: {exc}", fg=typer.colors.RED, err=True)

    if telemetry_png is not None:
        try:
            render_control_telemetry_chart(report=report, output_path=telemetry_png)
        except (OSError, ValueError) as exc:
            typer.secho(f"Telemetry PNG rendering error: {exc}", fg=typer.colors.RED, err=True)

    is_complete = report.status == ExecutionStatus.COMPLETED
    status_color = typer.colors.GREEN if is_complete else typer.colors.YELLOW
    dock_str = "YES" if report.kpis.is_docked_successfully else "NO"
    cte_cm = report.kpis.max_cross_track_error_m * 100
    typer.secho(
        f"Control execution {report.status.value}: duration={report.duration_s:.2f}s, "
        f"steps={report.total_steps}, max_cte={cte_cm:.1f}cm, "
        f"docked={dock_str}.",
        fg=status_color,
    )
