"""CLI commands for parking facility HD vector mapping, routing, and localization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.mapping.builder import build_benchmark_garage, build_surface_lot
from nearfield360.mapping.localization import LocalizationSimulator
from nearfield360.mapping.models import FacilityMap, GlobalRoute, SlotReservationStatus
from nearfield360.mapping.router import GlobalRouter
from nearfield360.mapping.viz import render_facility_bev, render_localization_dashboard
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

map_app = typer.Typer(
    help="Parking facility HD vector mapping, global topological routing, and localization.",
    no_args_is_help=True,
)

MapPathOption = Annotated[
    Path | None,
    typer.Option(
        "--map-path",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to facility map JSON file (defaults to benchmark garage if omitted).",
    ),
]

OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="Path to write JSON artifact atomically.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Path to render BEV facility layout PNG artifact.",
    ),
]


def _load_facility(map_path: Path | None) -> FacilityMap:
    """Load facility map from JSON path or default to benchmark indoor garage."""
    if map_path is not None:
        try:
            data = read_json(map_path)
            return FacilityMap.model_validate(data)
        except (ArtifactError, ValueError) as exc:
            typer.secho(f"Failed to load facility map: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None
    return build_benchmark_garage()


@map_app.command("info")
def map_info(
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Inspect facility map topology, bounds, slots, lanes, and integrity."""
    facility = _load_facility(map_path)
    issues = facility.validate_integrity()

    vacant_count = sum(1 for s in facility.slots if s.status == SlotReservationStatus.VACANT)
    occupied_count = sum(1 for s in facility.slots if s.status == SlotReservationStatus.OCCUPIED)

    summary = {
        "map_id": facility.map_id,
        "name": facility.name,
        "facility_type": facility.facility_type,
        "bounds": {
            "x_min": facility.bounds.x_min,
            "x_max": facility.bounds.x_max,
            "y_min": facility.bounds.y_min,
            "y_max": facility.bounds.y_max,
        },
        "slot_counts": {
            "total": len(facility.slots),
            "vacant": vacant_count,
            "occupied": occupied_count,
            "other": len(facility.slots) - vacant_count - occupied_count,
        },
        "lane_count": len(facility.lanes),
        "obstacle_count": len(facility.obstacles),
        "waypoint_count": len(facility.waypoints),
        "integrity": {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues,
        },
    }

    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.echo(f"Facility Map:   {facility.name} (ID: {facility.map_id})")
    typer.echo(f"Facility Type:  {facility.facility_type}")
    b = facility.bounds
    typer.echo(f"Bounds:         [{b.x_min:.1f}, {b.x_max:.1f}] x [{b.y_min:.1f}, {b.y_max:.1f}] m")
    typer.echo(
        f"Slots:          {len(facility.slots)} total "
        f"({vacant_count} vacant, {occupied_count} occupied)"
    )
    typer.echo(f"Lanes:          {len(facility.lanes)} driving corridors")
    typer.echo(f"Waypoints:      {len(facility.waypoints)} topological nodes")
    typer.echo(f"Obstacles:      {len(facility.obstacles)} structural boundaries")
    if not issues:
        typer.secho(
            "Integrity:      VALID (0 topological or geometric errors)", fg=typer.colors.GREEN
        )
    else:
        typer.secho(f"Integrity:      INVALID ({len(issues)} errors):", fg=typer.colors.RED)
        for issue in issues:
            typer.echo(f"  - {issue}")


@map_app.command("build")
def map_build(
    facility_type: Annotated[
        str, typer.Option("--type", help="Facility type: garage | lot.")
    ] = "garage",
    aisles: Annotated[int, typer.Option("--aisles", help="Number of aisles for garage.")] = 2,
    slots_per_aisle: Annotated[int, typer.Option("--slots-per-aisle", help="Slots per aisle.")] = 4,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Generate a benchmark indoor parking garage or surface parking lot map."""
    if facility_type.lower() == "lot":
        facility = build_surface_lot()
    else:
        facility = build_benchmark_garage(aisle_count=aisles, slots_per_aisle=slots_per_aisle)

    typer.echo(
        f"Generated {facility.name} ({len(facility.slots)} slots, {len(facility.lanes)} lanes)."
    )

    if output is not None:
        try:
            write_json(output, facility.model_dump(), overwrite=True)
            typer.echo(f"Wrote facility map JSON to {output}")
        except ArtifactError as exc:
            typer.secho(f"Failed to write map JSON: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None

    if png is not None:
        render_facility_bev(facility, output_path=png)
        typer.echo(f"Rendered facility BEV layout to {png}")


@map_app.command("route")
def map_route(
    context: typer.Context,
    map_path: MapPathOption = None,
    start_pose: Annotated[
        str, typer.Option("--start-pose", help="Start pose x,y,heading in meters and radians.")
    ] = "0.0,0.0,0.0",
    target_slot: Annotated[
        str | None, typer.Option("--target-slot", help="Target parking bay ID.")
    ] = "bay_a0_s0",
    target_waypoint: Annotated[
        str | None, typer.Option("--target-waypoint", help="Target waypoint ID.")
    ] = None,
    blocked_lanes: Annotated[
        str | None, typer.Option("--blocked-lanes", help="Comma-separated blocked lane IDs.")
    ] = None,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Plan an optimal global topological and metric route to a target bay or waypoint."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.mapping

    # Parse start pose
    try:
        parts = [float(p.strip()) for p in start_pose.split(",")]
        if len(parts) != 3:
            raise ValueError
        pose = (parts[0], parts[1], parts[2])
    except ValueError:
        typer.secho(
            "Invalid --start-pose. Must be format 'x,y,heading' (e.g. '0.0,0.0,0.0')",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None

    blocked_set = set(blocked_lanes.split(",")) if blocked_lanes else None

    router = GlobalRouter(facility, turn_penalty_weight=config.turn_penalty_weight)
    try:
        route = router.plan(
            start_pose=pose,
            target_slot_id=target_slot,
            target_waypoint_id=target_waypoint,
            blocked_lanes=blocked_set,
        )
    except Exception as exc:
        typer.secho(f"Route planning failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Planned route {route.route_id}:")
    typer.echo(f"  Total length:       {route.total_length_m:.2f} m")
    typer.echo(f"  Estimated duration: {route.estimated_duration_s:.1f} s")
    typer.echo(f"  Waypoints count:    {len(route.waypoints)}")
    typer.echo(f"  Corridors traversed: {' -> '.join(route.lane_sequence)}")

    if output is not None:
        try:
            write_json(output, route.model_dump(), overwrite=True)
            typer.echo(f"Wrote global route JSON to {output}")
        except ArtifactError as exc:
            typer.secho(f"Failed to write route JSON: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None

    if png is not None:
        render_facility_bev(facility, route=route, output_path=png)
        typer.echo(f"Rendered route BEV visualization to {png}")


@map_app.command("localize")
def map_localize(
    context: typer.Context,
    map_path: MapPathOption = None,
    route_json: Annotated[
        Path | None,
        typer.Option(
            "--route-json",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Precomputed route JSON.",
        ),
    ] = None,
    target_slot: Annotated[
        str, typer.Option("--target-slot", help="Target slot ID if route is auto-planned.")
    ] = "bay_a0_s0",
    seed: Annotated[int, typer.Option("--seed", help="Random seed for noise generation.")] = 42,
    output: OutputOption = None,
    dashboard_png: Annotated[
        Path | None,
        typer.Option(
            "--dashboard-png", resolve_path=True, help="Path for telemetry dashboard PNG."
        ),
    ] = None,
    bev_png: Annotated[
        Path | None,
        typer.Option("--bev-png", resolve_path=True, help="Path for BEV trajectory PNG."),
    ] = None,
) -> None:
    """Simulate vehicle motion with dead-reckoning drift and EKF slot landmark corrections."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.mapping

    # Load or generate route
    if route_json is not None:
        try:
            route_data = read_json(route_json)
            route = GlobalRoute.model_validate(route_data)
        except Exception as exc:
            typer.secho(f"Failed to load route JSON: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None
    else:
        router = GlobalRouter(facility, turn_penalty_weight=config.turn_penalty_weight)
        route = router.plan(start_pose=(0.0, 0.0, 0.0), target_slot_id=target_slot)

    wp_coords = [(w.x, w.y, w.heading_rad) for w in route.waypoints]
    sim = LocalizationSimulator(facility=facility, config=config, seed=seed)
    report, records = sim.simulate(wp_coords)

    typer.echo("Localization Simulation Complete:")
    typer.echo(
        f"  Trajectory length:   {report.trajectory_length_m:.2f} m ({report.step_count} steps)"
    )
    typer.echo(f"  Landmark fixes:      {report.total_landmark_updates}")
    typer.echo(
        f"  Mean position error: {report.mean_position_error_m:.3f} m "
        f"(Max: {report.max_position_error_m:.3f} m)"
    )
    typer.echo(
        f"  Mean heading error:  {math.degrees(report.mean_heading_error_rad):.2f} deg "
        f"(Max: {math.degrees(report.max_heading_error_rad):.2f} deg)"
    )
    typer.echo(f"  Max 1-sigma uncert:  {report.max_position_uncertainty_m:.3f} m")

    if output is not None:
        try:
            write_json(output, report.model_dump(), overwrite=True)
            typer.echo(f"Wrote localization report JSON to {output}")
        except ArtifactError as exc:
            typer.secho(
                f"Failed to write localization report: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=2) from None

    if dashboard_png is not None:
        render_localization_dashboard(
            report=report, records=records, facility=facility, output_path=dashboard_png
        )
        typer.echo(f"Rendered localization dashboard to {dashboard_png}")

    if bev_png is not None:
        est_poses = [r.estimated_pose for r in records]
        render_facility_bev(facility, route=route, estimated_poses=est_poses, output_path=bev_png)
        typer.echo(f"Rendered BEV trajectory visualization to {bev_png}")


__all__ = ["map_app"]
