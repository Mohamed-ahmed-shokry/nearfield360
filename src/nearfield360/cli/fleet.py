"""CLI commands for cloud fleet coordination, V2X dispatch, and multi-agent conflict management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.fleet.coordinator import FleetCoordinator
from nearfield360.fleet.dispatcher import SlotDispatcher
from nearfield360.fleet.viz import render_fleet_coordination_dashboard
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

fleet_app = typer.Typer(
    help="Cloud fleet coordination, V2X facility dispatch, and multi-agent conflict deconfliction.",
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
        help="Path to render fleet coordination BEV PNG artifact.",
    ),
]


def _load_facility(map_path: Path | None) -> FacilityMap:
    """Load facility map from JSON path or default to benchmark garage."""
    if map_path is not None:
        try:
            data = read_json(map_path)
            return FacilityMap.model_validate(data)
        except (ArtifactError, Exception) as exc:
            typer.echo(f"Error loading facility map from {map_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    return build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=(1, 5))


@fleet_app.command("info")
def fleet_info(
    context: typer.Context,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
    output: OutputOption = None,
) -> None:
    """Display cloud fleet coordination status and facility bay capacity."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.fleet
    dispatcher = SlotDispatcher(facility, config)
    summary = dispatcher.get_occupancy_summary(current_time_s=0.0)

    payload = {
        "facility_id": facility.map_id,
        "facility_name": facility.name,
        "total_slots": summary.total_slots,
        "vacant_slots": summary.vacant_slots,
        "reserved_slots": summary.reserved_slots,
        "occupied_slots": summary.occupied_slots,
        "blocked_slots": summary.blocked_slots,
        "heartbeat_interval_s": config.heartbeat_interval_s,
        "temporal_headway_s": config.temporal_headway_s,
        "priority_policy": config.priority_policy,
    }

    if output is not None:
        try:
            write_json(output, payload, overwrite=True)
            typer.echo(f"Wrote fleet info to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo("=== Fleet Dispatch & Capacity Info ===")
        typer.echo(f"Facility:     {payload['facility_name']} ({payload['facility_id']})")
        typer.echo(f"Total Slots:  {payload['total_slots']}")
        typer.echo(f"Vacant:       {payload['vacant_slots']}")
        typer.echo(f"Occupied:     {payload['occupied_slots']}")
        typer.echo(f"Reserved:     {payload['reserved_slots']}")
        typer.echo(f"Headway:      {payload['temporal_headway_s']} s")
        typer.echo(f"Policy:       {payload['priority_policy']}")


@fleet_app.command("dispatch")
def fleet_dispatch(
    context: typer.Context,
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", "-v", help="ID of incoming vehicle.")],
    map_path: MapPathOption = None,
    start_x: Annotated[float, typer.Option("--start-x", help="Start X coordinate.")] = 0.0,
    start_y: Annotated[float, typer.Option("--start-y", help="Start Y coordinate.")] = 0.0,
    start_yaw: Annotated[float, typer.Option("--start-yaw", help="Start heading (rad).")] = 0.0,
    target_slot: Annotated[
        str | None, typer.Option("--target-slot", help="Explicit target slot ID.")
    ] = None,
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
) -> None:
    """Allocate an optimal parking bay and issue a V2I dispatch assignment."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.fleet
    coordinator = FleetCoordinator(facility, config=config)

    coordinator.register_vehicle(vehicle_id, initial_pose=(start_x, start_y, start_yaw))
    try:
        asg = coordinator.dispatch_vehicle(vehicle_id, target_slot_id=target_slot)
    except Exception as exc:
        typer.echo(f"Dispatch error for vehicle {vehicle_id}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = asg.model_dump(mode="json")
    if output is not None:
        try:
            write_json(output, payload, overwrite=True)
            typer.echo(f"Wrote dispatch assignment to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"=== Dispatched {vehicle_id} ===")
        typer.echo(f"Assignment ID:  {asg.assignment_id}")
        typer.echo(f"Assigned Bay:   {asg.assigned_slot_id}")
        typer.echo(f"Lease Expires:  {asg.lease_expires_at_s:.1f} s")
        typer.echo(f"Waypoints:      {len(asg.route_waypoint_ids)}")


@fleet_app.command("simulate")
def fleet_simulate(
    context: typer.Context,
    map_path: MapPathOption = None,
    vehicle_count: Annotated[
        int, typer.Option("--vehicles", "-n", help="Number of vehicles to simulate.")
    ] = 3,
    steps: Annotated[int, typer.Option("--steps", "-s", help="Simulation steps to advance.")] = 40,
    dt: Annotated[float, typer.Option("--dt", help="Simulation time delta in seconds.")] = 0.5,
    output: OutputOption = None,
    png: PngOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON report.")] = False,
) -> None:
    """Run discrete-time multi-agent fleet simulation with V2X messaging and conflict handling."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.fleet
    coordinator = FleetCoordinator(facility, config=config)

    # Register and dispatch vehicles
    for i in range(vehicle_count):
        vid = f"veh_{i + 1:02d}"
        init_x = float(i * 3.0)
        coordinator.register_vehicle(vid, initial_pose=(init_x, 0.0, 0.0))
        coordinator.dispatch_vehicle(vid)

    # Step simulation forward
    for _ in range(steps):
        coordinator.step(dt_s=dt)

    report = coordinator.generate_report()
    payload = report.model_dump(mode="json")

    if png is not None:
        try:
            render_fleet_coordination_dashboard(
                facility=facility,
                vehicles=list(coordinator.vehicles.values()),
                conflicts=coordinator.active_conflicts,
                report=report,
                output_path=png,
            )
            typer.echo(f"Rendered fleet telemetry dashboard to {png}")
        except Exception as exc:
            typer.echo(f"Error rendering dashboard PNG: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if output is not None:
        try:
            write_json(output, payload, overwrite=True)
            typer.echo(f"Wrote simulation report to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo("=== Multi-Agent Fleet Simulation Complete ===")
        typer.echo(f"Facility:            {report.facility_id}")
        typer.echo(f"Elapsed Time:        {report.timestamp_s:.1f} s")
        typer.echo(f"Active Vehicles:     {report.total_vehicles_active}")
        typer.echo(f"Conflicts Resolved:  {report.resolved_conflicts_count}")
        typer.echo(f"Average Wait Time:   {report.average_wait_time_s:.2f} s")
        vac_cnt = report.occupancy_summary.vacant_slots
        tot_cnt = report.occupancy_summary.total_slots
        typer.echo(f"Vacant Bays:         {vac_cnt}/{tot_cnt}")


@fleet_app.command("deconflict")
def fleet_deconflict(
    context: typer.Context,
    map_path: MapPathOption = None,
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON conflict list.")] = False,
) -> None:
    """Analyze multi-vehicle trajectories and identify spatiotemporal conflict points."""
    facility = _load_facility(map_path)
    state = get_state(context)
    config = state.config.fleet
    coordinator = FleetCoordinator(facility, config=config)

    # Register test vehicles intersecting near entrance
    coordinator.register_vehicle("v1", (0.0, 0.0, 0.0))
    coordinator.register_vehicle("v2", (15.0, 5.0, 1.57))
    coordinator.dispatch_vehicle("v1")
    coordinator.dispatch_vehicle("v2")

    conflicts = coordinator.run_deconfliction_cycle()
    payload = [c.model_dump(mode="json") for c in conflicts]

    if output is not None:
        try:
            write_json(output, payload, overwrite=True)
            typer.echo(f"Wrote conflict analysis to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Identified {len(conflicts)} Spatiotemporal Conflict Point(s):")
        for c in conflicts:
            pair_str = f"{c.primary_vehicle_id} vs {c.secondary_vehicle_id}"
            typer.echo(f"- [{c.conflict_id}] {pair_str} at {c.conflict_location}")
            act_str = f"{c.action_primary.value} / {c.action_secondary.value}"
            typer.echo(f"  Arrival: {c.earliest_arrival_s:.1f}s | Action: {act_str}")
