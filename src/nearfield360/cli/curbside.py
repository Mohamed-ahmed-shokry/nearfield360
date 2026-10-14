"""CLI commands for curbside drop-off/pick-up staging, dwell, and pedestrian safety interlocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.curbside.manager import CurbsideManager
from nearfield360.curbside.models import PedestrianDetection
from nearfield360.curbside.viz import render_curbside_dashboard
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

curbside_app = typer.Typer(
    help="Curbside drop-off/pick-up staging, dwell, and pedestrian safety interlocks.",
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
        help="Path to render curbside BEV telemetry PNG artifact.",
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


@curbside_app.command("info")
def curbside_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active curbside staging parameters, dwell limits, and safety thresholds."""
    cfg = get_state(context).config.curbside

    info_payload = {
        "max_dropoff_dwell_s": cfg.max_dropoff_dwell_s,
        "max_pickup_dwell_s": cfg.max_pickup_dwell_s,
        "pedestrian_safety_radius_m": cfg.pedestrian_safety_radius_m,
        "interlock_clearance_hold_s": cfg.interlock_clearance_hold_s,
        "staging_capacity": cfg.staging_capacity,
        "overstay_penalty_per_min": cfg.overstay_penalty_per_min,
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("Curbside Staging & Pedestrian Safety Configuration:")
    typer.echo(f"  Max Drop-off Dwell:         {cfg.max_dropoff_dwell_s:.1f} s")
    typer.echo(f"  Max Pick-up Dwell:          {cfg.max_pickup_dwell_s:.1f} s")
    typer.echo(f"  Pedestrian Safety Radius:   {cfg.pedestrian_safety_radius_m:.1f} m")
    typer.echo(f"  Interlock Clearance Hold:   {cfg.interlock_clearance_hold_s:.1f} s")
    typer.echo(f"  Staging Bays Capacity:      {cfg.staging_capacity} bays")
    typer.echo(f"  Overstay Penalty Rate:      ${cfg.overstay_penalty_per_min:.2f}/min")


@curbside_app.command("dropoff")
def curbside_dropoff(
    context: typer.Context,
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "veh-user-01",
    dwell_s: Annotated[float, typer.Option("--dwell", help="Egress dwell seconds.")] = 60.0,
    doors_open: Annotated[bool, typer.Option("--doors-open", help="Simulate doors open.")] = False,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate vehicle arrival at drop-off bay, passenger egress, and autonomous handover."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.curbside
    mgr = CurbsideManager(facility_id=facility.map_id, config=cfg)

    # 1. Request drop-off staging
    mission, bay = mgr.request_dropoff(vehicle_id, current_time_s=0.0)

    # 2. Simulate door latch state
    mgr.interlock_engine.set_door_state(vehicle_id, is_open=doors_open)

    # 3. If doors closed, complete handover
    completed_mission = None
    if not doors_open:
        completed_mission = mgr.complete_dropoff(vehicle_id, current_time_s=dwell_s)

    report = mgr.generate_report(current_time_s=dwell_s)

    if png is not None:
        render_curbside_dashboard(facility, report, output_path=png)

    payload = {
        "mission": (completed_mission or mission).model_dump(mode="json"),
        "assigned_bay": bay.model_dump(mode="json") if bay else None,
        "interlock_status": bay.interlock_status.value if bay else "none",
        "overstay_penalty_usd": (
            completed_mission.overstay_penalty_usd if completed_mission else 0.0
        ),
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Curbside Drop-off Handover Execution:")
    typer.echo(f"  Mission ID:         {mission.mission_id}")
    typer.echo(f"  Vehicle ID:         {vehicle_id}")
    typer.echo(f"  Assigned Bay:       {bay.bay_id if bay else 'QUEUED'}")
    typer.echo(f"  Dwell Time:         {dwell_s:.1f} s")
    if completed_mission:
        typer.echo(
            "  Handover Status:    COMPLETED "
            f"(Overstay: ${completed_mission.overstay_penalty_usd:.2f})"
        )
    else:
        typer.echo("  Handover Status:    INHIBITED (Safety Interlock Engaged)")


@curbside_app.command("summon")
def curbside_summon(
    context: typer.Context,
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "veh-summon-01",
    dwell_s: Annotated[float, typer.Option("--dwell", help="Boarding dwell seconds.")] = 90.0,
    ped_dist: Annotated[
        float, typer.Option("--pedestrian-distance", help="Pedestrian distance (m).")
    ] = 3.5,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate summon request, pick-up bay dispatch, passenger boarding, and departure."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.curbside
    mgr = CurbsideManager(facility_id=facility.map_id, config=cfg)

    # 1. Request pick-up staging
    mission, bay = mgr.request_pickup(vehicle_id, current_time_s=10.0)

    # 2. Update pedestrian observation
    ped = PedestrianDetection(pedestrian_id="passenger-01", distance_m=ped_dist)
    mgr.interlock_engine.update_pedestrians(vehicle_id, [ped], current_time_s=10.0)

    # 3. Complete boarding if no pedestrian within safety radius
    completed_mission = None
    if ped_dist >= cfg.pedestrian_safety_radius_m:
        completed_mission = mgr.complete_pickup(vehicle_id, current_time_s=10.0 + dwell_s)

    report = mgr.generate_report(current_time_s=10.0 + dwell_s)

    if png is not None:
        render_curbside_dashboard(facility, report, output_path=png)

    payload = {
        "mission": (completed_mission or mission).model_dump(mode="json"),
        "assigned_bay": bay.model_dump(mode="json") if bay else None,
        "pedestrian_distance_m": ped_dist,
        "overstay_penalty_usd": (
            completed_mission.overstay_penalty_usd if completed_mission else 0.0
        ),
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Curbside Pick-up Summon Execution:")
    typer.echo(f"  Mission ID:         {mission.mission_id}")
    typer.echo(f"  Vehicle ID:         {vehicle_id}")
    typer.echo(f"  Pick-up Bay:        {bay.bay_id if bay else 'QUEUED'}")
    typer.echo(f"  Pedestrian Proximity: {ped_dist:.1f} m")
    if completed_mission:
        typer.echo(
            "  Summon Status:      COMPLETED "
            f"(Overstay: ${completed_mission.overstay_penalty_usd:.2f})"
        )
    else:
        typer.echo("  Summon Status:      INHIBITED (Pedestrian in Critical Zone)")


@curbside_app.command("audit")
def curbside_audit(
    context: typer.Context,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Perform a comprehensive facility curbside handover and pedestrian safety audit."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.curbside
    mgr = CurbsideManager(facility_id=facility.map_id, config=cfg)

    # Simulate multi-vehicle staging events
    _m1, _ = mgr.request_dropoff("veh-drop-01", current_time_s=0.0)
    mgr.complete_dropoff("veh-drop-01", current_time_s=80.0)

    _m2, _ = mgr.request_pickup("veh-pick-01", current_time_s=50.0)
    # Exceed dwell limit slightly to simulate overstay fee
    mgr.complete_pickup("veh-pick-01", current_time_s=50.0 + cfg.max_pickup_dwell_s + 60.0)

    report = mgr.generate_report(current_time_s=300.0)

    if png is not None:
        render_curbside_dashboard(facility, report, output_path=png)

    payload = report.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Curbside Handover Facility Audit Report:")
    typer.echo(f"  Facility:           {report.facility_id}")
    typer.echo(f"  Total Staging Bays: {report.total_bays_count}")
    typer.echo(f"  Vacant Bays:        {report.vacant_bays_count}")
    typer.echo(f"  Completed Missions: {report.completed_missions_count}")
    typer.echo(f"  Average Dwell:      {report.average_dwell_duration_s:.1f} s")
    typer.echo(f"  Total Overstay:     ${report.total_overstay_penalties_usd:.2f}")
