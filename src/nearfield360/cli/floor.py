"""CLI commands for multi-floor facility topology, vertical ramps, and vehicle elevators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.multilevel.arbiter import RampTokenArbiter
from nearfield360.multilevel.elevator import VehicleElevatorDispatcher
from nearfield360.multilevel.graph import build_default_facility_graph
from nearfield360.multilevel.models import (
    MultiFloorFacilityReport,
    RampDirection,
)
from nearfield360.multilevel.ramp import RampDynamicsEngine
from nearfield360.multilevel.viz import render_multilevel_dashboard
from nearfield360.utils.artifacts import write_json

floor_app = typer.Typer(
    help="Multi-floor garage topology, spiral ramp kinematics, and elevator dispatch.",
    no_args_is_help=True,
)

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
        help="Path to render multi-floor BEV dashboard PNG artifact.",
    ),
]


@floor_app.command("info")
def floor_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit configuration as JSON.")] = False,
) -> None:
    """Display configured multi-floor vertical transit parameters and thresholds."""
    config = get_state(context).config.floor
    data = config.model_dump(mode="json")

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo("[Multi-Floor Vertical Transit Configuration]")
    typer.echo(f"  Default Floor Height:          {config.default_floor_height_m} m")
    typer.echo(f"  Max Ramp Grade:               {config.max_ramp_grade_pct} %")
    typer.echo(f"  Min Breakover Clearance:       {config.min_breakover_clearance_m} m")
    typer.echo(f"  Ramp Speed Limit:              {config.ramp_speed_limit_mps} m/s")
    typer.echo(f"  Elevator Hoist Speed:          {config.elevator_hoist_speed_mps} m/s")
    typer.echo(f"  Elevator Door Dwell:           {config.elevator_door_dwell_s} s")
    typer.echo(f"  Leveling Tolerance:            {config.elevator_leveling_tolerance_m} m")
    typer.echo(f"  Ramp Token Timeout:            {config.ramp_token_timeout_s} s")
    typer.echo(f"  Max Supported Floors:          {config.max_floors_supported}")


@floor_app.command("route")
def floor_route(
    origin: Annotated[str, typer.Option("--origin", help="Origin node ID.")] = "L1-entry",
    destination: Annotated[
        str, typer.Option("--destination", help="Destination node ID.")
    ] = "L2-bay-42",
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", help="Navigating vehicle ID.")
    ] = "veh-avp-01",
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Plan an optimal 3D path across floor aisles and vertical portals."""
    graph = build_default_facility_graph()

    try:
        route = graph.find_route(
            origin_node=origin, destination_node=destination, vehicle_id=vehicle_id
        )
    except (KeyError, ValueError) as exc:
        typer.echo(f"Routing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = route.model_dump(mode="json")

    if output is not None:
        write_json(output, payload)

    if png is not None:
        render_multilevel_dashboard(graph, route=route, output_path=png)

    portals_str = ", ".join(route.portals_sequence) or "None (Intra-floor)"
    typer.echo(f"[Multi-Floor Route: {route.route_id}]")
    typer.echo(f"  Origin Floor:         {route.origin_floor_id}")
    typer.echo(f"  Destination Floor:    {route.destination_floor_id}")
    typer.echo(f"  Portals Traversed:    {portals_str}")
    typer.echo(f"  Waypoints Count:      {len(route.waypoints)}")
    typer.echo(f"  Total Distance:       {route.total_distance_m} m")
    typer.echo(f"  Estimated Transit:    {route.estimated_transit_time_s} s")


@floor_app.command("ramp")
def floor_ramp(
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "veh-ramp-01",
    grade_pct: Annotated[float, typer.Option("--grade", help="Ramp grade percentage.")] = 12.0,
    elevation_m: Annotated[float, typer.Option("--elevation", help="Current elevation.")] = 1.75,
    spiral_radius: Annotated[
        float | None, typer.Option("--spiral-radius", help="Spiral ramp radius if curved.")
    ] = None,
    velocity_mps: Annotated[
        float, typer.Option("--velocity", help="Speed traversing ramp.")
    ] = 2.5,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate vehicle pitch dynamics, gravity forces, and breakover clearances on a ramp."""
    engine = RampDynamicsEngine()
    pitch_state = engine.calculate_pitch_dynamics(
        vehicle_id=vehicle_id, grade_pct=grade_pct, elevation_m=elevation_m
    )

    spiral_metrics = None
    if spiral_radius is not None and spiral_radius > 0:
        spiral_metrics = engine.calculate_spiral_kinematics(
            radius_m=spiral_radius, velocity_mps=velocity_mps, grade_pct=grade_pct
        )

    payload = {
        "pitch_state": pitch_state.model_dump(mode="json"),
        "spiral_kinematics": spiral_metrics,
    }

    if output is not None:
        write_json(output, payload)

    if png is not None:
        graph = build_default_facility_graph()
        render_multilevel_dashboard(graph, pitch_state=pitch_state, output_path=png)

    status_str = "SAFE" if pitch_state.is_clearance_safe else "ALERT DANGER"
    typer.echo(f"[Ramp Dynamics Simulation: {vehicle_id}]")
    typer.echo(f"  Incline Grade:        {pitch_state.grade_pct}%")
    typer.echo(f"  Pitch Angle:          {pitch_state.pitch_angle_deg} deg")
    typer.echo(f"  Gravity Resistance:   {pitch_state.gravity_force_n} N")
    typer.echo(f"  Traction Margin:      {pitch_state.traction_margin}")
    typer.echo(f"  Breakover Clearance:  {pitch_state.breakover_clearance_m * 100:.1f} cm")
    typer.echo(f"  Chassis Grounding:    {status_str}")


@floor_app.command("elevator")
def floor_elevator(
    elevator_id: Annotated[str, typer.Option("--elevator-id", help="Elevator ID.")] = "lift-01",
    origin_floor: Annotated[str, typer.Option("--origin-floor", help="Pickup floor ID.")] = "L1",
    destination_floor: Annotated[
        str, typer.Option("--destination-floor", help="Dropoff floor ID.")
    ] = "L2",
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "veh-lift-01",
    vehicle_weight: Annotated[
        float, typer.Option("--weight", help="Vehicle mass in kg.")
    ] = 1800.0,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate vehicle elevator dispatch, hoisting lifecycle, and leveling interlocks."""
    graph = build_default_facility_graph()
    dispatcher = VehicleElevatorDispatcher()

    for floor in graph.floors.values():
        dispatcher.register_floor(floor)

    dispatcher.register_elevator(elevator_id=elevator_id, initial_floor_id=origin_floor)
    call_id = dispatcher.request_elevator(
        vehicle_id=vehicle_id,
        origin_floor_id=origin_floor,
        destination_floor_id=destination_floor,
        vehicle_weight_kg=vehicle_weight,
    )

    # Step through pickup doors open and boarding
    events = []
    events.extend(dispatcher.step(dt_s=5.2))
    drive_in_allowed = dispatcher.is_drive_in_permitted(elevator_id)
    events.extend(dispatcher.step(dt_s=5.2))

    cabin_after_load = dispatcher.get_cabin_state(elevator_id)

    payload = {
        "call_id": call_id,
        "elevator_id": elevator_id,
        "vehicle_id": vehicle_id,
        "drive_in_permitted": drive_in_allowed,
        "cabin_state": cabin_after_load.model_dump(mode="json"),
        "events": events,
    }

    if output is not None:
        write_json(output, payload)

    if png is not None:
        render_multilevel_dashboard(graph, active_cabin=cabin_after_load, output_path=png)

    typer.echo(f"[Elevator Dispatch Cycle: {elevator_id}]")
    typer.echo(f"  Call ID:              {call_id}")
    typer.echo(f"  Payload Vehicle:      {vehicle_id} ({vehicle_weight} kg)")
    typer.echo(f"  Drive-in Permitted:   {'YES (Level Aligned)' if drive_in_allowed else 'NO'}")
    typer.echo(f"  Cabin Current State:  {cabin_after_load.state.value.upper()}")
    typer.echo(f"  Leveling Offset:      {cabin_after_load.leveling_offset_m * 1000:.1f} mm")


@floor_app.command("audit")
def floor_audit(
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Generate structural audit and operational vertical transit telemetry report."""
    graph = build_default_facility_graph()
    arbiter = RampTokenArbiter()
    dispatcher = VehicleElevatorDispatcher()

    for floor in graph.floors.values():
        dispatcher.register_floor(floor)
    dispatcher.register_elevator("lift-01", initial_floor_id="L1")

    # Grant a sample token
    token = arbiter.request_token(
        portal_id="ramp-L1-to-L2",
        vehicle_id="veh-avp-01",
        direction=RampDirection.ASCENDING,
        timestamp_s=100.0,
    )

    report = MultiFloorFacilityReport(
        facility_id="facility-central-01",
        facility_name="Central Multi-Story Garage",
        timestamp_s=1700000000.0,
        floors_count=len(graph.floors),
        total_bays=sum(f.bays_count for f in graph.floors.values()),
        available_bays=sum(f.available_bays for f in graph.floors.values()),
        active_vehicles_count=3,
        ramp_token_events_count=arbiter.total_grants_count,
        elevator_cycles_count=dispatcher.total_cycles_completed,
        average_vertical_transit_time_s=16.8,
        floors=list(graph.floors.values()),
        portals=list(graph.portals.values()),
        elevators=[dispatcher.get_cabin_state("lift-01")],
    )

    payload = report.model_dump(mode="json")

    if output is not None:
        write_json(output, payload)

    if png is not None:
        render_multilevel_dashboard(graph, report=report, active_token=token, output_path=png)

    typer.echo(f"[Multi-Floor Facility Audit: {report.facility_name}]")
    typer.echo(f"  Total Floors:         {report.floors_count}")
    typer.echo(f"  Total Capacity:       {report.total_bays} Bays")
    typer.echo(f"  Available Capacity:   {report.available_bays} Bays")
    typer.echo(f"  Portals Count:        {len(report.portals)}")
    typer.echo(f"  Avg Vertical Transit: {report.average_vertical_transit_time_s} s")
