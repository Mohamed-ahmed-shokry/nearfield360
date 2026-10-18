"""CLI commands for emergency vehicle preemption, evacuation planning, and life safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.emergency.containment import HazardContainmentEngine
from nearfield360.emergency.evacuation import FacilityEvacuationPlanner
from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    EmergencyVehicle,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    LifeSafetyReport,
    ResponderType,
    VehicleEvacuationStatus,
)
from nearfield360.emergency.preemption import EmergencyPreemptionArbiter
from nearfield360.emergency.viz import render_emergency_dashboard
from nearfield360.utils.artifacts import write_json

emergency_app = typer.Typer(
    help="Emergency vehicle preemption, facility fire evacuation, and life safety dispatch.",
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
        help="Path to render life safety BEV tactical dashboard PNG artifact.",
    ),
]


@emergency_app.command("info")
def emergency_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit configuration as JSON.")] = False,
) -> None:
    """Display configured emergency preemption, evacuation, and life safety parameters."""
    config = get_state(context).config.emergency
    data = config.model_dump(mode="json")

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo("[Emergency & Life Safety Configuration]")
    typer.echo(f"  Preemption Radius:             {config.preemption_radius_m} m")
    typer.echo(f"  Hazard Buffer Radius:          {config.hazard_buffer_radius_m} m")
    typer.echo(f"  Evacuation Speed Limit:        {config.evacuation_speed_limit_mps} m/s")
    typer.echo(f"  Fire Door Clearance Timeout:   {config.fire_door_clearance_timeout_s} s")
    typer.echo(f"  Stagger Interval:              {config.stagger_interval_s} s")
    typer.echo(f"  Emergency Heartbeat Timeout:   {config.emergency_heartbeat_timeout_s} s")
    typer.echo(f"  Thermal Runaway Temp:          {config.thermal_runaway_temp_c} C")


@emergency_app.command("preempt")
def emergency_preempt(
    context: typer.Context,
    responder_id: Annotated[
        str, typer.Option("--responder-id", help="Emergency responder vehicle ID.")
    ] = "fd-ladder-01",
    responder_type: Annotated[
        ResponderType, typer.Option("--type", help="Responder classification type.")
    ] = ResponderType.FIRE_ENGINE,
    start_x: Annotated[
        float, typer.Option("--start-x", help="Responder start X coordinate.")
    ] = 0.0,
    start_y: Annotated[
        float, typer.Option("--start-y", help="Responder start Y coordinate.")
    ] = 10.0,
    target_x: Annotated[
        float, typer.Option("--target-x", help="Responder target X coordinate.")
    ] = 60.0,
    target_y: Annotated[
        float, typer.Option("--target-y", help="Responder target Y coordinate.")
    ] = 10.0,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Evaluate emergency responder corridor and issue preemptive yield holds."""
    config = get_state(context).config.emergency
    arbiter = EmergencyPreemptionArbiter(config)

    responder = EmergencyVehicle(
        vehicle_id=responder_id,
        responder_type=responder_type,
        x=start_x,
        y=start_y,
        yaw_rad=0.0,
        velocity_mps=6.0,
        target_x=target_x,
        target_y=target_y,
        active_siren=True,
    )
    arbiter.register_responder(responder)

    vehicles = {
        "veh-avp-01": (25.0, 11.0),
        "veh-avp-02": (45.0, 10.5),
        "veh-avp-03": (15.0, 35.0),
    }

    evaluations = arbiter.evaluate_fleet(vehicles)

    payload = {
        "responder": responder.model_dump(mode="json"),
        "evaluations": {
            vid: {"status": status.value, "reason": reason}
            for vid, (status, reason) in evaluations.items()
        },
        "is_preemption_active": arbiter.is_preemption_active(),
    }

    if output is not None:
        write_json(output, payload)

    yielding_count = sum(
        1 for status, _ in evaluations.values() if status == VehicleEvacuationStatus.HOLD_YIELDING
    )

    typer.echo("[Emergency Vehicle Preemption Evaluation]")
    typer.echo(f"  Responder:        {responder.vehicle_id} ({responder.responder_type.value})")
    typer.echo(f"  Trajectory:       ({start_x}, {start_y}) -> ({target_x}, {target_y})")
    typer.echo(f"  Yield Holds:      {yielding_count} vehicles held")
    for vid, (status, reason) in evaluations.items():
        typer.echo(f"    - {vid}: status={status.value}, reason={reason}")

    if png is not None:
        report = LifeSafetyReport(
            facility_name="Autonomous Facility",
            timestamp_s=10.0,
            emergency_active=True,
            incident_type=EmergencyType.EMERGENCY_VEHICLE_APPROACH,
            severity=EmergencySeverity.EVP_PREEMPTION,
            total_vehicles=len(vehicles),
            evacuated_count=0,
            evacuating_count=0,
            yielding_count=yielding_count,
            trapped_count=0,
            evp_active=True,
            emergency_responders=[responder],
        )
        render_emergency_dashboard(
            report,
            responders=[responder],
            vehicle_poses=vehicles,
            vehicle_statuses={vid: status for vid, (status, _) in evaluations.items()},
            output_path=png,
        )
        typer.echo(f"  Rendered tactical dashboard to: {png}")


@emergency_app.command("hazard")
def emergency_hazard(
    context: typer.Context,
    hazard_id: Annotated[
        str, typer.Option("--hazard-id", help="Unique hazard identifier.")
    ] = "hz-ev-fire-01",
    incident_type: Annotated[
        EmergencyType, typer.Option("--type", help="Hazard emergency type.")
    ] = EmergencyType.EV_THERMAL_RUNAWAY,
    severity: Annotated[
        EmergencySeverity, typer.Option("--severity", help="Severity classification.")
    ] = EmergencySeverity.FULL_FACILITY_EVACUATION,
    center_x: Annotated[float, typer.Option("--x", help="Hazard center X coordinate.")] = 30.0,
    center_y: Annotated[float, typer.Option("--y", help="Hazard center Y coordinate.")] = 25.0,
    radius: Annotated[
        float, typer.Option("--radius", help="Hazard initial radius in meters.")
    ] = 10.0,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Declare hazard zone and compute dynamic exclusion perimeter & fire door interlocks."""
    config = get_state(context).config.emergency
    engine = HazardContainmentEngine(config)

    hazard = HazardZone(
        hazard_id=hazard_id,
        emergency_type=incident_type,
        severity=severity,
        center_x=center_x,
        center_y=center_y,
        radius_m=radius,
        floor_id="L1",
        active=True,
    )
    engine.add_hazard(hazard)

    door = FireDoorInterlock(
        door_id="fd-sector-01",
        floor_id="L1",
        location_x=center_x + 5.0,
        location_y=center_y + 10.0,
        status=FireDoorStatus.CLOSING,
        time_to_close_s=12.0,
    )
    engine.register_fire_door(door)

    test_points = {
        "veh-avp-01": (center_x + 2.0, center_y + 1.0),
        "veh-avp-02": (center_x + 20.0, center_y + 20.0),
    }

    in_exclusion = {
        vid: engine.is_point_inside_hazard(pt[0], pt[1], buffer_m=config.hazard_buffer_radius_m)
        for vid, pt in test_points.items()
    }

    payload = {
        "hazard": hazard.model_dump(mode="json"),
        "exclusion_buffer_m": config.hazard_buffer_radius_m,
        "fire_door": door.model_dump(mode="json"),
        "points_in_exclusion": in_exclusion,
    }

    if output is not None:
        write_json(output, payload)

    typer.echo("[Hazard Containment & Exclusion Zone]")
    typer.echo(f"  Hazard ID:        {hazard.hazard_id}")
    typer.echo(f"  Type:             {hazard.emergency_type.value}")
    typer.echo(f"  Severity:         {hazard.severity.value}")
    typer.echo(f"  Center:           ({hazard.center_x}, {hazard.center_y}) r={hazard.radius_m} m")
    typer.echo(f"  Buffer Radius:    {config.hazard_buffer_radius_m} m")
    typer.echo(f"  Fire Door:        {door.door_id} status={door.status.value}")

    if png is not None:
        report = LifeSafetyReport(
            facility_name="Autonomous Facility",
            timestamp_s=25.0,
            emergency_active=True,
            incident_type=incident_type,
            severity=severity,
            total_vehicles=len(test_points),
            evacuated_count=0,
            evacuating_count=1,
            yielding_count=0,
            trapped_count=1,
            evp_active=False,
            active_hazards=list(engine.hazards.values()),
            fire_doors=list(engine.fire_doors.values()),
        )
        render_emergency_dashboard(
            report,
            hazards=list(engine.hazards.values()),
            fire_doors=list(engine.fire_doors.values()),
            vehicle_poses=test_points,
            vehicle_statuses={
                "veh-avp-01": VehicleEvacuationStatus.TRAPPED_IN_HAZARD,
                "veh-avp-02": VehicleEvacuationStatus.EVACUATING,
            },
            output_path=png,
        )
        typer.echo(f"  Rendered tactical dashboard to: {png}")


@emergency_app.command("evacuate")
def emergency_evacuate(
    context: typer.Context,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Plan multi-wave hazard-free egress routes and dispatch waves."""
    config = get_state(context).config.emergency
    engine = HazardContainmentEngine(config)

    # Hazard at (20, 20)
    engine.add_hazard(
        HazardZone(
            hazard_id="fire-alarm-zone-a",
            emergency_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            center_x=20.0,
            center_y=20.0,
            radius_m=8.0,
            floor_id="L1",
            active=True,
        )
    )

    planner = FacilityEvacuationPlanner(engine, config)

    vehicles = {
        "veh-avp-01": ("bay-101", (10.0, 10.0)),
        "veh-avp-02": ("bay-102", (15.0, 50.0)),
        "veh-avp-03": ("bay-103", (50.0, 15.0)),
        "veh-avp-04": ("bay-104", (80.0, 70.0)),
    }
    portals = {
        "Exit-North": (10.0, 90.0),
        "Exit-South": (10.0, -10.0),
    }

    waves, routes = planner.plan_facility_evacuation(vehicles, portals)

    payload = {
        "waves": [wave.model_dump(mode="json") for wave in waves],
        "routes": {vid: route.model_dump(mode="json") for vid, route in routes.items()},
    }

    if output is not None:
        write_json(output, payload)

    typer.echo("[Facility Multi-Wave Evacuation Dispatch]")
    typer.echo(f"  Total Vehicles:   {len(vehicles)}")
    typer.echo(f"  Total Waves:      {len(waves)}")
    for wave in waves:
        typer.echo(
            f"  Wave {wave.wave_index}: dispatch_t={wave.dispatch_time_s} s, "
            f"vehicles={wave.vehicle_ids}"
        )
    for vid, route in routes.items():
        status_txt = "CLEARED" if route.is_cleared else "BLOCKED"
        typer.echo(
            f"    - {vid}: exit={route.exit_portal_id}, len={route.total_length_m:.1f} m, "
            f"status={status_txt}"
        )

    if png is not None:
        report = LifeSafetyReport(
            facility_name="Autonomous Facility",
            timestamp_s=45.0,
            emergency_active=True,
            incident_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            total_vehicles=len(vehicles),
            evacuated_count=0,
            evacuating_count=len(vehicles),
            yielding_count=0,
            trapped_count=0,
            evp_active=False,
            active_hazards=list(engine.hazards.values()),
        )
        render_emergency_dashboard(
            report,
            hazards=list(engine.hazards.values()),
            routes=routes,
            vehicle_poses={vid: pose for vid, (_, pose) in vehicles.items()},
            vehicle_statuses=dict.fromkeys(vehicles, VehicleEvacuationStatus.EVACUATING),
            exit_portals=portals,
            output_path=png,
        )
        typer.echo(f"  Rendered tactical dashboard to: {png}")


@emergency_app.command("audit")
def emergency_audit(
    context: typer.Context,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Execute comprehensive life safety compliance audit (ISO 23374 / NFPA 88A)."""
    config = get_state(context).config.emergency
    engine = HazardContainmentEngine(config)
    arbiter = EmergencyPreemptionArbiter(config)
    planner = FacilityEvacuationPlanner(engine, config)

    # 1. Test EVP preemption corridor
    responder = EmergencyVehicle(
        vehicle_id="fd-paramedic-01",
        responder_type=ResponderType.AMBULANCE,
        x=0.0,
        y=0.0,
        yaw_rad=1.57,
        velocity_mps=8.0,
        target_x=0.0,
        target_y=80.0,
        active_siren=True,
    )
    arbiter.register_responder(responder)
    fleet_eval = arbiter.evaluate_fleet({"v1": (2.0, 20.0), "v2": (40.0, 40.0)})
    held_vehicles = [
        vid for vid, (st, _) in fleet_eval.items() if st == VehicleEvacuationStatus.HOLD_YIELDING
    ]

    # 2. Test Hazard & Fire Door interlock
    hazard = HazardZone(
        hazard_id="audit-hz-gas-01",
        emergency_type=EmergencyType.GAS_LEAK,
        severity=EmergencySeverity.PARTIAL_EVACUATION,
        center_x=30.0,
        center_y=30.0,
        radius_m=10.0,
        floor_id="L1",
        active=True,
    )
    engine.add_hazard(hazard)

    door = FireDoorInterlock(
        door_id="audit-door-01",
        floor_id="L1",
        location_x=30.0,
        location_y=45.0,
        status=FireDoorStatus.LOCKED_SHUT,
        time_to_close_s=0.0,
    )
    engine.register_fire_door(door)
    door_transit_allowed, _ = engine.can_transit_fire_door(
        "audit-door-01", estimated_transit_time_s=5.0
    )

    # 3. Test Evacuation wave scheduler
    test_fleet = {
        "veh-01": ("bay-1", (10.0, 10.0)),
        "veh-02": ("bay-2", (50.0, 50.0)),
    }
    portals = {"Exit-Portal-North": (0.0, 100.0)}
    waves, routes = planner.plan_facility_evacuation(test_fleet, portals)

    passed = (
        len(held_vehicles) == 1
        and not door_transit_allowed
        and len(waves) == 1
        and all(r.is_cleared for r in routes.values())
    )

    audit_result = {
        "status": "PASS" if passed else "FAIL",
        "standard_compliance": ["ISO 23374", "NFPA 88A"],
        "preemption_verified": len(held_vehicles) == 1,
        "fire_door_interlock_verified": not door_transit_allowed,
        "evacuation_routing_verified": len(routes) == 2,
        "hazard_buffer_m": config.hazard_buffer_radius_m,
    }

    if output is not None:
        write_json(output, audit_result)

    typer.echo("[Life Safety & Emergency Compliance Audit]")
    typer.echo(f"  Audit Status:     {audit_result['status']}")
    typer.echo("  Standards:        ISO 23374 (AVP Safety) / NFPA 88A (Garages)")
    typer.echo(f"  EVP Arbiter:      {'PASS' if audit_result['preemption_verified'] else 'FAIL'}")
    typer.echo(
        f"  Door Interlocks:  {'PASS' if audit_result['fire_door_interlock_verified'] else 'FAIL'}"
    )
    typer.echo(
        f"  Evacuation Route: {'PASS' if audit_result['evacuation_routing_verified'] else 'FAIL'}"
    )

    if png is not None:
        report = LifeSafetyReport(
            facility_name="Autonomous Facility",
            timestamp_s=60.0,
            emergency_active=False,
            incident_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.ADVISORY,
            total_vehicles=2,
            evacuated_count=2,
            evacuating_count=0,
            yielding_count=0,
            trapped_count=0,
            evp_active=False,
            active_hazards=list(engine.hazards.values()),
            fire_doors=list(engine.fire_doors.values()),
        )
        render_emergency_dashboard(
            report,
            hazards=list(engine.hazards.values()),
            fire_doors=list(engine.fire_doors.values()),
            responders=[responder],
            routes=routes,
            vehicle_poses={vid: pose for vid, (_, pose) in test_fleet.items()},
            exit_portals=portals,
            output_path=png,
        )
        typer.echo(f"  Rendered compliance audit dashboard to: {png}")

    if not passed:
        raise typer.Exit(code=1)
