"""CLI commands for autonomous EV charging dispatch, power grid balancing, and scheduling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.charging.grid import GridPowerManager
from nearfield360.charging.hardware import ChargingHardwareSimulator
from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargerType,
    ChargingDispatchReport,
    EVBatteryState,
    FacilityPowerSummary,
)
from nearfield360.charging.scheduler import ChargingRequest, ChargingScheduler
from nearfield360.charging.viz import render_charging_dashboard
from nearfield360.cli.state import get_state
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

charging_app = typer.Typer(
    help="Autonomous EV charging dispatch, power grid management, and bay scheduling.",
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
        help="Path to render EV charging BEV PNG artifact.",
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


def _build_default_bays(facility: FacilityMap) -> list[ChargerBay]:
    """Generate default charging bays matching initial facility slots."""
    bays: list[ChargerBay] = []
    # Pick first 2 slots as charging stations
    for idx, slot in enumerate(facility.slots[:2]):
        bay_id = f"BAY_{idx + 1:02d}"
        power = 150.0 if idx == 0 else 50.0
        chg_type = ChargerType.DC_FAST_150KW if idx == 0 else ChargerType.DC_FAST_50KW
        bays.append(
            ChargerBay(
                bay_id=bay_id,
                slot_id=slot.slot_id,
                charger_type=chg_type,
                max_power_rating_kw=power,
                status=ChargerPortStatus.AVAILABLE,
            )
        )
    return bays


@charging_app.command("info")
def charging_info(
    context: typer.Context,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect facility charging stations, ports, and power grid capacity."""
    state = get_state(context)
    facility = _load_facility(map_path)
    bays = _build_default_bays(facility)
    grid_mgr = GridPowerManager(config=state.config.charging, facility_id=facility.map_id)

    info_dict = {
        "facility_id": facility.map_id,
        "grid_capacity_kw": grid_mgr.grid_capacity_kw,
        "peak_shaving_threshold_kw": grid_mgr.peak_shaving_threshold_kw,
        "total_charging_bays": len(bays),
        "bays": [b.model_dump() for b in bays],
    }

    if output is not None:
        try:
            write_json(output, info_dict, overwrite=True)
            typer.echo(f"Wrote charging info to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(info_dict, indent=2))
    else:
        typer.echo(f"=== Facility {facility.name} ({facility.map_id}) Charging Info ===")
        typer.echo(f"Grid Capacity:        {grid_mgr.grid_capacity_kw:.1f} kW")
        typer.echo(f"Total Charging Bays:  {len(bays)}")
        for b in bays:
            msg = (
                f"  - {b.bay_id} (Slot {b.slot_id}): {b.charger_type.value} "
                f"({b.max_power_rating_kw:.0f} kW) -> {b.status.value}"
            )
            typer.echo(msg)


@charging_app.command("grid")
def charging_grid(
    context: typer.Context,
    map_path: MapPathOption = None,
    active_power: Annotated[
        float,
        typer.Option("--active-power", "-p", help="Simulate current active charging load kW."),
    ] = 120.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect real-time power grid balance, available headroom, and peak shaving limits."""
    state = get_state(context)
    facility = _load_facility(map_path)
    grid_mgr = GridPowerManager(config=state.config.charging, facility_id=facility.map_id)

    headroom = max(0.0, grid_mgr.grid_capacity_kw - active_power)
    util = min(1.0, active_power / max(1.0, grid_mgr.grid_capacity_kw))
    is_curtailed = active_power > grid_mgr.grid_capacity_kw

    summary = FacilityPowerSummary(
        facility_id=facility.map_id,
        grid_capacity_kw=grid_mgr.grid_capacity_kw,
        total_requested_power_kw=active_power,
        total_allocated_power_kw=min(active_power, grid_mgr.grid_capacity_kw),
        available_headroom_kw=headroom,
        active_charging_sessions=1 if active_power > 0 else 0,
        throttled_sessions=1 if is_curtailed else 0,
        grid_utilization_ratio=util,
        is_curtailed=is_curtailed,
    )

    data = summary.model_dump()
    if output is not None:
        try:
            write_json(output, data, overwrite=True)
            typer.echo(f"Wrote grid status to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"=== Grid Status ({facility.map_id}) ===")
        typer.echo(f"Grid Capacity:    {summary.grid_capacity_kw:.1f} kW")
        typer.echo(f"Allocated Power:  {summary.total_allocated_power_kw:.1f} kW")
        typer.echo(f"Headroom:         {summary.available_headroom_kw:.1f} kW")
        typer.echo(f"Utilization:      {int(summary.grid_utilization_ratio * 100)}%")
        typer.echo(f"Curtailment:      {'ACTIVE' if summary.is_curtailed else 'NORMAL'}")


@charging_app.command("schedule")
def charging_schedule(
    context: typer.Context,
    map_path: MapPathOption = None,
    vehicles: Annotated[
        int, typer.Option("--vehicles", "-v", help="Number of EV requests to queue.")
    ] = 3,
    output: OutputOption = None,
    png: PngOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
) -> None:
    """Schedule incoming EV requests to available charger bays with queue prioritization."""
    state = get_state(context)
    facility = _load_facility(map_path)
    bays = _build_default_bays(facility)

    scheduler = ChargingScheduler(config=state.config.charging, facility_map=facility)
    scheduler.register_charger_bays(bays)

    # Queue vehicles with varying current SoC
    for i in range(vehicles):
        veh_id = f"EV_{i + 1:02d}"
        # Make first vehicle critical
        soc = 0.12 if i == 0 else 0.40 + (i * 0.10)
        scheduler.enqueue_request(
            ChargingRequest(
                vehicle_id=veh_id,
                arrival_time_s=float(i * 10),
                current_soc=min(0.9, soc),
                target_soc=state.config.charging.default_target_soc,
            )
        )

    # Schedule assignments
    sessions = scheduler.schedule_assignments(current_time_s=100.0)
    grid_mgr = GridPowerManager(config=state.config.charging, facility_id=facility.map_id)

    batteries = [
        EVBatteryState(
            vehicle_id=s.vehicle_id,
            current_soc=s.current_soc,
            target_soc=s.target_soc,
        )
        for s in sessions
    ]

    power_summary = grid_mgr.balance_power(sessions, bays, batteries)

    report = ChargingDispatchReport(
        facility_id=facility.map_id,
        timestamp_s=100.0,
        power_summary=power_summary,
        total_bays=len(bays),
        available_bays=sum(1 for b in bays if b.status == ChargerPortStatus.AVAILABLE),
        occupied_charging_bays=sum(
            1 for b in bays if b.status == ChargerPortStatus.OCCUPIED_CHARGING
        ),
        waiting_relocation_bays=0,
        active_sessions=sessions,
        pending_queue_count=len(scheduler.queue),
    )

    data = report.model_dump()
    if output is not None:
        try:
            write_json(output, data, overwrite=True)
            typer.echo(f"Wrote schedule report to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if png is not None:
        render_charging_dashboard(
            facility=facility,
            bays=bays,
            sessions=sessions,
            report=report,
            output_path=png,
        )
        typer.echo(f"Rendered charging dashboard to {png}")

    if as_json:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"Scheduled {len(sessions)} vehicle charging sessions.")
        for s in sessions:
            typer.echo(
                f"  Vehicle {s.vehicle_id} -> Bay {s.bay_id} "
                f"(Allocated: {s.allocated_power_kw:.1f} kW)"
            )


@charging_app.command("simulate")
def charging_simulate(
    context: typer.Context,
    map_path: MapPathOption = None,
    vehicles: Annotated[int, typer.Option("--vehicles", "-v", help="Number of vehicles.")] = 3,
    steps: Annotated[int, typer.Option("--steps", "-s", help="Simulation time steps.")] = 20,
    dt: Annotated[float, typer.Option("--dt", help="Step interval in seconds.")] = 60.0,
    output: OutputOption = None,
    png: PngOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON payload.")] = False,
) -> None:
    """Run discrete-time multi-vehicle EV charging simulation with dynamic power balancing."""
    state = get_state(context)
    facility = _load_facility(map_path)
    bays = _build_default_bays(facility)

    scheduler = ChargingScheduler(config=state.config.charging, facility_map=facility)
    scheduler.register_charger_bays(bays)
    grid_mgr = GridPowerManager(config=state.config.charging, facility_id=facility.map_id)
    hardware_sim = ChargingHardwareSimulator()

    # Enqueue vehicles
    battery_dict: dict[str, EVBatteryState] = {}
    for i in range(vehicles):
        veh_id = f"EV_{i + 1:02d}"
        init_soc = 0.15 if i == 0 else 0.50 + (i * 0.10)
        target_soc = state.config.charging.default_target_soc
        battery_dict[veh_id] = EVBatteryState(
            vehicle_id=veh_id,
            current_soc=init_soc,
            target_soc=target_soc,
        )
        scheduler.enqueue_request(
            ChargingRequest(
                vehicle_id=veh_id,
                arrival_time_s=0.0,
                current_soc=init_soc,
                target_soc=target_soc,
            )
        )

    # Initial assignments
    sim_time_s = 0.0
    scheduler.schedule_assignments(current_time_s=sim_time_s)

    total_energy_delivered = 0.0
    # Run simulation steps
    for _ in range(steps):
        sim_time_s += dt
        active_sess = scheduler.active_sessions
        bats = [battery_dict[s.vehicle_id] for s in active_sess if s.vehicle_id in battery_dict]

        # Balance grid power
        grid_mgr.balance_power(active_sess, bays, bats)

        # Step hardware physics
        step_energy = hardware_sim.step_facility(active_sess, bats, bays, dt_s=dt)
        total_energy_delivered += step_energy

        # Check completions and relocations
        scheduler.check_completions_and_relocations(current_time_s=sim_time_s)

        # Assign newly freed bays if any
        scheduler.schedule_assignments(current_time_s=sim_time_s)

    final_sess = scheduler.active_sessions
    final_bats = [battery_dict[s.vehicle_id] for s in final_sess if s.vehicle_id in battery_dict]
    final_power = grid_mgr.balance_power(final_sess, bays, final_bats)

    report = ChargingDispatchReport(
        facility_id=facility.map_id,
        timestamp_s=sim_time_s,
        power_summary=final_power,
        total_bays=len(bays),
        available_bays=sum(1 for b in bays if b.status == ChargerPortStatus.AVAILABLE),
        occupied_charging_bays=sum(
            1 for b in bays if b.status == ChargerPortStatus.OCCUPIED_CHARGING
        ),
        waiting_relocation_bays=sum(
            1 for b in bays if b.status == ChargerPortStatus.OCCUPIED_COMPLETE_WAITING_RELOCATION
        ),
        active_sessions=final_sess,
        pending_queue_count=len(scheduler.queue),
        relocation_orders=scheduler.relocation_orders,
    )

    data = report.model_dump()
    data["simulation_metadata"] = {
        "steps": steps,
        "dt_s": dt,
        "sim_duration_s": sim_time_s,
        "total_energy_delivered_kwh": round(total_energy_delivered, 2),
    }

    if output is not None:
        try:
            write_json(output, data, overwrite=True)
            typer.echo(f"Wrote simulation report to {output}")
        except ArtifactError as exc:
            typer.echo(f"Error writing artifact to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if png is not None:
        render_charging_dashboard(
            facility=facility,
            bays=bays,
            sessions=final_sess,
            report=report,
            output_path=png,
        )
        typer.echo(f"Rendered simulation dashboard to {png}")

    if as_json:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(
            f"Simulation complete ({steps} steps, {sim_time_s:.0f}s elapsed). "
            f"Delivered: {total_energy_delivered:.2f} kWh"
        )
