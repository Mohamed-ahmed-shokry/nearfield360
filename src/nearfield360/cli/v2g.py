"""CLI commands for Vehicle-to-Grid (V2G) bi-directional power transfer and microgrid control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.utils.artifacts import write_json
from nearfield360.v2g.arbitrage import ArbitrageScheduler, get_default_tariff_schedule
from nearfield360.v2g.inverter import SmartInverter
from nearfield360.v2g.microgrid import (
    FacilityMicrogridManager,
    create_default_demonstrator_fleet,
)
from nearfield360.v2g.models import VehicleBPTProfile
from nearfield360.v2g.viz import render_v2g_dashboard

v2g_app = typer.Typer(
    help=(
        "Vehicle-to-Grid (V2G) bi-directional energy transfer, peak-shaving arbitrage, and "
        "smart inverter telemetry (ISO 15118-20 / OCPP 2.0.1)."
    ),
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
        "-p",
        resolve_path=True,
        help="Path to render tactical V2G energy dashboard PNG artifact.",
    ),
]


@v2g_app.command("info")
def v2g_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active V2G parameters, transformer limits, and Time-of-Use tariffs."""
    config = get_state(context).config.v2g
    tariffs = get_default_tariff_schedule()

    payload: dict[str, Any] = {
        "transformer_limit_kw": config.facility_transformer_limit_kw,
        "peak_shaving_target_kw": config.peak_shaving_target_kw,
        "min_departure_reserve_soc": config.min_departure_reserve_soc,
        "max_discharge_power_kw": config.max_discharge_power_kw,
        "max_charge_power_kw": config.max_charge_power_kw,
        "battery_degradation_penalty_per_kwh": config.battery_degradation_penalty_per_kwh,
        "grid_frequency_nominal_hz": config.grid_frequency_nominal_hz,
        "frequency_droop_gain_kw_per_hz": config.frequency_droop_gain_kw_per_hz,
        "arbitrage_interval_minutes": config.arbitrage_interval_minutes,
        "tariff_windows": [t.model_dump() for t in tariffs],
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    min_dep_pct = payload["min_departure_reserve_soc"] * 100.0
    deg_pen = payload["battery_degradation_penalty_per_kwh"]
    typer.echo("Vehicle-to-Grid (V2G) Configuration:")
    typer.echo(f"  Facility Transformer Limit : {payload['transformer_limit_kw']} kW")
    typer.echo(f"  Peak Shaving Target        : {payload['peak_shaving_target_kw']} kW")
    typer.echo(f"  Min Departure Reserve SoC  : {min_dep_pct:.0f}%")
    typer.echo(f"  Max Discharge Rate         : {payload['max_discharge_power_kw']} kW")
    typer.echo(f"  Max Charge Rate            : {payload['max_charge_power_kw']} kW")
    typer.echo(f"  Degradation Cost Penalty   : ${deg_pen:.2f}/kWh")
    typer.echo(f"  Grid Nominal Frequency     : {payload['grid_frequency_nominal_hz']} Hz")
    typer.echo(f"  Frequency Droop Gain       : {payload['frequency_droop_gain_kw_per_hz']} kW/Hz")
    typer.echo("\nStandard Time-of-Use Tariff Windows:")
    for w in tariffs:
        typer.echo(
            f"  [{w.window_id:18s}] {w.start_hour:02d}:00-{w.end_hour:02d}:00 | "
            f"Import: ${w.import_price_per_kwh:.2f}/kWh | Export: ${w.export_price_per_kwh:.2f}/kWh"
        )


@v2g_app.command("schedule")
def v2g_schedule(
    context: typer.Context,
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", help="Vehicle identifier.")
    ] = "EV-DEMO-01",
    battery_kwh: Annotated[
        float, typer.Option("--battery-kwh", help="Battery capacity in kWh.")
    ] = 75.0,
    current_soc: Annotated[
        float, typer.Option("--current-soc", help="Current battery state of charge (0.0-1.0).")
    ] = 0.85,
    min_departure_soc: Annotated[
        float, typer.Option("--min-departure-soc", help="Required SoC at departure.")
    ] = 0.60,
    arrival_hour: Annotated[
        float, typer.Option("--arrival-hour", help="Arrival hour of the day (0-24).")
    ] = 8.0,
    departure_hour: Annotated[
        float, typer.Option("--departure-hour", help="Departure hour (0-48).")
    ] = 17.0,
    max_discharge_kw: Annotated[
        float, typer.Option("--max-discharge-kw", help="Max discharge rate in kW.")
    ] = 22.0,
    max_charge_kw: Annotated[
        float, typer.Option("--max-charge-kw", help="Max charge rate in kW.")
    ] = 40.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Generate an optimal multi-interval V2G charging and discharging schedule for a vehicle."""
    config = get_state(context).config.v2g
    scheduler = ArbitrageScheduler(config=config)

    profile = VehicleBPTProfile(
        vehicle_id=vehicle_id,
        battery_capacity_kwh=battery_kwh,
        current_soc=current_soc,
        min_departure_soc=min_departure_soc,
        arrival_hour=arrival_hour,
        departure_hour=departure_hour,
        max_discharge_kw=max_discharge_kw,
        max_charge_kw=max_charge_kw,
    )

    sched = scheduler.schedule_vehicle(profile)
    payload = sched.model_dump()

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"V2G Arbitrage Dispatch Schedule for {sched.vehicle_id}:")
    typer.echo(f"  Slots Scheduled          : {len(sched.dispatches)}")
    typer.echo(f"  Total Energy Charged     : {sched.total_energy_charged_kwh:.2f} kWh")
    typer.echo(f"  Total Energy Discharged  : {sched.total_energy_discharged_kwh:.2f} kWh")
    typer.echo(f"  Net Electricity Cost     : ${sched.net_cost_dollars:.2f}")
    typer.echo(f"  Net Export Revenue       : ${sched.net_revenue_dollars:.2f}")
    typer.echo(f"  Net Arbitrage Profit     : ${sched.net_profit_dollars:.2f}")
    typer.echo(f"  Final Projected SoC      : {sched.final_soc * 100.0:.1f}%")
    status_str = "PASSED" if sched.min_departure_soc_met else "FAILED"
    typer.echo(f"  Departure SoC Guarantee  : {status_str}")


@v2g_app.command("arbitrage")
def v2g_arbitrage(
    context: typer.Context,
    hour: Annotated[float, typer.Option("--hour", help="Simulation reference hour.")] = 14.0,
    png: PngOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute multi-vehicle microgrid peak-shaving and tariff arbitrage simulation."""
    config = get_state(context).config.v2g
    manager = FacilityMicrogridManager(config=config)
    fleet = create_default_demonstrator_fleet()

    for idx, v in enumerate(fleet):
        manager.connect_vehicle(f"BAY-{idx + 1:02d}", v)

    report = manager.run_facility_simulation()
    payload = report.model_dump()

    if png is not None:
        render_v2g_dashboard(manager=manager, report=report, hour=hour, output_path=png)
        typer.echo(f"Exported V2G dashboard to {png}")

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Facility V2G Microgrid Arbitrage Summary:")
    typer.echo(f"  Connected Fleet Bays     : {report.total_bays_connected} bays")
    typer.echo(f"  Total Battery Capacity   : {report.total_battery_capacity_kwh:.1f} kWh")
    typer.echo(f"  Flexible Discharge Energy: {report.available_flexible_energy_kwh:.1f} kWh")
    typer.echo(f"  Baseline Facility Peak   : {report.baseline_facility_peak_kw:.1f} kW")
    typer.echo(f"  Managed Shaved Peak      : {report.peak_shaved_demand_kw:.1f} kW")
    typer.echo(f"  Transformer Utilization  : {report.transformer_utilization_percent:.1f}%")
    typer.echo(f"  Net Arbitrage Profit     : ${report.net_arbitrage_profit_dollars:.2f}")
    typer.echo(f"  Peaker CO2 Offset        : {report.carbon_offset_kg:.1f} kg")


@v2g_app.command("render")
def v2g_render(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", resolve_path=True, help="Destination PNG path."),
    ],
    hour: Annotated[float, typer.Option("--hour", help="Hour of the day (0-24).")] = 14.0,
) -> None:
    """Render high-contrast dual-panel 1280x720 V2G tactical energy dashboard to PNG."""
    config = get_state(context).config.v2g
    manager = FacilityMicrogridManager(config=config)
    fleet = create_default_demonstrator_fleet()

    for idx, v in enumerate(fleet):
        bay = manager.connect_vehicle(f"BAY-{idx + 1:02d}", v)
        if idx % 3 == 0:
            bay.inverter.convert_power(-20.0)  # Discharging
        elif idx % 3 == 1:
            bay.inverter.convert_power(35.0)  # Charging

    report = manager.run_facility_simulation()
    render_v2g_dashboard(manager=manager, report=report, hour=hour, output_path=output)
    typer.echo(f"Rendered V2G tactical energy dashboard to {output}")


@v2g_app.command("audit")
def v2g_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit inverter droop compliance, transformer limits, and departure SoC guarantees."""
    config = get_state(context).config.v2g
    manager = FacilityMicrogridManager(config=config)
    fleet = create_default_demonstrator_fleet()

    for idx, v in enumerate(fleet):
        manager.connect_vehicle(f"BAY-{idx + 1:02d}", v)

    report = manager.run_facility_simulation()

    # Inverter test
    inverter = SmartInverter("AUDIT-INV", rated_power_kw=50.0)
    adjusted_p = inverter.apply_frequency_droop(grid_frequency_hz=49.80)
    droop_ok = adjusted_p < 0.0

    # Shaving test
    shave_ok = report.peak_shaved_demand_kw <= config.peak_shaving_target_kw + 1e-3
    transformer_ok = report.peak_shaved_demand_kw < config.facility_transformer_limit_kw

    # Departure SoC test
    soc_ok = all(s.min_departure_soc_met for s in report.schedules)

    all_passed = droop_ok and shave_ok and transformer_ok and soc_ok

    payload: dict[str, Any] = {
        "status": "PASS" if all_passed else "FAIL",
        "inverter_droop_response": "PASS" if droop_ok else "FAIL",
        "peak_shaving_target_compliance": "PASS" if shave_ok else "FAIL",
        "transformer_headroom_safety": "PASS" if transformer_ok else "FAIL",
        "departure_soc_guarantees": "PASS" if soc_ok else "FAIL",
        "connected_bays": report.total_bays_connected,
        "total_battery_capacity_kwh": report.total_battery_capacity_kwh,
        "transformer_utilization_percent": report.transformer_utilization_percent,
        "net_arbitrage_profit_dollars": report.net_arbitrage_profit_dollars,
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        for check_name, res in payload.items():
            if isinstance(res, str) and res in ("PASS", "FAIL"):
                typer.echo(f"[{res}] {check_name}")
            else:
                typer.echo(f"  {check_name}: {res}")

    if not all_passed:
        raise typer.Exit(code=1)
