"""CLI commands for autonomous valet parking smart billing, dynamic tariffs, and receipt signing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.billing.calculator import BillingCalculator
from nearfield360.billing.models import ParkingReceipt
from nearfield360.billing.settlement import SettlementAggregator
from nearfield360.billing.viz import render_billing_dashboard
from nearfield360.cli.state import get_state
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

billing_app = typer.Typer(
    help="Autonomous Valet Parking smart billing, dynamic tariffs, and receipt signing.",
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
        help="Path to render financial telemetry BEV PNG artifact.",
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


@billing_app.command("info")
def billing_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active facility tariff schedule, pricing policies, and grace period settings."""
    cfg = get_state(context).config.billing

    info_payload = {
        "base_hourly_rate": cfg.base_hourly_rate,
        "peak_multiplier": cfg.peak_multiplier,
        "kwh_tariff": cfg.kwh_tariff,
        "idle_overstay_per_min": cfg.idle_overstay_per_min,
        "valet_service_fee": cfg.valet_service_fee,
        "grace_period_minutes": cfg.grace_period_minutes,
        "tax_rate": cfg.tax_rate,
        "receipt_hmac_configured": bool(cfg.receipt_hmac_secret),
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("Autonomous Valet Parking Billing & Tariff Schedule:")
    typer.echo(f"  Base Hourly Parking Rate:   ${cfg.base_hourly_rate:.2f}/hr")
    typer.echo(f"  Peak Congestion Multiplier: {cfg.peak_multiplier:.2f}x")
    typer.echo(f"  EV Energy Consumption Rate: ${cfg.kwh_tariff:.2f}/kWh")
    typer.echo(f"  Charging Overstay Penalty:  ${cfg.idle_overstay_per_min:.2f}/min")
    typer.echo(f"  Valet Service Fee:          ${cfg.valet_service_fee:.2f}")
    typer.echo(f"  Grace Period:               {cfg.grace_period_minutes:.1f} minutes")
    typer.echo(f"  Sales Tax Rate:             {cfg.tax_rate * 100:.1f}%")


@billing_app.command("calculate")
def billing_calculate(
    context: typer.Context,
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", "-v", help="Vehicle identifier.")
    ] = "VEH_01",
    duration_minutes: Annotated[
        float, typer.Option("--duration-minutes", "-d", help="Total parking dwell in minutes.")
    ] = 90.0,
    kwh_consumed: Annotated[
        float, typer.Option("--kwh", help="EV charging energy consumed in kWh.")
    ] = 15.0,
    overstay_minutes: Annotated[
        float, typer.Option("--overstay-minutes", help="Charging bay idle overstay in minutes.")
    ] = 0.0,
    valet_fee: Annotated[
        bool, typer.Option("--valet/--no-valet", help="Apply valet service fee.")
    ] = True,
    occupancy_ratio: Annotated[
        float, typer.Option("--occupancy-ratio", help="Facility occupancy load ratio.")
    ] = 0.80,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON receipt.")] = False,
    output: OutputOption = None,
) -> None:
    """Calculate parking, EV charging, and overstay charges for a session."""
    cfg = get_state(context).config.billing
    calc = BillingCalculator(cfg)

    entry_t = 0.0
    exit_t = duration_minutes * 60.0

    receipt = calc.calculate_session_charges(
        vehicle_id=vehicle_id,
        entry_timestamp=entry_t,
        exit_timestamp=exit_t,
        kwh_consumed=kwh_consumed,
        idle_overstay_minutes=overstay_minutes,
        include_valet_fee=valet_fee,
        occupancy_ratio=occupancy_ratio,
    )

    payload = receipt.model_dump()

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Calculated Parking Receipt: {receipt.receipt_id}")
    typer.echo(f"  Vehicle: {receipt.vehicle_id} | Tariff: {receipt.tariff_tier.value}")
    typer.echo(f"  Duration: {receipt.duration_minutes:.1f} min | Items: {len(receipt.items)}")
    for item in receipt.items:
        typer.echo(f"    - {item.description}: ${item.amount:.2f}")
    typer.echo(f"  Subtotal: ${receipt.subtotal:.2f}")
    typer.echo(f"  Tax:      ${receipt.tax:.2f}")
    typer.echo(f"  Total:    ${receipt.total_charged:.2f}")
    typer.echo(f"  HMAC-SHA256 Digest: {receipt.signature_digest[:16]}...")


@billing_app.command("receipt")
def billing_receipt(
    context: typer.Context,
    receipt_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to ParkingReceipt JSON file to verify.",
        ),
    ],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit raw JSON verification result.")
    ] = False,
) -> None:
    """Verify cryptographic HMAC-SHA256 signature of a parking receipt file."""
    cfg = get_state(context).config.billing
    calc = BillingCalculator(cfg)

    try:
        data = read_json(receipt_file)
        receipt = ParkingReceipt.model_validate(data)
    except Exception as exc:
        typer.echo(f"Error reading receipt file {receipt_file}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    is_valid = calc.verify_receipt(receipt)

    result = {
        "receipt_id": receipt.receipt_id,
        "vehicle_id": receipt.vehicle_id,
        "total_charged": receipt.total_charged,
        "signature_digest": receipt.signature_digest,
        "is_valid": is_valid,
    }

    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return

    if is_valid:
        typer.echo(
            f"[PASS] Receipt {receipt.receipt_id} signature is cryptographically valid."
        )
    else:
        typer.echo(
            f"[FAIL] Receipt {receipt.receipt_id} signature verification FAILED!",
            err=True,
        )
        raise typer.Exit(code=1)


@billing_app.command("settle")
def billing_settle(
    context: typer.Context,
    map_path: MapPathOption = None,
    receipts_count: Annotated[
        int, typer.Option("--sessions-count", "-n", help="Simulated sessions count.")
    ] = 5,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit raw JSON settlement report.")
    ] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Aggregate sessions into a facility financial settlement report and render BEV dashboard."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.billing
    calc = BillingCalculator(cfg)
    aggregator = SettlementAggregator(facility.map_id)

    receipts: list[ParkingReceipt] = []
    for i in range(receipts_count):
        dur = 30.0 + (i * 25.0)
        kwh = 10.0 + (i * 5.0) if i % 2 == 0 else 0.0
        overstay = 10.0 if i == receipts_count - 1 else 0.0
        r = calc.calculate_session_charges(
            vehicle_id=f"VEH_{i + 1:02d}",
            entry_timestamp=float(i * 3600),
            exit_timestamp=float(i * 3600 + dur * 60),
            kwh_consumed=kwh,
            idle_overstay_minutes=overstay,
            include_valet_fee=True,
            occupancy_ratio=min(0.95, 0.40 + (i * 0.12)),
            receipt_id=f"RCP-{i + 1:03d}",
        )
        receipts.append(r)

    report = aggregator.generate_settlement_report(receipts)
    payload = report.model_dump()

    if output is not None:
        write_json(output, payload, overwrite=True)

    if png is not None:
        render_billing_dashboard(facility=facility, report=report, output_path=png)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Financial Settlement Summary for Facility {report.facility_id}:")
    typer.echo(
        f"  Total Transactions:    {report.total_receipts_count} "
        f"(Settled: {report.settled_receipts_count})"
    )
    typer.echo(f"  Gross Revenue:         ${report.total_revenue:.2f}")
    typer.echo(f"  Parking Revenue:       ${report.parking_revenue:.2f}")
    typer.echo(f"  EV Energy Revenue:     ${report.energy_revenue:.2f}")
    typer.echo(f"  Penalty Fines:         ${report.penalty_revenue:.2f}")
    typer.echo(f"  Valet Service Revenue: ${report.valet_service_revenue:.2f}")
    typer.echo(f"  Tax Collected:         ${report.tax_collected:.2f}")
    typer.echo(f"  Average Dwell:         {report.average_dwell_minutes:.1f} minutes")
    if png is not None:
        typer.echo(f"  BEV Revenue Dashboard: {png}")
