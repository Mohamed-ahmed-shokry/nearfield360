"""CLI commands for SAE J2954 Wireless Power Transfer (WPT) and inductive docking."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from nearfield360.cli.state import get_state
from nearfield360.utils.artifacts import write_json
from nearfield360.wpt.beacon import MagneticBeaconEstimator
from nearfield360.wpt.controller import WPTInchingController
from nearfield360.wpt.efficiency import WPTEfficiencyModel
from nearfield360.wpt.fod import FODDetector
from nearfield360.wpt.models import (
    CoilOffset3D,
    WPTAlignmentStatus,
    WPTInterlockState,
    WPTPowerClass,
    WPTReport,
    WPTZClass,
)
from nearfield360.wpt.viz import render_wpt_dashboard

wpt_app = typer.Typer(
    help="SAE J2954 Wireless Power Transfer (WPT), magnetic resonance alignment, and docking.",
    no_args_is_help=True,
)

console = Console()

OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="Path to write JSON or PNG artifact atomically.",
    ),
]


@wpt_app.command("info")
def wpt_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display SAE J2954 specifications, alignment envelope, and WPT configuration."""
    state = get_state(context)
    cfg = state.config.wpt

    payload: dict[str, Any] = {
        "standard": "SAE J2954 / ISO 19363",
        "nominal_frequency_khz": cfg.nominal_frequency_khz,
        "max_power_kw": cfg.max_power_kw,
        "target_air_gap_mm": cfg.target_air_gap_mm,
        "alignment_envelope": {
            "max_longitudinal_offset_mm": cfg.max_longitudinal_offset_mm,
            "max_lateral_offset_mm": cfg.max_lateral_offset_mm,
            "max_heading_error_deg": cfg.max_heading_error_deg,
            "docked_radius_threshold_mm": cfg.docked_radius_threshold_mm,
        },
        "safety_thresholds": {
            "fod_q_drop_threshold_percent": cfg.fod_q_drop_threshold_percent,
            "fod_freq_shift_threshold_khz": cfg.fod_freq_shift_threshold_khz,
            "enable_emergency_shutoff": cfg.enable_emergency_shutoff,
        },
        "inching_speed_mps": cfg.inching_speed_mps,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title="SAE J2954 Wireless Power Transfer Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Standard", payload["standard"])
    table.add_row("Nominal Resonant Frequency", f"{cfg.nominal_frequency_khz} kHz")
    table.add_row("Max Power Rating", f"{cfg.max_power_kw} kW")
    table.add_row("Target Air Gap (Z2)", f"{cfg.target_air_gap_mm} mm")
    table.add_row(
        "Max Alignment Box",
        f"+/-{cfg.max_longitudinal_offset_mm}mm X, +/-{cfg.max_lateral_offset_mm}mm Y",
    )
    table.add_row("Max Heading Error", f"+/-{cfg.max_heading_error_deg} deg")
    table.add_row("Docked Radius Threshold", f"<= {cfg.docked_radius_threshold_mm} mm")
    table.add_row("FOD Q-Drop Trip Limit", f">= {cfg.fod_q_drop_threshold_percent} %")
    table.add_row("Inching Creep Speed", f"{cfg.inching_speed_mps} m/s")
    console.print(table)


@wpt_app.command("ping")
def wpt_ping(
    context: typer.Context,
    frequency_khz: Annotated[
        float, typer.Option("--freq", help="Probing frequency in kHz.")
    ] = 85.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Probe primary (GA) to secondary (VA) low-power magnetic resonance link."""
    state = get_state(context)
    cfg = state.config.wpt

    freq_dev = abs(frequency_khz - cfg.nominal_frequency_khz)
    link_ok = 81.38 <= frequency_khz <= 90.0 and freq_dev <= 1.5

    payload = {
        "probing_frequency_khz": frequency_khz,
        "nominal_frequency_khz": cfg.nominal_frequency_khz,
        "frequency_deviation_khz": round(freq_dev, 2),
        "beacon_link_active": link_ok,
        "baseline_q_factor": 280.0 if link_ok else 140.0,
        "link_margin_db": 18.5 if link_ok else -4.2,
        "status": "LINK_ESTABLISHED" if link_ok else "RESONANCE_OUT_OF_BAND",
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if link_ok else "red"
    console.print(f"[{color}]WPT Beacon Link: {payload['status']}[/{color}]")
    console.print(f"Frequency: {frequency_khz} kHz | Q-Factor: {payload['baseline_q_factor']}")


@wpt_app.command("align")
def wpt_align(
    context: typer.Context,
    dx: Annotated[float, typer.Option("--dx", help="Longitudinal offset in mm.")] = 0.0,
    dy: Annotated[float, typer.Option("--dy", help="Lateral offset in mm.")] = 0.0,
    dz: Annotated[float, typer.Option("--dz", help="Vertical air gap in mm.")] = 170.0,
    dtheta: Annotated[float, typer.Option("--dtheta", help="Heading error in deg.")] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Calculate 3D coil offset, radial misalignment, and SAE J2954 tolerance class."""
    state = get_state(context)
    estimator = MagneticBeaconEstimator(config=state.config.wpt)
    offset = estimator.compute_status(
        delta_x_mm=dx,
        delta_y_mm=dy,
        delta_z_mm=dz,
        delta_theta_deg=dtheta,
    )

    payload = offset.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if offset.is_within_tolerance else "yellow"
    console.print(f"[{color}]Coil Status: {offset.alignment_status.value}[/{color}]")
    console.print(
        f"Offset: dX={offset.delta_x_mm}mm, dY={offset.delta_y_mm}mm, dZ={offset.delta_z_mm}mm"
    )
    console.print(f"Radial: {offset.radial_offset_mm}mm | Heading: {offset.delta_theta_deg} deg")


@wpt_app.command("fod")
def wpt_fod(
    context: typer.Context,
    frequency_khz: Annotated[
        float, typer.Option("--freq", help="Measured frequency in kHz.")
    ] = 85.0,
    q_factor: Annotated[float, typer.Option("--q", help="Measured circuit Q-factor.")] = 280.0,
    living: Annotated[bool, typer.Option("--living", help="Living object trip flag.")] = False,
    emergency: Annotated[bool, typer.Option("--emergency", help="Emergency stop flag.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate Foreign Object Detection (FOD) and Living Object Detection (LOD) state."""
    state = get_state(context)
    detector = FODDetector(config=state.config.wpt)
    fod_status = detector.evaluate_resonance(
        measured_frequency_khz=frequency_khz,
        measured_q_factor=q_factor,
        living_object_present=living,
        manual_emergency_trip=emergency,
    )

    payload = fod_status.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if fod_status.interlock_state == WPTInterlockState.NORMAL else "red"
    console.print(f"[{color}]Safety Interlock: {fod_status.interlock_state.value}[/{color}]")
    console.print(
        f"Power Inhibited: {fod_status.power_inhibited} | Diagnostic: {fod_status.reason}"
    )


@wpt_app.command("dock")
def wpt_dock(
    context: typer.Context,
    init_x: Annotated[float, typer.Option("--init-x", help="Initial x in meters.")] = 0.20,
    init_y: Annotated[float, typer.Option("--init-y", help="Initial y in meters.")] = 0.04,
    init_theta: Annotated[float, typer.Option("--init-theta", help="Initial heading deg.")] = 1.5,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Max trajectory iterations.")] = 150,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute closed-loop creeping docking simulation to align vehicle coil with ground pad."""
    state = get_state(context)
    controller = WPTInchingController(config=state.config.wpt)
    steps = controller.simulate_docking(
        initial_x_m=init_x,
        initial_y_m=init_y,
        initial_heading_deg=init_theta,
        max_steps=max_steps,
    )

    docked = steps[-1].docked if steps else False
    payload = {
        "docked": docked,
        "iterations": len(steps),
        "final_step": steps[-1].model_dump(mode="json") if steps else None,
        "steps": [s.model_dump(mode="json") for s in steps],
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if docked else "red"
    console.print(f"[{color}]Docking Maneuver: {'SUCCESS' if docked else 'INCOMPLETE'}[/{color}]")
    console.print(f"Total Iterations: {len(steps)}")
    if steps:
        final_offset = steps[-1].coil_offset
        console.print(
            f"Final Radial Offset: {final_offset.radial_offset_mm} mm | "
            f"Heading: {final_offset.delta_theta_deg} deg"
        )


@wpt_app.command("efficiency")
def wpt_efficiency(
    context: typer.Context,
    dr: Annotated[float, typer.Option("--dr", help="Radial offset in mm.")] = 15.0,
    dz: Annotated[float, typer.Option("--dz", help="Air gap in mm.")] = 170.0,
    power_class: Annotated[
        str, typer.Option("--class", help="Power class (WPT1, WPT2, WPT3, WPT4).")
    ] = "WPT3",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate magnetic coupling k, DC-to-DC transfer efficiency, and power governor."""
    state = get_state(context)
    model = WPTEfficiencyModel(config=state.config.wpt)
    p_enum = WPTPowerClass(power_class.upper())

    offset = CoilOffset3D(
        delta_x_mm=dr * 0.707,
        delta_y_mm=dr * 0.707,
        delta_z_mm=dz,
        radial_offset_mm=dr,
        delta_theta_deg=0.5,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.ALIGNED,
    )
    report = model.evaluate(offset=offset, power_class=p_enum)

    payload = report.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[cyan]Coupling Factor (k): {report.coupling_factor_k}[/cyan]")
    console.print(f"[green]Transfer Efficiency: {report.transfer_efficiency * 100.0:.1f} %[/green]")
    console.print(
        f"Commanded: {report.commanded_power_kw} kW | Delivered: {report.delivered_power_kw} kW"
    )


@wpt_app.command("render")
def wpt_render(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", resolve_path=True, help="Path for rendered PNG dashboard."),
    ] = Path("reports/wpt_dashboard.png"),
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON summary.")] = False,
) -> None:
    """Generate tactical 1280x720 HUD dashboard visualizing coil alignment, FOD, and efficiency."""
    state = get_state(context)
    cfg = state.config.wpt

    controller = WPTInchingController(config=cfg)
    steps = controller.simulate_docking(0.20, 0.04, 1.5, max_steps=120)
    last_offset = (
        steps[-1].coil_offset
        if steps
        else CoilOffset3D(
            delta_x_mm=10.0,
            delta_y_mm=8.0,
            delta_z_mm=170.0,
            radial_offset_mm=12.8,
            delta_theta_deg=0.4,
            is_within_tolerance=True,
            alignment_status=WPTAlignmentStatus.DOCKED,
        )
    )

    fod_detector = FODDetector(config=cfg)
    fod = fod_detector.evaluate_resonance(measured_frequency_khz=85.2, measured_q_factor=275.0)

    eff_model = WPTEfficiencyModel(config=cfg)
    eff = eff_model.evaluate(offset=last_offset, fod_status=fod, power_class=WPTPowerClass.WPT3)

    report = WPTReport(
        vehicle_id="EV-DEMO-01",
        pad_id="PAD-BAY-03",
        power_class=WPTPowerClass.WPT3,
        z_class=WPTZClass.Z2,
        offset=last_offset,
        fod=fod,
        efficiency=eff,
        docking_history=steps,
    )

    canvas = render_wpt_dashboard(report=report, output_path=output)

    payload = {
        "rendered_image_path": str(output),
        "canvas_shape": list(canvas.shape),
        "vehicle_id": report.vehicle_id,
        "pad_id": report.pad_id,
        "status": last_offset.alignment_status.value,
        "efficiency_pct": round(eff.transfer_efficiency * 100.0, 1),
    }

    if as_json:
        console.print_json(data=payload)
    else:
        console.print(f"[green]Tactical WPT Dashboard rendered successfully to {output}[/green]")
        console.print(
            f"Canvas: {canvas.shape[1]}x{canvas.shape[0]} | "
            f"Efficiency: {payload['efficiency_pct']}%"
        )


@wpt_app.command("audit")
def wpt_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit WPT subsystem configuration, calibration, and safety compliance with SAE J2954."""
    state = get_state(context)
    cfg = state.config.wpt

    checks = [
        (
            "Nominal Frequency 85 kHz band (81.38 - 90.00 kHz)",
            81.38 <= cfg.nominal_frequency_khz <= 90.0,
        ),
        ("Rated Power WPT3 (<= 11.0 kW)", cfg.max_power_kw <= 11.1),
        ("Target Air Gap in Z2 Range (140 - 210 mm)", 140.0 <= cfg.target_air_gap_mm <= 210.0),
        (
            "Alignment Tolerance Box defined (<= 100mm X, <= 75mm Y)",
            cfg.max_longitudinal_offset_mm <= 100.0 and cfg.max_lateral_offset_mm <= 75.0,
        ),
        (
            "FOD Detection Limits enforced (<= 15% Q-drop, <= 1.5 kHz shift)",
            cfg.fod_q_drop_threshold_percent <= 15.0 and cfg.fod_freq_shift_threshold_khz <= 1.5,
        ),
        ("Creeping Speed governed (<= 0.08 m/s)", cfg.inching_speed_mps <= 0.081),
        ("Emergency Shutoff Interlock enabled", cfg.enable_emergency_shutoff is True),
    ]

    all_passed = all(status for _, status in checks)

    payload = {
        "audit_target": "SAE J2954 Wireless Power Transfer Subsystem",
        "compliant": all_passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for _, status in checks if status),
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    status_str = "[green]COMPLIANT[/green]" if all_passed else "[red]NON-COMPLIANT[/red]"
    console.print(f"SAE J2954 WPT Subsystem Audit: {status_str}")
    for name, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  [{icon}] {name}")
