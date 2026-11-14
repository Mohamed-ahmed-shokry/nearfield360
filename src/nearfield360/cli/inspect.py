"""CLI commands for automated vehicle condition inspection and liability audit."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from nearfield360.cli.state import get_state
from nearfield360.inspection.certificate import LiabilityCertificateManager
from nearfield360.inspection.glass import GlassIntegrityAnalyzer
from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    InspectionPhase,
    InspectionReport,
    LiabilityVerdict,
    OpticalAnomaly,
    TirePosition,
    TireWearStatus,
    VehicleConditionSnapshot,
)
from nearfield360.inspection.optical import OpticalDamageDetector
from nearfield360.inspection.tires import TireProfiler
from nearfield360.inspection.viz import render_inspection_dashboard
from nearfield360.utils.artifacts import write_json

inspect_app = typer.Typer(
    help="Automated vehicle condition inspection, 360 damage assessment, and liability audit.",
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


@inspect_app.command("info")
def inspect_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display vehicle condition inspection thresholds, tire wear standards, and audit config."""
    state = get_state(context)
    cfg = state.config.inspection

    payload: dict[str, Any] = {
        "standard": "ISO 23374 / ECE R43 / ASTM E177",
        "scratch_severity_threshold": cfg.scratch_severity_threshold,
        "dent_depth_threshold_mm": cfg.dent_depth_threshold_mm,
        "min_legal_tread_depth_mm": cfg.min_legal_tread_depth_mm,
        "recommended_tread_depth_mm": cfg.recommended_tread_depth_mm,
        "max_tread_wear_variance_mm": cfg.max_tread_wear_variance_mm,
        "nominal_cold_pressure_bar": cfg.nominal_cold_pressure_bar,
        "pressure_deviation_tolerance_bar": cfg.pressure_deviation_tolerance_bar,
        "glass_crack_length_threshold_mm": cfg.glass_crack_length_threshold_mm,
        "enable_cryptographic_certificate": cfg.enable_cryptographic_certificate,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title="Vehicle Condition Inspection Configuration (ISO 23374 / ECE R43)")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Standard", payload["standard"])
    table.add_row("Scratch Sensitivity Threshold", str(cfg.scratch_severity_threshold))
    table.add_row("Dent Depth Threshold", f"{cfg.dent_depth_threshold_mm} mm")
    table.add_row("Minimum Legal Tread Depth", f"{cfg.min_legal_tread_depth_mm} mm")
    table.add_row("Recommended Tread Depth", f"{cfg.recommended_tread_depth_mm} mm")
    table.add_row("Max Tread Variance", f"{cfg.max_tread_wear_variance_mm} mm")
    table.add_row("Nominal Cold TPMS Pressure", f"{cfg.nominal_cold_pressure_bar} bar")
    table.add_row("Pressure Tolerance Band", f"+/- {cfg.pressure_deviation_tolerance_bar} bar")
    table.add_row("Glass Crack Safety Limit", f"{cfg.glass_crack_length_threshold_mm} mm")
    table.add_row("Cryptographic Certificates", str(cfg.enable_cryptographic_certificate))
    console.print(table)


@inspect_app.command("scan")
def inspect_scan(
    context: typer.Context,
    vin: Annotated[
        str, typer.Option("--vin", help="Vehicle Identification Number.")
    ] = "WP0ZZZ99ZTS392001",
    phase: Annotated[
        str, typer.Option("--phase", help="Lifecycle phase (PRE_DROPOFF, POST_PICKUP).")
    ] = "PRE_DROPOFF",
    with_defects: Annotated[
        bool, typer.Option("--with-defects", help="Simulate exterior surface defects.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute 360-degree optical surround damage scan across 4 camera channels."""
    state = get_state(context)
    detector = OpticalDamageDetector(config=state.config.inspection)
    p_enum = InspectionPhase(phase.upper())

    sim_defects = []
    if with_defects:
        sim_defects = [
            {
                "anomaly_id": "ANO-01",
                "camera_id": "MVL",
                "surface_location": "driver_door",
                "damage_type": "SCRATCH",
                "estimated_area_cm2": 14.5,
                "depth_estimate_mm": 0.8,
            },
            {
                "anomaly_id": "ANO-02",
                "camera_id": "FV",
                "surface_location": "front_hood",
                "damage_type": "PAINT_CHIP",
                "estimated_area_cm2": 2.2,
                "depth_estimate_mm": 0.2,
            },
        ]

    anomalies = detector.scan_surround(sim_defects)
    score = detector.compute_surface_score(anomalies)

    snapshot = VehicleConditionSnapshot(
        vin=vin,
        inspection_phase=p_enum,
        mileage_km=15420.0,
        anomalies=anomalies,
        overall_condition_score=score,
    )

    payload = snapshot.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if score >= 90.0 else ("yellow" if score >= 75.0 else "red")
    console.print(f"[{color}]360 Optical Scan: Condition Score {score:.1f}%[/{color}]")
    console.print(f"VIN: {vin} | Phase: {phase} | Anomalies: {len(anomalies)}")
    for a in anomalies:
        console.print(
            f"  - [{color}]{a.severity.value}[/{color}] "
            f"{a.damage_type.value} on {a.surface_location} ({a.estimated_area_cm2} cm2)"
        )


@inspect_app.command("tire")
def inspect_tire(
    context: typer.Context,
    fl_depth: Annotated[float, typer.Option("--fl", help="Front-left tread depth mm.")] = 5.5,
    fr_depth: Annotated[float, typer.Option("--fr", help="Front-right tread depth mm.")] = 5.5,
    rl_depth: Annotated[float, typer.Option("--rl", help="Rear-left tread depth mm.")] = 4.8,
    rr_depth: Annotated[float, typer.Option("--rr", help="Rear-right tread depth mm.")] = 4.8,
    pressure: Annotated[float, typer.Option("--pressure", help="Cold tire pressure bar.")] = 2.40,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate 4-wheel multi-groove tread depth, shoulder asymmetry, and cold TPMS pressure."""
    state = get_state(context)
    profiler = TireProfiler(config=state.config.inspection)

    wheel_data = {
        TirePosition.FRONT_LEFT: {
            "inner": fl_depth,
            "center": fl_depth,
            "outer": fl_depth,
            "pressure": pressure,
        },
        TirePosition.FRONT_RIGHT: {
            "inner": fr_depth,
            "center": fr_depth,
            "outer": fr_depth,
            "pressure": pressure,
        },
        TirePosition.REAR_LEFT: {
            "inner": rl_depth,
            "center": rl_depth,
            "outer": rl_depth,
            "pressure": pressure,
        },
        TirePosition.REAR_RIGHT: {
            "inner": rr_depth,
            "center": rr_depth,
            "outer": rr_depth,
            "pressure": pressure,
        },
    }

    tires = profiler.evaluate_four_wheels(wheel_data)
    variance = profiler.compute_axle_variance(tires)

    payload = {
        "tires": [t.model_dump(mode="json") for t in tires],
        "variance": variance,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title="4-Wheel Tire Profiling & TPMS Telemetry")
    table.add_column("Position", style="cyan")
    table.add_column("Tread (mm)", style="green")
    table.add_column("Pressure (bar)", style="white")
    table.add_column("Status", style="yellow")

    for t in tires:
        col = (
            "green"
            if t.status == TireWearStatus.GOOD
            else ("yellow" if t.status == TireWearStatus.ACCEPTABLE else "red")
        )
        table.add_row(
            t.position.value,
            f"{t.mean_tread_depth_mm:.1f}",
            f"{t.pressure_bar:.2f}",
            f"[{col}]{t.status.value}[/{col}]",
        )

    console.print(table)


@inspect_app.command("glass")
def inspect_glass(
    context: typer.Context,
    transmittance: Annotated[
        float, typer.Option("--trans", help="Optical transmittance %.")
    ] = 85.0,
    chips: Annotated[int, typer.Option("--chips", help="Stone chip count.")] = 1,
    with_crack: Annotated[
        bool, typer.Option("--crack", help="Simulate structural glass crack.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate windshield and window glass optical clarity and ECE R43 compliance."""
    state = get_state(context)
    analyzer = GlassIntegrityAnalyzer(config=state.config.inspection)

    cracks = []
    if with_crack:
        cracks = [
            {
                "anomaly_id": "CRACK-01",
                "length_mm": 24.0,
                "depth_mm": 2.2,
                "location": "windshield_driver_zone_a",
            }
        ]

    rep = analyzer.evaluate_glass(
        optical_transmittance_pct=transmittance,
        raw_chips=chips,
        raw_cracks=cracks,
    )
    compliant, violations = analyzer.check_regulatory_compliance(rep)

    payload = {
        "report": rep.model_dump(mode="json"),
        "ece_r43_compliant": compliant,
        "violations": violations,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    color = "green" if compliant else "red"
    res = "COMPLIANT" if compliant else "NON-COMPLIANT"
    console.print(f"[{color}]ECE R43 Glass Integrity: {res}[/{color}]")
    console.print(f"Transmittance: {transmittance:.1f}% | Chips: {chips} | Cracks: {len(cracks)}")
    for v in violations:
        console.print(f"  - [red]{v}[/red]")


@inspect_app.command("certificate")
def inspect_certificate(
    context: typer.Context,
    vin: Annotated[str, typer.Option("--vin", help="Vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    session_id: Annotated[
        str, typer.Option("--session", help="AVP session identifier.")
    ] = "SESS-AVP-2026-99",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Generate cryptographically signed liability certificate for vehicle condition state."""
    state = get_state(context)
    manager = LiabilityCertificateManager(config=state.config.inspection)

    snapshot = VehicleConditionSnapshot(
        vin=vin,
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=14200.0,
        overall_condition_score=98.5,
    )

    cert = manager.issue_certificate(
        snapshot=snapshot,
        session_id=session_id,
        verdict=LiabilityVerdict.NO_NEW_DAMAGE,
    )

    payload = cert.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[green]Liability Certificate Issued: {cert.certificate_id}[/green]")
    console.print(f"VIN: {cert.vin} | Verdict: {cert.verdict.value}")
    console.print(f"SHA-256 Digest: {cert.payload_sha256[:32]}...")
    console.print(f"HMAC Signature: {cert.signature_hmac[:32]}...")


@inspect_app.command("compare")
def inspect_compare(
    context: typer.Context,
    with_new_damage: Annotated[
        bool, typer.Option("--damage", help="Simulate new damage on pickup.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Compare pre-dropoff and post-pickup snapshots to determine liability verdict."""
    state = get_state(context)
    manager = LiabilityCertificateManager(config=state.config.inspection)

    pre = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=14200.0,
        overall_condition_score=98.5,
    )

    post_anomalies = []
    post_score = 98.5
    if with_new_damage:
        post_anomalies = [
            OpticalAnomaly(
                anomaly_id="NEW-SCR-01",
                camera_id="MVL",
                surface_location="driver_door",
                damage_type=DamageType.SCRATCH,
                severity=DamageSeverity.MODERATE,
                bounding_box_norm=(0.3, 0.4, 0.5, 0.6),
                estimated_area_cm2=12.0,
                depth_estimate_mm=0.8,
                confidence=0.96,
                description="New door scratch occurred during parking",
            )
        ]
        post_score = 91.5

    post = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.POST_PICKUP,
        mileage_km=14203.0,
        anomalies=post_anomalies,
        overall_condition_score=post_score,
    )

    comp = manager.compare_snapshots(pre, post)
    if output is not None:
        write_json(output, comp, overwrite=True)

    if as_json:
        console.print_json(data=comp)
        return

    verdict_col = "green" if comp["verdict"] == LiabilityVerdict.NO_NEW_DAMAGE else "red"
    console.print(f"[{verdict_col}]Liability Verdict: {comp['verdict'].value}[/{verdict_col}]")
    console.print(
        f"New Defects: {comp['new_damage_count']} | Condition Delta: {comp['score_delta']:+.1f}%"
    )


@inspect_app.command("render")
def inspect_render(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", resolve_path=True, help="Path for rendered PNG dashboard."),
    ] = Path("reports/inspection_dashboard.png"),
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON summary.")] = False,
) -> None:
    """Generate tactical 1280x720 HUD dashboard visualizing vehicle condition and certificate."""
    state = get_state(context)
    cfg = state.config.inspection

    optical = OpticalDamageDetector(config=cfg)
    anomalies = optical.scan_surround(
        [
            {
                "anomaly_id": "A1",
                "camera_id": "MVL",
                "surface_location": "driver_door",
                "damage_type": "SCRATCH",
                "estimated_area_cm2": 8.0,
                "depth_estimate_mm": 0.5,
            },
            {
                "anomaly_id": "A2",
                "camera_id": "FV",
                "surface_location": "front_hood",
                "damage_type": "PAINT_CHIP",
                "estimated_area_cm2": 2.5,
                "depth_estimate_mm": 0.2,
            },
        ]
    )
    score = optical.compute_surface_score(anomalies)

    tires_profiler = TireProfiler(config=cfg)
    tires = tires_profiler.evaluate_four_wheels()

    glass_analyzer = GlassIntegrityAnalyzer(config=cfg)
    glass = glass_analyzer.evaluate_glass(optical_transmittance_pct=85.0, raw_chips=1)

    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=14200.0,
        anomalies=anomalies,
        tires=tires,
        glass=glass,
        overall_condition_score=score,
    )

    cert_mgr = LiabilityCertificateManager(config=cfg)
    cert = cert_mgr.issue_certificate(snapshot, session_id="SESS-DEMO-01")

    report = InspectionReport(
        vehicle_id="EV-101",
        facility_id="FAC-AIRPORT-AVP",
        bay_id="INSPECT-BAY-01",
        phase=InspectionPhase.PRE_DROPOFF,
        snapshot=snapshot,
        certificate=cert,
    )

    canvas = render_inspection_dashboard(report=report, output_path=output)

    payload = {
        "rendered_image_path": str(output),
        "canvas_shape": list(canvas.shape),
        "vin": snapshot.vin,
        "score": score,
        "anomalies_count": len(anomalies),
    }

    if as_json:
        console.print_json(data=payload)
    else:
        console.print(
            f"[green]Tactical Inspection Dashboard rendered successfully to {output}[/green]"
        )
        console.print(f"Canvas: {canvas.shape[1]}x{canvas.shape[0]} | Condition Score: {score}%")


@inspect_app.command("audit")
def inspect_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit vehicle condition inspection subsystem readiness and compliance against ECE R43."""
    state = get_state(context)
    cfg = state.config.inspection

    checks = [
        ("Tread Depth Legal Threshold Enforced (<= 1.6 mm)", cfg.min_legal_tread_depth_mm <= 1.61),
        (
            "Recommended Tread Depth Margin Enforced (>= 3.0 mm)",
            cfg.recommended_tread_depth_mm >= 2.99,
        ),
        (
            "Cold TPMS Pressure Baseline Validated (2.0 - 3.0 bar)",
            2.0 <= cfg.nominal_cold_pressure_bar <= 3.0,
        ),
        (
            "Scratch Severity Sensitivity Defined (0.1 - 0.5)",
            0.1 <= cfg.scratch_severity_threshold <= 0.5,
        ),
        (
            "Glass Crack Length Safety Limit (<= 20.0 mm)",
            cfg.glass_crack_length_threshold_mm <= 20.1,
        ),
        ("Cryptographic Certificate HMAC Salt Configured", len(cfg.certificate_signing_salt) >= 8),
        (
            "Cryptographic Liability Certificates Enabled",
            cfg.enable_cryptographic_certificate is True,
        ),
    ]

    all_passed = all(status for _, status in checks)

    payload = {
        "audit_target": "Automated Vehicle Condition Inspection Subsystem (ISO 23374 / ECE R43)",
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
    console.print(f"Vehicle Condition Inspection Audit: {status_str}")
    for name, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  [{icon}] {name}")
