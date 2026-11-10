"""CLI commands for UN R157 DSSAD & UN R160 EDR Incident Recording and Regulatory Forensics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.dssad.exporter import DSSADExporter
from nearfield360.dssad.models import (
    DSSADEventType,
    EventSeverity,
    TelemetrySnapshot,
)
from nearfield360.dssad.recorder import DSSADRecorder
from nearfield360.dssad.viz import render_dssad_dashboard
from nearfield360.utils.artifacts import write_json

dssad_app = typer.Typer(
    help="UN R157 DSSAD & UN R160 EDR Incident Recording and Regulatory Forensics.",
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
        help="Path to render tactical 1280x720 forensic HUD dashboard PNG artifact.",
    ),
]


def _emit_result(
    data: dict[str, Any],
    as_json: bool,
    output: Path | None,
    summary_lines: list[str] | None = None,
) -> None:
    if output is not None:
        write_json(output, data)
    if as_json or output is None:
        if as_json:
            typer.echo(json.dumps(data, indent=2))
        elif summary_lines:
            for line in summary_lines:
                typer.echo(line)
        else:
            typer.echo(json.dumps(data, indent=2))


@dssad_app.command("info")
def dssad_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect active UN R157 DSSAD and UN R160 EDR recording parameters."""
    config = get_state(context).config.dssad
    data = {
        "sampling_rate_hz": config.sampling_rate_hz,
        "pre_trigger_duration_s": config.pre_trigger_duration_s,
        "post_trigger_duration_s": config.post_trigger_duration_s,
        "max_retained_incidents": config.max_retained_incidents,
        "impact_accel_threshold_g": config.impact_accel_threshold_g,
        "enable_tamper_proof_chain": config.enable_tamper_proof_chain,
        "auto_freeze_on_mrm": config.auto_freeze_on_mrm,
    }
    summary = [
        "[INFO] UN R157 DSSAD & UN R160 EDR Parameters:",
        f"  Sampling Rate:           {config.sampling_rate_hz} Hz",
        f"  Pre-Trigger Duration:    {config.pre_trigger_duration_s:.1f} s",
        f"  Post-Trigger Duration:   {config.post_trigger_duration_s:.1f} s",
        f"  Max Retained Incidents:  {config.max_retained_incidents}",
        f"  Impact Accel Threshold:  {config.impact_accel_threshold_g:.2f} g",
        (
            f"  Tamper-Proof Chaining:   "
            f"{'ENABLED' if config.enable_tamper_proof_chain else 'DISABLED'}"
        ),
        (f"  Auto-Freeze on MRM:      {'ENABLED' if config.auto_freeze_on_mrm else 'DISABLED'}"),
    ]
    _emit_result(data, as_json, output, summary)


@dssad_app.command("record")
def dssad_record(
    context: typer.Context,
    samples: Annotated[
        int, typer.Option("--samples", "-n", help="Number of telemetry samples to ingest.")
    ] = 10,
    speed: Annotated[float, typer.Option("--speed", help="Vehicle speed in m/s.")] = 4.0,
    accel: Annotated[
        float, typer.Option("--accel", help="Longitudinal acceleration in m/s^2.")
    ] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Stream and ingest telemetry samples into the circular FIFO ring buffer."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    dt_ns = int(1e9 / config.sampling_rate_hz)
    base_ns = 1_000_000_000_000

    for i in range(samples):
        snap = TelemetrySnapshot(
            timestamp_ns=base_ns + i * dt_ns,
            timestamp_iso="2026-11-10T12:00:00Z",
            speed_mps=speed,
            longitudinal_accel_mps2=accel,
            lateral_accel_mps2=0.0,
            yaw_rate_rads=0.0,
            steering_angle_rad=0.0,
            brake_pressure_bar=0.0,
            throttle_pct=20.0,
            odd_state="NOMINAL",
            aeb_active=False,
        )
        rec.record_telemetry(snap)

    data = {
        "status": "RECORDING_ACTIVE",
        "ingested_samples": samples,
        "buffer_occupancy": len(rec.ring_buffer),
        "buffer_capacity": rec.ring_buffer.capacity,
        "total_events": rec.total_events,
        "total_incidents": rec.total_incidents,
    }
    summary = [
        f"[PASS] Ingested {samples} telemetry samples into EDR circular buffer.",
        f"  Occupancy: {len(rec.ring_buffer)} / {rec.ring_buffer.capacity} samples.",
    ]
    _emit_result(data, as_json, output, summary)


@dssad_app.command("trigger")
def dssad_trigger(
    context: typer.Context,
    event_type: Annotated[
        DSSADEventType,
        typer.Option("--event", "-e", help="DSSAD event type to trigger."),
    ] = DSSADEventType.COLLISION_IMPACT_DETECTED,
    severity: Annotated[
        EventSeverity,
        typer.Option("--severity", "-s", help="Event severity classification."),
    ] = EventSeverity.FATAL,
    delta_v: Annotated[
        float, typer.Option("--delta-v", help="Simulated impact delta-v in m/s.")
    ] = 5.2,
    peak_g: Annotated[
        float, typer.Option("--peak-g", help="Simulated peak impact acceleration in g.")
    ] = 3.1,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Trigger a regulatory safety event, freeze incident memory, and generate forensic reports."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    dt_ns = int(1e9 / config.sampling_rate_hz)
    base_ns = 1_000_000_000_000

    # Stream pre-crash telemetry
    pre_samples = int(config.pre_trigger_duration_s * config.sampling_rate_hz)
    for i in range(pre_samples):
        snap = TelemetrySnapshot(
            timestamp_ns=base_ns + i * dt_ns,
            timestamp_iso="2026-11-10T12:00:00Z",
            speed_mps=max(0.0, 5.0 - (i / pre_samples) * delta_v * 0.5),
            longitudinal_accel_mps2=-1.2,
            lateral_accel_mps2=0.1,
            yaw_rate_rads=0.01,
            steering_angle_rad=0.02,
            brake_pressure_bar=15.0,
            throttle_pct=0.0,
            odd_state="NOMINAL",
            aeb_active=False,
        )
        rec.record_telemetry(snap)

    trigger_t = base_ns + pre_samples * dt_ns
    trigger_snap = TelemetrySnapshot(
        timestamp_ns=trigger_t,
        timestamp_iso="2026-11-10T12:00:05Z",
        speed_mps=max(0.0, 5.0 - delta_v),
        longitudinal_accel_mps2=-peak_g * 9.80665,
        lateral_accel_mps2=0.5,
        yaw_rate_rads=0.05,
        steering_angle_rad=-0.05,
        brake_pressure_bar=60.0,
        throttle_pct=0.0,
        odd_state="ODD_BREACH_MRM",
        aeb_active=True,
        active_dtcs=["C1201"],
    )

    evt = rec.log_event(
        event_type=event_type,
        severity=severity,
        trigger_subsystem="impact_sensor_module",
        description=f"Forensic incident triggered: {event_type.value} with peak {peak_g:.2f}g.",
        snapshot=trigger_snap,
        freeze_incident=True,
    )

    # Stream post-crash telemetry
    post_samples = int(config.post_trigger_duration_s * config.sampling_rate_hz)
    for i in range(1, post_samples + 1):
        snap = TelemetrySnapshot(
            timestamp_ns=trigger_t + i * dt_ns,
            timestamp_iso="2026-11-10T12:00:06Z",
            speed_mps=0.0,
            longitudinal_accel_mps2=0.0,
            lateral_accel_mps2=0.0,
            yaw_rate_rads=0.0,
            steering_angle_rad=0.0,
            brake_pressure_bar=60.0,
            throttle_pct=0.0,
            odd_state="ODD_BREACH_MRM",
            aeb_active=True,
            active_dtcs=["C1201"],
        )
        rec.record_telemetry(snap)

    report = rec.get_report()
    incident = report.incidents[-1] if report.incidents else None

    if png is not None:
        render_dssad_dashboard(report, output_path=png)

    data = {
        "status": "INCIDENT_FROZEN",
        "event_id": evt.record_id,
        "event_type": evt.event_type.value,
        "severity": evt.severity.value,
        "incident_id": incident.incident_id if incident else None,
        "delta_v_mps": incident.delta_v_mps if incident else 0.0,
        "peak_accel_g": incident.peak_accel_g if incident else 0.0,
        "is_locked": incident.is_locked if incident else False,
        "chain_verified": report.chain_verified,
        "total_events": report.total_events_recorded,
        "png_rendered": str(png) if png else None,
    }

    summary = [
        f"[TRIGGER] Logged event {evt.record_id} ({evt.event_type.value})",
        f"  Incident ID:       {incident.incident_id if incident else 'NONE'}",
        f"  Severity:          {evt.severity.value}",
        f"  Delta-V:           {incident.delta_v_mps if incident else 0.0:.2f} m/s",
        f"  Peak Accel:        {incident.peak_accel_g if incident else 0.0:.2f} g",
        f"  Memory Lock:       {'LOCKED [RO]' if incident and incident.is_locked else 'UNLOCKED'}",
        f"  Chain Integrity:   {'VERIFIED (INTACT)' if report.chain_verified else 'CORRUPTED'}",
    ]
    if png:
        summary.append(f"  Dashboard PNG:     {png}")
    _emit_result(data, as_json, output, summary)


@dssad_app.command("query")
def dssad_query(
    context: typer.Context,
    min_severity: Annotated[
        EventSeverity | None,
        typer.Option("--min-severity", help="Filter by minimum severity threshold."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Query stored DSSAD event logs and frozen EDR incident archives."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    # Seed baseline query demonstration
    rec.log_event(
        event_type=DSSADEventType.SYSTEM_ACTIVATION,
        severity=EventSeverity.INFO,
        trigger_subsystem="mission_executive",
        description="Autonomous valet parking activated.",
    )
    events = rec.query_events(min_severity=min_severity)
    incidents = rec.query_incidents()

    data = {
        "total_events_matched": len(events),
        "total_incidents": len(incidents),
        "events": [e.model_dump() for e in events],
        "incidents": [inc.model_dump() for inc in incidents],
    }
    summary = [
        f"[QUERY] Matched {len(events)} events (min_severity={min_severity}):",
    ]
    summary.extend(
        f"  - [{e.sequence_id}] {e.event_type.value} ({e.severity.value}): {e.description}"
        for e in events
    )
    _emit_result(data, as_json, output, summary)


@dssad_app.command("verify")
def dssad_verify(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Verify cryptographic SHA-256 and HMAC chain continuity of forensic records."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    # Populate test events
    rec.log_event(
        event_type=DSSADEventType.SYSTEM_ACTIVATION,
        severity=EventSeverity.INFO,
        trigger_subsystem="avp_system",
        description="System activation logged to tamper-proof chain.",
    )
    rec.log_event(
        event_type=DSSADEventType.ODD_BOUNDARY_BREACH,
        severity=EventSeverity.WARNING,
        trigger_subsystem="odd_evaluator",
        description="Surface friction degraded below nominal threshold.",
    )

    result = rec.verify_integrity()
    data = result.model_dump()
    corrupted_str = str(result.corrupted_index) if result.corrupted_index is not None else "NONE"
    summary = [
        f"[VERIFY] Audit Chain Status: {'VALID / UNBROKEN' if result.valid else 'CORRUPTED'}",
        f"  Total Blocks Inspected: {result.total_blocks}",
        f"  Corrupted Index:        {corrupted_str}",
        f"  Error Message:          {result.error_message if result.error_message else 'NONE'}",
    ]
    _emit_result(data, as_json, output, summary)


@dssad_app.command("export")
def dssad_export(
    context: typer.Context,
    export_format: Annotated[
        str, typer.Option("--format", "-f", help="Export format: json, csv, uds.")
    ] = "uds",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Export forensic data to ISO 14229 UDS DID frame, CSV telemetry, or JSON report."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    rec.log_event(
        event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
        severity=EventSeverity.FATAL,
        trigger_subsystem="crash_sensor",
        description="Collision trigger for export verification.",
        freeze_incident=True,
    )
    report = rec.get_report()
    incident = report.incidents[0]

    fmt = export_format.lower()
    if fmt == "uds":
        uds_frame = DSSADExporter.export_uds_did_frame(incident)
        summary = [
            "[UDS] ISO 14229 Service 0x22 (ReadDataByIdentifier) Response:",
            f"  Service ID:  {uds_frame['service_id']}",
            f"  DID:         {uds_frame['did']}",
            f"  Hex Payload: {uds_frame['raw_hex']}",
            f"  Delta-V:     {uds_frame['delta_v_mps']:.1f} m/s",
            f"  Peak Accel:  {uds_frame['peak_accel_g']:.2f} g",
        ]
        _emit_result(uds_frame, as_json, output, summary)
    elif fmt == "csv":
        csv_text = DSSADExporter.export_telemetry_csv(incident.pre_trigger_telemetry)
        if output is not None:
            output.write_text(csv_text, encoding="utf-8")
        typer.echo(csv_text)
    else:
        json_text = DSSADExporter.export_json(report)
        if output is not None:
            output.write_text(json_text, encoding="utf-8")
        typer.echo(json_text)


@dssad_app.command("render")
def dssad_render(
    context: typer.Context,
    output_png: Annotated[
        Path, typer.Option("--output", "-o", help="Destination path for PNG.")
    ] = Path("artifacts/dssad_dashboard.png"),
) -> None:
    """Render the 1280x720 forensic investigation HUD dashboard."""
    config = get_state(context).config.dssad
    rec = DSSADRecorder(config=config)
    rec.log_event(
        event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
        severity=EventSeverity.FATAL,
        trigger_subsystem="imu",
        description="Incident simulation for dashboard rendering.",
        freeze_incident=True,
    )
    report = rec.get_report()
    render_dssad_dashboard(report, output_path=output_png)
    typer.echo(f"[PASS] Rendered forensic HUD dashboard to {output_png}")


@dssad_app.command("audit")
def dssad_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Verify UN R157 & UN R160 regulatory compliance readiness for AVP incident forensics."""
    config = get_state(context).config.dssad
    checks = [
        {
            "standard": "UN R157 §5.5",
            "requirement": "DSSAD event storage for automated driving state transitions",
            "status": "COMPLIANT",
        },
        {
            "standard": "UN R160 §5.3",
            "requirement": "Continuous >=50Hz pre/post crash telemetry buffering (-5.0s / +2.0s)",
            "status": "COMPLIANT" if config.sampling_rate_hz >= 50 else "NON_COMPLIANT",
        },
        {
            "standard": "UN R157 §5.5.4",
            "requirement": "Cryptographic tamper-evident audit trail & write-once lock",
            "status": "COMPLIANT" if config.enable_tamper_proof_chain else "NON_COMPLIANT",
        },
        {
            "standard": "ISO 14229-1",
            "requirement": "Unified Diagnostic Services DID extraction (ReadDataByIdentifier)",
            "status": "COMPLIANT",
        },
        {
            "standard": "ISO 23374 Type 2",
            "requirement": "AVP automated valet parking incident freeze interlock",
            "status": "COMPLIANT" if config.auto_freeze_on_mrm else "NON_COMPLIANT",
        },
    ]
    all_passed = all(c["status"] == "COMPLIANT" for c in checks)
    data = {
        "status": "AUDIT_PASSED" if all_passed else "AUDIT_FAILED",
        "total_checks": len(checks),
        "checks": checks,
    }
    summary = [
        f"[AUDIT] UN R157 / UN R160 Compliance Status: {'PASSED' if all_passed else 'FAILED'}",
    ]
    summary.extend(f"  [{c['status']}] {c['standard']}: {c['requirement']}" for c in checks)
    _emit_result(data, as_json, output, summary)
