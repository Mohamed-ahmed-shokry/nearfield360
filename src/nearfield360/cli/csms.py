"""CLI commands for UN R155 / ISO/SAE 21434 Cybersecurity Management System (CSMS)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.csms.ids import IntrusionDetectionSystem, compute_shannon_entropy
from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    SecurityEvent,
)
from nearfield360.csms.quarantine import QuarantineFilter
from nearfield360.csms.viz import render_csms_dashboard
from nearfield360.csms.vsoc import VSOCManager
from nearfield360.utils.artifacts import write_json

csms_app = typer.Typer(
    help="UN R155 / ISO/SAE 21434 Cybersecurity Management System (CSMS) & in-vehicle IDS.",
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
        help="Path to render tactical UN R155 CSMS HUD dashboard PNG artifact.",
    ),
]


@csms_app.command("info")
def csms_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active UN R155 CSMS parameters, anomaly limits, and quarantine thresholds."""
    config = get_state(context).config.csms

    payload: dict[str, Any] = {
        "max_can_frame_rate_per_sec": config.max_can_frame_rate_per_sec,
        "replay_window_tolerance_ms": config.replay_window_tolerance_ms,
        "payload_entropy_min": config.payload_entropy_min,
        "payload_entropy_max": config.payload_entropy_max,
        "anomaly_score_threshold": config.anomaly_score_threshold,
        "quarantine_duration_s": config.quarantine_duration_s,
        "vsoc_telemetry_batch_size": config.vsoc_telemetry_batch_size,
        "enable_defensive_mitigation": config.enable_defensive_mitigation,
        "max_active_security_incidents": config.max_active_security_incidents,
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("UN R155 / ISO/SAE 21434 CSMS & Intrusion Detection System Configuration:")
    typer.echo(f"  Max CAN Frame Rate        : {payload['max_can_frame_rate_per_sec']:.1f} fps")
    typer.echo(f"  Replay Tolerance Window    : {payload['replay_window_tolerance_ms']:.1f} ms")
    typer.echo(
        f"  Shannon Entropy Range      : [{payload['payload_entropy_min']:.2f}, "
        f"{payload['payload_entropy_max']:.2f}] bits"
    )
    typer.echo(f"  Anomaly Score Threshold    : {payload['anomaly_score_threshold']:.2f}")
    typer.echo(f"  Quarantine Blacklist Time  : {payload['quarantine_duration_s']:.1f} s")
    typer.echo(f"  vSOC Telemetry Batch Size  : {payload['vsoc_telemetry_batch_size']}")
    typer.echo(f"  Defensive Mitigation Active: {payload['enable_defensive_mitigation']}")


@csms_app.command("analyze")
def csms_analyze(
    context: typer.Context,
    arbitration_id: Annotated[
        int, typer.Option("--arbitration-id", "-a", help="CAN Arbitration ID.")
    ] = 0x120,
    payload_hex: Annotated[
        str, typer.Option("--payload-hex", help="CAN payload as hex string.")
    ] = "0102030405060708",
    speed: Annotated[float, typer.Option("--speed", help="Vehicle speed in m/s.")] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Analyze a single CAN frame with In-Vehicle IDS and evaluate anomaly score."""
    config = get_state(context).config.csms
    ids = IntrusionDetectionSystem(config)

    raw_bytes = bytes.fromhex(payload_hex)
    entropy = compute_shannon_entropy(raw_bytes)
    frame_dict = {"arbitration_id": arbitration_id, "payload": raw_bytes, "timestamp_s": 10.0}
    event = ids.inspect_frame(frame_dict, current_time_s=10.0, vehicle_speed_mps=speed)

    anomaly_detected = event is not None
    payload_res: dict[str, Any] = {
        "arbitration_id": f"0x{arbitration_id:03X}",
        "payload_hex": payload_hex,
        "shannon_entropy_bits": round(entropy, 4),
        "vehicle_speed_mps": speed,
        "anomaly_detected": anomaly_detected,
    }
    if event:
        payload_res["event"] = event.model_dump(mode="json")

    if output is not None:
        write_json(output, payload_res)

    if as_json:
        typer.echo(json.dumps(payload_res, indent=2))
        return

    status_tag = "ANOMALY DETECTED" if anomaly_detected else "NOMINAL"
    typer.echo(f"CAN Frame IDS Inspection Result: [{status_tag}]")
    typer.echo(f"  Arbitration ID : 0x{arbitration_id:03X}")
    typer.echo(f"  Payload Hex    : {payload_hex}")
    typer.echo(f"  Shannon Entropy: {entropy:.4f} bits/byte")
    if event:
        typer.echo(f"  Threat Category: {event.category.value.upper()}")
        typer.echo(f"  Severity       : {event.severity.value.upper()}")
        typer.echo(f"  Anomaly Score  : {event.anomaly_score:.2f}")
        typer.echo(f"  Defense Action : {event.defense_action.value.upper()}")
        typer.echo(f"  Details        : {event.message}")


@csms_app.command("quarantine")
def csms_quarantine(
    context: typer.Context,
    add_entity: Annotated[
        str | None, typer.Option("--add", help="Entity ID (e.g. '0x120') to quarantine.")
    ] = None,
    release_entity: Annotated[
        str | None, typer.Option("--release", help="Entity ID to release from quarantine.")
    ] = None,
    reason: Annotated[
        str, typer.Option("--reason", help="Quarantine reason.")
    ] = "Manual diagnostic quarantine",
    duration: Annotated[
        float, typer.Option("--duration-s", help="Quarantine duration in seconds.")
    ] = 15.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect or manage active network quarantine blacklist entries."""
    config = get_state(context).config.csms
    qf = QuarantineFilter(config)

    if add_entity:
        qf.quarantine_entity(add_entity, reason=reason, duration_s=duration, current_time_s=100.0)

    if release_entity:
        qf.release_quarantine(release_entity)

    active = qf.active_quarantines(current_time_s=100.0)
    data = [rec.model_dump(mode="json") for rec in active]

    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo("Active CSMS Network Quarantines:")
    if not active:
        typer.echo("  No active entity quarantines.")
    for rec in active:
        typer.echo(
            f"  - Entity [{rec.entity_id}]: {rec.reason} (Expires at t={rec.expires_at_s:.1f}s)"
        )


@csms_app.command("simulate")
def csms_simulate(
    context: typer.Context,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            "-s",
            help=(
                "Attack scenario: 'dos_flood', 'replay_attack', "
                "'unauthorized_uds', 'fuzzing_injection', 'nominal'."
            ),
        ),
    ] = "dos_flood",
    frame_count: Annotated[
        int, typer.Option("--frames", "-n", help="Number of simulation frames.")
    ] = 200,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON report.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate an in-vehicle cyber attack scenario and evaluate automated containment."""
    config = get_state(context).config.csms
    qf = QuarantineFilter(config)
    ids = IntrusionDetectionSystem(config)
    vsoc = VSOCManager(config=config, ids=ids, quarantine=qf)

    t = 10.0
    for i in range(frame_count):
        frame: dict[str, Any]
        if scenario == "dos_flood":
            # 400 fps high-rate flood on 0x120
            frame = {
                "arbitration_id": 0x120,
                "payload": bytes([0x01, i % 256, 0x03, 0x04]),
                "timestamp_s": t,
            }
            t += 0.0025
        elif scenario == "replay_attack":
            # Exact duplicate frame repeated every 0.005s
            frame = {
                "arbitration_id": 0x140,
                "payload": b"\xde\xad\xbe\xef\x00\x01\x02\x03",
                "timestamp_s": t,
            }
            t += 0.005 if i % 2 == 1 else 0.050
        elif scenario == "unauthorized_uds":
            # Diagnostic SecurityAccess while vehicle speed = 5.0 m/s
            frame = {
                "arbitration_id": 0x7DF,
                "payload": bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
                "timestamp_s": t,
            }
            t += 0.020
        elif scenario == "fuzzing_injection":
            # Random high-entropy payloads
            frame = {
                "arbitration_id": 0x250,
                "payload": bytes((i * 17 + j) % 256 for j in range(8)),
                "timestamp_s": t,
            }
            t += 0.020
        else:
            # Nominal normal telemetry at 50 fps
            frame = {
                "arbitration_id": 0x130,
                "payload": bytes([0x10, i % 256, 0x20, 0x30]),
                "timestamp_s": t,
            }
            t += 0.020

        # Check quarantine filter first
        arb_id = int(frame["arbitration_id"])
        is_filtered = qf.should_filter_frame(arb_id, current_time_s=t)
        if not is_filtered:
            speed = 5.0 if scenario == "unauthorized_uds" else 0.0
            evt = ids.inspect_frame(frame, current_time_s=t, vehicle_speed_mps=speed)
            if evt:
                vsoc.record_event(evt)
                qf.handle_security_event(evt, current_time_s=t)

    report = vsoc.generate_report(
        report_id=f"VSOC-SIM-{scenario.upper()}",
        total_frames_evaluated=frame_count,
        current_time_s=t,
    )

    if png is not None:
        render_csms_dashboard(report, output_path=png)

    data = report.model_dump(mode="json")
    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo(f"UN R155 CSMS Cyber Simulation Summary: [{scenario.upper()}]")
    typer.echo(f"  Frames Evaluated     : {report.threat_assessment.total_frames_evaluated}")
    typer.echo(f"  Anomalies Detected   : {report.threat_assessment.anomalous_frames_detected}")
    typer.echo(f"  Active Quarantines   : {report.threat_assessment.active_quarantines_count}")
    typer.echo(f"  Max Severity         : {report.threat_assessment.max_severity.value.upper()}")
    typer.echo(f"  Mean Anomaly Score   : {report.threat_assessment.mean_anomaly_score:.2f}")
    typer.echo("  Recent Incidents Log :")
    for e in report.events[-5:]:
        typer.echo(f"    - [{e.severity.value.upper()}] {e.category.value}: {e.message}")


@csms_app.command("render")
def csms_render(
    context: typer.Context,
    png: Annotated[
        Path,
        typer.Option(
            "--png",
            "-p",
            resolve_path=True,
            help="Destination path for 1280x720 PNG HUD dashboard.",
        ),
    ] = Path("artifacts/csms_hud.png"),
    scenario: Annotated[str, typer.Option("--scenario", help="Simulation scenario.")] = "dos_flood",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON status.")] = False,
    output: OutputOption = None,
) -> None:
    """Render a tactical 1280x720 UN R155 CSMS / IDS HUD dashboard PNG artifact."""
    config = get_state(context).config.csms
    qf = QuarantineFilter(config)
    ids = IntrusionDetectionSystem(config)
    vsoc = VSOCManager(config=config, ids=ids, quarantine=qf)

    # Generate sample scenario
    evt = SecurityEvent(
        event_id="SEC-RENDER-001",
        timestamp_s=15.0,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.CRITICAL,
        source_node="arb_0x120",
        arbitration_id=0x120,
        anomaly_score=0.96,
        defense_action=DefenseAction.REQUEST_SAFE_STOP,
        message="Critical rate flood on powertrain control bus",
    )
    vsoc.record_event(evt)
    qf.handle_security_event(evt, current_time_s=15.0)

    report = vsoc.generate_report(
        report_id=f"VSOC-HUD-{scenario.upper()}",
        total_frames_evaluated=2500,
        current_time_s=15.0,
    )
    render_csms_dashboard(report, output_path=png)

    payload = {
        "rendered_path": str(png),
        "report_id": report.report_id,
        "max_severity": report.threat_assessment.max_severity.value,
        "status": "rendered",
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Successfully rendered UN R155 CSMS HUD dashboard to: {png}")


@csms_app.command("audit")
def csms_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON audit results.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit vehicle Cybersecurity Management System (CSMS) compliance with UN R155."""
    config = get_state(context).config.csms

    checks: list[dict[str, Any]] = [
        {
            "name": "un_r155_csms_rate_bounds_configured",
            "passed": (
                config.max_can_frame_rate_per_sec > 0 and config.replay_window_tolerance_ms > 0
            ),
            "detail": (
                f"rate_limit={config.max_can_frame_rate_per_sec}fps, "
                f"replay_tol={config.replay_window_tolerance_ms}ms"
            ),
        },
        {
            "name": "in_vehicle_ids_entropy_active",
            "passed": config.payload_entropy_max > config.payload_entropy_min >= 0,
            "detail": (
                f"entropy_range=[{config.payload_entropy_min:.2f}, "
                f"{config.payload_entropy_max:.2f}] bits"
            ),
        },
        {
            "name": "defensive_containment_quarantine_enabled",
            "passed": (config.enable_defensive_mitigation and config.quarantine_duration_s >= 1.0),
            "detail": (
                f"mitigation={config.enable_defensive_mitigation}, "
                f"quarantine_dur={config.quarantine_duration_s}s"
            ),
        },
        {
            "name": "vsoc_telemetry_batching_conformance",
            "passed": (
                config.vsoc_telemetry_batch_size >= 1 and config.max_active_security_incidents >= 10
            ),
            "detail": (
                f"batch_size={config.vsoc_telemetry_batch_size}, "
                f"max_incidents={config.max_active_security_incidents}"
            ),
        },
        {
            "name": "iso_26262_safety_supervisor_interlock",
            "passed": config.anomaly_score_threshold >= 0.50,
            "detail": (
                f"escalation_threshold={config.anomaly_score_threshold:.2f} (Safe stop interlock)"
            ),
        },
    ]

    all_passed = all(c["passed"] for c in checks)
    audit_report = {
        "standard": "UN R155 / ISO/SAE 21434 Cybersecurity Management System (CSMS)",
        "compliant": all_passed,
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }

    if output is not None:
        write_json(output, audit_report)

    if as_json:
        typer.echo(json.dumps(audit_report, indent=2))
        return

    status_tag = "COMPLIANT" if all_passed else "NON-COMPLIANT"
    typer.echo(f"UN R155 CSMS Regulatory Compliance Audit: [{status_tag}]")
    for c in checks:
        icon = "[PASS]" if c["passed"] else "[FAIL]"
        typer.echo(f"  {icon} {c['name']}: {c['detail']}")
