"""CLI commands for facility access control, barrier gate interlocks, and ALPR enveloping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.gate.controller import BarrierInterlockController
from nearfield360.gate.envelope import VehicleEnvelopeProfiler
from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    BarrierKinematicsState,
    CredentialType,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    GateDirection,
    InductiveLoopId,
    InductiveLoopState,
    VehicleCredential,
    VehicleDimensionEnvelope,
)
from nearfield360.gate.validator import AccessCredentialValidator
from nearfield360.gate.viz import render_gate_dashboard
from nearfield360.utils.artifacts import write_json

gate_app = typer.Typer(
    help="Facility access control, barrier gate interlocks, ALPR, and clearance enveloping.",
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
        help="Path to render tactical BEV gateway dashboard PNG artifact.",
    ),
]


@gate_app.command("info")
def gate_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active gateway portal geometry, clearance limits, kinematics, and ALPR settings."""
    cfg = get_state(context).config.gate

    info_payload = {
        "portal_clearance_height_m": cfg.portal_clearance_height_m,
        "portal_clearance_width_m": cfg.portal_clearance_width_m,
        "ground_clearance_min_m": cfg.ground_clearance_min_m,
        "barrier_arm_length_m": cfg.barrier_arm_length_m,
        "barrier_opening_time_s": cfg.barrier_opening_time_s,
        "barrier_holding_time_s": cfg.barrier_holding_time_s,
        "barrier_closing_time_s": cfg.barrier_closing_time_s,
        "alpr_confidence_threshold": cfg.alpr_confidence_threshold,
        "anti_tailgating_timeout_s": cfg.anti_tailgating_timeout_s,
        "safety_loop_debounce_ms": cfg.safety_loop_debounce_ms,
        "uwb_rssi_threshold_dbm": cfg.uwb_rssi_threshold_dbm,
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("Facility Access Gateway Configuration:")
    clearance_str = (
        f"  Portal Clearance Limits:    {cfg.portal_clearance_height_m:.2f}m H x "
        f"{cfg.portal_clearance_width_m:.2f}m W"
    )
    typer.echo(clearance_str)
    typer.echo(f"  Min Ground Clearance:       {cfg.ground_clearance_min_m:.2f}m")
    typer.echo(f"  Barrier Arm Length:         {cfg.barrier_arm_length_m:.2f}m")
    timing_str = (
        f"  Arm Timing:                 Open {cfg.barrier_opening_time_s:.1f}s | "
        f"Hold {cfg.barrier_holding_time_s:.1f}s | Close {cfg.barrier_closing_time_s:.1f}s"
    )
    typer.echo(timing_str)
    typer.echo(f"  Anti-Tailgating Timeout:    {cfg.anti_tailgating_timeout_s:.1f}s")
    typer.echo(f"  Safety Loop Debounce:       {cfg.safety_loop_debounce_ms:.1f}ms")
    typer.echo(f"  ALPR Confidence Threshold:  {cfg.alpr_confidence_threshold * 100:.0f}%")
    typer.echo(f"  UWB RSSI Threshold:         {cfg.uwb_rssi_threshold_dbm:.1f} dBm")


@gate_app.command("verify")
def gate_verify(
    context: typer.Context,
    gate_id: Annotated[
        str, typer.Option("--gate-id", "-g", help="Gateway portal identifier.")
    ] = "GATE_01",
    plate: Annotated[
        str, typer.Option("--plate", "-p", help="Detected license plate string.")
    ] = "ABC-1234",
    confidence: Annotated[
        float, typer.Option("--confidence", help="ALPR optical match confidence (0.0 - 1.0).")
    ] = 0.95,
    token: Annotated[
        str, typer.Option("--token", "-t", help="Digital credential token / RFID hash.")
    ] = "",
    height: Annotated[
        float, typer.Option("--height", help="Measured vehicle height in meters.")
    ] = 1.65,
    width: Annotated[
        float, typer.Option("--width", help="Measured vehicle width in meters.")
    ] = 1.95,
    ground_clearance: Annotated[
        float, typer.Option("--ground-clearance", help="Measured ground clearance in meters.")
    ] = 0.15,
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", "-v", help="Vehicle identifier.")
    ] = "veh-001",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Validate vehicle license plate / credential token and evaluate dimensional envelope."""
    cfg = get_state(context).config.gate

    envelope = VehicleDimensionEnvelope(
        length_m=4.8,
        width_m=width,
        height_m=height,
        ground_clearance_m=ground_clearance,
    )

    credential = None
    if token:
        credential = VehicleCredential(
            credential_id="cred-cli",
            credential_type=CredentialType.UWB_DIGITAL_KEY,
            token_hash=token,
            valid_from_s=0.0,
            valid_until_s=86400.0,
        )

    request = GateAccessRequest(
        request_id="req-cli-verify",
        gate_id=gate_id,
        timestamp_s=100.0,
        vehicle_id=vehicle_id,
        detected_plate=plate,
        plate_confidence=confidence,
        credential=credential,
        envelope=envelope,
    )

    validator = AccessCredentialValidator(cfg)
    profiler = VehicleEnvelopeProfiler(cfg)

    # 1. Credential validation
    cred_decision = validator.validate_request(request)
    if not cred_decision.granted:
        final_decision = cred_decision
    else:
        # 2. Envelope profiling
        env_decision = profiler.evaluate_clearance(
            envelope, gate_id=gate_id, request_id=request.request_id
        )
        if not env_decision.granted:
            final_decision = env_decision
        else:
            final_decision = GateAccessDecision(
                decision_id="dec-cli-granted",
                request_id=request.request_id,
                status=AccessDecisionStatus.GRANTED,
                granted=True,
                message=f"Access GRANTED for {vehicle_id} ({plate}). Envelope clearances verified.",
                headroom_margin_m=env_decision.headroom_margin_m,
                width_margin_m=env_decision.width_margin_m,
                issued_pass_id=f"PASS-{plate.replace('-', '')}",
            )

    result_payload = final_decision.model_dump()

    if output is not None:
        write_json(output, result_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(result_payload, indent=2))
        return

    status_str = "GRANTED" if final_decision.granted else "REJECTED"
    typer.echo(f"Gate Access Verification: [{status_str}]")
    typer.echo(f"  Decision Status: {final_decision.status}")
    typer.echo(f"  Message:         {final_decision.message}")
    typer.echo(f"  Headroom Margin: {final_decision.headroom_margin_m:+.2f}m")
    typer.echo(f"  Width Margin:    {final_decision.width_margin_m:+.2f}m")
    if final_decision.issued_pass_id:
        typer.echo(f"  Issued Pass:     {final_decision.issued_pass_id}")


@gate_app.command("transit")
def gate_transit(
    context: typer.Context,
    gate_id: Annotated[
        str, typer.Option("--gate-id", "-g", help="Gateway portal identifier.")
    ] = "GATE_01",
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", "-v", help="Vehicle identifier.")
    ] = "veh-transit-01",
    plate: Annotated[str, typer.Option("--plate", "-p", help="License plate.")] = "ABC-1234",
    simulate_obstruction: Annotated[
        bool,
        typer.Option("--obstruction", help="Simulate Loop C obstacle triggering safety reversal."),
    ] = False,
    simulate_tailgating: Annotated[
        bool,
        typer.Option(
            "--tailgating",
            help="Simulate second vehicle unauthorized ingress triggering alarm lockout.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute complete 4-loop transit cycle with barrier kinematics and safety interlocks."""
    cfg = get_state(context).config.gate
    validator = AccessCredentialValidator(cfg)
    profiler = VehicleEnvelopeProfiler(cfg)
    controller = BarrierInterlockController(
        config=cfg,
        validator=validator,
        envelope_profiler=profiler,
        gate_id=gate_id,
        direction=GateDirection.ENTRY,
    )

    envelope = VehicleDimensionEnvelope(
        length_m=4.8, width_m=1.9, height_m=1.65, ground_clearance_m=0.15
    )
    request = GateAccessRequest(
        request_id="req-transit",
        gate_id=gate_id,
        vehicle_id=vehicle_id,
        detected_plate=plate,
        plate_confidence=0.98,
        envelope=envelope,
    )

    if simulate_tailgating:
        # Pre-trigger tailgating: occupy loop C without prior authorization
        controller.set_loop_state(InductiveLoopId.LOOP_C_SAFETY, True, timestamp_s=0.5)

    if simulate_obstruction:
        # Cause safety reversal on downward closing arm
        controller.process_access_request(request)
        controller.command_close()
        controller.set_loop_state(InductiveLoopId.LOOP_C_SAFETY, True, timestamp_s=2.0)

    event = controller.simulate_transit(request=request)

    result_payload = {
        "event_id": event.event_id,
        "gate_id": event.gate_id,
        "vehicle_id": event.vehicle_id,
        "granted": event.decision.granted,
        "decision_status": event.decision.status,
        "duration_s": event.duration_s,
        "tailgating_detected": event.tailgating_detected,
        "safety_reversal_triggered": controller.barrier.safety_interlock_active
        or simulate_obstruction,
        "barrier_angle_deg": controller.barrier.barrier_angle_deg,
    }

    if output is not None:
        write_json(output, result_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(result_payload, indent=2))
        return

    typer.echo(f"Gate Transit Simulation Cycle [{event.event_id}]:")
    typer.echo(f"  Gate Portal:     {event.gate_id}")
    typer.echo(f"  Vehicle:         {event.vehicle_id} ({plate})")
    typer.echo(f"  Decision:        {event.decision.status} (Granted: {event.decision.granted})")
    typer.echo(f"  Cycle Duration:  {event.duration_s:.2f}s")
    tailgating_banner = "DETECTED / ALARM ACTIVE" if event.tailgating_detected else "None (Secure)"
    reversal_status = "TRIGGERED" if result_payload["safety_reversal_triggered"] else "None"
    arm_angle_status = (
        f"  Final Arm Angle: {controller.barrier.barrier_angle_deg:.1f} deg "
        f"({controller.barrier.state})"
    )
    typer.echo(f"  Tailgating:      {tailgating_banner}")
    typer.echo(f"  Safety Reversal: {reversal_status}")
    typer.echo(arm_angle_status)


@gate_app.command("render")
def gate_render(
    context: typer.Context,
    gate_id: Annotated[
        str, typer.Option("--gate-id", "-g", help="Gateway portal identifier.")
    ] = "GATE_01",
    png: PngOption = None,
    barrier_angle: Annotated[
        float, typer.Option("--barrier-angle", help="Barrier arm angle in degrees (0.0 to 90.0).")
    ] = 0.0,
    occupied_loop: Annotated[
        str,
        typer.Option(
            "--loop",
            help="Active loop ID (approach, presence, safety, departure, none).",
        ),
    ] = "none",
    tailgating: Annotated[
        bool, typer.Option("--tailgating", help="Render tailgating alarm banner on HUD.")
    ] = False,
    reversal: Annotated[
        bool, typer.Option("--reversal", help="Render safety reversal indicator on HUD.")
    ] = False,
    granted: Annotated[
        bool, typer.Option("--granted/--rejected", help="Access authorization state for HUD.")
    ] = True,
) -> None:
    """Render tactical 1280x720 BEV gateway map and situational HUD to PNG."""
    del context
    arm_state = BarrierArmState.LOWERED_CLOSED

    if barrier_angle >= 85.0:
        arm_state = BarrierArmState.RAISED_OPEN
    elif reversal:
        arm_state = BarrierArmState.SAFETY_REVERSING
    elif barrier_angle > 0.0:
        arm_state = BarrierArmState.OPENING

    kinematics = BarrierKinematicsState(
        barrier_angle_deg=barrier_angle,
        angular_velocity_deg_s=0.0,
        state=arm_state,
        safety_interlock_active=reversal,
        obstruction_detected=reversal,
    )

    loops: dict[InductiveLoopId, InductiveLoopState] = {}
    for lid in InductiveLoopId:
        is_occ = occupied_loop.lower() == lid.value.lower()
        loops[lid] = InductiveLoopState(
            loop_id=lid,
            occupied=is_occ,
            activated_at_s=10.0 if is_occ else 0.0,
            dwell_duration_s=2.5 if is_occ else 0.0,
        )

    decision = GateAccessDecision(
        decision_id="dec-render",
        request_id="req-render",
        status=AccessDecisionStatus.GRANTED
        if granted
        else AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL,
        granted=granted,
        message="Access Authorized" if granted else "Access Denied: Unrecognized Credential",
        headroom_margin_m=0.75,
        width_margin_m=0.85,
        issued_pass_id="PASS-DEMO" if granted else "",
    )

    out_path = png or Path("gate_dashboard.png")
    render_gate_dashboard(
        gate_id=gate_id,
        barrier=kinematics,
        loops=loops,
        decision=decision,
        tailgating_active=tailgating,
        output_path=out_path,
    )

    typer.echo(f"Gateway tactical dashboard successfully rendered: {out_path}")


@gate_app.command("audit")
def gate_audit(
    context: typer.Context,
    gate_id: Annotated[
        str, typer.Option("--gate-id", "-g", help="Gateway portal identifier.")
    ] = "GATE_01",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display aggregated security metrics, transit volumes, and clearance audit stats."""
    cfg = get_state(context).config.gate
    validator = AccessCredentialValidator(cfg)
    profiler = VehicleEnvelopeProfiler(cfg)
    controller = BarrierInterlockController(
        config=cfg,
        validator=validator,
        envelope_profiler=profiler,
        gate_id=gate_id,
        direction=GateDirection.ENTRY,
    )

    # Populate sample transits for auditing
    envelope = VehicleDimensionEnvelope(length_m=4.8, width_m=1.9, height_m=1.65)
    r1 = GateAccessRequest(
        request_id="req-audit-1",
        gate_id=gate_id,
        vehicle_id="veh-audit-01",
        detected_plate="ABC-1234",
        plate_confidence=0.98,
        envelope=envelope,
    )
    r2 = GateAccessRequest(
        request_id="req-audit-2",
        gate_id=gate_id,
        vehicle_id="veh-audit-02",
        detected_plate="XYZ-9999",
        plate_confidence=0.95,
        envelope=envelope,
    )
    over_env = VehicleDimensionEnvelope(length_m=4.8, width_m=1.9, height_m=3.0)
    r3 = GateAccessRequest(
        request_id="req-audit-3",
        gate_id=gate_id,
        vehicle_id="veh-audit-03",
        detected_plate="ABC-1234",
        plate_confidence=0.98,
        envelope=over_env,
    )

    controller.simulate_transit(r1)
    controller.simulate_transit(r2)
    controller.simulate_transit(r3)

    report: GateAuditReport = controller.generate_audit_report()
    report_payload = report.model_dump()

    if output is not None:
        write_json(output, report_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(report_payload, indent=2))
        return

    typer.echo(f"Gateway Portal Operational Audit Report [{report.gate_id}]:")
    typer.echo(f"  Total Transits:              {report.total_transits}")
    typer.echo(f"  Granted Transits:            {report.granted_count}")
    typer.echo(f"  Rejected Transits:           {report.rejected_count}")
    typer.echo(f"  Tailgating Incidents:        {report.tailgating_incidents}")
    typer.echo(f"  Clearance Violations:        {report.clearance_violations}")
    typer.echo(f"  Avg Transit Duration:        {report.average_transit_duration_s:.2f}s")
