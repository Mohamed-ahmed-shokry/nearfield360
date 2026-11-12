"""CLI commands for CCC Digital Key 3.0, UWB distance bounding, and proximity interlock."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.digitalkey.crypto import DigitalKeyCrypto
from nearfield360.digitalkey.interlock import HandoverSupervisor
from nearfield360.digitalkey.models import (
    CCCDelegationToken,
    DelegationScope,
    DigitalKeyReport,
    HandoverState,
    UWBRangeMeasurement,
)
from nearfield360.digitalkey.solver import MultilaterationSolver, get_default_uwb_anchors
from nearfield360.digitalkey.uwb import UWBEngine
from nearfield360.digitalkey.viz import render_digitalkey_dashboard
from nearfield360.utils.artifacts import write_json

digitalkey_app = typer.Typer(
    help="CCC Digital Key 3.0, UWB distance bounding, and proximity interlock.",
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
        help="Path to render tactical 1280x720 HUD dashboard PNG artifact.",
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


@digitalkey_app.command("info")
def digitalkey_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect active CCC Digital Key 3.0 and UWB distance bounding configuration."""
    config = get_state(context).config.digitalkey
    data = {
        "enable_uwb_sts": config.enable_uwb_sts,
        "carrier_frequency_mhz": config.carrier_frequency_mhz,
        "touch_threshold_m": config.touch_threshold_m,
        "immediate_threshold_m": config.immediate_threshold_m,
        "near_threshold_m": config.near_threshold_m,
        "far_threshold_m": config.far_threshold_m,
        "max_relay_tof_discrepancy_ns": config.max_relay_tof_discrepancy_ns,
        "token_validity_window_s": config.token_validity_window_s,
        "enable_standstill_interlock": config.enable_standstill_interlock,
        "num_uwb_anchors": config.num_uwb_anchors,
    }
    summary = [
        "CCC Digital Key 3.0 & UWB Ranging Configuration:",
        f"  UWB STS Enabled: {config.enable_uwb_sts} ({config.carrier_frequency_mhz} MHz)",
        (
            f"  Thresholds: Touch={config.touch_threshold_m}m, "
            f"Immediate={config.immediate_threshold_m}m, "
            f"Near={config.near_threshold_m}m, Far={config.far_threshold_m}m"
        ),
        f"  Max Relay ToF Discrepancy: {config.max_relay_tof_discrepancy_ns} ns",
        f"  Standstill Interlock Enabled: {config.enable_standstill_interlock}",
        f"  UWB Anchors: {config.num_uwb_anchors}",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("pair")
def digitalkey_pair(
    context: typer.Context,
    vin: Annotated[str, typer.Option("--vin", help="Vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    user_id: Annotated[str, typer.Option("--user", help="User account ID.")] = "usr-owner-01",
    device_id: Annotated[str, typer.Option("--device", help="Device ID.")] = "iphone-16-pro",
    scopes: Annotated[
        str, typer.Option("--scopes", help="Comma-separated scopes (e.g. AVP_OPERATE,CABIN_LOCK).")
    ] = "AVP_OPERATE,CABIN_LOCK",
    duration_s: Annotated[
        float, typer.Option("--duration-s", help="Delegation validity duration in seconds.")
    ] = 3600.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Issue a cryptographically signed CCC Digital Key delegation token."""
    config = get_state(context).config.digitalkey
    crypto = DigitalKeyCrypto(config)

    parsed_scopes: list[DelegationScope] = []
    for s_name in scopes.split(","):
        s_clean = s_name.strip().upper()
        if s_clean in DelegationScope.__members__:
            parsed_scopes.append(DelegationScope[s_clean])

    if not parsed_scopes:
        parsed_scopes = [DelegationScope.AVP_OPERATE]

    token = crypto.issue_delegation_token(
        vin=vin,
        user_id=user_id,
        device_id=device_id,
        scopes=parsed_scopes,
        validity_duration_s=duration_s,
    )
    data = token.model_dump(mode="json")
    summary = [
        "CCC Digital Key Delegation Token Issued:",
        f"  Token ID: {token.token_id}",
        f"  VIN: {token.vin}",
        f"  User: {token.user_id}  |  Device: {token.device_id}",
        f"  Scopes: {', '.join(s.value for s in token.scopes)}",
        f"  Expires: {token.expires_at_utc}",
        f"  Signature: {token.signature[:16]}... (valid HMAC-SHA256)",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("authenticate")
def digitalkey_authenticate(
    context: typer.Context,
    token_file: Annotated[Path, typer.Option("--token-file", help="Path to token JSON.")],
    vin: Annotated[str, typer.Option("--vin", help="Expected vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    scope: Annotated[
        str, typer.Option("--scope", help="Required scope (e.g. AVP_OPERATE).")
    ] = "AVP_OPERATE",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Validate cryptographic signature, validity, and scope of a delegation token."""
    config = get_state(context).config.digitalkey
    crypto = DigitalKeyCrypto(config)

    if not token_file.is_file():
        typer.echo(f"Error: Token file not found: {token_file}", err=True)
        raise typer.Exit(code=1)

    with token_file.open("r", encoding="utf-8") as f:
        token_data = json.load(f)

    token = CCCDelegationToken.model_validate(token_data)
    req_scope = (
        DelegationScope[scope.upper()] if scope.upper() in DelegationScope.__members__ else None
    )

    valid, reason = crypto.verify_token(token, expected_vin=vin, required_scope=req_scope)
    data = {
        "token_id": token.token_id,
        "is_valid": valid,
        "reason": reason,
        "vin": token.vin,
        "user_id": token.user_id,
        "scopes": [s.value for s in token.scopes],
        "expires_at_utc": token.expires_at_utc,
    }
    summary = [
        "CCC Digital Key Authentication Result:",
        f"  Valid: {valid}",
        f"  Reason: {reason}",
        f"  Token: {token.token_id}  |  VIN: {token.vin}",
    ]
    _emit_result(data, as_json, output, summary)
    if not valid:
        raise typer.Exit(code=2)


@digitalkey_app.command("range")
def digitalkey_range(
    context: typer.Context,
    x: Annotated[
        float, typer.Option("--x", help="Simulated mobile device x offset in meters.")
    ] = 2.0,
    y: Annotated[
        float, typer.Option("--y", help="Simulated mobile device y offset in meters.")
    ] = 1.2,
    z: Annotated[
        float, typer.Option("--z", help="Simulated mobile device z offset in meters.")
    ] = 1.0,
    sts_valid: Annotated[
        bool, typer.Option("--sts-valid", help="Simulate STS cryptographic match.")
    ] = True,
    vin: Annotated[str, typer.Option("--vin", help="Vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Perform UWB 3D multilateration, anti-relay checks, and proximity classification."""
    config = get_state(context).config.digitalkey
    supervisor = HandoverSupervisor(vin=vin, config=config)

    anchors = get_default_uwb_anchors()
    ts = 1700000000000000
    meas: list[UWBRangeMeasurement] = []
    for a in anchors:
        dist = math.sqrt((x - a.x_m) ** 2 + (y - a.y_m) ** 2 + (z - a.z_m) ** 2)
        tof = dist / 0.299792458
        meas.append(
            UWBRangeMeasurement(
                anchor_id=a.anchor_id,
                distance_m=dist,
                timestamp_ns=ts,
                sts_valid=sts_valid,
                tof_ns=tof,
                snr_db=25.0,
            )
        )

    report = supervisor.evaluate(meas)
    data = report.to_dict()

    if png is not None:
        render_digitalkey_dashboard(report, output_path=png)

    summary = [
        "UWB 3D Multilateration & Ranging Report:",
        (
            f"  Resolved Position: ({report.device_position.x_m:.2f}, "
            f"{report.device_position.y_m:.2f}, {report.device_position.z_m:.2f}) m"
        ),
        f"  Distance to Center: {report.device_position.distance_m:.2f} m",
        (
            f"  Proximity Zone: {report.proximity_zone.value} "
            f"(Interior: {report.device_position.is_interior})"
        ),
        (
            f"  Confidence: {report.device_position.confidence:.2f} "
            f"(RMS Error: {report.device_position.residual_error_m:.3f} m)"
        ),
        f"  Relay Attack Detected: {report.relay_attack_detected}",
        f"  Standstill Command: {report.interlock.standstill_command}",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("interlock")
def digitalkey_interlock(
    context: typer.Context,
    distance_m: Annotated[
        float, typer.Option("--distance", help="Detected distance in meters.")
    ] = 1.2,
    relay_attack: Annotated[
        bool, typer.Option("--relay-attack", help="Simulate active relay attack.")
    ] = False,
    vin: Annotated[str, typer.Option("--vin", help="Vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate proximity standstill safety interlock during autonomous valet parking."""
    config = get_state(context).config.digitalkey
    supervisor = HandoverSupervisor(vin=vin, config=config)

    # Issue dummy delegation token and delegate
    crypto = DigitalKeyCrypto(config)
    tok = crypto.issue_delegation_token(
        vin=vin, user_id="usr-test", device_id="dev-test", scopes=[DelegationScope.AVP_OPERATE]
    )
    supervisor.delegate_to_avp(tok)

    anchors = get_default_uwb_anchors()
    meas = [
        UWBRangeMeasurement(
            anchor_id=a.anchor_id,
            distance_m=distance_m,
            timestamp_ns=1700000000000000,
            sts_valid=not relay_attack,
            tof_ns=(distance_m / 0.299792458) if not relay_attack else 25.0,
            snr_db=24.0,
        )
        for a in anchors
    ]

    report = supervisor.evaluate(meas)
    data = report.interlock.model_dump(mode="json")
    summary = [
        "Proximity Safety Interlock Decision:",
        f"  Standstill Command: {report.interlock.standstill_command}",
        f"  Interlock Active: {report.interlock.interlock_active}",
        f"  Detected Distance: {report.interlock.detected_distance_m:.2f} m",
        f"  Proximity Zone: {report.interlock.proximity_zone.value}",
        f"  Reason: {report.interlock.reason}",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("handover")
def digitalkey_handover(
    context: typer.Context,
    action: Annotated[
        str,
        typer.Option(
            "--action",
            "-a",
            help="Handover action: delegate, pickup-ready, takeover, manual.",
        ),
    ] = "delegate",
    token_file: Annotated[
        Path | None, typer.Option("--token-file", help="Path to token JSON.")
    ] = None,
    vin: Annotated[str, typer.Option("--vin", help="Vehicle VIN.")] = "WP0ZZZ99ZTS392001",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute drop-off delegation or pick-up takeover state transition."""
    config = get_state(context).config.digitalkey
    supervisor = HandoverSupervisor(vin=vin, config=config)
    crypto = DigitalKeyCrypto(config)

    action_lower = action.lower()
    if action_lower == "delegate":
        if token_file and token_file.is_file():
            with token_file.open("r", encoding="utf-8") as f:
                tok_data = json.load(f)
            tok = CCCDelegationToken.model_validate(tok_data)
        else:
            tok = crypto.issue_delegation_token(
                vin=vin,
                user_id="usr-alice",
                device_id="iphone-16",
                scopes=[DelegationScope.AVP_OPERATE],
            )
        ok, reason = supervisor.delegate_to_avp(tok)
    elif action_lower == "pickup-ready":
        # Simulate prior delegation
        tok = crypto.issue_delegation_token(
            vin=vin,
            user_id="usr-alice",
            device_id="iphone-16",
            scopes=[DelegationScope.AVP_OPERATE],
        )
        supervisor.delegate_to_avp(tok)
        supervisor.mark_pickup_ready()
        ok, reason = True, "Vehicle marked PICKUP_READY"
    elif action_lower == "takeover":
        tok = crypto.issue_delegation_token(
            vin=vin,
            user_id="usr-alice",
            device_id="iphone-16",
            scopes=[DelegationScope.AVP_OPERATE],
        )
        supervisor.delegate_to_avp(tok)
        supervisor.mark_pickup_ready()
        # Approach to 2.0m
        meas = [
            UWBRangeMeasurement(
                anchor_id=a.anchor_id,
                distance_m=2.0,
                timestamp_ns=100,
                sts_valid=True,
                tof_ns=6.67,
            )
            for a in get_default_uwb_anchors()
        ]
        supervisor.evaluate(meas)
        ok, reason = True, f"Takeover in progress: state={supervisor.state.value}"
    elif action_lower == "manual":
        tok = crypto.issue_delegation_token(
            vin=vin,
            user_id="usr-alice",
            device_id="iphone-16",
            scopes=[DelegationScope.AVP_OPERATE],
        )
        supervisor.delegate_to_avp(tok)
        supervisor.mark_pickup_ready()
        # Inside cabin measurements
        meas = [
            UWBRangeMeasurement(
                anchor_id=a.anchor_id,
                distance_m=math.sqrt(a.x_m**2 + a.y_m**2 + (1.0 - a.z_m) ** 2),
                timestamp_ns=100,
                sts_valid=True,
                tof_ns=4.0,
            )
            for a in get_default_uwb_anchors()
        ]
        supervisor.evaluate(meas)
        ok = supervisor.state == HandoverState.MANUAL_CONTROL
        reason = f"Handover to manual driver: state={supervisor.state.value}"
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(code=1)

    data = {
        "action": action_lower,
        "success": ok,
        "reason": reason,
        "handover_state": supervisor.state.value,
    }
    summary = [
        "AVP Handover State Transition:",
        f"  Action: {action_lower}",
        f"  Success: {ok}",
        f"  Current State: {supervisor.state.value}",
        f"  Details: {reason}",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("render")
def digitalkey_render(
    context: typer.Context,
    report_file: Annotated[
        Path | None, typer.Option("--report-file", help="Path to DigitalKeyReport JSON.")
    ] = None,
    png: PngOption = None,
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
) -> None:
    """Render a tactical 1280x720 HUD dashboard from report or live measurement."""
    config = get_state(context).config.digitalkey
    if report_file and report_file.is_file():
        with report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        report = DigitalKeyReport.model_validate(data)
    else:
        supervisor = HandoverSupervisor(config=config)
        crypto = DigitalKeyCrypto(config)
        tok = crypto.issue_delegation_token(
            vin="WP0ZZZ99ZTS392001",
            user_id="usr-alice",
            device_id="iphone-16",
            scopes=[DelegationScope.AVP_OPERATE],
        )
        supervisor.delegate_to_avp(tok)
        anchors = get_default_uwb_anchors()
        meas = [
            UWBRangeMeasurement(
                anchor_id=a.anchor_id,
                distance_m=2.4,
                timestamp_ns=1700000000000000,
                sts_valid=True,
                tof_ns=8.0,
                snr_db=25.0,
                aoa_azimuth_deg=15.0,
            )
            for a in anchors
        ]
        report = supervisor.evaluate(meas)

    target_png = png or Path("outputs/digitalkey_dashboard.png")
    render_digitalkey_dashboard(report, output_path=target_png)

    data = report.to_dict()
    summary = [
        "Tactical 1280x720 Digital Key HUD Rendered:",
        f"  PNG File: {target_png}",
        f"  Device: {report.device_id} ({report.proximity_zone.value})",
        f"  Handover State: {report.handover_state.value}",
    ]
    _emit_result(data, as_json, output, summary)


@digitalkey_app.command("audit")
def digitalkey_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Perform self-diagnostic audit of CCC Digital Key & UWB ranging subsystem."""
    config = get_state(context).config.digitalkey
    checks: list[dict[str, Any]] = []

    # 1. Check Config
    checks.append(
        {
            "name": "config_integrity",
            "passed": config.enable_uwb_sts and config.num_uwb_anchors >= 4,
            "detail": f"STS={config.enable_uwb_sts}, Anchors={config.num_uwb_anchors}",
        }
    )

    # 2. Check Cryptographic Engine
    crypto = DigitalKeyCrypto(config)
    tok = crypto.issue_delegation_token(
        vin="WP0ZZZ99ZTS392001",
        user_id="usr-audit",
        device_id="dev-audit",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    ok_tok, _ = crypto.verify_token(tok, expected_vin="WP0ZZZ99ZTS392001")
    checks.append(
        {
            "name": "crypto_token_signing",
            "passed": ok_tok,
            "detail": f"Token signing and HMAC verification: {ok_tok}",
        }
    )

    # 3. Check Multilateration Solver
    solver = MultilaterationSolver(config=config)
    anchors = solver.anchors
    checks.append(
        {
            "name": "uwb_anchor_layout",
            "passed": len(anchors) >= 4,
            "detail": f"Configured anchors: {sorted(anchors.keys())}",
        }
    )

    # 4. Check Relay Defense
    uwb = UWBEngine(config)
    test_meas = UWBRangeMeasurement(
        anchor_id="FL",
        distance_m=1.0,
        timestamp_ns=100,
        sts_valid=False,
        tof_ns=10.0,
    )
    is_relay, _, _ = uwb.detect_relay_attack(test_meas)
    checks.append(
        {
            "name": "anti_relay_defense",
            "passed": is_relay,
            "detail": "Invalid STS correctly triggered relay alarm",
        }
    )

    all_passed = all(c["passed"] for c in checks)
    data = {
        "overall_status": "PASS" if all_passed else "FAIL",
        "total_checks": len(checks),
        "checks": checks,
    }
    summary = [
        f"CCC Digital Key Subsystem Audit: {'PASS' if all_passed else 'FAIL'}",
        *[f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks],
    ]
    _emit_result(data, as_json, output, summary)
    if not all_passed:
        raise typer.Exit(code=1)
