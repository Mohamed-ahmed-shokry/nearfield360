"""CLI commands for UN R156 Software Update Management System (SUMS)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.ota.executive import OTAExecutive
from nearfield360.ota.models import (
    FirmwareManifest,
    PartitionSlot,
    PreflightCheck,
    PreflightStatus,
)
from nearfield360.ota.verifier import FirmwareVerifier
from nearfield360.ota.viz import render_ota_dashboard
from nearfield360.utils.artifacts import write_json

ota_app = typer.Typer(
    help="UN R156 Software Update Management System (SUMS), ECU firmware verification & rollback.",
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
        help="Path to render tactical UN R156 OTA HUD dashboard PNG artifact.",
    ),
]


@ota_app.command("info")
def ota_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active UN R156 SUMS parameters, crypto keys, and rollback limits."""
    config = get_state(context).config.ota

    payload: dict[str, Any] = {
        "min_battery_soc_pct": config.min_battery_soc_pct,
        "max_package_size_mb": config.max_package_size_mb,
        "verification_timeout_s": config.verification_timeout_s,
        "activation_dwell_s": config.activation_dwell_s,
        "rollback_timeout_s": config.rollback_timeout_s,
        "max_consecutive_failures": config.max_consecutive_failures,
        "enforce_monotonic_version": config.enforce_monotonic_version,
        "require_charging_or_parked": config.require_charging_or_parked,
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("UN R156 Software Update Management System (SUMS) Configuration:")
    typer.echo(f"  Minimum Battery SoC        : {payload['min_battery_soc_pct']:.1f}%")
    typer.echo(f"  Max Package Size           : {payload['max_package_size_mb']:.1f} MB")
    typer.echo(f"  Verification Timeout       : {payload['verification_timeout_s']:.1f} s")
    typer.echo(f"  Activation Dwell Time      : {payload['activation_dwell_s']:.1f} s")
    typer.echo(f"  Rollback Timeout           : {payload['rollback_timeout_s']:.1f} s")
    typer.echo(f"  Max Consecutive Failures   : {payload['max_consecutive_failures']}")
    typer.echo(f"  Anti-Rollback Monotonic    : {payload['enforce_monotonic_version']}")
    typer.echo(f"  Standstill / Park Required : {payload['require_charging_or_parked']}")


@ota_app.command("manifest")
def ota_manifest(
    context: typer.Context,
    component_id: Annotated[
        str, typer.Option("--component-id", help="Component identifier.")
    ] = "surround-perception-ecu",
    target_ecu: Annotated[
        str, typer.Option("--target-ecu", help="Target ECU identifier.")
    ] = "ecu-perception-01",
    rxswin: Annotated[
        str, typer.Option("--rxswin", help="UN R156 Regulation Software Identification Number.")
    ] = "RXSWIN-NF360-V2-4-0",
    version: Annotated[
        str, typer.Option("--version", help="Firmware semantic version string.")
    ] = "v2.4.0",
    revision: Annotated[int, typer.Option("--revision", help="Monotonic revision counter.")] = 42,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Generate and cryptographically sign a UN R156 firmware manifest."""
    config = get_state(context).config.ota
    verifier = FirmwareVerifier(config)
    payload_mock = f"firmware-payload-{target_ecu}-{version}-{revision}".encode()
    sha256 = verifier.compute_payload_sha256(payload_mock)
    sig_hex = verifier.sign_payload(payload_mock)

    manifest = FirmwareManifest(
        component_id=component_id,
        version=version,
        monotonic_revision=revision,
        rxswin=rxswin,
        target_ecu=target_ecu,
        payload_size_bytes=len(payload_mock),
        sha256_hash=sha256,
        signature_hex=sig_hex,
    )

    data = manifest.model_dump(mode="json")
    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo(f"Generated UN R156 Firmware Manifest for [{manifest.target_ecu}]:")
    typer.echo(f"  Component ID       : {manifest.component_id}")
    typer.echo(f"  RXSWIN             : {manifest.rxswin}")
    typer.echo(f"  Target Version     : {manifest.version}")
    typer.echo(f"  Monotonic Revision : {manifest.monotonic_revision}")
    typer.echo(f"  Payload Size       : {manifest.payload_size_bytes} bytes")
    typer.echo(f"  SHA-256 Digest     : {manifest.sha256_hash}")
    typer.echo(f"  HMAC Signature     : {manifest.signature_hex}")


@ota_app.command("verify")
def ota_verify(
    context: typer.Context,
    manifest_path: Annotated[
        Path | None, typer.Option("--manifest", "-m", help="Path to JSON firmware manifest file.")
    ] = None,
    current_revision: Annotated[
        int, typer.Option("--current-revision", help="Active ECU monotonic revision.")
    ] = 40,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Verify cryptographic signature and anti-rollback monotonic counter."""
    config = get_state(context).config.ota
    verifier = FirmwareVerifier(config)

    if manifest_path is not None and manifest_path.is_file():
        raw_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = FirmwareManifest.model_validate(raw_dict)
        mock_payload = (
            f"firmware-payload-{manifest.target_ecu}-{manifest.version}-"
            f"{manifest.monotonic_revision}"
        ).encode()
    else:
        target_ecu = "ecu-perception-01"
        ver_str = "v2.4.0"
        rev_num = 42
        mock_payload = f"firmware-payload-{target_ecu}-{ver_str}-{rev_num}".encode()
        sha256 = verifier.compute_payload_sha256(mock_payload)
        sig_hex = verifier.sign_payload(mock_payload)
        manifest = FirmwareManifest(
            component_id="surround-perception-ecu",
            version=ver_str,
            monotonic_revision=rev_num,
            rxswin="RXSWIN-NF360-V2-4-0",
            target_ecu=target_ecu,
            payload_size_bytes=len(mock_payload),
            sha256_hash=sha256,
            signature_hex=sig_hex,
        )

    manifest_ok, manifest_msg = verifier.verify_manifest(manifest, mock_payload)
    anti_rollback_ok, rollback_msg = verifier.verify_anti_rollback(
        manifest.monotonic_revision, current_revision
    )

    overall_ok = manifest_ok and anti_rollback_ok
    payload_res: dict[str, Any] = {
        "valid": overall_ok,
        "target_ecu": manifest.target_ecu,
        "rxswin": manifest.rxswin,
        "manifest_valid": manifest_ok,
        "manifest_message": manifest_msg,
        "anti_rollback_valid": anti_rollback_ok,
        "anti_rollback_message": rollback_msg,
    }

    if output is not None:
        write_json(output, payload_res)

    if as_json:
        typer.echo(json.dumps(payload_res, indent=2))
        return

    typer.echo("UN R156 Cryptographic Firmware Verification Result:")
    typer.echo(f"  Overall Valid      : {overall_ok}")
    typer.echo(f"  Target ECU         : {manifest.target_ecu}")
    typer.echo(f"  RXSWIN             : {manifest.rxswin}")
    typer.echo(f"  Manifest Check     : {manifest_msg}")
    typer.echo(f"  Anti-Rollback Check: {rollback_msg}")


@ota_app.command("preflight")
def ota_preflight(
    context: typer.Context,
    gear: Annotated[
        str, typer.Option("--gear", help="Current gear selector (PARK, REVERSE, NEUTRAL, DRIVE).")
    ] = "PARK",
    speed: Annotated[float, typer.Option("--speed", help="Vehicle speed in m/s.")] = 0.0,
    soc: Annotated[
        float, typer.Option("--soc", help="High voltage / 12V battery SoC percentage.")
    ] = 75.0,
    supervisor_ready: Annotated[
        bool,
        typer.Option(
            "--supervisor-ready/--supervisor-degraded",
            help="Functional safety supervisor readiness status.",
        ),
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate vehicle safety interlocks (gear, standstill, SoC, supervisor) for update staging."""
    config = get_state(context).config.ota
    executive = OTAExecutive(config=config)
    result: PreflightCheck = executive.check_preflight_interlocks(
        gear=gear,
        speed_mps=speed,
        battery_soc_pct=soc,
        safety_supervisor_ready=supervisor_ready,
    )

    data = result.model_dump(mode="json")
    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    status_tag = "READY" if result.status == PreflightStatus.PASSED else "BLOCKED"
    typer.echo(f"UN R156 Safety Preflight Interlock Check: [{status_tag}]")
    typer.echo(f"  Gear Is Park       : {result.gear_is_park}")
    typer.echo(f"  Vehicle Speed      : {result.speed_mps:.3f} m/s")
    typer.echo(f"  Battery SoC        : {result.battery_soc_pct:.1f}%")
    typer.echo(f"  Supervisor Ready   : {result.safety_supervisor_ready}")
    typer.echo(f"  Status             : {result.status.value.upper()}")
    if result.reasons:
        typer.echo(f"  Blocking Reasons   : {', '.join(result.reasons)}")


@ota_app.command("apply")
def ota_apply(
    context: typer.Context,
    campaign_id: Annotated[
        str, typer.Option("--campaign-id", help="Unique campaign execution ID.")
    ] = "OTA-CAMP-2026-001",
    target_ecu: Annotated[
        str, typer.Option("--target-ecu", help="Target ECU identifier.")
    ] = "ecu-perception-01",
    version: Annotated[str, typer.Option("--version", help="Target firmware version.")] = "v2.4.0",
    revision: Annotated[int, typer.Option("--revision", help="Target monotonic revision.")] = 42,
    simulate_failure: Annotated[
        bool,
        typer.Option(
            "--simulate-failure",
            help="Simulate post-reboot health check failure to trigger rollback.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Execute end-to-end UN R156 OTA campaign lifecycle with automated rollback guard."""
    config = get_state(context).config.ota
    executive = OTAExecutive(config=config)

    payload_mock = f"firmware-payload-{target_ecu}-{version}-{revision}".encode()
    sha256 = executive.verifier.compute_payload_sha256(payload_mock)
    sig_hex = executive.verifier.sign_payload(payload_mock)

    manifest = FirmwareManifest(
        component_id="surround-perception-ecu",
        version=version,
        monotonic_revision=revision,
        rxswin=f"RXSWIN-NF360-V{version.lstrip('v').replace('.', '-')}",
        target_ecu=target_ecu,
        payload_size_bytes=len(payload_mock),
        sha256_hash=sha256,
        signature_hex=sig_hex,
    )

    report = executive.execute_campaign(
        campaign_id=campaign_id,
        manifest=manifest,
        payload_bytes=payload_mock,
        vehicle_state={
            "gear": "PARK",
            "speed_mps": 0.0,
            "battery_soc_pct": 85.0,
            "safety_supervisor_ready": True,
        },
        simulate_reboot_health=not simulate_failure,
    )

    if png is not None:
        render_ota_dashboard(report, png)

    data = report.model_dump(mode="json")
    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    active_slot = report.partition_state.active_slot.value.upper()
    act_meta = report.partition_state.slots[report.partition_state.active_slot.value]
    typer.echo(f"UN R156 OTA Campaign Execution Report: [{report.campaign_id}]")
    typer.echo(f"  Target ECU         : {report.target_ecu}")
    typer.echo(f"  Final State        : {report.active_state.value.upper()}")
    typer.echo(f"  Active Partition   : Slot {active_slot}")
    typer.echo(f"  Active Version     : {act_meta.version}")
    typer.echo(f"  Active Revision    : {act_meta.monotonic_revision}")
    typer.echo(f"  Rollback Count     : {report.partition_state.rollback_count}")
    typer.echo(f"  Campaign Succeeded : {report.success}")
    typer.echo(f"  Total Duration     : {report.duration_s:.2f} s")
    typer.echo("  Execution Audit Trail:")
    for step in report.audit_trail:
        typer.echo(
            f"    - [{step.from_state.value} -> {step.to_state.value}] "
            f"{step.message} ({step.timestamp_s:.1f}s)"
        )


@ota_app.command("render")
def ota_render(
    context: typer.Context,
    png: Annotated[
        Path,
        typer.Option(
            "--png",
            "-p",
            resolve_path=True,
            help="Destination path for 1280x720 PNG HUD dashboard.",
        ),
    ] = Path("artifacts/ota_hud.png"),
    target_ecu: Annotated[
        str, typer.Option("--target-ecu", help="ECU identifier.")
    ] = "ecu-perception-01",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON status.")] = False,
    output: OutputOption = None,
) -> None:
    """Render a tactical 1280x720 UN R156 OTA HUD dashboard PNG artifact."""
    config = get_state(context).config.ota
    executive = OTAExecutive(config=config)

    ver_str = "v2.4.0"
    rev_num = 42
    payload_mock = f"firmware-payload-{target_ecu}-{ver_str}-{rev_num}".encode()
    sha256 = executive.verifier.compute_payload_sha256(payload_mock)
    sig_hex = executive.verifier.sign_payload(payload_mock)

    manifest = FirmwareManifest(
        component_id="surround-perception-ecu",
        version=ver_str,
        monotonic_revision=rev_num,
        rxswin="RXSWIN-NF360-V2-4-0",
        target_ecu=target_ecu,
        payload_size_bytes=len(payload_mock),
        sha256_hash=sha256,
        signature_hex=sig_hex,
    )

    report = executive.execute_campaign(
        campaign_id="CAMP-RENDER-SAMPLE-01",
        manifest=manifest,
        payload_bytes=payload_mock,
        vehicle_state={
            "gear": "PARK",
            "speed_mps": 0.0,
            "battery_soc_pct": 88.5,
            "safety_supervisor_ready": True,
        },
        simulate_reboot_health=True,
    )

    render_ota_dashboard(report, png)
    payload = {
        "rendered_path": str(png),
        "campaign_id": report.campaign_id,
        "status": "rendered",
        "active_slot": report.partition_state.active_slot.value,
        "active_version": report.partition_state.slots[PartitionSlot.SLOT_B.value].version,
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Successfully rendered UN R156 OTA HUD dashboard to: {png}")


@ota_app.command("audit")
def ota_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON audit results.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit vehicle software update management system (SUMS) compliance with UN R156."""
    config = get_state(context).config.ota

    checks: list[dict[str, Any]] = [
        {
            "name": "un_r156_sums_configured",
            "passed": config.verification_timeout_s > 0 and config.max_package_size_mb > 0,
            "detail": (
                f"timeout={config.verification_timeout_s}s, max_size={config.max_package_size_mb}MB"
            ),
        },
        {
            "name": "cryptographic_signatures_enabled",
            "passed": len(config.signing_secret_key) >= 16,
            "detail": f"secret_key_len={len(config.signing_secret_key)} (HMAC-SHA256)",
        },
        {
            "name": "anti_rollback_monotonic_guard",
            "passed": config.enforce_monotonic_version and config.max_consecutive_failures >= 1,
            "detail": (
                f"active={config.enforce_monotonic_version}, "
                f"max_consecutive_failures={config.max_consecutive_failures}"
            ),
        },
        {
            "name": "safety_preflight_interlocks",
            "passed": config.require_charging_or_parked and config.min_battery_soc_pct >= 20.0,
            "detail": (
                f"park_req={config.require_charging_or_parked}, "
                f"min_soc={config.min_battery_soc_pct}%"
            ),
        },
        {
            "name": "dual_partition_ab_failover",
            "passed": config.rollback_timeout_s >= 5.0,
            "detail": f"rollback_timeout={config.rollback_timeout_s}s (Dual-slot A/B compliant)",
        },
    ]

    all_passed = all(c["passed"] for c in checks)
    audit_report = {
        "standard": "UN R156 Software Update Management System (SUMS)",
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
    typer.echo(f"UN R156 SUMS Regulatory Audit: [{status_tag}]")
    for c in checks:
        icon = "[PASS]" if c["passed"] else "[FAIL]"
        typer.echo(f"  {icon} {c['name']}: {c['detail']}")
