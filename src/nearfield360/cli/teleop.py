"""CLI commands for remote teleoperation assistance, video fallback, and emergency MRM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.teleop.manager import TeleopManager
from nearfield360.teleop.models import (
    TeleopCommand,
    TeleopHeartbeat,
    TeleopMode,
    TeleopWaypoint,
)
from nearfield360.teleop.viz import render_teleop_dashboard
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

teleop_app = typer.Typer(
    help="Remote teleoperation assistance, video fallback streaming, and emergency MRM.",
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
        help="Path to render teleoperation cockpit BEV telemetry PNG artifact.",
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


@teleop_app.command("info")
def teleop_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active teleoperation latency limits, video streaming specs, and MRM thresholds."""
    cfg = get_state(context).config.teleop

    info_payload = {
        "heartbeat_timeout_s": cfg.heartbeat_timeout_s,
        "max_rtt_ms": cfg.max_rtt_ms,
        "critical_rtt_ms": cfg.critical_rtt_ms,
        "video_target_fps": cfg.video_target_fps,
        "video_fallback_fps": cfg.video_fallback_fps,
        "mrm_deceleration_mps2": cfg.mrm_deceleration_mps2,
        "safe_stop_standstill_speed_mps": cfg.safe_stop_standstill_speed_mps,
        "auto_mrm_on_heartbeat_loss": cfg.auto_mrm_on_heartbeat_loss,
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("Remote Teleoperation & Safety MRM Configuration:")
    typer.echo(f"  Heartbeat Timeout:          {cfg.heartbeat_timeout_s:.1f} s")
    typer.echo(f"  Max Latency (Direct Drive): {cfg.max_rtt_ms:.1f} ms")
    typer.echo(f"  Critical Latency Limit:     {cfg.critical_rtt_ms:.1f} ms")
    typer.echo(f"  Target Video Streaming:     {cfg.video_target_fps:.1f} fps")
    typer.echo(f"  Degraded Fallback Stream:   {cfg.video_fallback_fps:.1f} fps")
    typer.echo(f"  MRM Deceleration Rate:      {cfg.mrm_deceleration_mps2:.1f} m/s^2")
    typer.echo(f"  Failsafe Watchdog Active:   {cfg.auto_mrm_on_heartbeat_loss}")


@teleop_app.command("session")
def teleop_session(
    context: typer.Context,
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "veh-teleop-01",
    operator_id: Annotated[str, typer.Option("--operator-id", help="Operator ID.")] = "op-desk-01",
    rtt_ms: Annotated[float, typer.Option("--rtt", help="Simulated round trip time (ms).")] = 45.0,
    emergency_stop: Annotated[
        bool, typer.Option("--emergency-stop", help="Transmit emergency stop override.")
    ] = False,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Initiate and step a remote assistance session with guidance or direct commands."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.teleop
    mgr = TeleopManager(facility_id=facility.map_id, config=cfg)

    # 1. Start session
    session = mgr.start_session(vehicle_id, operator_id, TeleopMode.TELE_GUIDANCE, 0.0)

    # 2. Transmit heartbeat
    mgr.process_heartbeat(
        TeleopHeartbeat(vehicle_id=vehicle_id, sequence_number=1, timestamp_s=0.5, rtt_ms=rtt_ms)
    )

    # 3. Transmit command
    cmd = TeleopCommand(
        command_id="cmd-init-01",
        vehicle_id=vehicle_id,
        timestamp_s=0.6,
        mode=TeleopMode.TELE_GUIDANCE,
        emergency_stop=emergency_stop,
        suggested_waypoints=[TeleopWaypoint(x_m=12.0, y_m=6.0, target_speed_mps=1.5)],
    )
    ok, msg = mgr.dispatch_command(cmd, current_time_s=0.6, current_speed_mps=2.0)
    if emergency_stop:
        mgr.step(current_time_s=3.0)  # advance deceleration to standstill

    report = mgr.generate_report(current_time_s=3.0 if emergency_stop else 1.0)

    if png is not None:
        render_teleop_dashboard(facility, report, output_path=png)

    payload = {
        "session": mgr.active_sessions[vehicle_id].model_dump(mode="json"),
        "command_dispatched": ok,
        "dispatch_message": msg,
        "mrm_status": mgr.mrm_engines[vehicle_id].get_status(),
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    curr_session = mgr.active_sessions[vehicle_id]
    typer.echo("Remote Teleoperation Session Execution:")
    typer.echo(f"  Session ID:         {session.session_id}")
    typer.echo(f"  Vehicle ID:         {vehicle_id}")
    typer.echo(f"  Operator Desk:      {operator_id}")
    typer.echo(f"  Operating Mode:     {curr_session.active_mode.value.upper()}")
    typer.echo(f"  MRM Stage:          {curr_session.mrm_stage.value.upper()}")
    typer.echo(f"  Dispatch Result:    {msg}")


@teleop_app.command("fallback")
def teleop_fallback(
    context: typer.Context,
    vehicle_id: Annotated[
        str, typer.Option("--vehicle-id", help="Target vehicle ID.")
    ] = "veh-fallback-01",
    latency_ms: Annotated[float, typer.Option("--latency", help="Degraded RTT latency.")] = 280.0,
    packet_loss: Annotated[float, typer.Option("--packet-loss", help="Drop packets.")] = 0.25,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate bandwidth-adaptive video fallback and watchdog trigger under network severance."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.teleop
    mgr = TeleopManager(facility_id=facility.map_id, config=cfg)

    mgr.start_session(vehicle_id, "op-fallback", TeleopMode.DIRECT_DRIVING, 0.0)

    # Simulate heartbeat sequence with packet drops
    seq = 0
    if packet_loss > 0.0:
        seq = 5  # gap in sequence simulating dropped packets
    mgr.process_heartbeat(
        TeleopHeartbeat(
            vehicle_id=vehicle_id,
            sequence_number=seq,
            timestamp_s=0.5,
            rtt_ms=latency_ms,
        )
    )

    monitor = mgr.network_monitors[vehicle_id]
    streaming_rec = monitor.compute_streaming_recommendation(current_time_s=0.5)

    # Attempt direct driving command under degraded network
    cmd = TeleopCommand(
        command_id="cmd-direct-01",
        vehicle_id=vehicle_id,
        timestamp_s=0.6,
        mode=TeleopMode.DIRECT_DRIVING,
        steering_angle_rad=0.1,
        target_velocity_mps=2.0,
    )
    ok, msg = mgr.dispatch_command(cmd, current_time_s=0.6)

    # Advance time beyond timeout to trigger watchdog MRM
    mgr.step(current_time_s=0.5 + cfg.heartbeat_timeout_s + 0.5)

    report = mgr.generate_report(current_time_s=3.0)

    if png is not None:
        render_teleop_dashboard(facility, report, output_path=png)

    curr_session = mgr.active_sessions[vehicle_id]
    payload = {
        "session": curr_session.model_dump(mode="json"),
        "streaming_recommendation": streaming_rec,
        "command_accepted": ok,
        "rejection_reason": msg if not ok else None,
        "mrm_stage": curr_session.mrm_stage.value,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Teleoperation Video Fallback & Watchdog Execution:")
    typer.echo(f"  Vehicle ID:         {vehicle_id}")
    typer.echo(f"  Network Quality:    {streaming_rec['network_quality'].upper()}")
    typer.echo(f"  Recommended Video:  {streaming_rec['recommended_fps']:.1f} fps")
    typer.echo(f"  Direct Drive State: {'ACCEPTED' if ok else 'REJECTED'}")
    typer.echo(f"  Failsafe MRM Stage: {curr_session.mrm_stage.value.upper()}")


@teleop_app.command("audit")
def teleop_audit(
    context: typer.Context,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Perform a comprehensive facility teleoperation, network quality, and MRM audit."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.teleop
    mgr = TeleopManager(facility_id=facility.map_id, config=cfg)

    # 1. Normal guidance session
    mgr.start_session("veh-audit-01", "op-desk-1", TeleopMode.TELE_GUIDANCE, 0.0)
    mgr.process_heartbeat(
        TeleopHeartbeat(
            vehicle_id="veh-audit-01",
            sequence_number=1,
            timestamp_s=1.0,
            rtt_ms=42.0,
        )
    )

    # 2. Degraded session with MRM intervention
    mgr.start_session("veh-audit-02", "op-desk-2", TeleopMode.DIRECT_DRIVING, 2.0)
    mgr.process_heartbeat(
        TeleopHeartbeat(
            vehicle_id="veh-audit-02",
            sequence_number=1,
            timestamp_s=2.5,
            rtt_ms=180.0,
        )
    )
    cmd_e = TeleopCommand(
        command_id="cmd-audit-estop",
        vehicle_id="veh-audit-02",
        timestamp_s=3.0,
        emergency_stop=True,
    )
    mgr.dispatch_command(cmd_e, current_time_s=3.0, current_speed_mps=2.5)
    mgr.step(current_time_s=5.0)

    report = mgr.generate_report(current_time_s=10.0)

    if png is not None:
        render_teleop_dashboard(facility, report, output_path=png)

    payload = report.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Teleoperation Facility Operations & Safety Audit:")
    typer.echo(f"  Facility ID:        {report.facility_id}")
    typer.echo(f"  Active Sessions:    {report.active_sessions_count}")
    typer.echo(f"  Total Interventions:{report.total_interventions_count}")
    typer.echo(f"  Average Link RTT:   {report.average_rtt_ms:.1f} ms")
    typer.echo(f"  MRM Executions:     {report.mrm_executions_count}")
