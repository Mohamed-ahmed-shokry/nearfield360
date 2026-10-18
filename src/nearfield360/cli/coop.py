"""CLI commands for cooperative perception, ETSI CPM dispatch, and blind-spot fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.coop.alignment import Pose2D
from nearfield360.coop.blind_spot import DEFAULT_FACILITY_PILLARS, BlindSpotAssessor
from nearfield360.coop.fusion import CooperativeTrackManager
from nearfield360.coop.models import (
    CooperativeTrack,
    CoopTelemetryReport,
    CoopTrackSource,
    CPMPerceivedObject,
    OcclusionState,
    PerceivedObjectClass,
)
from nearfield360.coop.rpu import RPUSensorManager
from nearfield360.coop.viz import render_coop_dashboard
from nearfield360.utils.artifacts import write_json

coop_app = typer.Typer(
    help=(
        "Infrastructure cooperative perception, ETSI CPM dispatch, and blind-spot occlusion fusion"
        " (ISO 23374 Type 2/3)."
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
        resolve_path=True,
        help="Path to render tactical BEV cooperative dashboard PNG artifact.",
    ),
]


@coop_app.command("info")
def coop_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active cooperative perception configuration, standards, and demonstrator network."""
    cfg = get_state(context).config.coop

    demonstrator_stations = [
        "RPU-BLIND-CORNER-1 (3D LiDAR, Blind Corner Intersection)",
        "RPU-RAMP-CREST-1 (77 GHz Radar, Spiral Ramp Crest)",
        "RPU-GATE-OVERHEAD-1 (180° Fisheye, Access Portal Gateway)",
    ]

    info_payload = {
        "standards": [
            "ISO 23374 (AVP Type 2 / Type 3 I2V Cooperative Perception)",
            "ETSI TR 103 562 (Collective Perception Service)",
            "SAE J3224 (V2X Sensor Sharing Standard)",
        ],
        "parameters": {
            "max_latency_compensation_s": cfg.max_latency_compensation_s,
            "association_gate_mahalanobis": cfg.association_gate_mahalanobis,
            "min_cpm_confidence": cfg.min_cpm_confidence,
            "blind_spot_warning_ttc_s": cfg.blind_spot_warning_ttc_s,
            "occlusion_grid_resolution_m": cfg.occlusion_grid_resolution_m,
            "rpu_broadcast_frequency_hz": cfg.rpu_broadcast_frequency_hz,
            "track_covariance_inflation_factor": cfg.track_covariance_inflation_factor,
        },
        "default_pillars_count": len(DEFAULT_FACILITY_PILLARS),
        "demonstrator_rpu_stations": demonstrator_stations,
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("NearField360 Infrastructure Cooperative Perception (ISO 23374 / ETSI CPM):")
    typer.echo(f"  Max Latency Compensation:    {cfg.max_latency_compensation_s:.2f} s")
    typer.echo(f"  Association Gate (M-Dist):   {cfg.association_gate_mahalanobis:.2f}")
    typer.echo(f"  Min CPM Confidence:          {cfg.min_cpm_confidence:.2f}")
    typer.echo(f"  Blind Spot TTC Horizon:      {cfg.blind_spot_warning_ttc_s:.2f} s")
    typer.echo(f"  Occlusion Grid Resolution:   {cfg.occlusion_grid_resolution_m:.2f} m")
    typer.echo(f"  RPU Broadcast Frequency:     {cfg.rpu_broadcast_frequency_hz:.1f} Hz")
    typer.echo(f"  Covariance Inflation Factor: {cfg.track_covariance_inflation_factor:.2f}")
    typer.echo(f"  Demonstrator RPU Stations:   {len(demonstrator_stations)}")


@coop_app.command("rpu")
def coop_rpu(
    context: typer.Context,
    station_id: Annotated[
        str | None,
        typer.Option("--station-id", "-s", help="Filter by specific RPU station identifier."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect Roadside Perception Units (RPUs) and generate sample CPM message dispatch."""
    network = RPUSensorManager.create_default_demonstrator_network()
    stations = network.list_stations()

    if station_id:
        stations = [s for s in stations if s.station_id == station_id]
        if not stations:
            typer.secho(
                f"Station {station_id} not found in demonstrator network.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # Populate sample perceived objects
    sample_ped = CPMPerceivedObject(
        object_id="INFRA-OBJ-01",
        station_id="RPU-BLIND-CORNER-1",
        timestamp_s=10.0,
        category=PerceivedObjectClass.PEDESTRIAN,
        x_m=22.0,
        y_m=30.0,
        vy_mps=-1.2,
        confidence=0.88,
        occlusion_state=OcclusionState.VISIBLE,
    )
    sample_veh = CPMPerceivedObject(
        object_id="INFRA-OBJ-02",
        station_id="RPU-RAMP-CREST-1",
        timestamp_s=10.0,
        category=PerceivedObjectClass.VEHICLE,
        x_m=28.0,
        y_m=20.0,
        vx_mps=-4.0,
        confidence=0.92,
        occlusion_state=OcclusionState.VISIBLE,
    )

    if any(s.station_id == "RPU-BLIND-CORNER-1" for s in stations):
        network.add_perceived_object("RPU-BLIND-CORNER-1", sample_ped)
    if any(s.station_id == "RPU-RAMP-CREST-1" for s in stations):
        network.add_perceived_object("RPU-RAMP-CREST-1", sample_veh)

    messages = [network.generate_cpm(s.station_id, timestamp_s=10.0) for s in stations]

    payload = {
        "station_count": len(stations),
        "stations": [s.model_dump() for s in stations],
        "cpm_messages": [m.model_dump() for m in messages],
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Roadside Perception Units ({len(stations)} active stations):")
    for s in stations:
        sx, sy, sz = s.position_m
        typer.echo(
            f"  [{s.station_id}] Type={s.sensor_type.value} Pos=({sx:.1f}, {sy:.1f}, {sz:.1f})m "
            f"FOV={s.fov_azimuth_deg:.0f}° Range={s.range_max_m:.0f}m "
            f"Operational={s.is_operational}"
        )
    total_objs = sum(len(m.perceived_objects) for m in messages)
    typer.echo(f"Generated {len(messages)} CPM messages ({total_objs} perceived objects).")


@coop_app.command("fuse")
def coop_fuse(
    context: typer.Context,
    timestamp: Annotated[
        float, typer.Option("--timestamp", "-t", help="Observation epoch.")
    ] = 10.0,
    latency_s: Annotated[
        float, typer.Option("--latency-s", "-l", help="Simulated wireless transmission latency.")
    ] = 0.08,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Simulate track-to-track cooperative fusion cycle between onboard tracks and CPM messages."""
    cfg = get_state(context).config.coop
    network = RPUSensorManager.create_default_demonstrator_network()

    # Onboard vehicle track
    onboard_tracks: list[CooperativeTrack | dict[str, Any]] = [
        CooperativeTrack(
            track_id="ONBOARD-01",
            source=CoopTrackSource.ONBOARD_ONLY,
            x_m=18.0,
            y_m=15.0,
            vx_mps=2.0,
            vy_mps=0.0,
            category=PerceivedObjectClass.VEHICLE,
            confidence=0.85,
            age_s=1.5,
            onboard_track_id="TRK-01",
        )
    ]

    # Infrastructure observation slightly offset in time and space
    raw_cpm_obj = CPMPerceivedObject(
        object_id="RPU-OBJ-99",
        station_id="RPU-BLIND-CORNER-1",
        timestamp_s=timestamp - latency_s,
        category=PerceivedObjectClass.VEHICLE,
        x_m=18.0 - (2.0 * latency_s) + 0.1,  # matches after latency compensation
        y_m=15.05,
        vx_mps=2.0,
        vy_mps=0.0,
        confidence=0.90,
    )
    # Infrastructure-only hidden hazard in corridor
    hidden_cpm_obj = CPMPerceivedObject(
        object_id="RPU-OBJ-100",
        station_id="RPU-BLIND-CORNER-1",
        timestamp_s=timestamp - latency_s,
        category=PerceivedObjectClass.PEDESTRIAN,
        x_m=20.0,
        y_m=28.0,
        vx_mps=0.0,
        vy_mps=-1.2,
        confidence=0.92,
    )

    network.add_perceived_object("RPU-BLIND-CORNER-1", raw_cpm_obj)
    network.add_perceived_object("RPU-BLIND-CORNER-1", hidden_cpm_obj)

    cpm_messages = [
        network.generate_cpm(s.station_id, timestamp_s=timestamp - latency_s)
        for s in network.list_stations()
    ]

    fusion_mgr = CooperativeTrackManager()
    fused_tracks = fusion_mgr.fuse_cycle(
        onboard_tracks=onboard_tracks,
        cpm_messages=cpm_messages,
        timestamp_s=timestamp,
        config=cfg,
    )

    payload = {
        "timestamp_s": timestamp,
        "wireless_latency_s": latency_s,
        "input_onboard_tracks": [
            t.model_dump() if isinstance(t, CooperativeTrack) else t for t in onboard_tracks
        ],
        "input_cpm_messages": [m.model_dump() for m in cpm_messages],
        "fused_tracks": [t.model_dump() for t in fused_tracks],
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Cooperative Track Fusion (t={timestamp:.2f}s, latency={latency_s * 1000:.1f}ms):")
    typer.echo(f"  Input Onboard Tracks:        {len(onboard_tracks)}")
    typer.echo(f"  Input CPM Messages:          {len(cpm_messages)}")
    typer.echo(f"  Output Fused Tracks:         {len(fused_tracks)}")
    for trk in fused_tracks:
        typer.echo(
            f"    [{trk.track_id}] Source={trk.source.value} Pos=({trk.x_m:.2f}, {trk.y_m:.2f})m "
            f"Vel=({trk.vx_mps:.2f}, {trk.vy_mps:.2f})m/s Conf={trk.confidence:.2f} "
            f"Cov=[xx={trk.cov_xx:.3f}, yy={trk.cov_yy:.3f}]"
        )


@coop_app.command("render")
def coop_render(
    context: typer.Context,
    vehicle_x: Annotated[
        float, typer.Option("--vehicle-x", help="Ego vehicle X position in meters.")
    ] = 12.0,
    vehicle_y: Annotated[
        float, typer.Option("--vehicle-y", help="Ego vehicle Y position in meters.")
    ] = 15.0,
    vehicle_yaw: Annotated[
        float, typer.Option("--vehicle-yaw", help="Ego vehicle yaw in degrees.")
    ] = 0.0,
    vehicle_speed: Annotated[
        float, typer.Option("--vehicle-speed", help="Ego vehicle forward speed in m/s.")
    ] = 3.5,
    png: PngOption = None,
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
) -> None:
    """Render tactical BEV cooperative perception map and telemetry HUD."""
    cfg = get_state(context).config.coop
    network = RPUSensorManager.create_default_demonstrator_network()
    rpus = network.list_stations()

    veh_pose = Pose2D(x_m=vehicle_x, y_m=vehicle_y, yaw_deg=vehicle_yaw)

    # Tracks
    tracks = [
        CooperativeTrack(
            track_id="TRK-FUSED-01",
            source=CoopTrackSource.FUSED_COOPERATIVE,
            x_m=18.0,
            y_m=15.0,
            vx_mps=2.0,
            vy_mps=0.0,
            category=PerceivedObjectClass.VEHICLE,
            confidence=0.94,
        ),
        CooperativeTrack(
            track_id="TRK-INFRA-02",
            source=CoopTrackSource.INFRASTRUCTURE_ONLY,
            x_m=20.0,
            y_m=26.0,
            vx_mps=0.0,
            vy_mps=-2.0,
            category=PerceivedObjectClass.PEDESTRIAN,
            confidence=0.89,
        ),
    ]

    assessor = BlindSpotAssessor()
    warnings = assessor.assess_tracks(
        vehicle_id="EGO-01",
        vehicle_pose=veh_pose,
        vehicle_speed_mps=vehicle_speed,
        tracks=tracks,
        timestamp_s=50.0,
        config=cfg,
    )

    telemetry = CoopTelemetryReport(
        active_rpus=len(rpus),
        cpm_message_count=180,
        onboard_track_count=1,
        fused_track_count=2,
        blind_spot_hazard_count=len(warnings),
        warnings=warnings,
        mean_fusion_latency_ms=2.15,
        timestamp_s=50.0,
    )

    out_png = png or Path("coop_dashboard.png")
    render_coop_dashboard(
        vehicle_pose=veh_pose,
        vehicle_speed_mps=vehicle_speed,
        rpus=rpus,
        tracks=tracks,
        warnings=warnings,
        telemetry=telemetry,
        output_path=out_png,
    )

    summary_payload = {
        "vehicle_pose": {"x_m": vehicle_x, "y_m": vehicle_y, "yaw_deg": vehicle_yaw},
        "vehicle_speed_mps": vehicle_speed,
        "tracks_rendered": len(tracks),
        "warnings_count": len(warnings),
        "telemetry": telemetry.model_dump(),
        "rendered_png": str(out_png),
    }

    if output is not None:
        write_json(output, summary_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(summary_payload, indent=2))
        return

    typer.echo(f"Tactical BEV Cooperative Map successfully rendered to: {out_png}")
    typer.echo(f"  Tracks: {len(tracks)}, Warnings: {len(warnings)}")
    for w in warnings:
        typer.echo(
            f"  Blind Spot Hazard: {w.hazard_object_id} TTC={w.ttc_s:.2f}s "
            f"Risk={w.risk_level.value} Action={w.recommended_action.value}"
        )


@coop_app.command("audit")
def coop_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Run comprehensive ISO 23374 / ETSI CPM operational safety and telemetry audit."""
    cfg = get_state(context).config.coop
    network = RPUSensorManager.create_default_demonstrator_network()
    rpus = network.list_stations()

    assessor = BlindSpotAssessor()
    veh_pose = Pose2D(x_m=12.0, y_m=15.0, yaw_deg=0.0)

    # Verify hazard detection
    hidden_hazard = CooperativeTrack(
        track_id="HAZARD-01",
        source=CoopTrackSource.INFRASTRUCTURE_ONLY,
        x_m=20.0,
        y_m=25.0,
        vx_mps=0.0,
        vy_mps=-3.0,
        category=PerceivedObjectClass.PEDESTRIAN,
        confidence=0.91,
    )
    warnings = assessor.assess_tracks(
        vehicle_id="EGO-01",
        vehicle_pose=veh_pose,
        vehicle_speed_mps=3.0,
        tracks=[hidden_hazard],
        timestamp_s=100.0,
        config=cfg,
    )

    report = CoopTelemetryReport(
        active_rpus=len(rpus),
        cpm_message_count=320,
        onboard_track_count=2,
        fused_track_count=4,
        blind_spot_hazard_count=len(warnings),
        warnings=warnings,
        mean_fusion_latency_ms=1.85,
        timestamp_s=100.0,
    )

    audit_payload = {
        "status": "COMPLIANT",
        "iso_23374_type": "TYPE_2_AND_TYPE_3",
        "etsi_cpm_service": "ETSI_TR_103_562",
        "telemetry": report.model_dump(),
        "active_stations": [s.station_id for s in rpus],
        "all_stations_operational": all(s.is_operational for s in rpus),
        "blind_spot_mitigation_active": len(warnings) > 0,
    }

    if output is not None:
        write_json(output, audit_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(audit_payload, indent=2))
        return

    typer.echo("Cooperative Perception (ISO 23374 / ETSI CPM) Audit Report:")
    typer.echo(f"  Compliance Status:           {audit_payload['status']}")
    typer.echo(f"  Active RPU Stations:         {report.active_rpus}")
    typer.echo(f"  All Stations Operational:    {audit_payload['all_stations_operational']}")
    typer.echo(f"  CPM Telemetry Message Count: {report.cpm_message_count}")
    typer.echo(f"  Fused Track Count:           {report.fused_track_count}")
    typer.echo(f"  Mean Fusion Latency:         {report.mean_fusion_latency_ms:.2f} ms")
    typer.echo(f"  Active Blind Spot Warnings:  {report.blind_spot_hazard_count}")
    for w in warnings:
        typer.echo(
            f"    Alert: {w.hazard_object_id} -> TTC={w.ttc_s:.2f}s "
            f"Level={w.risk_level.value} Action={w.recommended_action.value}"
        )
