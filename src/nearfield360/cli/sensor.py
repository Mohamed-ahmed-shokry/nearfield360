"""CLI commands for synthetic radar and ultrasonic sensor simulation suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.sensor.cfar import CACFARDetector
from nearfield360.sensor.fusion import CrossModalSensorFusion
from nearfield360.sensor.models import RadarTarget
from nearfield360.sensor.radar import FMCWRadarSimulator
from nearfield360.sensor.ultrasonic import UltrasonicArraySimulator, calculate_speed_of_sound
from nearfield360.sensor.viz import render_sensor_dashboard
from nearfield360.utils.artifacts import write_json

sensor_app = typer.Typer(
    help="Synthetic FMCW radar and ultrasonic transducer array simulation suite.",
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
        help="Path to render tactical sensor dashboard PNG artifact.",
    ),
]


@sensor_app.command("info")
def sensor_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active radar transceiver and ultrasonic array configuration parameters."""
    config = get_state(context).config.sensor

    payload: dict[str, Any] = {
        "carrier_frequency_ghz": config.carrier_frequency_ghz,
        "sweep_bandwidth_mhz": config.sweep_bandwidth_mhz,
        "chirp_duration_us": config.chirp_duration_us,
        "num_chirps": config.num_chirps,
        "samples_per_chirp": config.samples_per_chirp,
        "cfar_guard_cells": config.cfar_guard_cells,
        "cfar_train_cells": config.cfar_train_cells,
        "cfar_threshold_factor": config.cfar_threshold_factor,
        "radar_max_range_m": config.radar_max_range_m,
        "ultrasonic_frequency_khz": config.ultrasonic_frequency_khz,
        "ultrasonic_beam_angle_deg": config.ultrasonic_beam_angle_deg,
        "ultrasonic_min_range_m": config.ultrasonic_min_range_m,
        "ultrasonic_max_range_m": config.ultrasonic_max_range_m,
        "speed_of_sound_mps": config.speed_of_sound_mps,
        "num_ultrasonic_sensors": config.num_ultrasonic_sensors,
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("NearField360 Sensor Simulation Suite Configuration:")
    typer.echo(f"  Radar Carrier Frequency     : {config.carrier_frequency_ghz:.1f} GHz")
    typer.echo(f"  Sweep Bandwidth             : {config.sweep_bandwidth_mhz:.1f} MHz")
    chirp_desc = f"{config.num_chirps} chirps x {config.samples_per_chirp} samples"
    typer.echo(f"  Chirp Configuration         : {chirp_desc}")
    typer.echo(f"  Chirp Duration              : {config.chirp_duration_us:.1f} us")
    cfar_cells = f"{config.cfar_guard_cells} / {config.cfar_train_cells}"
    typer.echo(f"  CA-CFAR Guard / Train Cells : {cfar_cells}")
    typer.echo(f"  CA-CFAR Threshold Factor    : {config.cfar_threshold_factor:.2f}")
    typer.echo(f"  Max Radar Range             : {config.radar_max_range_m:.1f} m")
    typer.echo(f"  Ultrasonic Transducer Count : {config.num_ultrasonic_sensors}")
    typer.echo(f"  Ultrasonic Frequency        : {config.ultrasonic_frequency_khz:.1f} kHz")
    typer.echo(f"  Ultrasonic Beam Cone Angle  : {config.ultrasonic_beam_angle_deg:.1f} deg")
    range_desc = f"{config.ultrasonic_min_range_m:.2f}m - {config.ultrasonic_max_range_m:.2f}m"
    typer.echo(f"  Ultrasonic Operating Range  : {range_desc}")
    typer.echo(f"  Nominal Speed of Sound      : {config.speed_of_sound_mps:.1f} m/s")


@sensor_app.command("radar")
def sensor_radar(
    context: typer.Context,
    target_range: Annotated[float, typer.Option("--range", "-r", help="Target range (m).")] = 12.0,
    target_velocity: Annotated[
        float, typer.Option("--velocity", "-v", help="Radial velocity (m/s).")
    ] = -2.5,
    target_rcs: Annotated[float, typer.Option("--rcs", help="RCS (dBm^2).")] = 15.0,
    noise_db: Annotated[float, typer.Option("--noise-db", help="Noise floor (dB).")] = -45.0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Simulate FMCW radar chirp transmission, 2D FFT, and CA-CFAR point cloud detection."""
    config = get_state(context).config.sensor
    sim = FMCWRadarSimulator(config)
    detector = CACFARDetector.from_config(config)

    target = RadarTarget(
        range_m=target_range,
        radial_velocity_mps=target_velocity,
        rcs_dbm2=target_rcs,
    )
    baseband = sim.simulate_baseband([target], noise_level_db=noise_db, seed=seed)
    power_db, range_axis_m, doppler_axis_mps = sim.compute_range_doppler(baseband)
    cloud = detector.detect(power_db, range_axis_m, doppler_axis_mps)

    payload = cloud.model_dump()
    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("FMCW Radar 2D FFT & CA-CFAR Detection Result:")
    typer.echo(f"  Detected Targets: {len(cloud.points)}")
    typer.echo(f"  Noise Floor     : {cloud.noise_floor_db:.1f} dB")
    for pt in cloud.points:
        line = (
            f"  -> Range: {pt.range_m:.2f} m | Velocity: {pt.radial_velocity_mps:.2f} m/s | "
            f"SNR: {pt.snr_db:.1f} dB | RCS: {pt.rcs_dbm2:.1f} dBm^2"
        )
        typer.echo(line)


@sensor_app.command("ultrasonic")
def sensor_ultrasonic(
    context: typer.Context,
    front_dist: Annotated[
        float, typer.Option("--front-dist", help="Front obstacle distance (m).")
    ] = 0.85,
    rear_dist: Annotated[
        float, typer.Option("--rear-dist", help="Rear obstacle distance (m).")
    ] = 1.20,
    temp_c: Annotated[float, typer.Option("--temp", help="Ambient air temperature (C).")] = 20.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Simulate perimeter ultrasonic acoustic array echoes and obstacle distances."""
    config = get_state(context).config.sensor
    sim = UltrasonicArraySimulator(config)

    # Front obstacle: x = 3.8 + front_dist, y = 0.3
    # Rear obstacle: x = -1.0 - rear_dist, y = 0.3
    obstacles = [(3.8 + front_dist, 0.3), (-1.0 - rear_dist, 0.3)]
    frame = sim.simulate_echoes(obstacles, ambient_temp_c=temp_c)

    payload = frame.model_dump()
    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    active_echoes = [e for e in frame.echoes if e.in_detection_cone]
    typer.echo("Perimeter Ultrasonic Transducer Array Echoes:")
    typer.echo(f"  Total Sensors     : {len(frame.echoes)}")
    typer.echo(f"  Active Detections : {len(active_echoes)}")
    typer.echo(f"  Speed of Sound    : {calculate_speed_of_sound(temp_c):.1f} m/s")
    for echo in active_echoes:
        typer.echo(
            f"  -> {echo.sensor_id:<22} : "
            f"Range = {echo.range_m:.2f} m | Amp = {echo.echo_amplitude:.2f}"
        )


@sensor_app.command("fuse")
def sensor_fuse(
    context: typer.Context,
    target_range: Annotated[
        float, typer.Option("--range", "-r", help="Radar target range (m).")
    ] = 10.0,
    target_velocity: Annotated[
        float, typer.Option("--velocity", "-v", help="Radar radial velocity (m/s).")
    ] = -1.8,
    front_uss_dist: Annotated[
        float, typer.Option("--uss-dist", help="Front USS distance (m).")
    ] = 1.05,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Simulate and cross-modally fuse radar point clouds with ultrasonic array echoes."""
    config = get_state(context).config.sensor
    r_sim = FMCWRadarSimulator(config)
    detector = CACFARDetector.from_config(config)
    u_sim = UltrasonicArraySimulator(config)
    fusion = CrossModalSensorFusion(config)

    t = RadarTarget(range_m=target_range, radial_velocity_mps=target_velocity, rcs_dbm2=15.0)
    baseband = r_sim.simulate_baseband([t], seed=42)
    power_db, r_axis, v_axis = r_sim.compute_range_doppler(baseband)
    cloud = detector.detect(power_db, r_axis, v_axis)

    u_frame = u_sim.simulate_echoes([(3.8 + front_uss_dist, 0.3)])

    report = fusion.fuse(cloud, u_frame)
    payload = report.model_dump()
    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Cross-Modal Radar-Ultrasonic Fused Obstacles:")
    typer.echo(f"  Fused Obstacles   : {report.fused_obstacles_count}")
    for obs in report.fused_obstacles:
        mods = ", ".join(obs.source_modalities)
        typer.echo(
            f"  -> {obs.obstacle_id}: ({obs.x_m:.2f}, {obs.y_m:.2f})m | "
            f"Dist: {obs.min_range_m:.2f}m | Vel: {obs.radial_velocity_mps:.2f} m/s | "
            f"Modalities: [{mods}] | Conf: {obs.confidence:.2f}"
        )


@sensor_app.command("render")
def sensor_render(
    context: typer.Context,
    target_range: Annotated[
        float, typer.Option("--range", "-r", help="Radar target range (m).")
    ] = 14.0,
    target_velocity: Annotated[
        float, typer.Option("--velocity", "-v", help="Radar radial velocity (m/s).")
    ] = -3.2,
    front_uss_dist: Annotated[
        float, typer.Option("--uss-dist", help="Front USS distance (m).")
    ] = 0.85,
    png: Annotated[
        Path,
        typer.Option(
            "--png",
            "-p",
            resolve_path=True,
            help="Path to render tactical dashboard PNG artifact.",
        ),
    ] = Path("outputs/sensor_dashboard.png"),
) -> None:
    """Render 1280x720 tactical FMCW Range-Doppler spectrum and ultrasonic BEV dashboard."""
    config = get_state(context).config.sensor
    r_sim = FMCWRadarSimulator(config)
    detector = CACFARDetector.from_config(config)
    u_sim = UltrasonicArraySimulator(config)
    fusion = CrossModalSensorFusion(config)

    t = RadarTarget(range_m=target_range, radial_velocity_mps=target_velocity, rcs_dbm2=18.0)
    baseband = r_sim.simulate_baseband([t], seed=42)
    power_db, r_axis, v_axis = r_sim.compute_range_doppler(baseband)
    cloud = detector.detect(power_db, r_axis, v_axis)
    rd_map = r_sim.generate_map([t], seed=42)

    u_frame = u_sim.simulate_echoes([(3.8 + front_uss_dist, 0.3), (-1.0 - 0.75, 0.0)])
    report = fusion.fuse(cloud, u_frame)

    render_sensor_dashboard(report, rd_map=rd_map, output_path=png)
    typer.echo(f"Rendered sensor dashboard PNG: {png}")


@sensor_app.command("audit")
def sensor_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Run comprehensive verification audit of radar physics and ultrasonic acoustic engines."""
    config = get_state(context).config.sensor
    r_sim = FMCWRadarSimulator(config)
    detector = CACFARDetector.from_config(config)
    u_sim = UltrasonicArraySimulator(config)
    fusion = CrossModalSensorFusion(config)

    # 1. FMCW Radar Physics & Peak Detection
    t_ref = RadarTarget(range_m=10.0, radial_velocity_mps=-2.0, rcs_dbm2=15.0)
    baseband = r_sim.simulate_baseband([t_ref], noise_level_db=-50.0, seed=42)
    p_db, r_axis, v_axis = r_sim.compute_range_doppler(baseband)
    cloud = detector.detect(p_db, r_axis, v_axis)
    radar_passed = len(cloud.points) > 0 and abs(cloud.points[0].range_m - 10.0) < 0.7

    # 2. Ultrasonic Acoustic ToF & Directivity Cone
    u_frame = u_sim.simulate_echoes([(4.85, 0.3)])
    uss_active = [e for e in u_frame.echoes if e.in_detection_cone]
    uss_passed = len(uss_active) > 0 and any(abs(e.range_m - 1.0) < 0.1 for e in uss_active)

    # 3. Cross-Modal Fusion Integrity
    report = fusion.fuse(cloud, u_frame)
    fusion_passed = report.fused_obstacles_count > 0

    audit_payload: dict[str, Any] = {
        "status": "PASS" if (radar_passed and uss_passed and fusion_passed) else "FAIL",
        "fmcw_radar_77ghz": {
            "passed": radar_passed,
            "carrier_frequency_ghz": config.carrier_frequency_ghz,
            "bandwidth_mhz": config.sweep_bandwidth_mhz,
            "detected_points": len(cloud.points),
        },
        "ultrasonic_array": {
            "passed": uss_passed,
            "num_sensors": len(u_frame.echoes),
            "active_echoes": len(uss_active),
            "speed_of_sound_mps": calculate_speed_of_sound(20.0),
        },
        "cross_modal_fusion": {
            "passed": fusion_passed,
            "fused_obstacles": report.fused_obstacles_count,
        },
    }

    if output is not None:
        write_json(output, audit_payload)

    if as_json:
        typer.echo(json.dumps(audit_payload, indent=2))
        return

    typer.echo(f"Sensor Suite Audit Status: {audit_payload['status']}")
    typer.echo(f"  FMCW Radar         : {'PASS' if radar_passed else 'FAIL'}")
    typer.echo(f"  Ultrasonic Array   : {'PASS' if uss_passed else 'FAIL'}")
    typer.echo(f"  Cross-Modal Fusion : {'PASS' if fusion_passed else 'FAIL'}")
