"""CLI commands for automotive acoustic sensing, siren detection, and DoA triangulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.acoustic.detector import SirenDetector, generate_synthetic_siren
from nearfield360.acoustic.doa import (
    AcousticDoAEstimator,
    build_default_microphone_array,
    simulate_multichannel_siren,
)
from nearfield360.acoustic.executive import AcousticExecutive
from nearfield360.acoustic.models import (
    CameraSector,
    SirenType,
    azimuth_to_camera_sector,
)
from nearfield360.acoustic.viz import render_acoustic_dashboard
from nearfield360.cli.state import get_state
from nearfield360.utils.artifacts import write_json

acoustic_app = typer.Typer(
    help="Automotive acoustic sensing, emergency siren detection, and DoA triangulation.",
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
        help="Path to render tactical acoustic HUD dashboard PNG artifact.",
    ),
]


@acoustic_app.command("info")
def acoustic_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect active acoustic sensing configuration and microphone array parameters."""
    config = get_state(context).config.acoustic
    mics = build_default_microphone_array(config.mic_array_radius_m)
    payload: dict[str, Any] = {
        "sampling_rate_hz": config.sampling_rate_hz,
        "sound_speed_mps": config.sound_speed_mps,
        "stft_window_samples": config.stft_window_samples,
        "stft_hop_samples": config.stft_hop_samples,
        "siren_freq_min_hz": config.siren_freq_min_hz,
        "siren_freq_max_hz": config.siren_freq_max_hz,
        "min_siren_snr_db": config.min_siren_snr_db,
        "min_confidence_threshold": config.min_confidence_threshold,
        "mic_array_radius_m": config.mic_array_radius_m,
        "enable_yield_interlock": config.enable_yield_interlock,
        "yield_distance_threshold_m": config.yield_distance_threshold_m,
        "num_microphones": config.num_microphones,
        "microphones": [m.model_dump(mode="json") for m in mics],
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Automotive Acoustic Sensing & DoA Configuration:")
    typer.echo(f"  Sampling Rate              : {payload['sampling_rate_hz']} Hz")
    typer.echo(f"  Sound Speed                : {payload['sound_speed_mps']:.1f} m/s")
    typer.echo(
        f"  STFT Window / Hop          : {payload['stft_window_samples']} / "
        f"{payload['stft_hop_samples']} samples"
    )
    typer.echo(
        f"  Siren Frequency Band       : [{payload['siren_freq_min_hz']:.1f}, "
        f"{payload['siren_freq_max_hz']:.1f}] Hz"
    )
    typer.echo(f"  Min Siren SNR              : {payload['min_siren_snr_db']:.1f} dB")
    typer.echo(f"  Min Confidence Threshold   : {payload['min_confidence_threshold']:.2f}")
    typer.echo(f"  Microphone Array Radius    : {payload['mic_array_radius_m']:.2f} m")
    typer.echo(f"  Number of Microphones      : {payload['num_microphones']}")
    typer.echo(f"  Emergency Yield Interlock  : {payload['enable_yield_interlock']}")
    typer.echo(f"  Yield Distance Threshold   : {payload['yield_distance_threshold_m']:.1f} m")


@acoustic_app.command("detect")
def acoustic_detect(
    context: typer.Context,
    siren_type: Annotated[
        str,
        typer.Option(
            "--siren-type",
            "-s",
            help="Siren profile: 'wail', 'yelp', 'hi_lo', 'horn', 'none'.",
        ),
    ] = "wail",
    duration: Annotated[
        float, typer.Option("--duration-s", help="Signal duration in seconds.")
    ] = 2.0,
    snr: Annotated[float, typer.Option("--snr-db", help="Signal-to-Noise Ratio in dB.")] = 25.0,
    doppler_shift: Annotated[
        float, typer.Option("--doppler-hz", help="Doppler shift in Hz.")
    ] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Analyze acoustic signal and classify emergency siren modulation pattern."""
    config = get_state(context).config.acoustic
    detector = SirenDetector(config)

    parsed_siren_type = SirenType(siren_type.lower())
    audio = generate_synthetic_siren(
        siren_type=parsed_siren_type,
        duration_s=duration,
        sampling_rate_hz=config.sampling_rate_hz,
        snr_db=snr,
        doppler_shift_hz=doppler_shift,
    )

    result = detector.analyze_signal(audio)
    data = result.model_dump(mode="json")

    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    status_tag = "SIREN DETECTED" if result.detected else "NO SIREN"
    typer.echo(f"Acoustic Siren Detection Result: [{status_tag}]")
    typer.echo(f"  Siren Classification : {result.siren_type.value.upper()}")
    typer.echo(f"  Fundamental Frequency: {result.fundamental_freq_hz:.1f} Hz")
    typer.echo(f"  Modulation Cycle Rate: {result.cycle_rate_hz:.2f} Hz")
    typer.echo(f"  Harmonic Energy Ratio: {result.harmonic_energy_ratio * 100:.1f}%")
    typer.echo(f"  Estimated SNR        : {result.snr_db:.1f} dB")
    typer.echo(f"  Confidence           : {result.confidence * 100:.1f}%")


@acoustic_app.command("doa")
def acoustic_doa(
    context: typer.Context,
    azimuth: Annotated[
        float,
        typer.Option("--azimuth-deg", "-a", help="Incident azimuth (-180 to +180 deg)."),
    ] = 45.0,
    siren_type: Annotated[str, typer.Option("--siren-type", "-s", help="Siren profile.")] = "wail",
    snr: Annotated[float, typer.Option("--snr-db", help="Sensor SNR in dB.")] = 25.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Triangulate Direction-of-Arrival (DoA) azimuth using multichannel GCC-PHAT."""
    config = get_state(context).config.acoustic
    mics = build_default_microphone_array(config.mic_array_radius_m)
    estimator = AcousticDoAEstimator(config, mic_specs=mics)

    parsed_siren = SirenType(siren_type.lower())
    source_sig = generate_synthetic_siren(
        siren_type=parsed_siren,
        duration_s=1.0,
        sampling_rate_hz=config.sampling_rate_hz,
    )
    multichannel_audio = simulate_multichannel_siren(
        source_sig,
        azimuth_deg=azimuth,
        mic_specs=mics,
        fs=config.sampling_rate_hz,
        sound_speed_mps=config.sound_speed_mps,
        snr_db=snr,
    )

    doa_res = estimator.estimate_doa(multichannel_audio)
    camera_sec = azimuth_to_camera_sector(doa_res.azimuth_deg)

    payload = doa_res.model_dump(mode="json")
    payload["camera_sector"] = camera_sec.value
    payload["ground_truth_azimuth_deg"] = azimuth
    payload["angular_error_deg"] = round(abs(doa_res.azimuth_deg - azimuth), 2)

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Acoustic Direction-of-Arrival (DoA) Triangulation:")
    typer.echo(f"  Estimated Azimuth    : {doa_res.azimuth_deg:+.1f}°")
    typer.echo(f"  Ground Truth Azimuth : {azimuth:+.1f}°")
    typer.echo(f"  Angular Error        : {payload['angular_error_deg']:.2f}°")
    typer.echo(f"  Matched Camera Sector: {camera_sec.value}")
    typer.echo(f"  Confidence           : {doa_res.confidence * 100:.1f}%")
    typer.echo(f"  Sound Pressure Level : {doa_res.sound_pressure_level_db:.1f} dB SPL")
    typer.echo(f"  Primary Mic Pair     : {doa_res.primary_pair[0]} - {doa_res.primary_pair[1]}")


@acoustic_app.command("simulate")
def acoustic_simulate(
    context: typer.Context,
    siren_type: Annotated[str, typer.Option("--siren-type", "-s", help="Siren profile.")] = "wail",
    azimuth: Annotated[float, typer.Option("--azimuth-deg", "-a", help="Incident azimuth.")] = 45.0,
    speed: Annotated[
        float, typer.Option("--speed-mps", help="Approaching velocity in m/s.")
    ] = 15.0,
    initial_distance: Annotated[
        float, typer.Option("--distance-m", help="Initial distance in m.")
    ] = 60.0,
    frames: Annotated[int, typer.Option("--frames", "-n", help="Number of steps.")] = 5,
    visual: Annotated[
        bool, typer.Option("--visual/--no-visual", help="Simulate visual camera confirmation.")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON report.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Simulate multi-step emergency vehicle approach with Doppler and yield interlock."""
    config = get_state(context).config.acoustic
    mics = build_default_microphone_array(config.mic_array_radius_m)
    exec_engine = AcousticExecutive(config)

    parsed_siren = SirenType(siren_type.lower())
    matched_sec = azimuth_to_camera_sector(azimuth)

    distance = initial_distance
    t = 1.0
    last_report = None

    for _f_idx in range(frames):
        # Doppler shift: delta_f = f0 * (v / c)
        f0 = 1000.0 if parsed_siren in (SirenType.WAIL, SirenType.YELP) else 500.0
        doppler_shift = f0 * (speed / config.sound_speed_mps) if speed > 0 else 0.0

        source_sig = generate_synthetic_siren(
            siren_type=parsed_siren,
            duration_s=1.0,
            sampling_rate_hz=config.sampling_rate_hz,
            doppler_shift_hz=doppler_shift,
        )
        audio = simulate_multichannel_siren(
            source_sig,
            azimuth_deg=azimuth,
            mic_specs=mics,
            fs=config.sampling_rate_hz,
            sound_speed_mps=config.sound_speed_mps,
            snr_db=25.0,
        )

        vis_sectors = [matched_sec] if (visual and distance <= 40.0) else []
        last_report = exec_engine.process_audio(
            multichannel_audio=audio,
            timestamp_s=t,
            visual_camera_sectors=vis_sectors,
            estimated_distance_m=distance,
        )

        distance = max(5.0, distance - speed * 1.0)
        t += 1.0

    if last_report is None:
        typer.echo("Error: simulation produced no reports.", err=True)
        raise typer.Exit(1)

    if png is not None:
        render_acoustic_dashboard(last_report, output_path=png)

    data = last_report.model_dump(mode="json")
    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo(f"Acoustic Emergency Simulation Summary: [{parsed_siren.value.upper()}]")
    typer.echo(f"  Doppler State            : {last_report.doppler.value.upper()}")
    typer.echo(f"  Estimated Radial Speed   : {last_report.estimated_radial_speed_mps:+.1f} m/s")
    typer.echo(f"  Yield Interlock State    : {last_report.yield_state.value.upper()}")
    typer.echo(f"  Emergency Interlock Active: {last_report.is_emergency_active()}")
    if last_report.sector_match:
        typer.echo(f"  Correlated Camera Sector : {last_report.sector_match.camera_sector.value}")
        typer.echo(f"  Visual Confirmation      : {last_report.sector_match.visual_confirmation}")
        typer.echo(
            f"  Fused Confidence         : {last_report.sector_match.fused_confidence * 100:.1f}%"
        )
    typer.echo(f"  Operational Message      : {last_report.summary_message}")


@acoustic_app.command("render")
def acoustic_render(
    context: typer.Context,
    png: Annotated[
        Path,
        typer.Option(
            "--png",
            "-p",
            resolve_path=True,
            help="Destination path for 1280x720 PNG HUD dashboard.",
        ),
    ] = Path("artifacts/acoustic_hud.png"),
    siren_type: Annotated[str, typer.Option("--siren-type", help="Siren profile.")] = "wail",
    azimuth: Annotated[float, typer.Option("--azimuth-deg", help="Azimuth in deg.")] = 45.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON status.")] = False,
    output: OutputOption = None,
) -> None:
    """Render a tactical 1280x720 acoustic HUD dashboard PNG artifact."""
    config = get_state(context).config.acoustic
    mics = build_default_microphone_array(config.mic_array_radius_m)
    exec_engine = AcousticExecutive(config)

    parsed_siren = SirenType(siren_type.lower())
    matched_sec = azimuth_to_camera_sector(azimuth)

    # Prime anchor frame
    sig_1 = generate_synthetic_siren(parsed_siren, duration_s=1.0, doppler_shift_hz=0.0)
    audio_1 = simulate_multichannel_siren(sig_1, azimuth_deg=azimuth, mic_specs=mics)
    exec_engine.process_audio(audio_1, timestamp_s=1.0)

    # Approaching frame with Doppler
    sig_2 = generate_synthetic_siren(parsed_siren, duration_s=1.0, doppler_shift_hz=50.0)
    audio_2 = simulate_multichannel_siren(sig_2, azimuth_deg=azimuth, mic_specs=mics)
    report = exec_engine.process_audio(
        audio_2,
        timestamp_s=2.0,
        visual_camera_sectors=[matched_sec],
        estimated_distance_m=20.0,
    )

    render_acoustic_dashboard(report, output_path=png)

    status_data = {
        "rendered_path": str(png),
        "siren_type": report.siren_detection.siren_type.value,
        "yield_state": report.yield_state.value,
        "is_emergency_active": report.is_emergency_active(),
        "status": "rendered",
    }

    if output is not None:
        write_json(output, status_data)

    if as_json:
        typer.echo(json.dumps(status_data, indent=2))
        return

    typer.echo(f"Successfully rendered acoustic HUD dashboard to: {png}")


@acoustic_app.command("audit")
def acoustic_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON audit results.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit vehicle acoustic emergency sensing compliance and safety interlocks."""
    config = get_state(context).config.acoustic

    f_min = config.siren_freq_min_hz
    f_max = config.siren_freq_max_hz
    r_mic = config.mic_array_radius_m
    n_mic = config.num_microphones
    win_sz = config.stft_window_samples
    hop_sz = config.stft_hop_samples
    y_dist = config.yield_distance_threshold_m

    checks: list[dict[str, Any]] = [
        {
            "name": "acoustic_siren_spectral_bounds_configured",
            "passed": 200.0 <= f_min < f_max <= 3500.0,
            "detail": f"siren_band=[{f_min:.1f}, {f_max:.1f}] Hz",
        },
        {
            "name": "acoustic_doa_array_geometry_configured",
            "passed": r_mic >= 0.10 and n_mic >= 2,
            "detail": f"array_radius={r_mic:.2f}m, num_mics={n_mic}",
        },
        {
            "name": "stft_resolution_conformance",
            "passed": win_sz >= 128 and hop_sz <= win_sz // 2,
            "detail": f"win_size={win_sz}, hop_size={hop_sz}",
        },
        {
            "name": "emergency_yield_interlock_enabled",
            "passed": config.enable_yield_interlock and y_dist >= 10.0,
            "detail": f"interlock={config.enable_yield_interlock}, threshold={y_dist:.1f}m",
        },
        {
            "name": "audio_visual_sector_mapping_conformance",
            "passed": len(CameraSector) == 4,
            "detail": "all 4 surround sectors (FV, MVR, RV, MVL) mapped to 360° azimuth",
        },
    ]

    all_passed = all(c["passed"] for c in checks)
    audit_report = {
        "standard": "Automotive Acoustic Emergency Siren Sensing & Audio-Visual Yield Interlock",
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
    typer.echo(f"Acoustic Emergency Sensing Regulatory Audit: [{status_tag}]")
    for c in checks:
        icon = "[PASS]" if c["passed"] else "[FAIL]"
        typer.echo(f"  {icon} {c['name']}: {c['detail']}")
