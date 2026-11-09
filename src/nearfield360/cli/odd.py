"""CLI commands for ISO 21448 SOTIF / ISO 23374 Dynamic ODD Supervisory Monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.odd.evaluator import ODDEvaluator
from nearfield360.odd.models import (
    ODDMetrics,
    ODDReport,
    ODDState,
    SensorHealthStatus,
)
from nearfield360.odd.supervisor import ODDSupervisor
from nearfield360.odd.viz import render_odd_dashboard
from nearfield360.utils.artifacts import write_json

odd_app = typer.Typer(
    help="ISO 21448 SOTIF / ISO 23374 Dynamic ODD Supervisory Monitoring and Safe Fallback.",
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
        help="Path to render tactical ODD HUD dashboard PNG artifact.",
    ),
]


@odd_app.command("info")
def odd_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Inspect active ODD configuration thresholds, bounds, and governor limits."""
    config = get_state(context).config.odd
    data = {
        "min_ambient_lux": config.min_ambient_lux,
        "max_glare_ratio": config.max_glare_ratio,
        "min_friction_coeff": config.min_friction_coeff,
        "critical_friction_coeff": config.critical_friction_coeff,
        "max_v2x_latency_ms": config.max_v2x_latency_ms,
        "critical_v2x_latency_ms": config.critical_v2x_latency_ms,
        "max_camera_occlusion_ratio": config.max_camera_occlusion_ratio,
        "critical_camera_occlusion_ratio": config.critical_camera_occlusion_ratio,
        "nominal_speed_limit_mps": config.nominal_speed_limit_mps,
        "advisory_speed_limit_mps": config.advisory_speed_limit_mps,
        "restricted_speed_limit_mps": config.restricted_speed_limit_mps,
        "degradation_confirm_frames": config.degradation_confirm_frames,
        "restoration_confirm_frames": config.restoration_confirm_frames,
        "enable_mrm_interlock": config.enable_mrm_interlock,
    }

    if output is not None:
        write_json(output, data)

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo("Dynamic ODD Supervisory Monitor Configuration (ISO 21448 / ISO 23374):")
    typer.echo(f"  Ambient Lux Threshold:      >= {config.min_ambient_lux:.1f} lux")
    typer.echo(f"  Max Glare Ratio:            <= {config.max_glare_ratio:.2f}")
    fric_str = (
        f"Nominal >= {config.min_friction_coeff:.2f}, "
        f"Critical >= {config.critical_friction_coeff:.2f}"
    )
    typer.echo(f"  Surface Friction Limits:    {fric_str}")
    lat_str = (
        f"Nominal <= {config.max_v2x_latency_ms:.0f} ms, "
        f"Critical <= {config.critical_v2x_latency_ms:.0f} ms"
    )
    typer.echo(f"  V2X Latency Bounds:         {lat_str}")
    occ_str = (
        f"Nominal <= {config.max_camera_occlusion_ratio:.2f}, "
        f"Critical <= {config.critical_camera_occlusion_ratio:.2f}"
    )
    typer.echo(f"  Camera Occlusion Limits:    {occ_str}")
    speed_str = (
        f"Nominal {config.nominal_speed_limit_mps:.1f} m/s, "
        f"Advisory {config.advisory_speed_limit_mps:.1f} m/s, "
        f"Restricted {config.restricted_speed_limit_mps:.1f} m/s"
    )
    typer.echo(f"  Governor Speed Limits:      {speed_str}")
    deg_f = config.degradation_confirm_frames
    res_f = config.restoration_confirm_frames
    typer.echo(f"  Hysteresis Frames:          Degrade {deg_f}, Restore {res_f}")
    mrm_str = "Enabled" if config.enable_mrm_interlock else "Disabled"
    typer.echo(f"  ASIL-D MRM Interlock:       {mrm_str}")


@odd_app.command("evaluate")
def odd_evaluate(
    context: typer.Context,
    lux: Annotated[float, typer.Option("--lux", help="Ambient lux.")] = 500.0,
    glare: Annotated[float, typer.Option("--glare", help="Glare ratio [0, 1].")] = 0.02,
    friction: Annotated[float, typer.Option("--friction", help="Friction coeff [0, 1].")] = 0.85,
    rain_mmph: Annotated[float, typer.Option("--rain-mmph", help="Rain rate mm/h.")] = 0.0,
    v2x_latency_ms: Annotated[float, typer.Option("--latency-ms", help="V2X latency ms.")] = 45.0,
    packet_loss: Annotated[float, typer.Option("--packet-loss", help="V2X loss [0, 1].")] = 0.0,
    camera_occlusion: Annotated[float, typer.Option("--occlusion", help="Occlusion ratio.")] = 0.05,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate instantaneous multi-factor environmental and sensor telemetry against ODD."""
    config = get_state(context).config.odd
    evaluator = ODDEvaluator(config)

    metrics = ODDMetrics(
        ambient_lux=lux,
        glare_ratio=glare,
        friction_coeff=friction,
        precipitation_rate_mmph=rain_mmph,
        v2x_latency_ms=v2x_latency_ms,
        v2x_packet_loss_ratio=packet_loss,
        max_camera_occlusion=camera_occlusion,
    )
    sensors = {
        "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=camera_occlusion),
        "RV": SensorHealthStatus(
            sensor_id="RV", is_healthy=True, occlusion_ratio=camera_occlusion * 0.5
        ),
        "MVL": SensorHealthStatus(
            sensor_id="MVL", is_healthy=True, occlusion_ratio=camera_occlusion * 0.3
        ),
        "MVR": SensorHealthStatus(
            sensor_id="MVR", is_healthy=True, occlusion_ratio=camera_occlusion * 0.3
        ),
    }

    state, limiters, illum, precip, surf = evaluator.evaluate_instantaneous(metrics, sensors)
    speed_map = {
        ODDState.NOMINAL: config.nominal_speed_limit_mps,
        ODDState.ADVISORY_DEGRADED: config.advisory_speed_limit_mps,
        ODDState.CRITICAL_RESTRICTED: config.restricted_speed_limit_mps,
        ODDState.ODD_BREACH_MRM: 0.0,
    }
    headway_map = {
        ODDState.NOMINAL: 1.0,
        ODDState.ADVISORY_DEGRADED: 1.5,
        ODDState.CRITICAL_RESTRICTED: 2.0,
        ODDState.ODD_BREACH_MRM: 3.0,
    }

    payload: dict[str, Any] = {
        "state": state.value,
        "illumination": illum.value,
        "precipitation": precip.value,
        "surface": surf.value,
        "active_limiters": limiters,
        "governor_speed_limit_mps": speed_map[state],
        "safety_headway_multiplier": headway_map[state],
        "mrm_required": (state == ODDState.ODD_BREACH_MRM),
        "metrics": metrics.model_dump(),
    }

    if output is not None:
        write_json(output, payload)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"ODD Operating Tier:          {state.value.upper()}")
    typer.echo(f"Illumination Condition:      {illum.value}")
    typer.echo(f"Precipitation Condition:     {precip.value}")
    typer.echo(f"Surface Friction Condition:  {surf.value}")
    speed_val = speed_map[state]
    typer.echo(f"Governor Speed Limit:        {speed_val:.1f} m/s ({speed_val * 3.6:.1f} km/h)")
    typer.echo(f"Safety Headway Multiplier:   {headway_map[state]:.2f}x")
    mrm_flag = "YES" if state == ODDState.ODD_BREACH_MRM else "NO"
    typer.echo(f"MRM Safe Stop Required:      {mrm_flag}")
    lim_str = ", ".join(limiters) if limiters else "None"
    typer.echo(f"Active Limiting Factors:     {lim_str}")


@odd_app.command("simulate")
def odd_simulate(
    context: typer.Context,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            "-s",
            help="Simulation scenario: nominal, light_rain, heavy_rain_glare, black_ice_breach",
        ),
    ] = "heavy_rain_glare",
    steps: Annotated[int, typer.Option("--steps", help="Simulation frame count.")] = 20,
    png: PngOption = None,
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON report.")] = False,
) -> None:
    """Simulate dynamic environmental progression and verify supervisory adaptation."""
    config = get_state(context).config.odd
    supervisor = ODDSupervisor(config)

    for step_idx in range(steps):
        t = step_idx * 0.1
        if scenario == "nominal":
            metrics = ODDMetrics(ambient_lux=600.0, friction_coeff=0.85)
            sensors = {
                "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=0.02),
                "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.03),
            }
        elif scenario == "light_rain":
            metrics = ODDMetrics(ambient_lux=80.0, friction_coeff=0.60, precipitation_rate_mmph=5.0)
            sensors = {
                "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=0.15),
                "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.10),
            }
        elif scenario == "black_ice_breach":
            friction = 0.85 if step_idx < 5 else 0.10
            metrics = ODDMetrics(ambient_lux=10.0, friction_coeff=friction)
            sensors = {
                "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=0.05),
                "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.05),
            }
        elif scenario == "camera_failure":
            occ = 0.05 if step_idx < 5 else 0.85
            metrics = ODDMetrics(ambient_lux=500.0, friction_coeff=0.85)
            sensors = {
                "FV": SensorHealthStatus(
                    sensor_id="FV", is_healthy=(step_idx < 5), occlusion_ratio=occ
                ),
                "RV": SensorHealthStatus(
                    sensor_id="RV", is_healthy=(step_idx < 5), occlusion_ratio=occ
                ),
            }
        else:  # heavy_rain_glare
            rain = 0.0 if step_idx < 4 else (15.0 if step_idx < 12 else 20.0)
            glare = 0.02 if step_idx < 6 else 0.35
            friction = 0.85 if step_idx < 4 else 0.45
            metrics = ODDMetrics(
                ambient_lux=250.0,
                glare_ratio=glare,
                friction_coeff=friction,
                precipitation_rate_mmph=rain,
                v2x_latency_ms=65.0,
            )
            sensors = {
                "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=0.10),
                "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.08),
            }

        supervisor.update(metrics, sensors=sensors, timestamp_s=t)

    report = supervisor.generate_report(report_id=f"SIM-{scenario.upper()}")

    if png is not None:
        render_odd_dashboard(report, png)
        typer.echo(f"Rendered tactical ODD HUD dashboard to {png}")

    report_dict = report.model_dump()
    if output is not None:
        write_json(output, report_dict)

    if as_json:
        typer.echo(json.dumps(report_dict, indent=2))
        return

    latest = report.latest_evaluation
    typer.echo(f"Simulation completed ({steps} steps, scenario='{scenario}'):")
    typer.echo(f"  Final ODD State:            {latest.current_state.value.upper()}")
    typer.echo(f"  Governed Speed Limit:       {latest.governor_speed_limit_mps:.1f} m/s")
    typer.echo(f"  Safety Headway Multiplier:  {latest.safety_headway_multiplier:.2f}x")
    mrm_str = "YES" if latest.mrm_required else "NO"
    typer.echo(f"  MRM Safe Stop Required:     {mrm_str}")
    typer.echo(f"  Total MRM Invocations:      {report.mrm_triggered_count}")
    lim_str = ", ".join(latest.active_limiters) if latest.active_limiters else "None"
    typer.echo(f"  Active Limiting Factors:    {lim_str}")


@odd_app.command("render")
def odd_render(
    input_report: Annotated[
        Path, typer.Option("--input", "-i", resolve_path=True, help="Input ODD report JSON.")
    ],
    output_png: Annotated[
        Path, typer.Option("--output", "-o", resolve_path=True, help="Target PNG destination.")
    ],
) -> None:
    """Render an existing ODD report JSON to tactical 1280x720 HUD PNG."""
    if not input_report.is_file():
        typer.echo(f"Error: report file not found: {input_report}", err=True)
        raise typer.Exit(code=1)

    raw_data = json.loads(input_report.read_text(encoding="utf-8"))
    report = ODDReport.model_validate(raw_data)
    render_odd_dashboard(report, output_png)
    typer.echo(f"Rendered ODD HUD dashboard from {input_report} to {output_png}")


@odd_app.command("audit")
def odd_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit ODD configuration, thresholds, and safety interlock readiness under ISO 21448."""
    config = get_state(context).config.odd
    crit_f = config.critical_friction_coeff
    min_f = config.min_friction_coeff
    lat_max = config.max_v2x_latency_ms
    lat_crit = config.critical_v2x_latency_ms
    spd_rest = config.restricted_speed_limit_mps
    spd_adv = config.advisory_speed_limit_mps
    spd_nom = config.nominal_speed_limit_mps

    checks: list[dict[str, Any]] = [
        {
            "name": "min_lux_threshold",
            "passed": config.min_ambient_lux > 0.0,
            "detail": f"{config.min_ambient_lux:.1f} lux",
        },
        {
            "name": "max_glare_ratio",
            "passed": 0.0 < config.max_glare_ratio < 1.0,
            "detail": f"{config.max_glare_ratio:.2f}",
        },
        {
            "name": "friction_consistency",
            "passed": crit_f < min_f,
            "detail": f"critical={crit_f:.2f} < min={min_f:.2f}",
        },
        {
            "name": "latency_consistency",
            "passed": lat_max < lat_crit,
            "detail": f"max={lat_max:.0f}ms < critical={lat_crit:.0f}ms",
        },
        {
            "name": "speed_limits_hierarchy",
            "passed": spd_rest < spd_adv < spd_nom,
            "detail": f"{spd_rest:.1f} < {spd_adv:.1f} < {spd_nom:.1f} m/s",
        },
        {
            "name": "hysteresis_damping",
            "passed": config.restoration_confirm_frames > config.degradation_confirm_frames,
            "detail": (
                f"restore={config.restoration_confirm_frames} > "
                f"degrade={config.degradation_confirm_frames}"
            ),
        },
        {
            "name": "mrm_safety_interlock",
            "passed": config.enable_mrm_interlock is True,
            "detail": "ASIL-D emergency stop interlock enabled",
        },
    ]

    all_passed = all(c["passed"] for c in checks)
    audit_data = {
        "status": "pass" if all_passed else "fail",
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }

    if output is not None:
        write_json(output, audit_data)

    if as_json:
        typer.echo(json.dumps(audit_data, indent=2))
        return

    for c in checks:
        status_tag = "[PASS]" if c["passed"] else "[FAIL]"
        typer.echo(f"{status_tag} {c['name']}: {c['detail']}")

    if all_passed:
        typer.echo("Dynamic ODD supervisory monitor audit passed.")
    else:
        typer.echo("Dynamic ODD supervisory monitor audit failed.", err=True)
        raise typer.Exit(code=1)
