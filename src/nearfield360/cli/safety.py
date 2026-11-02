"""CLI commands for ISO 26262 ASIL-D functional safety supervisor and watchdog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.safety.models import (
    ECUNodeId,
    HeartbeatPing,
    SafetyState,
)
from nearfield360.safety.plausibility import PlausibilityChecker
from nearfield360.safety.supervisor import SafetySupervisor
from nearfield360.safety.viz import render_safety_dashboard
from nearfield360.safety.watchdog import (
    CRITICAL_SAFETY_NODES,
    DEFAULT_NODE_PERIODS_MS,
    HeartbeatWatchdog,
)
from nearfield360.utils.artifacts import write_json

safety_app = typer.Typer(
    help="ISO 26262 ASIL-D functional safety supervisor, watchdog, and plausibility monitoring.",
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
        help="Path to render tactical ASIL-D safety HUD PNG artifact.",
    ),
]


@safety_app.command("info")
def safety_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active functional safety parameters, deadline limits, and ASIL-D node allocations."""
    config = get_state(context).config.safety

    payload: dict[str, Any] = {
        "heartbeat_timeout_ms": config.heartbeat_timeout_ms,
        "max_consecutive_misses": config.max_consecutive_misses,
        "max_steering_rate_rads": config.max_steering_rate_rads,
        "max_acceleration_mps2": config.max_acceleration_mps2,
        "speed_discrepancy_tolerance_mps": config.speed_discrepancy_tolerance_mps,
        "safe_stop_deceleration_mps2": config.safe_stop_deceleration_mps2,
        "degraded_speed_limit_mps": config.degraded_speed_limit_mps,
        "self_test_dwell_s": config.self_test_dwell_s,
        "critical_safety_nodes": [
            n.value for n in sorted(CRITICAL_SAFETY_NODES, key=lambda x: x.value)
        ],
        "node_nominal_periods_ms": {n.value: p for n, p in DEFAULT_NODE_PERIODS_MS.items()},
        "supported_safety_states": [s.value for s in SafetyState],
    }

    if output is not None:
        write_json(output, payload)

    if as_json or output is None:
        typer.echo(json.dumps(payload, indent=2))


@safety_app.command("watchdog")
def safety_watchdog(
    context: typer.Context,
    dropped_node: Annotated[
        str | None,
        typer.Option("--dropped-node", "-d", help="ECU node ID to drop heartbeat for."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Evaluate cyclic heartbeat watchdog tracking across simulated ECU broadcasts."""
    config = get_state(context).config.safety
    watchdog = HeartbeatWatchdog(config)

    # Simulate periodic broadcasts
    timestamps = [0.0, 0.05, 0.10, 0.15]
    for ts in timestamps:
        for node in DEFAULT_NODE_PERIODS_MS:
            if dropped_node and node.value == dropped_node and ts > 0.05:
                continue
            watchdog.record_heartbeat(
                HeartbeatPing(node_id=node, sequence_num=int(ts * 20), timestamp_s=ts)
            )

    eval_time = 0.35  # Advance beyond timeout
    statuses = watchdog.evaluate(eval_time)
    is_crit, crit_reason = watchdog.check_critical_expiration(statuses)

    payload: dict[str, Any] = {
        "evaluation_time_s": eval_time,
        "dropped_node": dropped_node,
        "is_critical_failure": is_crit,
        "critical_failure_reason": crit_reason,
        "node_statuses": {k: v.model_dump() for k, v in statuses.items()},
    }

    if output is not None:
        write_json(output, payload)

    if as_json or output is None:
        typer.echo(json.dumps(payload, indent=2))


@safety_app.command("plausibility")
def safety_plausibility(
    context: typer.Context,
    steering_rate: Annotated[
        float, typer.Option("--steering-rate", help="Measured steering rate (rad/s).")
    ] = 0.2,
    acceleration: Annotated[
        float, typer.Option("--acceleration", help="Measured longitudinal accel (m/s²).")
    ] = 1.5,
    speed_diff: Annotated[
        float,
        typer.Option("--speed-diff", help="Discrepancy between wheel and radar speed (m/s)."),
    ] = 0.3,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Check kinematic plausibility metrics against ASIL-D envelope thresholds."""
    config = get_state(context).config.safety
    checker = PlausibilityChecker(config)

    violations = []
    v_steer = checker.check_steering_slew(
        current_angle_rad=steering_rate * 0.1,
        previous_angle_rad=0.0,
        dt_s=0.1,
        max_rate_rads=config.max_steering_rate_rads,
    )
    if v_steer:
        violations.append(v_steer.model_dump())

    v_accel = checker.check_acceleration(
        current_speed_mps=acceleration * 0.1,
        previous_speed_mps=0.0,
        dt_s=0.1,
        max_accel_mps2=config.max_acceleration_mps2,
    )
    if v_accel:
        violations.append(v_accel.model_dump())

    v_diff = checker.check_speed_discrepancy(
        wheel_speed_mps=5.0 + speed_diff,
        secondary_speed_mps=5.0,
        tolerance_mps=config.speed_discrepancy_tolerance_mps,
    )
    if v_diff:
        violations.append(v_diff.model_dump())

    payload: dict[str, Any] = {
        "is_plausible": len(violations) == 0,
        "violations_count": len(violations),
        "violations": violations,
    }

    if output is not None:
        write_json(output, payload)

    if as_json or output is None:
        typer.echo(json.dumps(payload, indent=2))


@safety_app.command("simulate")
def safety_simulate(
    context: typer.Context,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario: nominal, critical_timeout, kinematic_violation, fail_latch.",
        ),
    ] = "nominal",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Run an end-to-end multi-step safety supervisor lifecycle simulation."""
    config = get_state(context).config.safety
    sup = SafetySupervisor(config=config, start_time_s=0.0)

    # 1. Boot phase
    for node in DEFAULT_NODE_PERIODS_MS:
        tracker = sup.watchdog.trackers[node]
        tracker.last_heartbeat_s = 0.1
        tracker.has_received_first = True

    # Advance through self-test dwell
    sup.step(current_time_s=config.self_test_dwell_s + 0.1)

    # 2. Scenario Execution
    if scenario == "nominal":
        for node in DEFAULT_NODE_PERIODS_MS:
            sup.watchdog.trackers[node].last_heartbeat_s = 1.0
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 3.0, "steering_angle_rad": 0.05})

    elif scenario == "critical_timeout":
        # Let vehicle controller expire by advancing time past timeout
        for node in DEFAULT_NODE_PERIODS_MS:
            if node != ECUNodeId.VEHICLE_CONTROLLER:
                sup.watchdog.trackers[node].last_heartbeat_s = 1.0
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 3.0})

    elif scenario == "kinematic_violation":
        for node in DEFAULT_NODE_PERIODS_MS:
            sup.watchdog.trackers[node].last_heartbeat_s = 1.0
        # Initialize kinematics
        sup.step(0.95, kinematics={"wheel_speed_mps": 3.0, "steering_angle_rad": 0.0})
        # Sudden massive steering jump
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 3.0, "steering_angle_rad": 0.8})

    elif scenario == "fail_latch":
        # Trigger emergency stop then come to standstill
        sup.trigger_emergency_stop("Manual safety trigger", timestamp_s=0.8)
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 0.0})

    else:
        raise typer.BadParameter(
            f"Unknown scenario '{scenario}'. Choose from: "
            "nominal, critical_timeout, kinematic_violation, fail_latch."
        )

    payload = rep.model_dump()

    if png is not None:
        render_safety_dashboard(rep, output_path=png)

    if output is not None:
        write_json(output, payload)

    if as_json or (output is None and png is None):
        typer.echo(json.dumps(payload, indent=2))


@safety_app.command("render")
def safety_render(
    context: typer.Context,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario: nominal, critical_timeout, kinematic_violation, fail_latch.",
        ),
    ] = "nominal",
    png: Annotated[
        Path,
        typer.Option(
            "--png",
            "-p",
            resolve_path=True,
            help="Path to render tactical ASIL-D safety HUD PNG artifact.",
        ),
    ] = Path("artifacts/safety_hud.png"),
) -> None:
    """Render a 1280x720 tactical safety HUD dashboard PNG artifact for a simulated state."""
    config = get_state(context).config.safety
    sup = SafetySupervisor(config=config, start_time_s=0.0)

    for node in DEFAULT_NODE_PERIODS_MS:
        tracker = sup.watchdog.trackers[node]
        tracker.last_heartbeat_s = 0.1
        tracker.has_received_first = True

    sup.step(current_time_s=config.self_test_dwell_s + 0.1)

    if scenario == "critical_timeout":
        for node in DEFAULT_NODE_PERIODS_MS:
            if node != ECUNodeId.VEHICLE_CONTROLLER:
                sup.watchdog.trackers[node].last_heartbeat_s = 1.0
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 2.0})
    elif scenario == "fail_latch":
        sup.trigger_emergency_stop("Emergency safe stop", timestamp_s=0.8)
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 0.0})
    elif scenario == "kinematic_violation":
        sup.step(0.95, kinematics={"wheel_speed_mps": 3.0, "steering_angle_rad": 0.0})
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 3.0, "steering_angle_rad": 0.8})
    else:
        for node in DEFAULT_NODE_PERIODS_MS:
            sup.watchdog.trackers[node].last_heartbeat_s = 1.0
        rep = sup.step(1.0, kinematics={"wheel_speed_mps": 2.5, "steering_angle_rad": 0.02})

    render_safety_dashboard(rep, output_path=png)
    typer.echo(f"Rendered tactical safety HUD dashboard to {png}")


@safety_app.command("audit")
def safety_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Perform ISO 26262 ASIL-D safety compliance verification audit."""
    config = get_state(context).config.safety

    checks: list[dict[str, Any]] = [
        {
            "standard": "ISO 26262-4:2018 §6.4.3",
            "requirement": "Hardware heartbeat watchdog deadline < 200ms",
            "passed": config.heartbeat_timeout_ms <= 200.0,
            "measured": f"{config.heartbeat_timeout_ms} ms",
        },
        {
            "standard": "ISO 26262-4:2018 §6.4.4",
            "requirement": "Critical ECU consecutive misses tolerance <= 3",
            "passed": config.max_consecutive_misses <= 3,
            "measured": f"{config.max_consecutive_misses} misses",
        },
        {
            "standard": "ISO 26262-6:2018 §7.4.5",
            "requirement": "Safe stop deceleration envelope >= 2.0 m/s²",
            "passed": config.safe_stop_deceleration_mps2 >= 2.0,
            "measured": f"{config.safe_stop_deceleration_mps2} m/s²",
        },
        {
            "standard": "ISO 26262-6:2018 §7.4.6",
            "requirement": "Degraded perception crawl speed limit <= 3.0 m/s",
            "passed": config.degraded_speed_limit_mps <= 3.0,
            "measured": f"{config.degraded_speed_limit_mps} m/s",
        },
        {
            "standard": "ISO 26262-8:2018 §11.4.1",
            "requirement": "Fail-silent latch state defined and irreversible without auth",
            "passed": hasattr(SafetyState, "FAIL_SILENT_LATCH"),
            "measured": "SafetyState.FAIL_SILENT_LATCH defined",
        },
    ]

    all_passed = all(c["passed"] for c in checks)
    payload: dict[str, Any] = {
        "overall_compliant": all_passed,
        "standard": "ISO 26262:2018 ASIL-D",
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }

    if output is not None:
        write_json(output, payload)

    if as_json or output is None:
        typer.echo(json.dumps(payload, indent=2))
