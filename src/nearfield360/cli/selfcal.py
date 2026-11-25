"""CLI commands for map-anchored online extrinsic self-calibration and drift monitoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from nearfield360.cli.state import get_state
from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.icp import ImageLandmarkCalibrator
from nearfield360.selfcal.models import CalibrationEstimate, RotationDelta
from nearfield360.selfcal.monitor import DriftMonitor
from nearfield360.selfcal.scene import (
    demo_fisheye_camera,
    demo_parking_aisle_map,
    demo_parking_aisle_polylines,
    synthesize_pixel_detections,
)
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations
from nearfield360.selfcal.viz import CAMERA_ORDER, render_selfcal_dashboard
from nearfield360.utils.artifacts import write_json

selfcal_app = typer.Typer(
    help="Map-anchored online extrinsic self-calibration and camera drift monitoring.",
    no_args_is_help=True,
)

console = Console()

OutputOption = Annotated[
    Path | None,
    typer.Option("--output", "-o", resolve_path=True, help="Path to write a JSON artifact."),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")]
CameraOption = Annotated[CameraId, typer.Option("--camera", help="Camera to misalign.")]
SeedOption = Annotated[int, typer.Option("--seed", help="Random seed for synthetic landmarks.")]


def _simulate(
    config: SelfCalConfig,
    camera: CameraId,
    final: RotationDelta,
    *,
    frames: int,
    points: int,
    outliers: float,
    noise_deg: float,
    seed: int,
) -> tuple[DriftMonitor, list[CalibrationEstimate]]:
    """Ramp ``camera`` from nominal to ``final`` while the others stay near nominal."""
    monitor = DriftMonitor(config)
    calibrator = ExtrinsicSelfCalibrator(config)
    latest: list[CalibrationEstimate] = []
    for frame in range(frames):
        latest = []
        fraction = (frame + 1) / frames
        for index, cam in enumerate(CAMERA_ORDER):
            truth = (
                RotationDelta.from_vector(final.as_vector() * fraction)
                if cam is camera
                else RotationDelta.from_degrees(pitch=0.05)
            )
            observations = synthesize_observations(
                cam,
                truth,
                num_points=points,
                outlier_fraction=outliers,
                noise_deg=noise_deg,
                seed=seed + frame * 10 + index,
            )
            estimate = calibrator.estimate(DEFAULT_CAMERA_ORIGIN, observations)
            monitor.update(estimate)
            latest.append(estimate)
    return monitor, latest


def _payload(monitor: DriftMonitor, estimates: list[CalibrationEstimate]) -> dict[str, Any]:
    return {
        "estimates": [e.model_dump(mode="json") for e in estimates],
        "reports": {c.value: r.model_dump(mode="json") for c, r in monitor.reports().items()},
        "history_deg": {c.value: monitor.history_deg(c) for c in CAMERA_ORDER},
    }


@selfcal_app.command("info")
def selfcal_info(
    context: typer.Context, as_json: JsonOption = False, output: OutputOption = None
) -> None:
    """Show self-calibration estimator and drift threshold configuration."""
    cfg = get_state(context).config.selfcal
    payload = cfg.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)
    if as_json:
        console.print_json(data=payload)
        return
    table = Table(title="Extrinsic Self-Calibration Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@selfcal_app.command("estimate")
def selfcal_estimate(
    context: typer.Context,
    camera: CameraOption = CameraId.FRONT,
    roll: Annotated[float, typer.Option("--roll", help="True roll error in degrees.")] = 0.5,
    pitch: Annotated[float, typer.Option("--pitch", help="True pitch error in degrees.")] = -0.3,
    yaw: Annotated[float, typer.Option("--yaw", help="True yaw error in degrees.")] = 0.4,
    points: Annotated[int, typer.Option("--points", min=1, help="Landmark count.")] = 40,
    outliers: Annotated[float, typer.Option("--outliers", min=0.0, max=0.95)] = 0.0,
    noise: Annotated[float, typer.Option("--noise", min=0.0, help="Ray noise in degrees.")] = 0.02,
    seed: SeedOption = 0,
    as_json: JsonOption = False,
    output: OutputOption = None,
) -> None:
    """Estimate a synthetic camera misalignment from map-anchored ground landmarks."""
    cfg = get_state(context).config.selfcal
    truth = RotationDelta.from_degrees(roll=roll, pitch=pitch, yaw=yaw)
    observations = synthesize_observations(
        camera, truth, num_points=points, outlier_fraction=outliers, noise_deg=noise, seed=seed
    )
    estimate = ExtrinsicSelfCalibrator(cfg).estimate(DEFAULT_CAMERA_ORIGIN, observations)
    status = DriftMonitor(cfg).classify(estimate.delta.magnitude_deg)
    payload = {
        "truth_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "status": status.value if estimate.observable else "UNOBSERVABLE",
        "estimate": estimate.model_dump(mode="json"),
    }
    if output is not None:
        write_json(output, payload, overwrite=True)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]{camera.value}[/bold] status: {payload['status']}")
    console.print(
        f"Estimated roll/pitch/yaw: {math.degrees(estimate.delta.roll_rad):+.3f} / "
        f"{math.degrees(estimate.delta.pitch_rad):+.3f} / "
        f"{math.degrees(estimate.delta.yaw_rad):+.3f} deg"
    )
    console.print(
        f"RMS {estimate.rms_before_m * 100:.2f} cm -> {estimate.rms_after_m * 100:.2f} cm | "
        f"inliers {estimate.inlier_ratio * 100:.0f}% | iterations {estimate.iterations}"
    )
    if estimate.reason:
        console.print(f"[yellow]{estimate.reason}[/yellow]")


@selfcal_app.command("monitor")
def selfcal_monitor(
    context: typer.Context,
    camera: CameraOption = CameraId.FRONT,
    pitch: Annotated[float, typer.Option("--pitch", help="Final pitch drift in degrees.")] = 2.0,
    frames: Annotated[int, typer.Option("--frames", min=1, max=200)] = 8,
    seed: SeedOption = 0,
    as_json: JsonOption = False,
    output: OutputOption = None,
) -> None:
    """Simulate gradual drift of one camera and report per-frame monitor status."""
    cfg = get_state(context).config.selfcal
    monitor, estimates = _simulate(
        cfg,
        camera,
        RotationDelta.from_degrees(pitch=pitch),
        frames=frames,
        points=40,
        outliers=0.1,
        noise_deg=0.02,
        seed=seed,
    )
    payload = _payload(monitor, estimates)
    if output is not None:
        write_json(output, payload, overwrite=True)
    if as_json:
        console.print_json(data=payload)
        return
    table = Table(title=f"Drift monitor: {camera.value} ramping to {pitch:g} deg pitch")
    table.add_column("Camera", style="cyan")
    table.add_column("Status")
    table.add_column("Smoothed drift (deg)", justify="right")
    table.add_column("Accepted", justify="right")
    table.add_column("Rejected", justify="right")
    for cam, report in monitor.reports().items():
        table.add_row(
            cam.value,
            report.status.value,
            f"{report.magnitude_deg:.3f}",
            str(report.accepted_updates),
            str(report.rejected_updates),
        )
    console.print(table)


@selfcal_app.command("render")
def selfcal_render(
    context: typer.Context,
    camera: CameraOption = CameraId.MIRROR_RIGHT,
    pitch: Annotated[float, typer.Option("--pitch", help="Final pitch drift in degrees.")] = 3.5,
    output: Annotated[
        Path, typer.Option("--output", "-o", resolve_path=True, help="PNG destination.")
    ] = Path("outputs/selfcal_dashboard.png"),
    as_json: JsonOption = False,
) -> None:
    """Render the 1280x720 self-calibration HUD dashboard PNG."""
    cfg = get_state(context).config.selfcal
    monitor, estimates = _simulate(
        cfg,
        camera,
        RotationDelta.from_degrees(pitch=pitch, roll=pitch / 3),
        frames=8,
        points=40,
        outliers=0.1,
        noise_deg=0.02,
        seed=0,
    )
    canvas = render_selfcal_dashboard(
        estimates,
        monitor.reports(),
        {c: monitor.history_deg(c) for c in CAMERA_ORDER},
        cfg,
        output_path=output,
    )
    payload = {"rendered_image_path": str(output), "canvas_shape": list(canvas.shape)}
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[green]Self-calibration dashboard rendered to {output}[/green]")
    console.print(f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}")


@selfcal_app.command("image")
def selfcal_image(
    context: typer.Context,
    camera: CameraOption = CameraId.FRONT,
    roll: Annotated[float, typer.Option("--roll", help="True roll error in degrees.")] = 0.5,
    pitch: Annotated[float, typer.Option("--pitch", help="True pitch error in degrees.")] = -0.3,
    yaw: Annotated[float, typer.Option("--yaw", help="True yaw error in degrees.")] = 0.4,
    points_per_segment: Annotated[
        int, typer.Option("--points-per-segment", min=1, help="Samples per map segment.")
    ] = 40,
    noise_px: Annotated[
        float, typer.Option("--noise-px", min=0.0, help="Pixel detection noise (px).")
    ] = 1.0,
    clutter: Annotated[
        float, typer.Option("--clutter", min=0.0, max=0.95, help="Clutter detection fraction.")
    ] = 0.2,
    seed: SeedOption = 0,
    as_json: JsonOption = False,
    output: OutputOption = None,
) -> None:
    """Run pixel-space point-to-line ICP from fisheye detections to the parking-aisle HD map."""
    cfg = get_state(context).config.selfcal
    cam = demo_fisheye_camera(camera)
    truth = RotationDelta.from_degrees(roll=roll, pitch=pitch, yaw=yaw)
    pixels = synthesize_pixel_detections(
        cam,
        demo_parking_aisle_polylines(),
        truth,
        points_per_segment=points_per_segment,
        pixel_noise_px=noise_px,
        clutter_fraction=clutter,
        seed=seed,
    )
    result = ImageLandmarkCalibrator(cfg).calibrate(cam, pixels, demo_parking_aisle_map())
    status = (
        DriftMonitor(cfg).classify(result.estimate.delta.magnitude_deg).value
        if result.estimate.observable
        else "UNOBSERVABLE"
    )
    payload = {
        "truth_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "num_detections": len(pixels),
        "status": status,
        "result": result.model_dump(mode="json"),
    }
    if output is not None:
        write_json(output, payload, overwrite=True)
    if as_json:
        console.print_json(data=payload)
        return
    est = result.estimate
    console.print(
        f"[bold]{camera.value}[/bold] image-ICP status: {status} | "
        f"converged: {result.converged} | detections: {len(pixels)}"
    )
    console.print(
        f"Estimated roll/pitch/yaw: {math.degrees(est.delta.roll_rad):+.3f} / "
        f"{math.degrees(est.delta.pitch_rad):+.3f} / "
        f"{math.degrees(est.delta.yaw_rad):+.3f} deg"
    )
    table = Table(title="Coarse-to-Fine Association Trace")
    table.add_column("Round", justify="right", style="cyan")
    table.add_column("Gate (m)", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Inliers", justify="right")
    table.add_column("RMS (cm)", justify="right")
    table.add_column("Step (deg)", justify="right")
    for rd in result.rounds:
        table.add_row(
            str(rd.round_index),
            f"{rd.gate_m:.3f}",
            str(rd.num_associated),
            f"{rd.inlier_ratio * 100:.0f}%",
            f"{rd.rms_after_m * 100:.2f}",
            f"{rd.step_deg:.4f}",
        )
    console.print(table)


@selfcal_app.command("audit")
def selfcal_audit(
    context: typer.Context, as_json: JsonOption = False, output: OutputOption = None
) -> None:
    """Audit self-calibration configuration and verify estimator accuracy end to end."""
    cfg = get_state(context).config.selfcal
    truth = RotationDelta.from_degrees(roll=0.5, pitch=-0.3, yaw=0.4)
    observations = synthesize_observations(
        CameraId.FRONT, truth, num_points=60, outlier_fraction=0.2, noise_deg=0.02, seed=0
    )
    estimate = ExtrinsicSelfCalibrator(cfg).estimate(DEFAULT_CAMERA_ORIGIN, observations)
    error_deg = math.degrees(float(abs(estimate.delta.as_vector() - truth.as_vector()).max()))
    checks = [
        (
            "Drift thresholds strictly ordered",
            cfg.warning_deg < cfg.recalibrate_deg < cfg.fault_deg,
        ),
        ("Warning threshold below 1 degree", cfg.warning_deg < 1.0),
        ("Minimum correspondences at least 8", cfg.min_correspondences >= 8),
        ("EWMA smoothing in (0, 1]", 0.0 < cfg.ewma_alpha <= 1.0),
        ("Estimator converges on 20% outlier scenario", estimate.converged and estimate.observable),
        ("Recovered rotation within 0.05 degrees of truth", error_deg < 0.05),
        ("Reprojection RMS reduced by estimate", estimate.rms_after_m < estimate.rms_before_m),
    ]
    compliant = all(passed for _, passed in checks)
    payload = {
        "audit_target": "Online Extrinsic Self-Calibration & Drift Monitoring Subsystem",
        "compliant": compliant,
        "total_checks": len(checks),
        "passed_checks": sum(1 for _, passed in checks if passed),
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
    }
    if output is not None:
        write_json(output, payload, overwrite=True)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        "Self-Calibration Audit: "
        + ("[green]COMPLIANT[/green]" if compliant else "[red]NON-COMPLIANT[/red]")
    )
    for name, passed in checks:
        console.print(f"  [{'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}] {name}")
