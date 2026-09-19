"""CLI commands for multi-camera dynamic obstacle tracking and risk evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import load_calibration
from nearfield360.data.detection import load_detection_annotations
from nearfield360.data.woodscape import CameraId, WoodScapeSample
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.occupancy.risk import surround_parking_zones
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.tracking.models import GroundFootprint, TrackedObstacle, TrackState
from nearfield360.tracking.projection import project_detections
from nearfield360.tracking.risk import forecast_all_trajectories
from nearfield360.tracking.tracker import MultiObjectTracker
from nearfield360.tracking.viz import render_tracking_bev_overlay
from nearfield360.utils.artifacts import ArtifactError, write_json

track_app = typer.Typer(
    help="Multi-camera dynamic obstacle tracking, trajectory forecasting, and collision risk.",
    no_args_is_help=True,
)


@track_app.command("run")
def run_tracking_command(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Path to write JSON tracking report.",
        ),
    ],
    camera: Annotated[
        CameraId,
        typer.Option("--camera", "-c", help="Specific camera view to track from."),
    ] = CameraId.FRONT,
    all_cameras: Annotated[
        bool,
        typer.Option(
            "--all-cameras",
            help="Fuse and track detections across all 4 calibrated surround cameras.",
        ),
    ] = False,
    max_frames: Annotated[
        int,
        typer.Option(
            "--max-frames",
            "-n",
            min=1,
            help="Maximum number of frames / time steps to evaluate.",
        ),
    ] = 10,
    png: Annotated[
        Path | None,
        typer.Option(
            "--png",
            resolve_path=True,
            help="Optional path to render final BEV tracking overlay PNG image.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing report output."),
    ] = False,
    root: DatasetRootOption = None,
) -> None:
    """Track dynamic obstacles across temporal frames and forecast safety zone ingress."""
    state = get_state(context)
    config = state.config
    dataset = discover_dataset(context, root)

    bev_cfg = config.bev
    grid = BevGrid(
        x_min=bev_cfg.x_min,
        x_max=bev_cfg.x_max,
        y_min=bev_cfg.y_min,
        y_max=bev_cfg.y_max,
        resolution=bev_cfg.resolution,
    )

    zones = list(
        surround_parking_zones(
            grid,
            front_length=config.risk.front_length,
            rear_length=config.risk.rear_length,
            half_width=config.risk.half_width,
            start_x=config.risk.start_x,
            rear_start_x=config.risk.rear_start_x,
            lateral_width=config.risk.lateral_width,
            vehicle_x_min=config.risk.vehicle_x_min,
            vehicle_x_max=config.risk.vehicle_x_max,
            near_radius=config.risk.near_radius,
            warning_radius=config.risk.warning_radius,
        )
    )

    tracker = MultiObjectTracker(config.tracking)

    # Group dataset samples by frame_id across camera views
    frames_dict: dict[str, list[WoodScapeSample]] = {}
    for sample in dataset:
        if not all_cameras and sample.key.camera != camera:
            continue
        frame_id = sample.key.frame_id
        frames_dict.setdefault(frame_id, []).append(sample)

    sorted_frames = sorted(frames_dict.keys())[:max_frames]
    if not sorted_frames:
        typer.secho(
            f"No matching samples found for camera={camera.value if not all_cameras else 'all'}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    frames_evaluated = 0
    active_obstacles: list[TrackedObstacle] = []

    for frame_id in sorted_frames:
        frame_samples = frames_dict[frame_id]
        footprints: list[GroundFootprint] = []

        for sample in frame_samples:
            if sample.calibration_path is None or sample.detection_path is None:
                continue

            try:
                calib = load_calibration(sample.calibration_path)
                cam_model = CalibratedCamera.from_calibration(
                    calib, theta_max=config.geometry.theta_max
                )
                detections = load_detection_annotations(sample.detection_path)
            except Exception as exc:
                typer.secho(
                    f"Error loading {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True
                )
                raise typer.Exit(code=1) from None

            fps = project_detections(
                detections,
                camera=cam_model,
                camera_name=sample.key.camera.value,
                ground_z=config.geometry.ground_z,
                max_distance=config.geometry.max_distance,
            )
            footprints.extend(fps)

        active_obstacles = tracker.update(footprints)
        frames_evaluated += 1

    forecasts = forecast_all_trajectories(
        active_obstacles,
        grid=grid,
        zones=zones,
        horizon_s=config.tracking.forecast_horizon_s,
        step_s=config.tracking.dt,
        confirmed_only=False,
    )

    confirmed_count = sum(1 for obs in active_obstacles if obs.state == TrackState.CONFIRMED)
    intrusions = [f for f in forecasts if f.min_ttc_s is not None]
    min_ttc = min((f.min_ttc_s for f in intrusions if f.min_ttc_s is not None), default=None)

    report_payload = {
        "environment": environment_metadata(),
        "config": config.model_dump(mode="json"),
        "camera_mode": "all" if all_cameras else camera.value,
        "frames_evaluated": frames_evaluated,
        "summary": {
            "total_active_tracks": len(active_obstacles),
            "confirmed_tracks": confirmed_count,
            "intrusions_detected": len(intrusions),
            "minimum_ttc_s": min_ttc,
        },
        "tracks": [obs.model_dump(mode="json") for obs in active_obstacles],
        "forecasts": [f.model_dump(mode="json") for f in forecasts],
    }

    try:
        write_json(output, report_payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if png is not None:
        render_tracking_bev_overlay(
            grid,
            active_obstacles,
            forecasts=forecasts,
            zones=zones,
            output_path=png,
        )

    typer.echo(
        f"Tracking complete: {frames_evaluated} frames evaluated, "
        f"{len(active_obstacles)} active tracks ({confirmed_count} confirmed). "
        f"Wrote report to {output}"
    )
    if intrusions:
        typer.secho(
            f"WARNING: {len(intrusions)} collision intrusions predicted! (Min TTC: {min_ttc}s)",
            fg=typer.colors.YELLOW,
            bold=True,
        )


__all__ = ["track_app"]
