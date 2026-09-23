"""Integrated four-camera perception demo with a measured performance report."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.occupancy import (
    _configured_grid,
    _configured_policy,
    _configured_zones,
    _evidence_summary,
    _frame_evidence,
    _render_occupancy_png,
)
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import load_calibration
from nearfield360.data.detection import DetectionAnnotation, load_detection_annotations
from nearfield360.data.images import load_rgb_image
from nearfield360.data.woodscape import CameraId, WoodScapeSample
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.health import CameraHealthReport
from nearfield360.occupancy import OccupancyEvidence, OccupancyPolicy, RiskZone, risk_report
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.perception.inference import (
    InferenceBackendType,
    InferenceDevice,
    ObjectDetectionEngine,
    SemanticSegmentationEngine,
    create_backend,
)
from nearfield360.tracking.models import (
    GroundFootprint,
    TrackedObstacle,
    TrackState,
    TrajectoryForecast,
)
from nearfield360.tracking.projection import project_detections
from nearfield360.tracking.risk import forecast_all_trajectories
from nearfield360.tracking.tracker import MultiObjectTracker
from nearfield360.utils.artifacts import ArtifactError, write_json

pipeline_app = typer.Typer(
    help="Run the integrated four-camera surround perception pipeline with a timed report.",
    no_args_is_help=True,
)

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="JSON pipeline report (created atomically; refuses to overwrite).",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Optional PNG visualization of the fused occupancy and zones.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace an existing pipeline report."),
]

HealthAwareOption = Annotated[
    bool,
    typer.Option(
        "--health-aware",
        help="Assess camera optical health and discount degraded or soiled camera evidence.",
    ),
]

SegModelOption = Annotated[
    Path | None,
    typer.Option(
        "--seg-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="ONNX semantic segmentation model for live occupancy (replaces semantic masks).",
    ),
]

DetModelOption = Annotated[
    Path | None,
    typer.Option(
        "--det-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="ONNX object detection model for live tracking (replaces detection files).",
    ),
]


def _load_segmentation_engine(model: Path) -> SemanticSegmentationEngine:
    try:
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=InferenceDevice.CPU,
        )
        return SemanticSegmentationEngine(backend=backend)
    except Exception as exc:
        typer.secho(
            f"Failed to load segmentation model {model}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None


def _load_detection_engine(model: Path) -> ObjectDetectionEngine:
    try:
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=InferenceDevice.CPU,
        )
        return ObjectDetectionEngine(backend=backend)
    except Exception as exc:
        typer.secho(
            f"Failed to load detection model {model}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None


def _latency_stats(samples_ms: list[float]) -> dict[str, Any]:
    """Compute latency distribution statistics from per-frame samples in milliseconds."""
    if not samples_ms:
        return {
            "samples": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "samples": len(samples_ms),
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def _group_complete_frames(
    context: typer.Context,
    root: Path | None,
    samples_limit: int,
    *,
    require_semantic: bool,
    require_detection: bool,
) -> list[tuple[str, list[WoodScapeSample]]]:
    """Group samples by frame_id, keeping only frames with all four cameras annotated."""
    dataset = discover_dataset(context, root)
    by_frame: dict[str, list[WoodScapeSample]] = defaultdict(list)
    for sample in dataset:
        if sample.calibration_path is None:
            continue
        if require_semantic and sample.semantic_mask_path is None:
            continue
        if require_detection and sample.detection_path is None:
            continue
        by_frame[sample.key.frame_id].append(sample)
    complete: list[tuple[str, list[WoodScapeSample]]] = []
    for frame_id in sorted(by_frame):
        cameras = {sample.key.camera for sample in by_frame[frame_id]}
        missing = set(CameraId) - cameras
        if missing:
            missing_names = ", ".join(cam.value for cam in sorted(missing, key=lambda c: c.value))
            typer.echo(f"Skipping frame {frame_id}: missing cameras {missing_names}", err=True)
            continue
        complete.append((frame_id, by_frame[frame_id]))
    if not complete:
        requirements = ["all four cameras and calibration"]
        if require_semantic:
            requirements.append("semantic masks")
        if require_detection:
            requirements.append("detections")
        typer.secho(
            "No frames with " + ", ".join(requirements) + " found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    if samples_limit > 0 and len(complete) < samples_limit:
        typer.secho(
            f"Requested {samples_limit} frames but only {len(complete)} complete "
            "multi-camera frames found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    return complete[:samples_limit] if samples_limit > 0 else complete


def _load_detections(
    context: typer.Context,
    sample: WoodScapeSample,
    det_engine: ObjectDetectionEngine | None,
) -> tuple[DetectionAnnotation, ...] | None:
    if det_engine is not None:
        try:
            rgb = load_rgb_image(sample.image_path)
            return det_engine.predict_annotations(rgb)
        except Exception as exc:
            typer.secho(
                f"Detection inference error for {sample.key.stem}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
    if sample.detection_path is None:
        return None
    try:
        return load_detection_annotations(sample.detection_path)
    except Exception as exc:
        typer.secho(
            f"Error loading detections for {sample.key.stem}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None


def _frame_footprints(
    context: typer.Context,
    sample: WoodScapeSample,
    det_engine: ObjectDetectionEngine | None = None,
) -> list[GroundFootprint]:
    """Load detections for one sample and project them onto the ground plane."""
    config = get_state(context).config
    if sample.calibration_path is None:
        return []
    detections = _load_detections(context, sample, det_engine)
    if detections is None:
        return []
    try:
        calib = load_calibration(sample.calibration_path)
        camera = CalibratedCamera.from_calibration(calib, theta_max=config.geometry.theta_max)
    except Exception as exc:
        typer.secho(
            f"Error processing {sample.key.stem}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    return project_detections(
        detections,
        camera=camera,
        camera_name=sample.key.camera.value,
        ground_z=config.geometry.ground_z,
        max_distance=config.geometry.max_distance,
    )


def _zone_summary_payload(grid: Any, zone: RiskZone) -> dict[str, Any]:
    rows, cols = np.nonzero(zone.mask)
    cell_area = grid.resolution * grid.resolution
    if rows.size == 0:
        return {
            "name": zone.name,
            "cells": 0,
            "area_m2": 0.0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }
    centers = grid.grid_to_world(np.stack((rows, cols), axis=1))
    return {
        "name": zone.name,
        "cells": int(rows.size),
        "area_m2": float(rows.size * cell_area),
        "x_min": float(centers[:, 0].min()),
        "x_max": float(centers[:, 0].max()),
        "y_min": float(centers[:, 1].min()),
        "y_max": float(centers[:, 1].max()),
    }


def _summary_payload(
    active_obstacles: list[TrackedObstacle],
    forecasts: list[TrajectoryForecast],
) -> dict[str, Any]:
    confirmed = sum(1 for obs in active_obstacles if obs.state == TrackState.CONFIRMED)
    intrusions = [f for f in forecasts if f.min_ttc_s is not None]
    min_ttc = min((f.min_ttc_s for f in intrusions if f.min_ttc_s is not None), default=None)
    return {
        "total_active_tracks": len(active_obstacles),
        "confirmed_tracks": confirmed,
        "intrusions_detected": len(intrusions),
        "minimum_ttc_s": min_ttc,
    }


@pipeline_app.command("run")
def run_pipeline(
    context: typer.Context,
    output: OutputOption,
    root: DatasetRootOption = None,
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            min=0,
            help="Number of four-camera frames to process (0 for all).",
        ),
    ] = 0,
    overwrite: OverwriteOption = False,
    png: PngOption = None,
    health_aware: HealthAwareOption = False,
    seg_model: SegModelOption = None,
    det_model: DetModelOption = None,
) -> None:
    """Run health, occupancy fusion, tracking, and risk over complete four-camera frames."""
    config = get_state(context).config

    model_engine: SemanticSegmentationEngine | None = None
    det_engine: ObjectDetectionEngine | None = None
    if seg_model is not None:
        model_engine = _load_segmentation_engine(seg_model)
    if det_model is not None:
        det_engine = _load_detection_engine(det_model)

    timings: dict[str, float] = {}
    frame_latencies_ms: list[float] = []
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    grid = _configured_grid(context)
    policy: OccupancyPolicy = _configured_policy(context)
    frames = _group_complete_frames(
        context,
        root,
        samples,
        require_semantic=model_engine is None,
        require_detection=det_engine is None,
    )
    timings["discovery_ms"] = (time.perf_counter() - stage_start) * 1000.0

    zones = _configured_zones(grid, context)
    tracker = MultiObjectTracker(config.tracking)

    fused: OccupancyEvidence | None = None
    health_reports: list[CameraHealthReport] = []
    per_camera_counts: dict[str, int] = {cam.value: 0 for cam in CameraId}
    active_obstacles: list[TrackedObstacle] = []
    frames_evaluated = 0
    occupancy_ms = 0.0
    tracking_ms = 0.0

    stage_start = time.perf_counter()
    for frame_index, (frame_id, frame_samples) in enumerate(frames, start=1):
        frame_start = time.perf_counter()
        frame_evidence: OccupancyEvidence | None = None
        footprints: list[GroundFootprint] = []
        for sample in frame_samples:
            t_occ = time.perf_counter()
            evidence, report = _frame_evidence(
                context,
                sample,
                grid,
                policy,
                theta_max=None,
                health_aware=health_aware,
                model_engine=model_engine,
            )
            occupancy_ms += (time.perf_counter() - t_occ) * 1000.0
            if report is not None:
                health_reports.append(report)
            per_camera_counts[sample.key.camera.value] += 1
            frame_evidence = evidence if frame_evidence is None else frame_evidence.add(evidence)

            t_trk = time.perf_counter()
            footprints.extend(_frame_footprints(context, sample, det_engine))
            tracking_ms += (time.perf_counter() - t_trk) * 1000.0

        if frame_evidence is not None:
            fused = frame_evidence if fused is None else fused.add(frame_evidence)
        active_obstacles = tracker.update(footprints)
        frames_evaluated += 1
        frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
        typer.echo(f"[{frame_index}/{len(frames)}] processed frame {frame_id}")
    timings["perception_ms"] = (time.perf_counter() - stage_start) * 1000.0
    timings["occupancy_ms"] = occupancy_ms
    timings["tracking_ms"] = tracking_ms

    if fused is None:
        typer.secho(
            "No occupancy evidence could be fused.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    stage_start = time.perf_counter()
    min_evidence = config.occupancy.min_evidence
    risk_reports = risk_report(
        zones,
        fused,
        min_evidence=min_evidence,
        danger_occupancy=config.risk.danger_occupancy,
    )
    timings["risk_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    forecasts = forecast_all_trajectories(
        active_obstacles,
        grid=grid,
        zones=zones,
        horizon_s=config.tracking.forecast_horizon_s,
        step_s=config.tracking.dt,
        confirmed_only=False,
    )
    timings["forecast_ms"] = (time.perf_counter() - stage_start) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

    samples_payload: dict[str, Any] = {
        "requested": len(frames) if samples == 0 else samples,
        "evaluated": frames_evaluated,
        "camera": "all",
        "per_camera": per_camera_counts,
    }
    if health_aware:
        samples_payload["health_aware"] = True
    if seg_model is not None:
        samples_payload["seg_model"] = str(seg_model)
    if det_model is not None:
        samples_payload["det_model"] = str(det_model)

    fps = (
        round(frames_evaluated / (timings["total_ms"] / 1000.0), 3)
        if timings["total_ms"] > 0
        else 0.0
    )
    payload: dict[str, Any] = {
        "environment": environment_metadata(),
        "config": config.model_dump(mode="json"),
        "samples": samples_payload,
        "timings": {
            **{k: round(v, 3) for k, v in timings.items()},
            "frames_per_second": fps,
            "frame_latency": _latency_stats(frame_latencies_ms),
        },
        "grid": {
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "y_min": grid.y_min,
            "y_max": grid.y_max,
            "resolution": grid.resolution,
            "width": grid.width,
            "height": grid.height,
        },
        "evidence": _evidence_summary(fused, min_evidence),
        "zones": [_zone_summary_payload(grid, zone) for zone in zones],
        "risk": [
            {
                "name": report.name,
                "cells": report.cells,
                "observed_cells": report.observed_cells,
                "occupied_cells": report.occupied_cells,
                "area_m2": report.area_m2,
                "observed_area_m2": report.observed_area_m2,
                "occupied_area_m2": report.occupied_area_m2,
                "mean_occupancy": report.mean_occupancy,
                "max_occupancy": report.max_occupancy,
                "mean_uncertainty": report.mean_uncertainty,
                "max_uncertainty": report.max_uncertainty,
            }
            for report in risk_reports
        ],
        "summary": _summary_payload(active_obstacles, forecasts),
        "tracks": [obs.model_dump(mode="json") for obs in active_obstacles],
        "forecasts": [f.model_dump(mode="json") for f in forecasts],
    }
    if health_aware:
        payload["health"] = [r.model_dump(mode="json") for r in health_reports]

    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Wrote {output}")
    summary = payload["summary"]
    latency = payload["timings"]["frame_latency"]
    typer.echo(
        f"Pipeline complete: {frames_evaluated} frames, "
        f"{summary['total_active_tracks']} tracks "
        f"({summary['confirmed_tracks']} confirmed), "
        f"total {timings['total_ms']:.1f} ms ({fps} fps), "
        f"frame p50={latency['p50_ms']} ms p95={latency['p95_ms']} ms."
    )
    if summary["intrusions_detected"]:
        typer.secho(
            f"WARNING: {summary['intrusions_detected']} collision intrusions predicted! "
            f"(Min TTC: {summary['minimum_ttc_s']}s)",
            fg=typer.colors.YELLOW,
            bold=True,
        )

    if png is not None:
        try:
            _render_occupancy_png(grid, fused, zones, png)
        except (OSError, ValueError) as exc:
            typer.secho(f"PNG error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None


__all__ = ["pipeline_app"]
