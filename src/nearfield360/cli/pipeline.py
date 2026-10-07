"""Integrated four-camera perception demo with a measured performance report."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.inference_common import BackendOption, load_backend
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
from nearfield360.occupancy import (
    OccupancyEvidence,
    OccupancyPolicy,
    RiskZone,
    TemporalOccupancyForecaster,
    risk_report,
)
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.perception.inference import (
    InferenceBackendType,
    ObjectDetectionEngine,
    SemanticSegmentationEngine,
)
from nearfield360.planning.models import ParkingTrajectoryPlan
from nearfield360.planning.planner import ParkingTrajectoryPlanner
from nearfield360.planning.viz import render_parking_plan_bev_overlay
from nearfield360.slots.classifier import SlotOccupancyClassifier
from nearfield360.slots.corridor import ApproachCorridorEvaluator
from nearfield360.slots.detector import ParkingSlotDetector
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotType,
    SlotDetectionSummary,
    SlotOccupancyStatus,
)
from nearfield360.slots.viz import render_slots_bev_overlay
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


def _load_segmentation_engine(
    context: typer.Context,
    model: Path,
    backend: InferenceBackendType | None = None,
) -> SemanticSegmentationEngine:
    loaded = load_backend(context, model, backend=backend)
    return SemanticSegmentationEngine(backend=loaded)


def _load_detection_engine(
    context: typer.Context,
    model: Path,
    backend: InferenceBackendType | None = None,
) -> ObjectDetectionEngine:
    config = get_state(context).config
    loaded = load_backend(context, model, backend=backend)
    return ObjectDetectionEngine(
        backend=loaded,
        confidence_threshold=config.inference.confidence_threshold,
        nms_threshold=config.inference.nms_threshold,
    )


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
    precomputed_detections: tuple[DetectionAnnotation, ...] | None = None,
) -> list[GroundFootprint]:
    """Load detections for one sample and project them onto the ground plane."""
    config = get_state(context).config
    if sample.calibration_path is None:
        return []
    detections: tuple[DetectionAnnotation, ...] | None
    if precomputed_detections is not None:
        detections = precomputed_detections
    else:
        detections = _load_detections(context, sample, det_engine)
    if not detections:
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


def _render_pipeline_png_with_slots(
    grid: Any,
    evidence: OccupancyEvidence,
    zones: list[RiskZone],
    slots: Sequence[ParkingSlot],
    path: Path,
) -> None:
    occupancy = evidence.occupancy()
    observed = evidence.observed > 0
    with np.errstate(invalid="ignore", over="ignore"):
        red = np.asarray(np.where(observed, occupancy * 255.0, 0.0), dtype=np.uint8)
        green = np.asarray(np.where(observed, (1.0 - occupancy) * 255.0, 0.0), dtype=np.uint8)
    canvas = np.zeros((*grid.shape, 3), dtype=np.uint8)
    canvas[observed, 0] = red[observed]
    canvas[observed, 1] = green[observed]
    for zone in zones:
        from nearfield360.cli.occupancy import _ZONE_COLORS, _border

        color = _ZONE_COLORS.get(zone.name, (128, 128, 128))
        for row, col in np.argwhere(_border(zone.mask)):
            canvas[row, col] = color
    canvas = render_slots_bev_overlay(grid, slots, base_canvas=canvas)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"unable to write PNG: {path}")
    typer.echo(f"Wrote {path}")


def _render_pipeline_png_with_plan(
    grid: Any,
    evidence: OccupancyEvidence,
    zones: list[RiskZone],
    slots: Sequence[ParkingSlot],
    plan: ParkingTrajectoryPlan,
    path: Path,
) -> None:
    occupancy = evidence.occupancy()
    observed = evidence.observed > 0
    with np.errstate(invalid="ignore", over="ignore"):
        red = np.asarray(np.where(observed, occupancy * 255.0, 0.0), dtype=np.uint8)
        green = np.asarray(np.where(observed, (1.0 - occupancy) * 255.0, 0.0), dtype=np.uint8)
    canvas = np.zeros((*grid.shape, 3), dtype=np.uint8)
    canvas[observed, 0] = red[observed]
    canvas[observed, 1] = green[observed]
    for zone in zones:
        from nearfield360.cli.occupancy import _ZONE_COLORS, _border

        color = _ZONE_COLORS.get(zone.name, (128, 128, 128))
        for row, col in np.argwhere(_border(zone.mask)):
            canvas[row, col] = color
    canvas = render_parking_plan_bev_overlay(grid, plan, slots=slots, base_canvas=canvas)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"unable to write PNG: {path}")
    typer.echo(f"Wrote {path}")


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
    backend: BackendOption = None,
    batch_cameras: Annotated[
        bool,
        typer.Option(
            "--batch-cameras/--no-batch-cameras",
            help="Batch multi-camera frames during live model perception.",
        ),
    ] = False,
    temporal_forecast: Annotated[
        bool,
        typer.Option(
            "--temporal-forecast/--no-temporal-forecast",
            help="Generate 4D recurrent spatiotemporal BEV occupancy forecast grids.",
        ),
    ] = False,
    slots: Annotated[
        bool,
        typer.Option(
            "--slots/--no-slots",
            help=(
                "Delineate 3D metric parking slots, evaluate occupancy, "
                "and check approach corridors."
            ),
        ),
    ] = False,
    plan_parking: Annotated[
        bool,
        typer.Option(
            "--plan-parking/--no-plan-parking",
            help=(
                "Plan an autonomous, collision-free parking trajectory "
                "into the most feasible vacant slot."
            ),
        ),
    ] = False,
) -> None:
    """Run health, occupancy fusion, tracking, and risk over complete four-camera frames."""
    config = get_state(context).config

    model_engine: SemanticSegmentationEngine | None = None
    det_engine: ObjectDetectionEngine | None = None
    if seg_model is not None:
        model_engine = _load_segmentation_engine(context, seg_model, backend)
    if det_model is not None:
        det_engine = _load_detection_engine(context, det_model, backend)

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
    forecaster = (
        TemporalOccupancyForecaster(grid, config=config.forecast) if temporal_forecast else None
    )

    fused: OccupancyEvidence | None = None
    health_reports: list[CameraHealthReport] = []
    per_camera_counts: dict[str, int] = {cam.value: 0 for cam in CameraId}
    active_obstacles: list[TrackedObstacle] = []
    frames_evaluated = 0
    occupancy_ms = 0.0
    tracking_ms = 0.0
    markings_mask: np.ndarray | None = np.zeros(grid.shape, dtype=np.uint8) if slots else None

    stage_start = time.perf_counter()
    for frame_index, (frame_id, frame_samples) in enumerate(frames, start=1):
        frame_start = time.perf_counter()
        frame_evidence: OccupancyEvidence | None = None
        footprints: list[GroundFootprint] = []

        batch_masks: list[np.ndarray] | None = None
        batch_dets: list[tuple[DetectionAnnotation, ...]] | None = None

        if batch_cameras and (model_engine is not None or det_engine is not None):
            rgb_images = [load_rgb_image(s.image_path) for s in frame_samples]
            if model_engine is not None:
                t_occ_b = time.perf_counter()
                model_preds = model_engine.predict_batch(rgb_images)
                batch_masks = [mask for mask, _conf in model_preds]
                occupancy_ms += (time.perf_counter() - t_occ_b) * 1000.0
            if det_engine is not None:
                t_trk_b = time.perf_counter()
                batch_dets = det_engine.predict_batch_annotations(rgb_images)
                tracking_ms += (time.perf_counter() - t_trk_b) * 1000.0

        for i, sample in enumerate(frame_samples):
            t_occ = time.perf_counter()
            precomputed_m = batch_masks[i] if batch_masks is not None else None
            evidence, report = _frame_evidence(
                context,
                sample,
                grid,
                policy,
                theta_max=None,
                health_aware=health_aware,
                model_engine=model_engine if batch_masks is None else None,
                precomputed_mask=precomputed_m,
                markings_out=markings_mask if slots else None,
            )
            occupancy_ms += (time.perf_counter() - t_occ) * 1000.0
            if report is not None:
                health_reports.append(report)
            per_camera_counts[sample.key.camera.value] += 1
            frame_evidence = evidence if frame_evidence is None else frame_evidence.add(evidence)

            t_trk = time.perf_counter()
            precomputed_d = batch_dets[i] if batch_dets is not None else None
            footprints.extend(
                _frame_footprints(
                    context,
                    sample,
                    det_engine if batch_dets is None else None,
                    precomputed_detections=precomputed_d,
                )
            )
            tracking_ms += (time.perf_counter() - t_trk) * 1000.0

        active_obstacles = tracker.update(footprints)
        if frame_evidence is not None:
            fused = frame_evidence if fused is None else fused.add(frame_evidence)
            if forecaster is not None:
                timestamp = (frame_index - 1) * config.tracking.dt
                forecaster.update(frame_evidence, timestamp=timestamp, obstacles=active_obstacles)
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

    temporal_forecast_payload: dict[str, Any] | None = None
    temporal_forecast_grid_payload: dict[str, Any] | None = None
    if forecaster is not None:
        stage_start = time.perf_counter()
        temporal_forecast_grid = forecaster.forecast()
        curr_st = forecaster.current_state
        obs_c = int(np.sum(curr_st.occupancy >= 0.0)) if curr_st is not None else 0
        dyn_c = int(np.sum(curr_st.dynamic_mask)) if curr_st is not None else 0
        temporal_summary = temporal_forecast_grid.summary(
            zones,
            danger_occupancy=config.risk.danger_occupancy,
            observed_cells=obs_c,
            dynamic_cells=dyn_c,
        )
        timings["temporal_forecast_ms"] = (time.perf_counter() - stage_start) * 1000.0
        temporal_forecast_payload = temporal_summary.model_dump(mode="json")
        temporal_forecast_grid_payload = temporal_forecast_grid.to_dict()

    slots_payload: dict[str, Any] | None = None
    evaluated_slots: list[ParkingSlot] = []
    if slots or plan_parking:
        stage_start = time.perf_counter()
        clean_markings: np.ndarray | None = None
        if markings_mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean_markings = np.asarray(
                cv2.morphologyEx(markings_mask, cv2.MORPH_CLOSE, kernel), dtype=np.uint8
            )
        detector = ParkingSlotDetector(config.slots)
        candidate_slots = detector.detect_slots(
            grid=grid,
            markings_mask=clean_markings,
            obstacles=active_obstacles,
        )
        classifier = SlotOccupancyClassifier(config.slots)
        classified_slots = classifier.classify_slots(
            slots=candidate_slots,
            occupancy=fused.occupancy(),
            uncertainty=fused.uncertainty(),
            grid=grid,
            obstacles=active_obstacles,
            danger_threshold=config.risk.danger_occupancy,
        )
        corridor_evaluator = ApproachCorridorEvaluator(config.slots)
        evaluated_slots = corridor_evaluator.evaluate_slots(
            slots=classified_slots,
            occupancy=fused.occupancy(),
            grid=grid,
            obstacles=active_obstacles,
            danger_threshold=config.risk.danger_occupancy,
        )
        timings["slots_ms"] = (time.perf_counter() - stage_start) * 1000.0

        slot_summary = SlotDetectionSummary(
            total_slots=len(evaluated_slots),
            vacant_slots=sum(1 for s in evaluated_slots if s.status == SlotOccupancyStatus.VACANT),
            occupied_slots=sum(
                1 for s in evaluated_slots if s.status == SlotOccupancyStatus.OCCUPIED
            ),
            uncertain_slots=sum(
                1 for s in evaluated_slots if s.status == SlotOccupancyStatus.UNCERTAIN
            ),
            parallel_slots=sum(
                1 for s in evaluated_slots if s.slot_type == ParkingSlotType.PARALLEL
            ),
            perpendicular_slots=sum(
                1 for s in evaluated_slots if s.slot_type == ParkingSlotType.PERPENDICULAR
            ),
            slanted_slots=sum(1 for s in evaluated_slots if s.slot_type == ParkingSlotType.SLANTED),
            feasible_approaches=sum(
                1
                for s in evaluated_slots
                if s.approach_path is not None and s.approach_path.is_feasible
            ),
        )
        slots_payload = {
            "summary": slot_summary.model_dump(mode="json"),
            "slots": [s.model_dump(mode="json") for s in evaluated_slots],
        }

    plan_payload: dict[str, Any] | None = None
    evaluated_plan: ParkingTrajectoryPlan | None = None
    if plan_parking:
        stage_start = time.perf_counter()
        planner = ParkingTrajectoryPlanner(config.planner)
        plan_report = planner.plan_parking(
            slots=evaluated_slots,
            occupancy=fused.occupancy(),
            grid=grid,
            obstacles=active_obstacles,
            uncertainty=fused.uncertainty(),
        )
        timings["plan_ms"] = (time.perf_counter() - stage_start) * 1000.0
        plan_payload = plan_report.model_dump(mode="json")
        evaluated_plan = plan_report.plan

    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

    samples_payload: dict[str, Any] = {
        "requested": len(frames) if samples == 0 else samples,
        "evaluated": frames_evaluated,
        "camera": "all",
        "per_camera": per_camera_counts,
        "batch_cameras": batch_cameras,
    }
    if temporal_forecast:
        samples_payload["temporal_forecast"] = True
    if slots:
        samples_payload["slots"] = True
    if plan_parking:
        samples_payload["plan_parking"] = True
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
    if temporal_forecast_payload is not None:
        payload["temporal_forecast"] = temporal_forecast_payload
        payload["temporal_forecast_grid"] = temporal_forecast_grid_payload
    if slots_payload is not None:
        payload["slots"] = slots_payload
    if plan_payload is not None:
        payload["plan"] = plan_payload
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
    if (slots or plan_parking) and slots_payload is not None:
        slot_summ = slots_payload["summary"]
        typer.echo(
            f"Slots complete: {slot_summ['total_slots']} detected "
            f"({slot_summ['vacant_slots']} vacant, "
            f"{slot_summ['feasible_approaches']} feasible corridors)."
        )
    if plan_parking and plan_payload is not None:
        if evaluated_plan is not None:
            typer.echo(
                f"Plan complete: slot '{evaluated_plan.slot_id}' ({evaluated_plan.slot_type}), "
                f"length {evaluated_plan.total_length_m:.2f}m, "
                f"duration {evaluated_plan.total_duration_s:.2f}s, "
                f"gears {evaluated_plan.gear_switches + 1}, "
                f"clearance {evaluated_plan.min_clearance_m:.2f}m."
            )
        else:
            typer.echo("Plan complete: no executable parking trajectory found.")
    if summary["intrusions_detected"]:
        typer.secho(
            f"WARNING: {summary['intrusions_detected']} collision intrusions predicted! "
            f"(Min TTC: {summary['minimum_ttc_s']}s)",
            fg=typer.colors.YELLOW,
            bold=True,
        )

    if png is not None:
        try:
            if evaluated_plan is not None:
                _render_pipeline_png_with_plan(
                    grid, fused, zones, evaluated_slots, evaluated_plan, png
                )
            elif (slots or plan_parking) and evaluated_slots:
                _render_pipeline_png_with_slots(grid, fused, zones, evaluated_slots, png)
            else:
                _render_occupancy_png(grid, fused, zones, png)
        except (OSError, ValueError) as exc:
            typer.secho(f"PNG error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None


__all__ = ["pipeline_app"]
