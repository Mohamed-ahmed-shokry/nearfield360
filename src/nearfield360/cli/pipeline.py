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
from nearfield360.control import (
    ManeuverExecutionReport,
    ManeuverExecutionStep,
    ManeuverExecutor,
    render_control_execution_bev_overlay,
)
from nearfield360.data.calibration import load_calibration
from nearfield360.data.detection import DetectionAnnotation, load_detection_annotations
from nearfield360.data.images import load_rgb_image
from nearfield360.data.woodscape import CameraId, WoodScapeSample
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.health import CameraHealthReport
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.localization import LocalizationSimulator
from nearfield360.mapping.models import FacilityMap
from nearfield360.mapping.router import GlobalRouter
from nearfield360.mission import (
    MissionExecutive,
    MissionSummaryReport,
    render_mission_dashboard_overlay,
)
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
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

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


def _render_pipeline_png_with_control(
    grid: Any,
    evidence: OccupancyEvidence,
    zones: list[RiskZone],
    slots: Sequence[ParkingSlot],
    plan: ParkingTrajectoryPlan,
    report: ManeuverExecutionReport,
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
    canvas = render_control_execution_bev_overlay(
        grid, plan, report, slots=slots, base_canvas=canvas
    )
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"unable to write PNG: {path}")
    typer.echo(f"Wrote {path}")


def _render_pipeline_png_with_mission(
    grid: Any,
    evidence: OccupancyEvidence,
    zones: list[RiskZone],
    slots: Sequence[ParkingSlot],
    plan: ParkingTrajectoryPlan | None,
    report: MissionSummaryReport,
    steps: Sequence[ManeuverExecutionStep],
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
    canvas = render_mission_dashboard_overlay(
        grid, report, steps, slots=slots, plan=plan, base_canvas=canvas
    )
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
    simulate_control: Annotated[
        bool,
        typer.Option(
            "--simulate-control/--no-simulate-control",
            help=(
                "Simulate closed-loop Stanley tracking control for the planned parking trajectory "
                "and verify docking accuracy."
            ),
        ),
    ] = False,
    mission: Annotated[
        bool,
        typer.Option(
            "--mission/--no-mission",
            help=(
                "Orchestrate end-to-end Autonomous Valet Parking (AVP) mission lifecycle, "
                "dynamic replanning, and slot memory."
            ),
        ),
    ] = False,
    map_file: Annotated[
        Path | None,
        typer.Option(
            "--map",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to facility HD vector map JSON for global routing and localization.",
        ),
    ] = None,
    target_slot: Annotated[
        str | None,
        typer.Option(
            "--target-slot",
            help="Target parking bay ID in the facility map.",
        ),
    ] = None,
    fleet: Annotated[
        bool,
        typer.Option(
            "--fleet/--no-fleet",
            help="Simulate multi-vehicle V2X cloud dispatch and conflict deconfliction.",
        ),
    ] = False,
    fleet_vehicles: Annotated[
        int,
        typer.Option(
            "--fleet-vehicles",
            min=1,
            help="Number of simulated fleet vehicles under coordination when --fleet is enabled.",
        ),
    ] = 2,
    charging: Annotated[
        bool,
        typer.Option(
            "--charging/--no-charging",
            help="Simulate autonomous EV charging dispatch, grid power balancing, and sessions.",
        ),
    ] = False,
    billing: Annotated[
        bool,
        typer.Option(
            "--billing/--no-billing",
            help="Calculate itemized session billing and generate signed receipts.",
        ),
    ] = False,
    tenant: Annotated[
        bool,
        typer.Option(
            "--tenant/--no-tenant",
            help="Enforce multi-tenant facility partitioning, SLA monitoring, and VIP corridors.",
        ),
    ] = False,
    curbside: Annotated[
        bool,
        typer.Option(
            "--curbside/--no-curbside",
            help="Simulate curbside drop-off/pick-up staging, dwell, and safety interlocks.",
        ),
    ] = False,
    teleop: Annotated[
        bool,
        typer.Option(
            "--teleop/--no-teleop",
            help="Simulate remote teleoperation assistance, video fallback, and failsafe MRM.",
        ),
    ] = False,
    floor: Annotated[
        bool,
        typer.Option(
            "--floor/--no-floor",
            help="Simulate multi-floor facility topology, vertical ramps, and vehicle elevators.",
        ),
    ] = False,
    emergency: Annotated[
        bool,
        typer.Option(
            "--emergency/--no-emergency",
            help="Simulate emergency preemption, evacuation planning, and life safety dispatch.",
        ),
    ] = False,
    gate: Annotated[
        bool,
        typer.Option(
            "--gate/--no-gate",
            help="Simulate facility access control, barrier gate interlocks, and ALPR enveloping.",
        ),
    ] = False,
    coop: Annotated[
        bool,
        typer.Option(
            "--coop/--no-coop",
            help=(
                "Simulate infrastructure cooperative perception, ETSI CPM fusion, "
                "and blind-spot occlusion warnings (ISO 23374 Type 2/3)."
            ),
        ),
    ] = False,
) -> None:
    """Run health, occupancy fusion, tracking, and risk over complete four-camera frames."""
    config = get_state(context).config

    if mission:
        simulate_control = True
        plan_parking = True
        slots = True
    elif simulate_control:
        plan_parking = True
        slots = True

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
    if slots or plan_parking or simulate_control:
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
    control_payload: dict[str, Any] | None = None
    evaluated_control: ManeuverExecutionReport | None = None
    mission_payload: dict[str, Any] | None = None
    evaluated_mission: MissionSummaryReport | None = None
    mission_steps: list[ManeuverExecutionStep] = []
    if plan_parking or simulate_control:
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

        if simulate_control and evaluated_plan is not None:
            c_stage_start = time.perf_counter()
            executor = ManeuverExecutor(config.control, config.planner)
            control_report = executor.execute_plan(
                plan=evaluated_plan,
                occupancy=fused.occupancy(),
                grid=grid,
                obstacles=active_obstacles,
            )
            timings["control_ms"] = (time.perf_counter() - c_stage_start) * 1000.0
            control_payload = control_report.model_dump(mode="json")
            evaluated_control = control_report

        if mission and evaluated_slots:
            m_stage_start = time.perf_counter()
            executive = MissionExecutive(
                mission_config=config.mission,
                planner_config=config.planner,
                control_config=config.control,
            )
            mission_report, mission_steps = executive.execute_mission(
                slots=evaluated_slots,
                initial_plan=evaluated_plan,
                occupancy=fused.occupancy(),
                grid=grid,
                obstacles=active_obstacles,
                mission_id=f"mission_{frames[-1][0]}",
            )
            timings["mission_ms"] = (time.perf_counter() - m_stage_start) * 1000.0
            mission_payload = mission_report.model_dump(mode="json")
            evaluated_mission = mission_report

    map_payload: dict[str, Any] | None = None
    if map_file is not None or target_slot is not None:
        map_stage_start = time.perf_counter()
        if map_file is not None:
            map_data = read_json(map_file)
            facility_map = FacilityMap.model_validate(map_data)
        else:
            facility_map = build_benchmark_garage()

        router = GlobalRouter(facility_map, turn_penalty_weight=config.mapping.turn_penalty_weight)
        eff_target = target_slot or (facility_map.slots[0].slot_id if facility_map.slots else None)
        if eff_target:
            global_route = router.plan(start_pose=(0.0, 0.0, 0.0), target_slot_id=eff_target)
            sim = LocalizationSimulator(facility_map, config=config.mapping)
            wp_coords = [(w.x, w.y, w.heading_rad) for w in global_route.waypoints]
            loc_report, _ = sim.simulate(wp_coords)
            map_payload = {
                "map_id": facility_map.map_id,
                "name": facility_map.name,
                "route": global_route.model_dump(mode="json"),
                "localization": loc_report.model_dump(mode="json"),
            }
        else:
            map_payload = {
                "map_id": facility_map.map_id,
                "name": facility_map.name,
                "slots_count": len(facility_map.slots),
            }
        timings["mapping_ms"] = (time.perf_counter() - map_stage_start) * 1000.0

    fleet_payload: dict[str, Any] | None = None
    if fleet:
        fleet_stage_start = time.perf_counter()
        from nearfield360.fleet.coordinator import FleetCoordinator

        if map_file is not None:
            f_map_data = read_json(map_file)
            f_map = FacilityMap.model_validate(f_map_data)
        else:
            f_map = build_benchmark_garage()

        fleet_coord = FleetCoordinator(f_map, config=config.fleet)
        for fi in range(fleet_vehicles):
            f_vid = f"veh_{fi + 1:02d}"
            fleet_coord.register_vehicle(f_vid, initial_pose=(float(fi * 3.0), 0.0, 0.0))
            fleet_coord.dispatch_vehicle(f_vid)

        for _ in range(30):
            fleet_coord.step(dt_s=0.5)

        flt_report = fleet_coord.generate_report()
        fleet_payload = flt_report.model_dump(mode="json")
        timings["fleet_ms"] = (time.perf_counter() - fleet_stage_start) * 1000.0

    charging_payload: dict[str, Any] | None = None
    if charging:
        charging_stage_start = time.perf_counter()
        from nearfield360.charging.grid import GridPowerManager
        from nearfield360.charging.hardware import ChargingHardwareSimulator
        from nearfield360.charging.models import (
            ChargerBay,
            ChargerPortStatus,
            ChargerType,
            ChargingDispatchReport,
            EVBatteryState,
        )
        from nearfield360.charging.scheduler import ChargingRequest, ChargingScheduler

        if map_file is not None:
            c_map_data = read_json(map_file)
            c_map = FacilityMap.model_validate(c_map_data)
        else:
            c_map = build_benchmark_garage()

        c_bays = [
            ChargerBay(
                bay_id=f"BAY_{idx + 1:02d}",
                slot_id=slot.slot_id,
                charger_type=ChargerType.DC_FAST_150KW if idx == 0 else ChargerType.DC_FAST_50KW,
                max_power_rating_kw=150.0 if idx == 0 else 50.0,
                status=ChargerPortStatus.AVAILABLE,
            )
            for idx, slot in enumerate(c_map.slots[:2])
        ]
        c_scheduler = ChargingScheduler(config=config.charging, facility_map=c_map)
        c_scheduler.register_charger_bays(c_bays)
        c_grid = GridPowerManager(config=config.charging, facility_id=c_map.map_id)
        c_hw = ChargingHardwareSimulator()

        c_bats: dict[str, EVBatteryState] = {}
        for ci in range(2):
            c_vid = f"EV_{ci + 1:02d}"
            c_soc = 0.20 + (ci * 0.25)
            c_bats[c_vid] = EVBatteryState(
                vehicle_id=c_vid,
                current_soc=c_soc,
                target_soc=config.charging.default_target_soc,
            )
            c_scheduler.enqueue_request(
                ChargingRequest(
                    vehicle_id=c_vid,
                    arrival_time_s=0.0,
                    current_soc=c_soc,
                    target_soc=config.charging.default_target_soc,
                )
            )

        c_scheduler.schedule_assignments(current_time_s=0.0)
        for _ in range(10):
            act_s = c_scheduler.active_sessions
            act_b = [c_bats[s.vehicle_id] for s in act_s if s.vehicle_id in c_bats]
            c_grid.balance_power(act_s, c_bays, act_b)
            c_hw.step_facility(act_s, act_b, c_bays, dt_s=60.0)

        c_final_s = c_scheduler.active_sessions
        c_final_b = [c_bats[s.vehicle_id] for s in c_final_s if s.vehicle_id in c_bats]
        c_power = c_grid.balance_power(c_final_s, c_bays, c_final_b)

        c_report = ChargingDispatchReport(
            facility_id=c_map.map_id,
            timestamp_s=600.0,
            power_summary=c_power,
            total_bays=len(c_bays),
            available_bays=sum(1 for b in c_bays if b.status == ChargerPortStatus.AVAILABLE),
            occupied_charging_bays=sum(
                1 for b in c_bays if b.status == ChargerPortStatus.OCCUPIED_CHARGING
            ),
            waiting_relocation_bays=0,
            active_sessions=c_final_s,
            pending_queue_count=len(c_scheduler.queue),
        )
        charging_payload = c_report.model_dump(mode="json")
        timings["charging_ms"] = (time.perf_counter() - charging_stage_start) * 1000.0

    billing_payload: dict[str, Any] | None = None
    if billing:
        billing_stage_start = time.perf_counter()
        from nearfield360.billing.calculator import BillingCalculator
        from nearfield360.billing.settlement import SettlementAggregator

        b_calc = BillingCalculator(config.billing)
        b_aggregator = SettlementAggregator("AVP-FACILITY-01")
        b_receipts = [
            b_calc.calculate_session_charges(
                vehicle_id=f"VEH_{bi + 1:02d}",
                entry_timestamp=float(bi * 3600),
                exit_timestamp=float(bi * 3600 + (45.0 + bi * 30.0) * 60.0),
                kwh_consumed=15.0 if bi % 2 == 0 else 0.0,
                idle_overstay_minutes=5.0 if bi == 1 else 0.0,
                include_valet_fee=True,
                occupancy_ratio=0.80,
                receipt_id=f"RCP-{bi + 1:03d}",
            )
            for bi in range(3)
        ]
        b_report = b_aggregator.generate_settlement_report(b_receipts)
        billing_payload = b_report.model_dump(mode="json")
        timings["billing_ms"] = (time.perf_counter() - billing_stage_start) * 1000.0

    tenant_payload: dict[str, Any] | None = None
    if tenant:
        tenant_stage_start = time.perf_counter()
        from nearfield360.tenant.corridor import VIPCorridorManager
        from nearfield360.tenant.manager import TenantManager
        from nearfield360.tenant.models import (
            TenantPartitionReport,
            TenantSLAContract,
            TenantTier,
        )
        from nearfield360.tenant.sla import SLAMonitor

        t_facility = build_benchmark_garage()
        t_mgr = TenantManager(facility_map=t_facility, config=config.tenant)
        t_p1 = t_mgr.create_partition(
            tenant_id="fleet-alpha",
            tenant_name="Alpha Fleet",
            tier=TenantTier.COMMERCIAL_FLEET,
            slot_ids=[s.slot_id for s in t_facility.slots[:4]],
            quota=4,
        )
        t_sla = TenantSLAContract(
            contract_id="sla-alpha",
            tenant_id="fleet-alpha",
            max_dispatch_latency_s=config.tenant.sla_max_dispatch_wait_s,
            breach_penalty_per_min=config.tenant.sla_breach_penalty_per_min,
        )
        t_sla_mon = SLAMonitor([t_sla])
        t_sla_mon.record_dispatch(
            "fleet-alpha", latency_s=config.tenant.sla_max_dispatch_wait_s * 0.9
        )

        t_corr_mgr = VIPCorridorManager()
        t_lanes = (
            [lane.lane_id for lane in t_facility.lanes[:1]] if t_facility.lanes else ["lane_1"]
        )
        t_corr_mgr.register_corridor("vip-express-1", t_lanes)
        t_c1 = t_corr_mgr.reserve_corridor(
            "vip-express-1",
            vehicle_id="vip-fleet-01",
            start_time_s=0.0,
            duration_s=60.0,
            priority_level=2,
        )

        t_report = TenantPartitionReport(
            facility_id=t_facility.map_id,
            timestamp_s=0.0,
            total_slots_count=len(t_facility.slots),
            total_partitions_count=len(t_mgr.partitions),
            partitions=[t_p1],
            sla_contracts=t_sla_mon.contracts,
            express_corridors=[t_c1],
            overall_sla_compliance=t_sla_mon.compute_overall_compliance(),
            total_breach_penalties=t_sla_mon.compute_total_penalties(),
        )
        tenant_payload = t_report.model_dump(mode="json")
        timings["tenant_ms"] = (time.perf_counter() - tenant_stage_start) * 1000.0

    curbside_payload: dict[str, Any] | None = None
    if curbside:
        curbside_stage_start = time.perf_counter()
        from nearfield360.curbside.manager import CurbsideManager

        c_mgr = CurbsideManager(facility_id="benchmark_garage_avp", config=config.curbside)
        c_mgr.request_dropoff("veh-curbside-01", current_time_s=0.0)
        c_mgr.complete_dropoff("veh-curbside-01", current_time_s=45.0)
        curb_report = c_mgr.generate_report(current_time_s=50.0)
        curbside_payload = curb_report.model_dump(mode="json")
        timings["curbside_ms"] = (time.perf_counter() - curbside_stage_start) * 1000.0

    teleop_payload: dict[str, Any] | None = None
    if teleop:
        teleop_stage_start = time.perf_counter()
        from nearfield360.teleop.manager import TeleopManager
        from nearfield360.teleop.models import TeleopHeartbeat, TeleopMode

        teleop_mgr = TeleopManager(facility_id="benchmark_garage_avp", config=config.teleop)
        teleop_mgr.start_session("veh-pipeline-01", "op-center-01", TeleopMode.TELE_GUIDANCE, 0.0)
        teleop_mgr.process_heartbeat(
            TeleopHeartbeat(
                vehicle_id="veh-pipeline-01",
                sequence_number=1,
                timestamp_s=0.5,
                rtt_ms=45.0,
            )
        )
        teleop_report = teleop_mgr.generate_report(current_time_s=1.0)
        teleop_payload = teleop_report.model_dump(mode="json")
        timings["teleop_ms"] = (time.perf_counter() - teleop_stage_start) * 1000.0

    floor_payload: dict[str, Any] | None = None
    if floor:
        floor_stage_start = time.perf_counter()
        from nearfield360.multilevel.graph import build_default_facility_graph
        from nearfield360.multilevel.ramp import RampDynamicsEngine

        m_graph = build_default_facility_graph()
        multi_route = m_graph.find_route("L1-entry", "L2-bay-42", vehicle_id="veh-pipeline-01")
        r_engine = RampDynamicsEngine()
        pitch = r_engine.calculate_pitch_dynamics(
            "veh-pipeline-01", grade_pct=11.0, elevation_m=1.75
        )
        floor_payload = {
            "route": multi_route.model_dump(mode="json"),
            "pitch_dynamics": pitch.model_dump(mode="json"),
            "floors_count": len(m_graph.floors),
            "total_bays": sum(f.bays_count for f in m_graph.floors.values()),
        }
        timings["floor_ms"] = (time.perf_counter() - floor_stage_start) * 1000.0

    emergency_payload: dict[str, Any] | None = None
    if emergency:
        emergency_stage_start = time.perf_counter()
        from nearfield360.emergency.containment import HazardContainmentEngine
        from nearfield360.emergency.evacuation import FacilityEvacuationPlanner
        from nearfield360.emergency.models import (
            EmergencySeverity,
            EmergencyType,
            EmergencyVehicle,
            HazardZone,
            LifeSafetyReport,
            ResponderType,
        )
        from nearfield360.emergency.preemption import EmergencyPreemptionArbiter

        e_arbiter = EmergencyPreemptionArbiter(config.emergency)
        e_resp = EmergencyVehicle(
            vehicle_id="fd-pipeline-01",
            responder_type=ResponderType.FIRE_ENGINE,
            x=0.0,
            y=10.0,
            target_x=40.0,
            target_y=40.0,
        )
        e_arbiter.register_responder(e_resp)
        e_contain = HazardContainmentEngine(config.emergency)
        e_hz = HazardZone(
            hazard_id="fire-zone-01",
            emergency_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            center_x=40.0,
            center_y=40.0,
            radius_m=12.0,
            floor_id="L1",
            active=True,
        )
        e_contain.add_hazard(e_hz)
        e_planner = FacilityEvacuationPlanner(e_contain, config.emergency)
        e_vehicles = {
            "veh-pipeline-01": ("bay-01", (15.0, 15.0)),
            "veh-pipeline-02": ("bay-02", (38.0, 38.0)),
        }
        e_portals = {"Exit-North": (0.0, 90.0), "Exit-South": (0.0, -10.0)}
        e_waves, e_routes = e_planner.plan_facility_evacuation(e_vehicles, e_portals)
        e_report = LifeSafetyReport(
            facility_name="Pipeline Multi-Camera Facility",
            emergency_active=True,
            incident_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            active_hazards=[e_hz],
            emergency_responders=[e_resp],
            total_vehicles=len(e_vehicles),
            evacuating_count=len(e_vehicles),
            evp_active=True,
        )
        emergency_payload = {
            "report": e_report.model_dump(mode="json"),
            "waves_count": len(e_waves),
            "routes_count": len(e_routes),
            "preemption_active": e_arbiter.is_preemption_active(),
        }
        timings["emergency_ms"] = (time.perf_counter() - emergency_stage_start) * 1000.0

    gate_payload: dict[str, Any] | None = None
    if gate:
        gate_stage_start = time.perf_counter()
        from nearfield360.gate.controller import BarrierInterlockController
        from nearfield360.gate.envelope import VehicleEnvelopeProfiler
        from nearfield360.gate.models import (
            CredentialType,
            GateAccessRequest,
            GateDirection,
            VehicleCredential,
            VehicleDimensionEnvelope,
        )
        from nearfield360.gate.validator import AccessCredentialValidator

        g_cred = VehicleCredential(
            credential_id="cred-pipeline-01",
            credential_type=CredentialType.ALPR_PLATE,
            plate_number="ABC-1234",
            valid_from_s=0.0,
            valid_until_s=86400.0,
            tenant_id="PUBLIC",
        )
        g_validator = AccessCredentialValidator(config.gate, initial_credentials=[g_cred])
        g_profiler = VehicleEnvelopeProfiler(config.gate)
        g_controller = BarrierInterlockController(
            config=config.gate,
            validator=g_validator,
            envelope_profiler=g_profiler,
            gate_id="GATE_ENTRY_01",
            direction=GateDirection.ENTRY,
        )

        g_envelope = VehicleDimensionEnvelope(
            length_m=config.slots.vehicle_length,
            width_m=config.slots.vehicle_width,
            height_m=1.65,
            ground_clearance_m=0.15,
        )
        g_request = GateAccessRequest(
            request_id="req-pipeline-gate",
            gate_id="GATE_ENTRY_01",
            vehicle_id="veh-pipeline",
            detected_plate="ABC-1234",
            plate_confidence=0.98,
            envelope=g_envelope,
        )
        g_event = g_controller.simulate_transit(g_request)
        g_report = g_controller.generate_audit_report()

        gate_payload = {
            "gate_id": g_event.gate_id,
            "vehicle_id": g_event.vehicle_id,
            "decision": g_event.decision.model_dump(mode="json"),
            "transit_duration_s": g_event.duration_s,
            "tailgating_detected": g_event.tailgating_detected,
            "barrier_state": str(g_controller.barrier.state.value),
            "barrier_angle_deg": g_controller.barrier.barrier_angle_deg,
            "audit_report": g_report.model_dump(mode="json"),
        }
        timings["gate_ms"] = (time.perf_counter() - gate_stage_start) * 1000.0

    coop_payload: dict[str, Any] | None = None
    if coop:
        coop_stage_start = time.perf_counter()
        from nearfield360.coop.alignment import Pose2D
        from nearfield360.coop.blind_spot import BlindSpotAssessor
        from nearfield360.coop.fusion import CooperativeTrackManager
        from nearfield360.coop.models import (
            CooperativeTrack,
            CoopTelemetryReport,
            CoopTrackSource,
            CPMPerceivedObject,
            PerceivedObjectClass,
        )
        from nearfield360.coop.rpu import RPUSensorManager

        c_network = RPUSensorManager.create_default_demonstrator_network()
        c_stations = c_network.list_stations()

        # Convert active obstacles from surround pipeline into CooperativeTrack objects
        c_onboard: list[CooperativeTrack | dict[str, Any]] = [
            CooperativeTrack(
                track_id=f"ONBOARD-{obs.track_id}",
                source=CoopTrackSource.ONBOARD_ONLY,
                x_m=obs.position[0],
                y_m=obs.position[1],
                vx_mps=obs.velocity[0],
                vy_mps=obs.velocity[1],
                category=PerceivedObjectClass.VEHICLE,
                confidence=min(0.99, 0.5 + 0.1 * obs.hits),
                onboard_track_id=str(obs.track_id),
            )
            for obs in active_obstacles
        ]

        # Simulated roadside perception unit observation (blind corner)
        c_hidden = CPMPerceivedObject(
            object_id="RPU-BLIND-OBJ-01",
            station_id="RPU-BLIND-CORNER-1",
            timestamp_s=0.0,
            category=PerceivedObjectClass.PEDESTRIAN,
            x_m=20.0,
            y_m=25.0,
            vx_mps=0.0,
            vy_mps=-1.5,
            confidence=0.92,
        )
        c_network.add_perceived_object("RPU-BLIND-CORNER-1", c_hidden)
        c_cpm_messages = [c_network.generate_cpm(s.station_id, timestamp_s=0.0) for s in c_stations]

        c_fusion = CooperativeTrackManager()
        c_fused_tracks = c_fusion.fuse_cycle(
            onboard_tracks=c_onboard,
            cpm_messages=c_cpm_messages,
            timestamp_s=0.0,
            config=config.coop,
        )

        c_assessor = BlindSpotAssessor()
        c_ego_pose = Pose2D(x_m=12.0, y_m=15.0, yaw_deg=0.0)
        c_warnings = c_assessor.assess_tracks(
            vehicle_id="EGO-SURROUND",
            vehicle_pose=c_ego_pose,
            vehicle_speed_mps=2.5,
            tracks=c_fused_tracks,
            timestamp_s=0.0,
            config=config.coop,
        )

        c_telemetry = CoopTelemetryReport(
            active_rpus=len(c_stations),
            cpm_message_count=len(c_cpm_messages),
            onboard_track_count=len(c_onboard),
            fused_track_count=len(c_fused_tracks),
            blind_spot_hazard_count=len(c_warnings),
            warnings=c_warnings,
            mean_fusion_latency_ms=(time.perf_counter() - coop_stage_start) * 1000.0,
            timestamp_s=0.0,
        )

        coop_payload = {
            "telemetry": c_telemetry.model_dump(mode="json"),
            "active_rpus": [s.station_id for s in c_stations],
            "fused_tracks": [t.model_dump(mode="json") for t in c_fused_tracks],
            "warnings": [w.model_dump(mode="json") for w in c_warnings],
        }
        timings["coop_ms"] = (time.perf_counter() - coop_stage_start) * 1000.0

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
    if simulate_control:
        samples_payload["simulate_control"] = True
    if mission:
        samples_payload["mission"] = True
    if map_file is not None or target_slot is not None:
        samples_payload["map"] = True
    if fleet:
        samples_payload["fleet"] = True
    if charging:
        samples_payload["charging"] = True
    if billing:
        samples_payload["billing"] = True
    if tenant:
        samples_payload["tenant"] = True
    if curbside:
        samples_payload["curbside"] = True
    if teleop:
        samples_payload["teleop"] = True
    if floor:
        samples_payload["floor"] = True
    if emergency:
        samples_payload["emergency"] = True
    if gate:
        samples_payload["gate"] = True
    if coop:
        samples_payload["coop"] = True
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
    if control_payload is not None:
        payload["control"] = control_payload
    if mission_payload is not None:
        payload["mission"] = mission_payload
    if map_payload is not None:
        payload["map"] = map_payload
    if fleet_payload is not None:
        payload["fleet"] = fleet_payload
    if charging_payload is not None:
        payload["charging"] = charging_payload
    if billing_payload is not None:
        payload["billing"] = billing_payload
    if tenant_payload is not None:
        payload["tenant"] = tenant_payload
    if curbside_payload is not None:
        payload["curbside"] = curbside_payload
    if teleop_payload is not None:
        payload["teleop"] = teleop_payload
    if floor_payload is not None:
        payload["floor"] = floor_payload
    if emergency_payload is not None:
        payload["emergency"] = emergency_payload
    if gate_payload is not None:
        payload["gate"] = gate_payload
    if coop_payload is not None:
        payload["coop"] = coop_payload
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
    if simulate_control and control_payload is not None:
        if evaluated_control is not None:
            is_dock = evaluated_control.kpis.is_docked_successfully
            cte_cm = evaluated_control.kpis.max_cross_track_error_m * 100
            typer.echo(
                f"Control complete: status={evaluated_control.status.value}, "
                f"duration={evaluated_control.duration_s:.2f}s, "
                f"steps={evaluated_control.total_steps}, "
                f"max_cte={cte_cm:.1f}cm, "
                f"docked={'YES' if is_dock else 'NO'}."
            )
        else:
            typer.echo("Control complete: no trajectory executed.")
    if mission and evaluated_mission is not None:
        typer.echo(
            f"Mission complete: status={evaluated_mission.final_state.value}, "
            f"duration={evaluated_mission.total_duration_s:.2f}s, "
            f"steps={evaluated_mission.total_steps}, "
            f"replans={evaluated_mission.replan_count}, "
            f"docked={'YES' if evaluated_mission.is_success else 'NO'}."
        )
    if coop and coop_payload is not None:
        c_telem = coop_payload["telemetry"]
        typer.echo(
            f"Coop complete: {c_telem['active_rpus']} RPUs active, "
            f"{c_telem['fused_track_count']} fused tracks, "
            f"{c_telem['blind_spot_hazard_count']} blind spot hazards."
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
            if evaluated_mission is not None:
                _render_pipeline_png_with_mission(
                    grid,
                    fused,
                    zones,
                    evaluated_slots,
                    evaluated_plan,
                    evaluated_mission,
                    mission_steps,
                    png,
                )
            elif evaluated_control is not None and evaluated_plan is not None:
                _render_pipeline_png_with_control(
                    grid, fused, zones, evaluated_slots, evaluated_plan, evaluated_control, png
                )
            elif evaluated_plan is not None:
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
