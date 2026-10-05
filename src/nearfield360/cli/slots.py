"""CLI commands for 3D metric parking slot detection, occupancy, and feasibility."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.inference_common import BackendOption, load_backend
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import load_calibration
from nearfield360.data.detection import load_detection_annotations
from nearfield360.data.images import load_rgb_image
from nearfield360.data.semantic import load_semantic_mask
from nearfield360.data.woodscape import CameraId, WoodScapeDataset, WoodScapeSample
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    distance_weights,
    rasterize_occupancy,
)
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.perception.inference import ObjectDetectionEngine, SemanticSegmentationEngine
from nearfield360.slots.classifier import SlotOccupancyClassifier
from nearfield360.slots.corridor import ApproachCorridorEvaluator
from nearfield360.slots.detector import ParkingSlotDetector
from nearfield360.slots.models import (
    ParkingSlotType,
    SlotDetectionReport,
    SlotDetectionSummary,
    SlotOccupancyStatus,
)
from nearfield360.slots.viz import render_slots_bev_overlay
from nearfield360.tracking.models import TrackedObstacle, TrackState
from nearfield360.tracking.projection import project_detections
from nearfield360.utils.artifacts import ArtifactError, write_json

slots_app = typer.Typer(
    help="3D metric parking slot detection, occupancy classification, and feasibility.",
    no_args_is_help=True,
)

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="JSON path to write parking slot detection report.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Optional PNG visualization of detected parking slots and approach corridors.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace an existing output artifact."),
]

VacantOnlyOption = Annotated[
    bool,
    typer.Option(
        "--vacant-only",
        help="Filter detection report to include only vacant slots.",
    ),
]

AllCamerasOption = Annotated[
    bool,
    typer.Option(
        "--all-cameras",
        help="Fuse evidence and obstacles across all 4 calibrated surround cameras.",
    ),
]

SamplesOption = Annotated[
    int,
    typer.Option(
        "--samples",
        "-n",
        min=1,
        help="Number of multi-camera frames or single-camera samples to fuse.",
    ),
]


def _build_camera(
    context: typer.Context, sample: WoodScapeSample, theta_max: float | None = None
) -> CalibratedCamera:
    state = get_state(context)
    config = state.config
    limit = config.geometry.theta_max if theta_max is None else theta_max
    if sample.calibration_path is None:
        typer.secho(f"Missing calibration for {sample.key.stem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    try:
        calib = load_calibration(sample.calibration_path)
    except Exception as exc:
        typer.secho(
            f"Failed to load calibration {sample.calibration_path}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    return CalibratedCamera.from_calibration(calib, theta_max=limit)


def _collect_samples(
    dataset: WoodScapeDataset,
    camera: CameraId,
    all_cameras: bool,
    samples_limit: int,
) -> list[tuple[str, list[WoodScapeSample]]]:
    samples_list = list(dataset)
    if not all_cameras:
        single_samples = [s for s in samples_list if s.key.camera == camera]
        if not single_samples:
            typer.secho(
                f"No samples found for camera {camera.value}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        chosen = single_samples[:samples_limit] if samples_limit > 0 else single_samples
        return [(s.key.frame_id, [s]) for s in chosen]

    by_frame: dict[str, list[WoodScapeSample]] = defaultdict(list)
    for sample in samples_list:
        by_frame[sample.key.frame_id].append(sample)

    complete_frames: list[tuple[str, list[WoodScapeSample]]] = []
    for frame_id in sorted(by_frame.keys()):
        cams = {s.key.camera for s in by_frame[frame_id]}
        if set(CameraId).issubset(cams):
            complete_frames.append((frame_id, by_frame[frame_id]))

    if not complete_frames:
        typer.secho(
            "No complete multi-camera frames found in dataset.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from None

    return complete_frames[:samples_limit] if samples_limit > 0 else complete_frames


@slots_app.command("detect")
def detect_slots_command(
    context: typer.Context,
    output: OutputOption,
    camera: Annotated[
        CameraId,
        typer.Option("--camera", "-c", help="Camera view to use (FV, RV, MVL, MVR)."),
    ] = CameraId.FRONT,
    all_cameras: AllCamerasOption = False,
    samples: SamplesOption = 1,
    vacant_only: VacantOnlyOption = False,
    png: PngOption = None,
    overwrite: OverwriteOption = False,
    root: DatasetRootOption = None,
    model: Annotated[
        Path | None,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional ONNX semantic segmentation model for live inference.",
        ),
    ] = None,
    det_model: Annotated[
        Path | None,
        typer.Option(
            "--det-model",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional ONNX object detection model for live inference.",
        ),
    ] = None,
    backend: BackendOption = None,
) -> None:
    """Delineate 3D metric parking slots, evaluate occupancy, and check approach corridors."""
    state = get_state(context)
    config = state.config
    dataset = discover_dataset(context, root)

    grid = BevGrid(
        x_min=config.bev.x_min,
        x_max=config.bev.x_max,
        y_min=config.bev.y_min,
        y_max=config.bev.y_max,
        resolution=config.bev.resolution,
    )

    policy = OccupancyPolicy.from_names(
        free=config.occupancy.free_classes,
        occupied=config.occupancy.occupied_classes,
    )

    chosen_frames = _collect_samples(dataset, camera, all_cameras, samples)

    # Optional neural perception engines
    seg_engine: SemanticSegmentationEngine | None = None
    det_engine: ObjectDetectionEngine | None = None
    if model is not None:
        seg_backend = load_backend(context, model, backend=backend)
        seg_engine = SemanticSegmentationEngine(backend=seg_backend)
    if det_model is not None:
        det_backend = load_backend(context, det_model, backend=backend)
        det_engine = ObjectDetectionEngine(
            backend=det_backend,
            confidence_threshold=config.inference.confidence_threshold,
            nms_threshold=config.inference.nms_threshold,
        )

    accumulated_evidence: OccupancyEvidence | None = None
    markings_mask = np.zeros(grid.shape, dtype=np.uint8)
    collected_obstacles: list[TrackedObstacle] = []
    track_counter = 1

    last_frame_id = chosen_frames[-1][0] if chosen_frames else None
    camera_sources: set[str] = set()

    for _frame_id, frame_samples in chosen_frames:
        for sample in frame_samples:
            camera_sources.add(sample.key.camera.value)
            cam = _build_camera(context, sample)

            # 1. Semantic mask
            if seg_engine is not None:
                rgb = load_rgb_image(sample.image_path)
                mask, _ = seg_engine.predict(rgb)
            elif sample.semantic_mask_path is not None:
                mask = load_semantic_mask(sample.semantic_mask_path)
            else:
                continue

            # Ground ray intersection
            rows, cols = np.meshgrid(
                np.arange(mask.shape[0], dtype=np.float32),
                np.arange(mask.shape[1], dtype=np.float32),
                indexing="ij",
            )
            pixels = np.stack((cols, rows), axis=-1)
            rays = cam.pixel_rays_vehicle(pixels, check_image_bounds=True)
            footprints_grid = intersect_ground(
                rays.origins,
                rays.directions,
                ground_z=config.geometry.ground_z,
                max_distance=config.geometry.max_distance,
            )
            valid = rays.valid & footprints_grid.valid
            weights = distance_weights(
                footprints_grid.distances, slope=config.occupancy.confidence_slope
            )

            ev = rasterize_occupancy(
                grid=grid,
                points=footprints_grid.points[..., :2],
                labels=mask,
                policy=policy,
                weights=weights,
                valid=valid,
            )

            accumulated_evidence = (
                ev if accumulated_evidence is None else accumulated_evidence.add(ev)
            )

            # Markings mask (lanemarks=1, curb=2)
            is_marking = np.isin(mask, [1, 2]) & valid
            marking_ground = footprints_grid.points[..., :2][is_marking]
            if marking_ground.size > 0:
                indexed = grid.world_to_grid(marking_ground)
                valid_idx = indexed.valid
                if np.any(valid_idx):
                    rows_m = indexed.indices[valid_idx, 0]
                    cols_m = indexed.indices[valid_idx, 1]
                    markings_mask[rows_m, cols_m] = 255

            # 2. Obstacles / detections
            if det_engine is not None:
                rgb = load_rgb_image(sample.image_path)
                boxes = det_engine.predict_annotations(rgb)
                footprints = project_detections(
                    boxes, camera=cam, camera_name=sample.key.camera.value
                )
            elif sample.detection_path is not None:
                try:
                    boxes = load_detection_annotations(sample.detection_path)
                    footprints = project_detections(
                        boxes, camera=cam, camera_name=sample.key.camera.value
                    )
                except Exception:
                    footprints = []
            else:
                footprints = []

            for fp in footprints:
                collected_obstacles.append(
                    TrackedObstacle(
                        track_id=track_counter,
                        class_id=fp.class_id,
                        class_name=fp.class_name,
                        state=TrackState.CONFIRMED,
                        position=(fp.center_x, fp.center_y),
                        velocity=(0.0, 0.0),
                        speed=0.0,
                        hits=3,
                        age=3,
                        time_since_update=0,
                    )
                )
                track_counter += 1

    if accumulated_evidence is None:
        typer.secho("Failed to rasterize occupancy evidence.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    # Clean markings mask with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    markings_mask = np.asarray(
        cv2.morphologyEx(markings_mask, cv2.MORPH_CLOSE, kernel), dtype=np.uint8
    )

    # 1. Detect candidate parking slots
    detector = ParkingSlotDetector(config.slots)
    candidate_slots = detector.detect_slots(
        grid=grid,
        markings_mask=markings_mask,
        obstacles=collected_obstacles,
    )

    # 2. Classify slot occupancy & uncertainty
    classifier = SlotOccupancyClassifier(config.slots)
    classified_slots = classifier.classify_slots(
        slots=candidate_slots,
        occupancy=accumulated_evidence.occupancy(),
        uncertainty=accumulated_evidence.uncertainty(),
        grid=grid,
        obstacles=collected_obstacles,
        danger_threshold=config.risk.danger_occupancy,
    )

    # 3. Evaluate approach corridor kinematics and feasibility
    corridor_evaluator = ApproachCorridorEvaluator(config.slots)
    evaluated_slots = corridor_evaluator.evaluate_slots(
        slots=classified_slots,
        occupancy=accumulated_evidence.occupancy(),
        grid=grid,
        obstacles=collected_obstacles,
        danger_threshold=config.risk.danger_occupancy,
    )

    # Filter vacant-only if requested
    reported_slots = (
        [s for s in evaluated_slots if s.status == SlotOccupancyStatus.VACANT]
        if vacant_only
        else evaluated_slots
    )

    # Calculate summary
    summary = SlotDetectionSummary(
        total_slots=len(evaluated_slots),
        vacant_slots=sum(1 for s in evaluated_slots if s.status == SlotOccupancyStatus.VACANT),
        occupied_slots=sum(1 for s in evaluated_slots if s.status == SlotOccupancyStatus.OCCUPIED),
        uncertain_slots=sum(
            1 for s in evaluated_slots if s.status == SlotOccupancyStatus.UNCERTAIN
        ),
        parallel_slots=sum(1 for s in evaluated_slots if s.slot_type == ParkingSlotType.PARALLEL),
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

    report = SlotDetectionReport(
        frame_id=last_frame_id,
        camera_sources=tuple(sorted(camera_sources)),
        slots=reported_slots,
        summary=summary,
        metadata={
            "environment": environment_metadata(),
            "config": {
                "min_slot_width": config.slots.min_slot_width,
                "max_slot_width": config.slots.max_slot_width,
                "min_slot_length": config.slots.min_slot_length,
                "max_slot_length": config.slots.max_slot_length,
                "occupied_ratio_threshold": config.slots.occupied_ratio_threshold,
                "approach_lead_distance": config.slots.approach_lead_distance,
                "vehicle_width": config.slots.vehicle_width,
                "safety_margin": config.slots.safety_margin,
            },
        },
    )

    try:
        write_json(output, report.model_dump(), overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    # Render PNG visualization if requested
    if png is not None:
        occ = accumulated_evidence.occupancy()
        # Create base occupancy canvas
        valid_occ = np.nan_to_num(occ, nan=0.5)
        # 0 (free) -> green (0, 200, 0), 1 (occupied) -> red (0, 0, 200)
        base_canvas = np.zeros((*grid.shape, 3), dtype=np.uint8)
        base_canvas[:, :, 1] = ((1.0 - valid_occ) * 180).astype(np.uint8)
        base_canvas[:, :, 2] = (valid_occ * 200).astype(np.uint8)

        render_slots_bev_overlay(
            grid=grid,
            slots=evaluated_slots,
            base_canvas=base_canvas,
            output_path=png,
        )

    typer.echo(
        f"Detected {summary.total_slots} parking slots "
        f"({summary.vacant_slots} vacant, {summary.feasible_approaches} feasible corridors)."
    )


__all__ = ["slots_app"]
