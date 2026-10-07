"""CLI commands for autonomous parking trajectory planning and maneuver generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer
from numpy.typing import NDArray

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.slots import _build_camera
from nearfield360.cli.state import get_state
from nearfield360.data.semantic import load_semantic_mask
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.ground import intersect_ground
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    distance_weights,
    rasterize_occupancy,
)
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.planning.kinematics import AckermannVehicle
from nearfield360.planning.models import PlanStatus
from nearfield360.planning.planner import ParkingTrajectoryPlanner
from nearfield360.planning.viz import render_parking_plan_bev_overlay
from nearfield360.slots.classifier import SlotOccupancyClassifier
from nearfield360.slots.corridor import ApproachCorridorEvaluator
from nearfield360.slots.detector import ParkingSlotDetector
from nearfield360.slots.models import ParkingSlot
from nearfield360.tracking.models import TrackedObstacle
from nearfield360.utils.artifacts import ArtifactError, write_json

plan_app = typer.Typer(
    help="Autonomous parking motion trajectory planning and multi-stage maneuvers.",
    no_args_is_help=True,
)

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="JSON path to write parking trajectory plan report.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Optional destination PNG path for BEV trajectory visualization.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace an existing output artifact."),
]

SlotIdOption = Annotated[
    str | None,
    typer.Option(
        "--slot-id",
        help="Explicit parking slot identifier to target (defaults to best vacant slot).",
    ),
]

SlotsJsonOption = Annotated[
    Path | None,
    typer.Option(
        "--slots-json",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional precomputed parking slots JSON report to plan for directly.",
    ),
]

StartXOption = Annotated[
    float,
    typer.Option("--start-x", help="Initial ego vehicle X coordinate in metres."),
]

StartYOption = Annotated[
    float,
    typer.Option("--start-y", help="Initial ego vehicle Y coordinate in metres."),
]

StartYawOption = Annotated[
    float,
    typer.Option("--start-yaw", help="Initial ego vehicle yaw heading in radians."),
]


@plan_app.command(name="parking")
def plan_parking_command(
    context: typer.Context,
    output: OutputOption,
    root: DatasetRootOption = None,
    png: PngOption = None,
    overwrite: OverwriteOption = False,
    slot_id: SlotIdOption = None,
    slots_json: SlotsJsonOption = None,
    start_x: StartXOption = 0.0,
    start_y: StartYOption = 0.0,
    start_yaw: StartYawOption = 0.0,
) -> None:
    """Plan an autonomous, collision-free parking trajectory into a vacant parking slot."""
    config = get_state(context).config
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
    vehicle = AckermannVehicle(config.planner)
    planner = ParkingTrajectoryPlanner(config.planner)

    slots: list[ParkingSlot] = []
    occupancy_grid = np.zeros(grid.shape, dtype=np.float64)
    uncertainty_grid: NDArray[np.float64] | None = None
    obstacles: list[TrackedObstacle] = []

    # Path 1: Load precomputed slots JSON report if provided
    if slots_json is not None:
        try:
            doc = json.loads(slots_json.read_text(encoding="utf-8"))
            slots_raw = doc.get("slots", [])
            slots = [ParkingSlot(**s) for s in slots_raw]
        except Exception as exc:
            typer.secho(
                f"Failed to read slots JSON {slots_json}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None

    # Path 2: Discover dataset and extract perception evidence
    elif root is not None or config.paths.dataset_root is not None:
        dataset = discover_dataset(context, root)
        samples = [
            s
            for s in dataset
            if s.semantic_mask_path is not None and s.calibration_path is not None
        ]
        if not samples:
            typer.secho("No annotated dataset samples found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None

        accumulated_evidence: OccupancyEvidence | None = None
        markings_mask = np.zeros(grid.shape, dtype=np.uint8)

        sample = samples[0]
        if sample.semantic_mask_path is not None:
            cam = _build_camera(context, sample)
            mask = load_semantic_mask(sample.semantic_mask_path)

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

            is_marking = np.isin(mask, [1, 2]) & valid
            marking_ground = footprints_grid.points[..., :2][is_marking]
            if marking_ground.size > 0:
                indexed = grid.world_to_grid(marking_ground)
                valid_idx = indexed.valid
                if np.any(valid_idx):
                    rows_m = indexed.indices[valid_idx, 0]
                    cols_m = indexed.indices[valid_idx, 1]
                    markings_mask[rows_m, cols_m] = 255

        if accumulated_evidence is not None:
            occupancy_grid = accumulated_evidence.occupancy()
            uncertainty_grid = accumulated_evidence.uncertainty()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_markings = cv2.morphologyEx(markings_mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

        detector = ParkingSlotDetector(config.slots)
        candidate_slots = detector.detect_slots(
            grid=grid, markings_mask=clean_markings, obstacles=[]
        )
        classifier = SlotOccupancyClassifier(config.slots)
        classified_slots = classifier.classify_slots(
            slots=candidate_slots,
            occupancy=occupancy_grid,
            uncertainty=uncertainty_grid,
            grid=grid,
            obstacles=[],
            danger_threshold=config.risk.danger_occupancy,
        )
        corridor_eval = ApproachCorridorEvaluator(config.slots)
        slots = corridor_eval.evaluate_slots(
            slots=classified_slots,
            occupancy=occupancy_grid,
            grid=grid,
            obstacles=[],
            danger_threshold=config.risk.danger_occupancy,
        )

    if not slots:
        typer.secho("No parking slots provided or delineated.", fg=typer.colors.YELLOW, err=True)

    start_pose = (start_x, start_y, start_yaw)
    report = planner.plan_parking(
        slots=slots,
        occupancy=occupancy_grid,
        grid=grid,
        obstacles=obstacles,
        target_slot_id=slot_id,
        start_pose=start_pose,
        uncertainty=uncertainty_grid,
    )

    payload: dict[str, Any] = {
        "environment": environment_metadata(),
        "config": config.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }

    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Wrote {output}")
    if report.status == PlanStatus.SUCCESS and report.plan is not None:
        plan = report.plan
        typer.secho(
            f"Parking plan generated successfully for slot '{plan.slot_id}' "
            f"({plan.slot_type}): length {plan.total_length_m:.2f}m, "
            f"duration {plan.total_duration_s:.2f}s, "
            f"gears {plan.gear_switches + 1}, clearance {plan.min_clearance_m:.2f}m.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    elif report.status == PlanStatus.NO_VACANT_SLOT:
        typer.secho("No vacant parking slots available to target.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"Planning failed: {report.status}", fg=typer.colors.RED)

    if png is not None:
        if report.plan is not None:
            render_parking_plan_bev_overlay(
                grid=grid,
                plan=report.plan,
                slots=slots,
                vehicle=vehicle,
                output_path=png,
            )
            typer.echo(f"Wrote visualization {png}")
        else:
            typer.secho(
                "Skipping PNG rendering (no executable plan generated).",
                fg=typer.colors.YELLOW,
            )


__all__ = ["plan_app"]
