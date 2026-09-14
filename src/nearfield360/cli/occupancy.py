"""Fuse camera evidence onto a local BEV grid and score spatial risk zones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import cv2
import numpy as np
import typer
from numpy.typing import NDArray

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import CalibrationError, load_calibration
from nearfield360.data.images import ImageReadError
from nearfield360.data.semantic import SemanticMaskError, load_semantic_mask
from nearfield360.data.woodscape import CameraId, WoodScapeSample
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.occupancy import (
    OccupancyEvidence,
    OccupancyPolicy,
    OccupancyPolicyError,
    RiskZone,
    circular_zone,
    corridor_zone,
    distance_weights,
    rasterize_occupancy,
    risk_report,
)
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.utils.artifacts import ArtifactError, write_json

occupancy_app = typer.Typer(
    help="Fuse camera evidence onto a local BEV grid and score risk zones.",
    no_args_is_help=True,
)

occupancy_app = typer.Typer(
    help="Fuse camera evidence onto a local BEV grid and score risk zones.",
    no_args_is_help=True,
)

CameraOption = Annotated[
    CameraId,
    typer.Option(
        "--camera",
        case_sensitive=True,
        help="Camera view to fuse evidence from (FV, MV, ML, MR).",
    ),
]

ThetaMaxOption = Annotated[
    float | None,
    typer.Option(
        "--theta-max",
        min=0.0,
        max=3.141592653589793,
        help="Usable angular limit in radians (defaults to configured geometry.theta_max).",
    ),
]

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        resolve_path=True,
        help="JSON occupancy-artifact (created atomically; refuses to overwrite).",
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
    typer.Option("--overwrite", help="Replace an existing occupancy artifact."),
]


def _camera_choice(value: str) -> CameraId:
    try:
        return CameraId(value)
    except ValueError as exc:  # pragma: no cover - typer already constrains choices
        raise typer.BadParameter(
            f"camera must be one of {', '.join(camera.value for camera in CameraId)}"
        ) from exc


def _configured_grid(context: typer.Context) -> BevGrid:
    config = get_state(context).config.bev
    return BevGrid(
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        resolution=config.resolution,
    )


def _configured_policy(context: typer.Context) -> OccupancyPolicy:
    config = get_state(context).config.occupancy
    try:
        return OccupancyPolicy.from_names(config.free_classes, config.occupied_classes)
    except OccupancyPolicyError as exc:
        typer.secho(f"Occupancy policy error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None


def _build_camera(
    context: typer.Context, sample: WoodScapeSample, theta_max: float | None
) -> CalibratedCamera:
    calibration_path = sample.calibration_path
    if calibration_path is None:
        typer.secho(f"Sample {sample.key.stem} lacks calibration.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    limit = theta_max if theta_max is not None else get_state(context).config.geometry.theta_max
    try:
        calibration = load_calibration(calibration_path)
        return CalibratedCamera.from_calibration(calibration, theta_max=limit)
    except (CalibrationError, ValueError) as exc:
        typer.secho(
            f"Calibration error for {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from None


def _select_samples(
    context: typer.Context, root: Path | None, camera: CameraId, samples_limit: int
) -> list[WoodScapeSample]:
    dataset = discover_dataset(context, root)
    selected: list[WoodScapeSample] = []
    for sample in dataset:
        if sample.key.camera is not camera:
            continue
        if sample.calibration_path is None or sample.semantic_mask_path is None:
            typer.secho(
                f"Sample {sample.key.stem} lacks calibration or semantic mask.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        selected.append(sample)
    if not selected:
        typer.secho(
            f"No {camera.value} samples with calibration and semantic mask found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    if samples_limit > 0 and len(selected) < samples_limit:
        typer.secho(
            f"Requested {samples_limit} samples but only {len(selected)} {camera.value} "
            "samples have annotations.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    return selected[:samples_limit] if samples_limit > 0 else selected


def _frame_evidence(
    context: typer.Context,
    sample: WoodScapeSample,
    grid: BevGrid,
    policy: OccupancyPolicy,
    theta_max: float | None,
) -> OccupancyEvidence:
    config = get_state(context).config
    camera = _build_camera(context, sample, theta_max)
    mask_path = sample.semantic_mask_path
    if mask_path is None:
        typer.secho(f"Sample {sample.key.stem} lacks semantic mask.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    try:
        mask = load_semantic_mask(mask_path)
    except (ImageReadError, SemanticMaskError, OSError) as exc:
        typer.secho(f"Mask error for {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    rows, cols = np.meshgrid(
        np.arange(mask.shape[0], dtype=np.float32),
        np.arange(mask.shape[1], dtype=np.float32),
        indexing="ij",
    )
    pixels = np.stack((cols, rows), axis=-1)
    rays = camera.pixel_rays_vehicle(pixels, check_image_bounds=True)
    footprints = intersect_ground(
        rays.origins,
        rays.directions,
        ground_z=config.geometry.ground_z,
        max_distance=config.geometry.max_distance,
    )
    valid = rays.valid & footprints.valid
    weights = distance_weights(footprints.distances, slope=config.occupancy.confidence_slope)
    return rasterize_occupancy(
        grid,
        footprints.points[..., :2],
        labels=mask,
        policy=policy,
        weights=weights,
        valid=valid,
    )


def _configured_zones(grid: BevGrid, context: typer.Context) -> list[RiskZone]:
    config = get_state(context).config.risk
    return [
        corridor_zone(
            grid,
            front_length=config.front_length,
            half_width=config.half_width,
            start=config.start_x,
        ),
        circular_zone(grid, center_xy=(0.0, 0.0), radius=0.5, name="near_circle"),
        circular_zone(grid, center_xy=(0.0, 0.0), radius=1.5, name="warning_circle"),
    ]


def _zone_summary(grid: BevGrid, zone: RiskZone) -> dict[str, Any]:
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


def _border(mask: np.ndarray) -> NDArray[np.bool_]:
    padded = np.pad(np.asarray(mask, dtype=np.bool_), 1)
    dilated = np.maximum.reduce(
        (padded[1:-1, :-2], padded[1:-1, 2:], padded[:-2, 1:-1], padded[2:, 1:-1])
    )
    return cast(NDArray[np.bool_], dilated & ~padded[1:-1, 1:-1])


_ZONE_COLORS = {
    "forward_corridor": (0, 255, 255),
    "near_circle": (255, 255, 255),
    "warning_circle": (255, 0, 255),
}


def _render_occupancy_png(
    grid: BevGrid, evidence: OccupancyEvidence, zones: list[RiskZone], path: Path
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
        color = _ZONE_COLORS.get(zone.name, (128, 128, 128))
        for row, col in np.argwhere(_border(zone.mask)):
            canvas[row, col] = color
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"unable to write PNG: {path}")
    typer.echo(f"Wrote {path}")


def _evidence_summary(evidence: OccupancyEvidence, min_evidence: int) -> dict[str, Any]:
    confident = evidence.observed >= min_evidence
    occupancy = evidence.occupancy()[confident]
    mean = float(np.mean(occupancy)) if occupancy.size else None
    return {
        "cells": evidence.grid.height * evidence.grid.width,
        "observed_cells": int(confident.sum()),
        "observations": int(evidence.observed.sum()),
        "mean_occupancy": mean,
    }


@occupancy_app.command("layer")
def occupancy_layer(
    context: typer.Context,
    output: OutputOption,
    root: DatasetRootOption = None,
    camera: CameraOption = CameraId.FRONT,
    samples: Annotated[
        int,
        typer.Option("--samples", min=0, help="Number of frames to fuse (0 for all)."),
    ] = 0,
    overwrite: OverwriteOption = False,
    png: PngOption = None,
    theta_max: ThetaMaxOption = None,
) -> None:
    """Fuse semantic-grounded evidence from all frames into one occupancy layer."""
    grid = _configured_grid(context)
    policy = _configured_policy(context)
    selected = _select_samples(context, root, camera, samples)
    first = selected[0]
    fused = _frame_evidence(context, first, grid, policy, theta_max)
    typer.echo(f"[1/{len(selected)}] fused {first.key.stem}")
    for index, sample in enumerate(selected[1:], start=2):
        frame = _frame_evidence(context, sample, grid, policy, theta_max)
        fused = fused.add(frame)
        typer.echo(f"[{index}/{len(selected)}] fused {sample.key.stem}")
    min_evidence = get_state(context).config.occupancy.min_evidence
    zones = _configured_zones(grid, context)
    reports = risk_report(
        zones,
        fused,
        min_evidence=min_evidence,
        danger_occupancy=get_state(context).config.risk.danger_occupancy,
    )
    payload: dict[str, Any] = {
        "environment": environment_metadata(),
        "config": get_state(context).config.model_dump(mode="json"),
        "samples": {
            "requested": len(selected) if samples == 0 else samples,
            "evaluated": len(selected),
            "camera": camera.value,
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
        "zones": [_zone_summary(grid, zone) for zone in zones],
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
            }
            for report in reports
        ],
    }
    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Wrote {output}")
    if png is not None:
        try:
            _render_occupancy_png(grid, fused, zones, png)
        except (OSError, ValueError) as exc:
            typer.secho(f"PNG error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None


@occupancy_app.command("zones")
def occupancy_zones(
    context: typer.Context,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable zone output.")
    ] = False,
) -> None:
    """Report configured risk zones over the BEV grid without needing a dataset."""
    grid = _configured_grid(context)
    zones = _configured_zones(grid, context)
    payload: dict[str, Any] = {
        "grid": {
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "y_min": grid.y_min,
            "y_max": grid.y_max,
            "resolution": grid.resolution,
            "width": grid.width,
            "height": grid.height,
        },
        "zones": [_zone_summary(grid, zone) for zone in zones],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for zone in payload["zones"]:
        typer.echo(
            f"{zone['name']}: {zone['cells']} cells, {zone['area_m2']:.3f} m^2, "
            f"x=[{zone['x_min']}, {zone['x_max']}] y=[{zone['y_min']}, {zone['y_max']}]"
        )


__all__ = ["occupancy_app"]
