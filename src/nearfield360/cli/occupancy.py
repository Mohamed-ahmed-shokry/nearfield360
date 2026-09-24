"""Fuse camera evidence onto a local BEV grid and score spatial risk zones."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, cast

import cv2
import numpy as np
import typer
from numpy.typing import NDArray

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.inference_common import BackendOption, load_backend
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import CalibrationError, load_calibration
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import SemanticMaskError, load_semantic_mask
from nearfield360.data.woodscape import CameraId, WoodScapeSample
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.health import (
    CameraHealthReport,
    CameraHealthStatus,
    apply_health_discount,
    assess_camera_health,
)
from nearfield360.occupancy import (
    OccupancyEvidence,
    OccupancyPolicy,
    OccupancyPolicyError,
    RiskZone,
    distance_weights,
    rasterize_occupancy,
    risk_report,
    surround_parking_zones,
)
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.perception.inference import SemanticSegmentationEngine
from nearfield360.utils.artifacts import ArtifactError, write_json

occupancy_app = typer.Typer(
    help="Fuse camera evidence onto a local BEV grid and score risk zones.",
    no_args_is_help=True,
)

CameraOption = Annotated[
    CameraId,
    typer.Option(
        "--camera",
        case_sensitive=True,
        help="Camera view to fuse evidence from (FV, RV, MVL, MVR).",
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

UncertaintyPngOption = Annotated[
    Path | None,
    typer.Option(
        "--uncertainty-png",
        resolve_path=True,
        help="Optional PNG visualization of per-cell Bayesian occupancy uncertainty.",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace an existing occupancy artifact."),
]

HealthAwareOption = Annotated[
    bool,
    typer.Option(
        "--health-aware",
        help="Assess camera optical health and discount degraded or soiled camera evidence.",
    ),
]

ModelOption = Annotated[
    Path | None,
    typer.Option(
        "--model",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="ONNX semantic segmentation model path for live perception.",
    ),
]


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
    context: typer.Context,
    root: Path | None,
    camera: CameraId,
    samples_limit: int,
    allow_live_model: bool = False,
) -> list[WoodScapeSample]:
    dataset = discover_dataset(context, root)
    selected: list[WoodScapeSample] = []
    for sample in dataset:
        if sample.key.camera is not camera:
            continue
        missing_ann = sample.calibration_path is None or (
            False if allow_live_model else sample.semantic_mask_path is None
        )
        if missing_ann:
            desc = "calibration" if allow_live_model else "calibration or semantic mask"
            typer.secho(
                f"Sample {sample.key.stem} lacks {desc}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        selected.append(sample)
    if not selected:
        desc = "image" if allow_live_model else "semantic mask"
        typer.secho(
            f"No {camera.value} samples with calibration and {desc} found.",
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


def _select_all_cameras(
    context: typer.Context,
    root: Path | None,
    samples_limit: int,
    allow_live_model: bool = False,
) -> list[tuple[str, list[WoodScapeSample]]]:
    """Group samples by frame_id across all cameras."""
    dataset = discover_dataset(context, root)
    by_frame: dict[str, list[WoodScapeSample]] = defaultdict(list)
    for sample in dataset:
        missing_ann = sample.calibration_path is None or (
            False if allow_live_model else sample.semantic_mask_path is None
        )
        if missing_ann:
            continue
        by_frame[sample.key.frame_id].append(sample)
    frames = sorted(by_frame.keys())
    complete_frames: list[tuple[str, list[WoodScapeSample]]] = []
    for frame_id in frames:
        cameras = {sample.key.camera for sample in by_frame[frame_id]}
        missing = set(CameraId) - cameras
        if missing:
            missing_names = ", ".join(cam.value for cam in sorted(missing, key=lambda c: c.value))
            typer.echo(
                f"Skipping frame {frame_id}: missing cameras {missing_names}",
                err=True,
            )
            continue
        complete_frames.append((frame_id, by_frame[frame_id]))
    if not complete_frames:
        typer.secho(
            "No frames with all four cameras and required annotations found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    if samples_limit > 0 and len(complete_frames) < samples_limit:
        typer.secho(
            f"Requested {samples_limit} frames but only {len(complete_frames)} complete "
            "multi-camera frames found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    return complete_frames[:samples_limit] if samples_limit > 0 else complete_frames


def _frame_evidence(
    context: typer.Context,
    sample: WoodScapeSample,
    grid: BevGrid,
    policy: OccupancyPolicy,
    theta_max: float | None,
    health_aware: bool = False,
    model_engine: SemanticSegmentationEngine | None = None,
) -> tuple[OccupancyEvidence, CameraHealthReport | None]:
    config = get_state(context).config
    camera = _build_camera(context, sample, theta_max)
    if model_engine is not None:
        try:
            rgb_img = load_rgb_image(sample.image_path)
            mask, _ = model_engine.predict(rgb_img)
        except Exception as exc:
            typer.secho(
                f"Segmentation inference error for {sample.key.stem}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
    else:
        mask_path = sample.semantic_mask_path
        if mask_path is None:
            typer.secho(
                f"Sample {sample.key.stem} lacks semantic mask.",
                fg=typer.colors.RED,
                err=True,
            )
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
    evidence = rasterize_occupancy(
        grid,
        footprints.points[..., :2],
        labels=mask,
        policy=policy,
        weights=weights,
        valid=valid,
    )

    report: CameraHealthReport | None = None
    if health_aware:
        try:
            rgb = load_rgb_image(sample.image_path)
            report = assess_camera_health(rgb, camera=sample.key.camera.value, config=config.health)
            evidence = apply_health_discount(evidence, report)
            if report.status != CameraHealthStatus.HEALTHY:
                typer.echo(
                    f"Camera {sample.key.camera.value} ({sample.key.stem}) degraded: "
                    f"{report.status.value} (discount factor: "
                    f"{report.discount_weight:.2f})",
                    err=True,
                )
        except (ImageReadError, OSError) as exc:
            typer.secho(
                f"Warning: could not read image for health assessment {sample.key.stem}: {exc}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    return evidence, report


def _configured_zones(grid: BevGrid, context: typer.Context) -> list[RiskZone]:
    config = get_state(context).config.risk
    return list(
        surround_parking_zones(
            grid,
            front_length=config.front_length,
            rear_length=config.rear_length,
            half_width=config.half_width,
            start_x=config.start_x,
            rear_start_x=config.rear_start_x,
            lateral_width=config.lateral_width,
            vehicle_x_min=config.vehicle_x_min,
            vehicle_x_max=config.vehicle_x_max,
            near_radius=config.near_radius,
            warning_radius=config.warning_radius,
        )
    )


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
    "rear_corridor": (0, 165, 255),
    "left_clearance": (255, 191, 0),
    "right_clearance": (255, 144, 30),
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


def _render_uncertainty_png(
    grid: BevGrid, evidence: OccupancyEvidence, zones: list[RiskZone], path: Path
) -> None:
    uncertainty = evidence.uncertainty()
    observed = evidence.observed > 0
    norm_unc = np.zeros(grid.shape, dtype=np.uint8)
    if np.any(observed):
        clipped = np.clip(np.nan_to_num(uncertainty, nan=0.0) / 0.5, 0.0, 1.0)
        norm_unc[observed] = np.asarray(clipped[observed] * 255.0, dtype=np.uint8)
    colored = cv2.applyColorMap(norm_unc, cv2.COLORMAP_INFERNO)
    colored[~observed] = (0, 0, 0)
    for zone in zones:
        color = _ZONE_COLORS.get(zone.name, (128, 128, 128))
        for row, col in np.argwhere(_border(zone.mask)):
            colored[row, col] = color
    if not cv2.imwrite(str(path), colored):
        raise ValueError(f"unable to write PNG: {path}")
    typer.echo(f"Wrote {path}")


def _evidence_summary(evidence: OccupancyEvidence, min_evidence: int) -> dict[str, Any]:
    confident = evidence.observed >= min_evidence
    occupancy = evidence.occupancy()[confident]
    uncertainty = evidence.uncertainty()[confident]
    valid_occ = occupancy[np.isfinite(occupancy)]
    valid_unc = uncertainty[np.isfinite(uncertainty)]
    mean_occ = float(np.mean(valid_occ)) if valid_occ.size else None
    mean_unc = float(np.mean(valid_unc)) if valid_unc.size else None
    max_unc = float(np.max(valid_unc)) if valid_unc.size else None
    return {
        "cells": evidence.grid.height * evidence.grid.width,
        "observed_cells": int(confident.sum()),
        "observations": int(evidence.observed.sum()),
        "mean_occupancy": mean_occ,
        "mean_uncertainty": mean_unc,
        "max_uncertainty": max_unc,
    }


@occupancy_app.command("layer")
def occupancy_layer(
    context: typer.Context,
    output: OutputOption,
    root: DatasetRootOption = None,
    camera: CameraOption = CameraId.FRONT,
    all_cameras: Annotated[
        bool,
        typer.Option(
            "--all-cameras",
            help="Fuse evidence from all four cameras per frame (overrides --camera).",
        ),
    ] = False,
    samples: Annotated[
        int,
        typer.Option("--samples", min=0, help="Number of frames to fuse (0 for all)."),
    ] = 0,
    overwrite: OverwriteOption = False,
    png: PngOption = None,
    uncertainty_png: UncertaintyPngOption = None,
    theta_max: ThetaMaxOption = None,
    health_aware: HealthAwareOption = False,
    model: ModelOption = None,
    backend: BackendOption = None,
) -> None:
    """Fuse semantic-grounded evidence into one occupancy layer."""
    grid = _configured_grid(context)
    policy = _configured_policy(context)

    model_engine: SemanticSegmentationEngine | None = None
    if model is not None:
        loaded = load_backend(context, model, backend=backend)
        model_engine = SemanticSegmentationEngine(backend=loaded)

    health_reports: list[CameraHealthReport] = []
    if all_cameras:
        frames = _select_all_cameras(context, root, samples, allow_live_model=model is not None)
        per_camera_counts: dict[str, int] = {cam.value: 0 for cam in CameraId}
        fused: OccupancyEvidence | None = None
        for frame_index, (frame_id, frame_samples) in enumerate(frames, start=1):
            frame_evidence: OccupancyEvidence | None = None
            for sample in frame_samples:
                evidence, report = _frame_evidence(
                    context,
                    sample,
                    grid,
                    policy,
                    theta_max,
                    health_aware=health_aware,
                    model_engine=model_engine,
                )
                if report is not None:
                    health_reports.append(report)
                per_camera_counts[sample.key.camera.value] += 1
                frame_evidence = (
                    evidence if frame_evidence is None else frame_evidence.add(evidence)
                )
            if frame_evidence is not None:
                fused = frame_evidence if fused is None else fused.add(frame_evidence)
            typer.echo(f"[{frame_index}/{len(frames)}] fused frame {frame_id}")
        if fused is None:
            typer.secho(
                "No occupancy evidence could be fused.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        samples_payload: dict[str, Any] = {
            "requested": len(frames) if samples == 0 else samples,
            "evaluated": len(frames),
            "camera": "all",
            "per_camera": per_camera_counts,
        }
        if health_aware:
            samples_payload["health_aware"] = True
        if model is not None:
            samples_payload["model"] = str(model)
    else:
        selected = _select_samples(
            context, root, camera, samples, allow_live_model=model is not None
        )
        first = selected[0]
        fused, first_report = _frame_evidence(
            context,
            first,
            grid,
            policy,
            theta_max,
            health_aware=health_aware,
            model_engine=model_engine,
        )
        if first_report is not None:
            health_reports.append(first_report)
        typer.echo(f"[1/{len(selected)}] fused {first.key.stem}")
        for index, sample in enumerate(selected[1:], start=2):
            frame, report = _frame_evidence(
                context,
                sample,
                grid,
                policy,
                theta_max,
                health_aware=health_aware,
                model_engine=model_engine,
            )
            if report is not None:
                health_reports.append(report)
            fused = fused.add(frame)
            typer.echo(f"[{index}/{len(selected)}] fused {sample.key.stem}")
        samples_payload = {
            "requested": len(selected) if samples == 0 else samples,
            "evaluated": len(selected),
            "camera": camera.value,
        }
        if health_aware:
            samples_payload["health_aware"] = True
        if model is not None:
            samples_payload["model"] = str(model)

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
        "samples": samples_payload,
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
                "mean_uncertainty": report.mean_uncertainty,
                "max_uncertainty": report.max_uncertainty,
            }
            for report in reports
        ],
    }
    if health_aware:
        payload["health"] = [r.model_dump(mode="json") for r in health_reports]
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
    if uncertainty_png is not None:
        try:
            _render_uncertainty_png(grid, fused, zones, uncertainty_png)
        except (OSError, ValueError) as exc:
            typer.secho(f"Uncertainty PNG error: {exc}", fg=typer.colors.RED, err=True)
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
