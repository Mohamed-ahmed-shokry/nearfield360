"""CLI commands for controlled robustness testing, corruptions, and plot exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.state import get_state
from nearfield360.data.calibration import (
    CalibrationError,
    load_calibration,
)
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import load_semantic_mask
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    distance_weights,
    rasterize_occupancy,
)
from nearfield360.occupancy.risk import surround_parking_zones
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.robustness.calibration import (
    perturb_camera_calibration,
)
from nearfield360.robustness.corruptions import (
    CorruptionType,
    apply_sensor_corruption,
)
from nearfield360.robustness.plots import (
    generate_robustness_html_dashboard,
    render_raster_line_chart,
    render_svg_line_chart,
)
from nearfield360.utils.artifacts import ArtifactError, write_json

robustness_app = typer.Typer(
    help="Controlled sensor corruptions, calibration perturbations, and diagnostic plots.",
    no_args_is_help=True,
)


@robustness_app.command("corrupt")
def corrupt_image_command(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to input image.",
        ),
    ],
    corruption_type: Annotated[
        CorruptionType,
        typer.Option(
            "--type",
            "-t",
            help="Synthetic sensor corruption model to apply.",
        ),
    ] = CorruptionType.LENS_SOILING,
    severity: Annotated[
        int,
        typer.Option(
            "--severity",
            "-s",
            min=1,
            max=5,
            help="Severity level from 1 (minor) to 5 (extreme).",
        ),
    ] = 1,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for deterministic degradation."),
    ] = 42,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Output path for the corrupted image.",
        ),
    ] = Path("outputs/corrupted.png"),
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing image file."),
    ] = False,
) -> None:
    """Apply a synthetic sensor corruption to an input image and save the result."""
    if output.exists() and not overwrite:
        typer.secho(f"Output file already exists: {output}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        rgb_image = load_rgb_image(image_path)
    except ImageReadError as exc:
        typer.secho(f"Image read error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    corrupted_rgb = apply_sensor_corruption(
        rgb_image,
        corruption_type=corruption_type,
        severity=severity,
        seed=seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output), bgr)
    typer.echo(
        f"Saved corrupted image ({corruption_type.value}, severity {severity}) to {output}"
    )


@robustness_app.command("perturb-calibration")
def perturb_calibration_command(
    calibration_path: Annotated[
        Path,
        typer.Option(
            "--calibration",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to original WoodScape calibration JSON.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="Output path for perturbed calibration JSON.",
        ),
    ],
    roll: Annotated[float, typer.Option("--roll", help="Roll rotation in degrees.")] = 0.0,
    pitch: Annotated[float, typer.Option("--pitch", help="Pitch rotation in degrees.")] = 0.0,
    yaw: Annotated[float, typer.Option("--yaw", help="Yaw rotation in degrees.")] = 0.0,
    dx: Annotated[
        float,
        typer.Option("--dx", help="Translation delta along vehicle X in metres."),
    ] = 0.0,
    dy: Annotated[
        float,
        typer.Option("--dy", help="Translation delta along vehicle Y in metres."),
    ] = 0.0,
    dz: Annotated[
        float,
        typer.Option("--dz", help="Translation delta along vehicle Z in metres."),
    ] = 0.0,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing calibration output."),
    ] = False,
) -> None:
    """Generate a perturbed camera calibration file with modified extrinsics."""
    try:
        original = load_calibration(calibration_path)
    except CalibrationError as exc:
        typer.secho(f"Calibration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    perturbed = perturb_camera_calibration(
        original,
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
        translation_m=(dx, dy, dz),
    )

    payload = perturbed.model_dump(mode="json")
    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        f"Wrote perturbed calibration ({original.name.value}: "
        f"roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°, "
        f"dxyz=({dx:.2f}, {dy:.2f}, {dz:.2f})m) to {output}"
    )


@robustness_app.command("benchmark")
def benchmark_command(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="JSON robustness benchmark artifact (created atomically; refuses to overwrite).",
        ),
    ],
    camera: Annotated[
        CameraId,
        typer.Option("--camera", help="WoodScape camera view to evaluate."),
    ] = CameraId.FRONT,
    samples_limit: Annotated[
        int,
        typer.Option(
            "--samples",
            min=1,
            help="Maximum samples to fuse into the clean baseline layer.",
        ),
    ] = 5,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing benchmark artifact."),
    ] = False,
    root: DatasetRootOption = None,
) -> None:
    """Run full synthetic corruption and calibration perturbation benchmark sweeps."""
    state = get_state(context)
    dataset = discover_dataset(context, root)
    config = state.config

    # Find calibration and sample rays for requested camera
    bev_config = config.bev
    grid = BevGrid(
        x_min=bev_config.x_min,
        x_max=bev_config.x_max,
        y_min=bev_config.y_min,
        y_max=bev_config.y_max,
        resolution=bev_config.resolution,
    )
    policy = OccupancyPolicy.from_names(
        free=config.occupancy.free_classes,
        occupied=config.occupancy.occupied_classes,
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

    clean_evidence: OccupancyEvidence | None = None
    calibrated_cam: CalibratedCamera | None = None
    sample_pixels_list: list[Any] = []
    pixel_classes_list: list[Any] = []
    fused_count = 0

    for sample in dataset:
        if sample.key.camera != camera:
            continue
        if sample.calibration_path is None or sample.semantic_mask_path is None:
            continue

        try:
            calib_record = load_calibration(sample.calibration_path)
            cam = CalibratedCamera.from_calibration(
                calib_record, theta_max=config.geometry.theta_max
            )
            mask = load_semantic_mask(sample.semantic_mask_path)
        except Exception as exc:
            typer.secho(f"Sample error for {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None

        if calibrated_cam is None:
            calibrated_cam = cam

        # Subsample pixels for fast ray projection
        step = 16
        ys, xs = mask.nonzero()
        if len(ys) > 0:
            sub_y = ys[::step]
            sub_x = xs[::step]
            coords = np.column_stack((sub_x, sub_y)).astype(np.float64)
            labels = mask[sub_y, sub_x]
            sample_pixels_list.append(coords)
            pixel_classes_list.append(labels)

            rays = cam.pixel_rays_vehicle(coords)
            footprints = intersect_ground(
                rays.origins,
                rays.directions,
                ground_z=config.geometry.ground_z,
                max_distance=config.geometry.max_distance,
            )
            valid = rays.valid & footprints.valid
            weights = distance_weights(
                footprints.distances, slope=config.occupancy.confidence_slope
            )
            frame_evidence = rasterize_occupancy(
                grid,
                footprints.points[..., :2],
                labels=labels,
                policy=policy,
                weights=weights,
                valid=valid,
            )
            clean_evidence = (
                frame_evidence
                if clean_evidence is None
                else clean_evidence.add(frame_evidence)
            )
            fused_count += 1
            if fused_count >= samples_limit:
                break

    if clean_evidence is None or calibrated_cam is None or not sample_pixels_list:
        typer.secho(
            f"No valid annotated samples found for camera {camera.value}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    all_pixels = np.concatenate(sample_pixels_list, axis=0)
    all_classes = np.concatenate(pixel_classes_list, axis=0)

    # Run sweeps
    from nearfield360.robustness.benchmark import (
        RobustnessReport,
        run_calibration_sweep,
        run_corruption_sweep,
    )

    corr_records = run_corruption_sweep(
        clean_evidence,
        grid=grid,
        zones=zones,
        config=config,
        severities=config.robustness.severities,
        seed=config.robustness.seed,
    )

    calib_records = run_calibration_sweep(
        clean_evidence,
        calibrated_cam,
        all_pixels,
        all_classes,
        grid=grid,
        zones=zones,
        config=config,
        rotation_perturbations_deg=config.robustness.rotation_perturbations_deg,
        translation_perturbations_m=config.robustness.translation_perturbations_m,
    )

    report = RobustnessReport(
        environment=environment_metadata(),
        config=config.model_dump(mode="json"),
        corruption_sweeps=corr_records,
        calibration_sweeps=calib_records,
    )

    try:
        write_json(output, report.as_dict(), overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        f"Robustness benchmark complete ({len(corr_records)} corruption sweeps, "
        f"{len(calib_records)} calibration sweeps). Wrote report to {output}"
    )


@robustness_app.command("plot")
def plot_command(
    report_path: Annotated[
        Path,
        typer.Option(
            "--report",
            "-r",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to input JSON robustness benchmark report.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            resolve_path=True,
            help="Directory to save generated charts and HTML dashboard.",
        ),
    ] = Path("outputs/robustness"),
    html_flag: Annotated[
        bool,
        typer.Option("--html/--no-html", help="Generate interactive HTML dashboard."),
    ] = True,
    svg_flag: Annotated[
        bool,
        typer.Option("--svg/--no-svg", help="Generate vector SVG charts."),
    ] = True,
    png_flag: Annotated[
        bool,
        typer.Option("--png/--no-png", help="Generate raster PNG charts."),
    ] = True,
) -> None:
    """Generate diagnostic plots and an interactive HTML report from a benchmark artifact."""
    try:
        text = report_path.read_text(encoding="utf-8")
        report_dict: dict[str, Any] = json.loads(text)
    except Exception as exc:
        typer.secho(f"Failed to load report {report_path}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    output_dir.mkdir(parents=True, exist_ok=True)

    corr_sweeps = report_dict.get("corruption_sweeps", [])
    calib_sweeps = report_dict.get("calibration_sweeps", [])

    corr_series: dict[str, list[tuple[float, float]]] = {}
    for rec in corr_sweeps:
        c_type = rec["corruption"]
        sev = float(rec["severity"])
        mae = float(rec["occupancy_metrics"]["mae"])
        corr_series.setdefault(c_type, []).append((sev, mae))

    calib_series: dict[str, list[tuple[float, float]]] = {}
    for rec in calib_sweeps:
        axis = f"{rec['axis']} ({rec['unit']})"
        mag = float(rec["magnitude"])
        mae = float(rec["occupancy_metrics"]["mae"])
        calib_series.setdefault(axis, []).append((mag, mae))

    generated_files: list[Path] = []

    if svg_flag:
        svg_corr = render_svg_line_chart(
            title="BEV Occupancy Degradation (MAE) vs. Sensor Corruption Severity",
            x_label="Corruption Severity (1 to 5)",
            y_label="Occupancy Mean Absolute Error (MAE)",
            series_dict=corr_series,
        )
        svg_corr_path = output_dir / "corruption_degradation.svg"
        svg_corr_path.write_text(svg_corr, encoding="utf-8")
        generated_files.append(svg_corr_path)

        svg_calib = render_svg_line_chart(
            title="BEV Occupancy Degradation (MAE) vs. Calibration Misalignment",
            x_label="Perturbation Magnitude",
            y_label="Occupancy Mean Absolute Error (MAE)",
            series_dict=calib_series,
        )
        svg_calib_path = output_dir / "calibration_sensitivity.svg"
        svg_calib_path.write_text(svg_calib, encoding="utf-8")
        generated_files.append(svg_calib_path)

    if png_flag:
        png_corr_path = output_dir / "corruption_degradation.png"
        render_raster_line_chart(
            title="BEV Occupancy Degradation vs. Corruption Severity",
            x_label="Severity",
            y_label="Occupancy MAE",
            series_dict=corr_series,
            output_path=png_corr_path,
        )
        generated_files.append(png_corr_path)

        png_calib_path = output_dir / "calibration_sensitivity.png"
        render_raster_line_chart(
            title="BEV Occupancy Degradation vs. Calibration Misalignment",
            x_label="Perturbation",
            y_label="Occupancy MAE",
            series_dict=calib_series,
            output_path=png_calib_path,
        )
        generated_files.append(png_calib_path)

    if html_flag:
        html_content = generate_robustness_html_dashboard(report_dict)
        html_path = output_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        generated_files.append(html_path)

    typer.echo(f"Rendered {len(generated_files)} visual artifacts to {output_dir}")
    for f in generated_files:
        typer.echo(f"  - {f.name}")


__all__ = ["robustness_app"]
