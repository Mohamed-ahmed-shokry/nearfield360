"""Calibrated fisheye inspection and pixel-to-ground debugging commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from pydantic import ValidationError

from nearfield360.cli.state import get_state
from nearfield360.data.calibration import CalibrationError, load_calibration
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground

geometry_app = typer.Typer(
    help="Inspect fisheye calibration and map pixels to vehicle ground.",
    no_args_is_help=True,
)

CalibrationOption = Annotated[
    Path,
    typer.Option(
        "--calibration",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="WoodScape per-image calibration JSON.",
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


def _build_camera(
    calibration_path: Path, theta_max: float | None, context: typer.Context
) -> CalibratedCamera:
    limit = theta_max if theta_max is not None else get_state(context).config.geometry.theta_max
    try:
        calibration = load_calibration(calibration_path)
        return CalibratedCamera.from_calibration(calibration, theta_max=limit)
    except (CalibrationError, ValidationError, ValueError) as exc:
        typer.secho(f"Calibration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None


def _camera_payload(camera: CalibratedCamera) -> dict[str, Any]:
    intrinsics = camera.model.intrinsics
    return {
        "camera": camera.name.value,
        "image": {"width": intrinsics.width, "height": intrinsics.height},
        "principal_point": list(intrinsics.principal_point),
        "coefficients": list(intrinsics.coefficients),
        "aspect_ratio": intrinsics.aspect_ratio,
        "theta_max": camera.model.theta_max,
        "radius_max": camera.model.radius_max,
        "translation": [float(value) for value in camera.camera_to_vehicle.translation],
        "rotation": [[float(value) for value in row] for row in camera.camera_to_vehicle.rotation],
    }


@geometry_app.command("info")
def geometry_info(
    context: typer.Context,
    calibration: CalibrationOption,
    theta_max: ThetaMaxOption = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable calibration output.")
    ] = False,
) -> None:
    """Validate a calibration file and report its fisheye and vehicle geometry."""
    camera = _build_camera(calibration, theta_max, context)
    payload = _camera_payload(camera)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Camera: {payload['camera']}")
    typer.echo(f"Image: {payload['image']['width']}x{payload['image']['height']}")
    typer.echo(f"Theta max: {payload['theta_max']:.6g} rad")
    typer.echo(f"Radius max: {payload['radius_max']:.6g} px")
    typer.echo(f"Translation: {payload['translation']}")
    typer.echo("Rotation:")
    for row in payload["rotation"]:
        typer.echo(f"  {row}")


def _parse_pixels(values: tuple[str, ...]) -> np.ndarray:
    pixels: list[list[float]] = []
    for value in values:
        try:
            first, second = value.split(",", maxsplit=1)
            pixels.append([float(first.strip()), float(second.strip())])
        except ValueError as exc:
            raise typer.BadParameter(f"Pixel must look like 'x,y', got {value!r}") from exc
    try:
        array = np.asarray(pixels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"Pixels must be finite numbers: {exc}") from exc
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise typer.BadParameter("Provide at least one 'x,y' pixel")
    if not np.all(np.isfinite(array)):
        raise typer.BadParameter("Pixels must contain only finite coordinates")
    return array


@geometry_app.command("ground")
def geometry_ground(
    context: typer.Context,
    calibration: CalibrationOption,
    pixel: Annotated[
        list[str] | None,
        typer.Option(
            "--pixel",
            help="Pixel as 'x,y' in stored-image coordinates; repeat for batches.",
        ),
    ] = None,
    theta_max: ThetaMaxOption = None,
    ground_z: Annotated[
        float | None,
        typer.Option(help="Ground plane height in metres (defaults to configured value)."),
    ] = None,
    max_distance: Annotated[
        float | None,
        typer.Option(min=0.0, help="Maximum Euclidean ray length in metres (defaults to config)."),
    ] = None,
    check_bounds: Annotated[
        bool,
        typer.Option(
            "--check-bounds/--ignore-bounds",
            help="Enforce the half-open image rectangle.",
        ),
    ] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable footprint output.")
    ] = False,
) -> None:
    """Unproject pixels to vehicle rays and intersect them with the ground plane."""
    if not pixel:
        typer.secho("Provide at least one --pixel 'x,y'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    camera = _build_camera(calibration, theta_max, context)
    config = get_state(context).config.geometry
    plane_z = config.ground_z if ground_z is None else ground_z
    limit = config.max_distance if max_distance is None else max_distance
    pixels = _parse_pixels(tuple(pixel))
    rays = camera.pixel_rays_vehicle(pixels, check_image_bounds=check_bounds)
    footprints = intersect_ground(
        rays.origins, rays.directions, ground_z=plane_z, max_distance=limit
    )
    valid = rays.valid & footprints.valid
    payload: dict[str, Any] = {
        "camera": camera.name.value,
        "ground_z": plane_z,
        "max_distance": limit,
        "footprints": [
            {
                "pixel": [float(pixels[index, 0]), float(pixels[index, 1])],
                "valid": bool(valid[index]),
                "origin": [float(value) for value in rays.origins[index]]
                if bool(valid[index])
                else [None, None, None],
                "ground": [float(value) for value in footprints.points[index]]
                if bool(valid[index])
                else [None, None, None],
                "distance": float(footprints.distances[index]) if bool(valid[index]) else None,
            }
            for index in range(len(pixels))
        ],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Camera: {payload['camera']}; ground_z={plane_z:.6g}; max_distance={limit:.6g}")
    for item in payload["footprints"]:
        if item["valid"]:
            typer.echo(f"{item['pixel']} -> ground {item['ground']} ({item['distance']:.3f} m)")
        else:
            typer.echo(f"{item['pixel']} -> unknown (invalid ray or no ground intersection)")


@geometry_app.command("bev")
def geometry_bev(
    context: typer.Context,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable grid output.")
    ] = False,
) -> None:
    """Report the configured local BEV grid extent and cell counts."""
    config = get_state(context).config.bev
    grid = BevGrid(
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        resolution=config.resolution,
    )
    payload = {
        "x_min": grid.x_min,
        "x_max": grid.x_max,
        "y_min": grid.y_min,
        "y_max": grid.y_max,
        "resolution": grid.resolution,
        "width": grid.width,
        "height": grid.height,
        "shape": list(grid.shape),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(
        f"BEV grid: x=[{grid.x_min:.6g}, {grid.x_max:.6g}) "
        f"y=[{grid.y_min:.6g}, {grid.y_max:.6g}) "
        f"resolution={grid.resolution:.6g} m"
    )
    typer.echo(f"Cells: width={grid.width} height={grid.height} shape={grid.shape}")


__all__ = ["geometry_app"]
