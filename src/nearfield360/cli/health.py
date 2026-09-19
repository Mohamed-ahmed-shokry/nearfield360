"""CLI commands for camera optical and sensor health assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.woodscape import CameraId
from nearfield360.health.detector import assess_camera_health
from nearfield360.utils.artifacts import ArtifactError, write_json

health_app = typer.Typer(
    help="Optical health, lens soiling, blur, and blockage assessment for surround cameras.",
    no_args_is_help=True,
)


@health_app.command("assess")
def assess_command(
    context: typer.Context,
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
            help="Path to camera image.",
        ),
    ],
    camera: Annotated[
        CameraId,
        typer.Option("--camera", "-c", help="Camera view identifier."),
    ] = CameraId.FRONT,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Optional path to write JSON health report.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print health assessment as JSON to standard output."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing output file."),
    ] = False,
) -> None:
    """Assess optical health, soiling, blur, and exposure for a single camera frame."""
    state = get_state(context)
    config = state.config.health

    try:
        rgb_image = load_rgb_image(image_path)
    except ImageReadError as exc:
        typer.secho(f"Image read error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    report = assess_camera_health(rgb_image, camera=camera.value, config=config)
    payload = report.model_dump(mode="json")

    if output is not None:
        try:
            write_json(output, payload, overwrite=overwrite)
        except ArtifactError as exc:
            typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        status_color = typer.colors.GREEN
        if report.status.value == "degraded":
            status_color = typer.colors.YELLOW
        elif report.status.value == "blocked":
            status_color = typer.colors.RED

        typer.secho(
            f"Camera {report.camera}: {report.status.value.upper()} "
            f"(Confidence: {report.metrics.confidence:.2f}, "
            f"Discount: {report.discount_weight:.2f})",
            fg=status_color,
            bold=True,
        )
        typer.echo(
            f"  Soiling score:   {report.metrics.soiling_score:.4f}\n"
            f"  Blur score:      {report.metrics.blur_score:.1f}\n"
            f"  Mean brightness: {report.metrics.mean_brightness:.1f}\n"
            f"  Contrast:        {report.metrics.contrast:.1f}\n"
            f"  Blockage ratio:  {report.metrics.blockage_ratio:.4f}"
        )
        if report.anomalies:
            typer.secho(
                f"  Anomalies:       {', '.join(a.value for a in report.anomalies)}",
                fg=typer.colors.YELLOW,
            )
        if output is not None:
            typer.echo(f"Wrote report to {output}")


__all__ = ["health_app"]
