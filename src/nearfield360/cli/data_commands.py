"""Commands for inspecting locally acquired external datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from nearfield360.cli.state import get_state
from nearfield360.data import (
    DatasetLayoutError,
    DatasetValidationReport,
    ValidationPolicy,
    WoodScapeDataset,
    validate_dataset,
)

data_app = typer.Typer(help="Discover and validate local dataset files.", no_args_is_help=True)


def _resolve_dataset_root(context: typer.Context, override: Path | None) -> Path:
    if override is not None:
        return override
    configured = get_state(context).config.paths.dataset_root
    if configured is None:
        typer.secho(
            "Dataset root is not configured. Use --root or NEARFIELD360_PATHS__DATASET_ROOT.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return configured


def _report_payload(report: DatasetValidationReport) -> dict[str, Any]:
    return {
        "valid": report.is_valid,
        "sample_count": report.sample_count,
        "camera_counts": {camera.value: count for camera, count in report.camera_counts.items()},
        "available": {
            "previous_images": report.previous_image_count,
            "semantic_masks": report.semantic_mask_count,
            "calibrations": report.calibration_count,
        },
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "issues_truncated": report.issues_truncated,
        "issues": [
            {
                "severity": issue.severity.value,
                "code": issue.code,
                "sample": None if issue.sample_key is None else issue.sample_key.stem,
                "message": issue.message,
            }
            for issue in report.issues
        ],
    }


def _render_report(report: DatasetValidationReport) -> None:
    status = "VALID" if report.is_valid else "INVALID"
    typer.echo(f"WoodScape dataset: {status}")
    typer.echo(f"Samples: {report.sample_count}")
    camera_counts = ", ".join(
        f"{camera.value}={count}" for camera, count in report.camera_counts.items()
    )
    typer.echo(f"Cameras: {camera_counts}")
    typer.echo(
        "Available: "
        f"previous={report.previous_image_count}, "
        f"semantic={report.semantic_mask_count}, "
        f"calibration={report.calibration_count}"
    )
    for issue in report.issues:
        sample = "dataset" if issue.sample_key is None else issue.sample_key.stem
        typer.echo(f"[{issue.severity.value}] {issue.code} ({sample}): {issue.message}")
    if report.issues_truncated:
        typer.echo("Additional issues were truncated; increase --max-issues to inspect them.")


@data_app.command("verify")
def verify_dataset(
    context: typer.Context,
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Override the configured WoodScape root.",
        ),
    ] = None,
    require_previous: Annotated[
        bool,
        typer.Option("--require-previous", help="Require a previous frame for every RGB sample."),
    ] = False,
    require_semantic: Annotated[
        bool,
        typer.Option("--require-semantic", help="Require a semantic mask for every RGB sample."),
    ] = False,
    require_calibration: Annotated[
        bool,
        typer.Option(
            "--require-calibration", help="Require calibration JSON for every RGB sample."
        ),
    ] = False,
    max_issues: Annotated[
        int, typer.Option(min=1, max=100_000, help="Maximum number of retained findings.")
    ] = 1000,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable validation output.")
    ] = False,
) -> None:
    """Decode and cross-check indexed WoodScape images, masks, and calibration files."""
    dataset_root = _resolve_dataset_root(context, root)
    try:
        dataset = WoodScapeDataset.discover(dataset_root)
    except DatasetLayoutError as exc:
        typer.secho(f"Dataset discovery error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    report = validate_dataset(
        dataset,
        policy=ValidationPolicy(
            require_previous_images=require_previous,
            require_semantic_masks=require_semantic,
            require_calibrations=require_calibration,
            max_issues=max_issues,
        ),
    )
    if as_json:
        typer.echo(json.dumps(_report_payload(report), indent=2, sort_keys=True))
    else:
        _render_report(report)
    if not report.is_valid:
        raise typer.Exit(code=1)


__all__ = ["data_app"]
