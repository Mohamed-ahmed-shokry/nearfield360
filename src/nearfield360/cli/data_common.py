"""Shared local-dataset CLI options and contextual discovery errors."""

from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.data.woodscape import DatasetLayoutError, WoodScapeDataset

DatasetRootOption = Annotated[
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
]


def discover_dataset(context: typer.Context, root: Path | None) -> WoodScapeDataset:
    if root is None:
        root = get_state(context).config.paths.dataset_root
    if root is None:
        typer.secho(
            "Dataset root is not configured. Use --root or NEARFIELD360_PATHS__DATASET_ROOT.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        return WoodScapeDataset.discover(root)
    except (DatasetLayoutError, OSError) as exc:
        typer.secho(f"Dataset discovery error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
