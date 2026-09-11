"""Creation and verification of portable local split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.data.manifests import load_split_manifest, write_split_manifest
from nearfield360.data.splits import DatasetSplits, SplitError, SplitRatios, create_splits
from nearfield360.data.woodscape import SampleKey, WoodScapeDataset
from nearfield360.utils.artifacts import read_json


class _GroupFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1, max_length=2048)
    groups: dict[str, str]


def _read_groups(path: Path, dataset: WoodScapeDataset) -> tuple[dict[SampleKey, str], str]:
    payload = _GroupFile.model_validate(read_json(path))
    keys = {sample.key.stem: sample.key for sample in dataset}
    if set(payload.groups) != set(keys):
        raise SplitError("Group file keys must exactly match dataset sample identifiers")
    return {keys[name]: group for name, group in payload.groups.items()}, payload.source


def _summary(splits: DatasetSplits, path: Path, *, as_json: bool) -> None:
    counts = {split.value: count for split, count in splits.counts.items()}
    if splits.grouping == "filename_id":
        typer.secho(
            "Warning: filename identifiers do not establish synchronized cameras or "
            "sequence-level split isolation. Supply verified recording groups for evaluation.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "manifest": str(path),
                    "counts": counts,
                    "grouping": splits.grouping,
                    "group_source": splits.group_source,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(f"Split manifest: {path}")
        typer.echo("Samples: " + ", ".join(f"{key}={count}" for key, count in counts.items()))
        typer.echo(f"Grouping: {splits.grouping}; source: {splits.group_source or 'not supplied'}")


def split_dataset(
    context: typer.Context,
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    root: DatasetRootOption = None,
    seed: Annotated[int, typer.Option(min=0, max=2**64 - 1)] = 42,
    train: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    validation: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    test: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    groups: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="JSON containing source and sample groups."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing manifest.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Persist deterministic group-hash assignments; requested proportions are approximate."""
    dataset = discover_dataset(context, root)
    try:
        group_map, source = (None, None) if groups is None else _read_groups(groups, dataset)
        splits = create_splits(
            dataset,
            ratios=SplitRatios(train, validation, test),
            seed=seed,
            groups=group_map,
            group_source=source,
        )
        write_split_manifest(output, dataset, splits, overwrite=overwrite)
    except ValueError as exc:
        typer.secho(f"Split error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    _summary(splits, output, as_json=as_json)


def verify_split(
    context: typer.Context,
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    root: DatasetRootOption = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate a saved split against the currently discovered sample identities."""
    dataset = discover_dataset(context, root)
    try:
        splits = load_split_manifest(manifest, dataset)
    except SplitError as exc:
        typer.secho(f"Split error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    _summary(splits, manifest, as_json=as_json)
