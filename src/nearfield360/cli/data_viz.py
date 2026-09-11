"""Render semantic-mask overlays for locally acquired WoodScape samples."""

from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.state import get_state
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import (
    SemanticMaskError,
    colorize_semantic_mask,
    load_semantic_mask,
)


def _viz_error(sample_stem: str, message: object) -> None:
    typer.secho(f"Visualization error for {sample_stem}: {message}", fg=typer.colors.RED, err=True)


def render_viz(
    context: typer.Context,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="Overlay directory (defaults to <output_root>/viz).",
        ),
    ] = None,
    root: DatasetRootOption = None,
    max_images: Annotated[
        int, typer.Option(min=1, max=10_000, help="Maximum overlays to render.")
    ] = 20,
    blend: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Semantic color weight in the overlay."),
    ] = 0.5,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing overlay files.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Blend RGB images with official-palette semantic masks for inspection.

    Only samples with an available semantic mask are rendered; samples without
    masks are skipped, not fabricated. Overlays are visual debugging aids, not
    model metrics.
    """
    dataset = discover_dataset(context, root)
    state = get_state(context)
    output_dir = output if output is not None else state.config.paths.output_root / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped_no_mask = 0
    skipped_existing = 0
    rendered = 0
    for sample in dataset:
        if rendered >= max_images:
            break
        if sample.semantic_mask_path is None:
            skipped_no_mask += 1
            continue
        destination = output_dir / f"{sample.key.stem}_overlay.png"
        if destination.exists() and not overwrite:
            skipped_existing += 1
            continue
        try:
            rgb = load_rgb_image(sample.image_path)
            mask = load_semantic_mask(sample.semantic_mask_path)
        except (ImageReadError, SemanticMaskError, OSError) as exc:
            _viz_error(sample.key.stem, exc)
            raise typer.Exit(code=1) from None
        if mask.shape != rgb.shape[:2]:
            _viz_error(
                sample.key.stem,
                f"mask shape {mask.shape} does not match RGB shape {rgb.shape[:2]}",
            )
            raise typer.Exit(code=1) from None
        color_rgb = colorize_semantic_mask(mask, color_order="rgb")
        with np.errstate(all="ignore"):
            blended = (1.0 - blend) * rgb.astype(np.float64)
            blended += blend * color_rgb.astype(np.float64)
            overlay_rgb = np.clip(blended, 0, 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        try:
            success = cv2.imwrite(str(destination), overlay_bgr)
        except cv2.error as exc:
            _viz_error(sample.key.stem, exc)
            raise typer.Exit(code=1) from None
        if not success:
            _viz_error(sample.key.stem, f"unable to write {destination}")
            raise typer.Exit(code=1) from None
        written.append(sample.key.stem)
        rendered += 1

    if as_json:
        typer.echo(
            json_module.dumps(
                {
                    "output": str(output_dir),
                    "written": written,
                    "skipped_no_mask": skipped_no_mask,
                    "skipped_existing": skipped_existing,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(f"Overlays: {len(written)} written to {output_dir}")
    if skipped_no_mask:
        typer.echo(f"Skipped without semantic mask: {skipped_no_mask}")
    if skipped_existing:
        typer.echo(f"Skipped existing (use --overwrite to replace): {skipped_existing}")


__all__ = ["render_viz"]
