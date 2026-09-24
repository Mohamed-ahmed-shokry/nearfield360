"""Shared CLI options for selecting and loading neural inference backends."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.perception.inference.backend import (
    InferenceBackend,
    InferenceError,
    create_backend,
)
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice

BackendOption = Annotated[
    InferenceBackendType | None,
    typer.Option(
        "--backend",
        case_sensitive=False,
        help="Inference backend (defaults to config.inference.backend).",
    ),
]

DeviceOption = Annotated[
    InferenceDevice | None,
    typer.Option(
        "--device",
        case_sensitive=False,
        help="Inference device (defaults to config.inference.device).",
    ),
]


def resolve_backend_type(
    context: typer.Context,
    backend: InferenceBackendType | None,
) -> InferenceBackendType:
    """Return the CLI override or the configured inference backend."""
    if backend is not None:
        return backend
    configured = get_state(context).config.inference.backend
    return InferenceBackendType(configured)


def resolve_device(
    context: typer.Context,
    device: InferenceDevice | None,
) -> InferenceDevice:
    """Return the CLI override or the configured inference device."""
    if device is not None:
        return device
    configured = get_state(context).config.inference.device
    return InferenceDevice(configured)


def load_backend(
    context: typer.Context,
    model: Path,
    *,
    backend: InferenceBackendType | None = None,
    device: InferenceDevice | None = None,
) -> InferenceBackend:
    """Create an inference backend from CLI overrides or project configuration."""
    b_type = resolve_backend_type(context, backend)
    dev = resolve_device(context, device)
    try:
        return create_backend(model, backend_type=b_type, device=dev)
    except InferenceError as exc:
        typer.secho(f"Failed to load backend: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None


__all__ = [
    "BackendOption",
    "DeviceOption",
    "load_backend",
    "resolve_backend_type",
    "resolve_device",
]
