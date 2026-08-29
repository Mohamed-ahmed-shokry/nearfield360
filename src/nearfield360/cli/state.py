"""Validated state shared by independently implemented CLI command groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import typer

from nearfield360.config import ProjectConfig


@dataclass(frozen=True)
class CliState:
    config: ProjectConfig
    config_path: Path | None


def get_state(context: typer.Context) -> CliState:
    """Return the root callback state with a single typed cast boundary."""
    return cast(CliState, context.obj)


__all__ = ["CliState", "get_state"]
