"""Root commands shared by every NearField360 workflow."""

from __future__ import annotations

import json
import platform
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from nearfield360 import __version__
from nearfield360.cli.state import CliState, get_state
from nearfield360.config import ConfigurationError, LoggingConfig, ProjectConfig, load_config
from nearfield360.logging import configure_logging


class LogLevel(StrEnum):
    """Log levels exposed as constrained CLI values."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


app = typer.Typer(
    name="nearfield360",
    help="Automotive fisheye surround-view perception toolkit.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Inspect and validate project configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"NearField360 {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    context: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="YAML configuration overlay.",
        ),
    ] = None,
    log_level: Annotated[
        LogLevel | None,
        typer.Option(
            "--log-level", case_sensitive=False, help="Override the configured log level."
        ),
    ] = None,
    structured_logs: Annotated[
        bool | None,
        typer.Option(
            "--structured-logs/--human-logs",
            help="Override the configured log rendering mode.",
        ),
    ] = None,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = None,
) -> None:
    """Load shared configuration before dispatching a command."""
    del version
    try:
        config = load_config(config_path)
    except (ConfigurationError, ValidationError) as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    if log_level is not None or structured_logs is not None:
        logging_values = config.logging.model_dump()
        if log_level is not None:
            logging_values["level"] = log_level.value
        if structured_logs is not None:
            logging_values["structured"] = structured_logs
        config = config.model_copy(update={"logging": LoggingConfig.model_validate(logging_values)})

    configure_logging(config.logging)
    context.obj = CliState(config=config, config_path=config_path)


@config_app.command("validate")
def validate_config(context: typer.Context) -> None:
    """Validate configuration syntax, fields, types, and environment overrides."""
    state = get_state(context)
    source = state.config_path if state.config_path is not None else "built-in defaults"
    typer.echo(f"Configuration is valid: {source}")


@config_app.command("show")
def show_config(context: typer.Context) -> None:
    """Print the effective validated configuration as JSON."""
    typer.echo(get_state(context).config.model_dump_json(indent=2))


def _doctor_payload(config: ProjectConfig) -> dict[str, Any]:
    import cv2  # Imported only for the command that needs it.

    dataset_root = config.paths.dataset_root
    return {
        "nearfield360_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "dataset_root_exists": dataset_root is not None and dataset_root.is_dir(),
        "tools": {
            name: shutil.which(name)
            for name in ("uv", "git", "cmake", "ninja", "nvidia-smi", "trtexec")
        },
    }


@app.command()
def doctor(
    context: typer.Context,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable diagnostic output.")
    ] = False,
) -> None:
    """Report required and optional runtime capabilities without changing the host."""
    payload = _doctor_payload(get_state(context).config)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"NearField360: {payload['nearfield360_version']}")
    typer.echo(f"Python:       {payload['python_version']}")
    typer.echo(f"Platform:     {payload['platform']}")
    typer.echo(f"OpenCV:       {payload['opencv_version']}")
    typer.echo(f"Dataset root: {payload['dataset_root'] or 'not configured'}")
    typer.echo("Optional tools:")
    for name, location in payload["tools"].items():
        typer.echo(f"  {name:<12} {location or 'not found'}")


__all__ = ["app"]
