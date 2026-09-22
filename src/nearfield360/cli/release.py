"""Release readiness checks for packaging, metadata, and CLI surface consistency."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from nearfield360 import __version__
from nearfield360.config import ConfigurationError, load_config
from nearfield360.utils.artifacts import ArtifactError, write_json

release_app = typer.Typer(
    help="Verify packaging metadata, license, and CLI surface for a release.",
    no_args_is_help=True,
)

OUTPUT_OPTION = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="Optional JSON path for the audit report (atomically written).",
    ),
]

OVERWRITE_OPTION = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace an existing audit report."),
]


def _project_root() -> Path | None:
    """Locate the repository root by walking up from this module."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "LICENSE").is_file():
            return parent
    return None


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "detail": detail}


def _run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    root = _project_root()

    if root is None:
        checks.append(
            _check(
                "project_root",
                False,
                "Could not locate pyproject.toml and LICENSE above the package.",
            )
        )
        return checks
    checks.append(_check("project_root", True, str(root)))

    license_path = root / "LICENSE"
    if license_path.is_file():
        text = license_path.read_text(encoding="utf-8", errors="replace")
        apache = "Apache License" in text and "2.0" in text
        checks.append(_check("license_file", apache, str(license_path)))
    else:
        checks.append(_check("license_file", False, f"Missing {license_path}"))

    pyproject_path = root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        name_ok = project.get("name") == "nearfield360"
        version_ok = project.get("version") == __version__
        license_meta = str(project.get("license", ""))
        license_ok = "Apache" in license_meta or license_meta == "Apache-2.0"
        readme_ok = bool(project.get("readme"))
        scripts = project.get("scripts", {})
        entry_ok = scripts.get("nearfield360") == "nearfield360.cli:app"
        checks.append(_check("project_name", name_ok, str(project.get("name"))))
        checks.append(
            _check(
                "version_consistency",
                version_ok,
                f"pyproject={project.get('version')} package={__version__}",
            )
        )
        checks.append(_check("license_metadata", license_ok, license_meta))
        checks.append(_check("readme_metadata", readme_ok, str(project.get("readme"))))
        checks.append(
            _check(
                "console_script",
                entry_ok,
                str(scripts.get("nearfield360")),
            )
        )
    except (OSError, tomllib.TOMLDecodeError) as exc:
        checks.append(_check("pyproject_parse", False, str(exc)))

    readme = root / "README.md"
    checks.append(_check("readme_exists", readme.is_file(), str(readme)))

    py_typed = Path(__file__).resolve().parents[1] / "py.typed"
    checks.append(_check("py_typed", py_typed.is_file(), str(py_typed)))

    config_path = root / "configs" / "default.yaml"
    if config_path.is_file():
        try:
            load_config(config_path)
            checks.append(_check("default_config", True, str(config_path)))
        except (ConfigurationError, ValidationError) as exc:
            checks.append(_check("default_config", False, str(exc)))
    else:
        checks.append(_check("default_config", False, f"Missing {config_path}"))

    expected_groups = {
        "config",
        "data",
        "eval",
        "geometry",
        "health",
        "infer",
        "occupancy",
        "pipeline",
        "release",
        "robustness",
        "track",
    }
    from nearfield360.cli.app import app as root_app

    registered = {g.name for g in root_app.registered_groups if g.name}
    missing = sorted(expected_groups - registered)
    checks.append(
        _check(
            "cli_surface",
            not missing,
            "registered="
            + ",".join(sorted(registered))
            + ("" if not missing else f" missing={','.join(missing)}"),
        )
    )

    return checks


@release_app.command("audit")
def release_audit(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable audit output.")
    ] = False,
    output: OUTPUT_OPTION = None,
    overwrite: OVERWRITE_OPTION = False,
) -> None:
    """Run packaging, license, config, and CLI-surface checks; exit 1 on any failure."""
    checks = _run_checks()
    failed = [c for c in checks if c["status"] != "pass"]
    payload: dict[str, Any] = {
        "nearfield360_version": __version__,
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "ok": not failed,
    }

    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in checks:
            marker = "PASS" if check["status"] == "pass" else "FAIL"
            typer.echo(f"[{marker}] {check['name']}: {check['detail']}")
        if failed:
            typer.echo(f"Release audit FAILED ({len(failed)}).")
        else:
            typer.echo("Release audit passed.")

    if output is not None:
        try:
            write_json(output, payload, overwrite=overwrite)
        except ArtifactError as exc:
            typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        if not as_json:
            typer.echo(f"Wrote {output}")

    if failed:
        raise typer.Exit(code=1)


__all__ = ["release_app"]
