import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360 import __version__
from nearfield360.cli import app

runner = CliRunner()


def test_version_option_does_not_require_a_command() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"NearField360 {__version__}"


def test_config_validate_accepts_repository_defaults() -> None:
    result = runner.invoke(app, ["--config", "configs/default.yaml", "config", "validate"])

    assert result.exit_code == 0
    assert "Configuration is valid:" in result.stdout
    assert "default.yaml" in result.stdout


def test_config_show_applies_environment_override() -> None:
    result = runner.invoke(
        app,
        ["config", "show"],
        env={"NEARFIELD360_RUNTIME__SEED": "123"},
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["runtime"]["seed"] == 123


def test_global_logging_options_override_configuration() -> None:
    result = runner.invoke(
        app,
        ["--log-level", "debug", "--structured-logs", "config", "show"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["logging"] == {"level": "DEBUG", "structured": True}


def test_invalid_configuration_has_stable_exit_code(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("runtime:\n  device: quantum\n", encoding="utf-8")

    result = runner.invoke(app, ["--config", str(config_path), "config", "validate"])

    assert result.exit_code == 2
    assert "Configuration error:" in result.stderr
    assert "runtime.device" in result.stderr


def test_doctor_json_reports_capabilities_without_dataset() -> None:
    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nearfield360_version"] == __version__
    assert payload["opencv_version"]
    assert payload["dataset_root"] is None
    assert payload["dataset_root_exists"] is False
    assert set(payload["tools"]) == {"cmake", "git", "ninja", "nvidia-smi", "trtexec", "uv"}


def test_doctor_human_output_is_scannable(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["doctor"],
        env={"NEARFIELD360_PATHS__DATASET_ROOT": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert f"Dataset root: {tmp_path}" in result.stdout
    assert "Optional tools:" in result.stdout
    assert "cmake" in result.stdout
