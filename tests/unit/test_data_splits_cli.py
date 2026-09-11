import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app

runner = CliRunner()


def _dataset(root: Path) -> None:
    (root / "rgb_images").mkdir()
    for index in range(10):
        (root / f"rgb_images/{index}_FV.png").touch()


def test_cli_split_saves_and_verifies_a_portable_manifest(tmp_path: Path) -> None:
    _dataset(tmp_path)
    path = tmp_path / "split.json"
    result = runner.invoke(
        app, ["data", "split", "--root", str(tmp_path), "-o", str(path), "--json"]
    )

    assert result.exit_code == 0, result.output
    assert sum(json.loads(result.stdout)["counts"].values()) == 10
    assert "do not establish synchronized cameras" in result.stderr

    verified = runner.invoke(app, ["data", "verify-split", str(path), "--root", str(tmp_path)])
    assert verified.exit_code == 0
    assert "Grouping: filename_id" in verified.stdout


def test_cli_split_preserves_explicit_recording_provenance(tmp_path: Path) -> None:
    _dataset(tmp_path)
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps(
            {"source": "recording log", "groups": {f"{i}_FV": "sequence-a" for i in range(10)}}
        ),
        encoding="utf-8",
    )
    path = tmp_path / "split.json"
    result = runner.invoke(
        app, ["data", "split", "--root", str(tmp_path), "-o", str(path), "--groups", str(groups)]
    )

    assert result.exit_code == 0, result.output
    assert "source: recording log" in result.stdout
    assert "Warning:" not in result.stderr


def test_cli_split_rejects_invalid_ratios_groups_and_existing_output(tmp_path: Path) -> None:
    _dataset(tmp_path)
    path = tmp_path / "split.json"
    arguments = ["data", "split", "--root", str(tmp_path), "-o", str(path)]
    bad_ratio = runner.invoke(app, [*arguments, "--train", "0.1"])
    assert bad_ratio.exit_code == 2
    assert "must sum to 1.0" in bad_ratio.stderr
    assert not path.exists()

    groups = tmp_path / "groups.json"
    groups.write_text('{"source":"recordings", "groups":{}}', encoding="utf-8")
    bad_group = runner.invoke(app, [*arguments, "--groups", str(groups)])
    assert bad_group.exit_code == 2
    assert "exactly match" in bad_group.stderr

    assert runner.invoke(app, arguments).exit_code == 0
    assert runner.invoke(app, arguments).exit_code == 2
    assert runner.invoke(app, [*arguments, "--overwrite"]).exit_code == 0


def test_cli_split_verification_rejects_malformed_manifest(tmp_path: Path) -> None:
    _dataset(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["data", "verify-split", str(path), "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "Invalid split manifest" in result.stderr
