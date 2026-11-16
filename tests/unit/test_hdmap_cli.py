"""Unit tests for nearfield360 hdmap CLI command group (ISO 23374 / Lanelet2)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_hdmap_info(tmp_path: Path) -> None:
    # 1. Plain text output
    res = runner.invoke(app, ["hdmap", "info"])
    assert res.exit_code == 0
    assert "High-Definition (HD) Map & Localization Configuration" in res.stdout
    assert "50.0 m" in res.stdout

    # 2. JSON output
    json_res = runner.invoke(app, ["hdmap", "info", "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["tile_size_m"] == 50.0
    assert data["num_particles"] == 200

    # 3. File output
    out_file = tmp_path / "hdmap_info.json"
    file_res = runner.invoke(app, ["hdmap", "info", "--output", str(out_file)])
    assert file_res.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["douglas_peucker_epsilon_m"] == 0.15


def test_cli_hdmap_vectorize(tmp_path: Path) -> None:
    # 1. Plain text
    res = runner.invoke(app, ["hdmap", "vectorize", "--id", "LANE-X", "--speed", "4.0"])
    assert res.exit_code == 0
    assert "Lanelet Vectorized: LANE-X" in res.stdout
    assert "Speed Limit: 4.0 m/s" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "vect_out.json"
    json_res = runner.invoke(app, ["hdmap", "vectorize", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert "lanelet_id" in data
    assert "centerline" in data
    assert out_file.is_file()


def test_cli_hdmap_tile(tmp_path: Path) -> None:
    # 1. Plain text
    res = runner.invoke(app, ["hdmap", "tile", "--facility", "FAC-B1", "--level", "L3"])
    assert res.exit_code == 0
    assert "HD Map Partitioned" in res.stdout
    assert "FAC-B1" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "tiles.json"
    json_res = runner.invoke(app, ["hdmap", "tile", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert "tiles" in data
    assert len(data["tiles"]) >= 2
    assert out_file.is_file()


def test_cli_hdmap_localize(tmp_path: Path) -> None:
    # 1. Plain text execution
    res = runner.invoke(app, ["hdmap", "localize", "--x", "15.0", "--y", "0.0", "--steps", "3"])
    assert res.exit_code == 0
    assert "Localization State:" in res.stdout
    assert "Estimated Pose:" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "loc.json"
    json_res = runner.invoke(app, ["hdmap", "localize", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert "estimated_x" in data
    assert "std_x" in data
    assert out_file.is_file()


def test_cli_hdmap_export(tmp_path: Path) -> None:
    # 1. GeoJSON format
    geo_file = tmp_path / "map.geojson"
    geo_res = runner.invoke(app, ["hdmap", "export", "--format", "geojson", "-o", str(geo_file)])
    assert geo_res.exit_code == 0
    assert geo_file.is_file()
    geo_data = json.loads(geo_file.read_text(encoding="utf-8"))
    assert geo_data["type"] == "FeatureCollection"

    # 2. Lanelet2 OSM format
    osm_file = tmp_path / "map.osm"
    osm_res = runner.invoke(app, ["hdmap", "export", "--format", "lanelet2", "-o", str(osm_file)])
    assert osm_res.exit_code == 0
    assert osm_file.is_file()
    assert "<osm" in osm_file.read_text(encoding="utf-8")

    # 3. OpenDRIVE XML format
    xodr_file = tmp_path / "map.xodr"
    xodr_res = runner.invoke(
        app, ["hdmap", "export", "--format", "opendrive", "-o", str(xodr_file)]
    )
    assert xodr_res.exit_code == 0
    assert xodr_file.is_file()
    assert "<OpenDRIVE>" in xodr_file.read_text(encoding="utf-8")

    # 4. Invalid format error
    bad_res = runner.invoke(app, ["hdmap", "export", "--format", "unknown_fmt"])
    assert bad_res.exit_code == 1


def test_cli_hdmap_render(tmp_path: Path) -> None:
    out_png = tmp_path / "test_hdmap_dashboard.png"
    res = runner.invoke(app, ["hdmap", "render", "-o", str(out_png)])
    assert res.exit_code == 0
    assert out_png.is_file()
    assert out_png.stat().st_size > 0

    # JSON render summary
    json_png = tmp_path / "test_hdmap_dashboard2.png"
    json_res = runner.invoke(app, ["hdmap", "render", "-o", str(json_png), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["canvas_shape"] == [720, 1280, 3]


def test_cli_hdmap_audit(tmp_path: Path) -> None:
    # 1. Text output
    res = runner.invoke(app, ["hdmap", "audit"])
    assert res.exit_code == 0
    assert "COMPLIANT" in res.stdout
    assert "PASS" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "audit.json"
    json_res = runner.invoke(app, ["hdmap", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["compliant"] is True
    assert data["total_checks"] == 8
    assert out_file.is_file()
