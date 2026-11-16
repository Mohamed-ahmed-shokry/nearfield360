"""CLI commands for HD vector map tiling, OpenDRIVE/Lanelet2 export, and localization."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from nearfield360.cli.state import get_state
from nearfield360.hdmap.formats import MapFormatExporter
from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    LaneletType,
    ParkingSlotType,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
    TurnDirection,
)
from nearfield360.hdmap.particle_filter import FisheyeParticleFilter
from nearfield360.hdmap.tiler import MapTiler
from nearfield360.hdmap.vectorizer import BEVVectorizer
from nearfield360.hdmap.viz import render_hdmap_dashboard
from nearfield360.utils.artifacts import write_json

hdmap_app = typer.Typer(
    help="High-Definition (HD) vector map tiling, export, and fisheye feature localization.",
    no_args_is_help=True,
)

console = Console()

OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="Path to write JSON, XML, or PNG artifact atomically.",
    ),
]


def _build_demo_map() -> tuple[MapTiler, HDVectorMap]:
    """Helper to construct a standard multi-tile demo HDVectorMap."""
    tiler = MapTiler()
    left = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=60.0, y=2.0)])
    right = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=60.0, y=-2.0)])
    center = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=60.0, y=0.0)])
    lane = Lanelet(
        lanelet_id="LANE-MAIN-01",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
        speed_limit_mps=3.0,
        lanelet_type=LaneletType.PARKING_AISLE,
        turn_direction=TurnDirection.STRAIGHT,
    )

    slot_poly = Polyline2D(
        points=[
            Point2D(x=20.0, y=4.0),
            Point2D(x=22.5, y=4.0),
            Point2D(x=22.5, y=9.0),
            Point2D(x=20.0, y=9.0),
        ],
        is_closed=True,
    )
    space = ParkingSpaceFeature(
        slot_id="BAY-EV-10",
        boundary=slot_poly,
        center=Point2D(x=21.25, y=6.5),
        slot_type=ParkingSlotType.EV_CHARGING,
    )

    hd_map = tiler.tile_map("MAP-DEMO-01", lanelets=[lane], parking_spaces=[space])
    return tiler, hd_map


@hdmap_app.command("info")
def hdmap_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display HD vector map tiling parameters and particle filter configuration."""
    state = get_state(context)
    cfg = state.config.hdmap

    payload: dict[str, Any] = {
        "standard": "ISO 23374 / Lanelet2 / ASAM OpenDRIVE 1.7",
        "tile_size_m": cfg.tile_size_m,
        "douglas_peucker_epsilon_m": cfg.douglas_peucker_epsilon_m,
        "min_lane_width_m": cfg.min_lane_width_m,
        "max_lane_width_m": cfg.max_lane_width_m,
        "default_speed_limit_mps": cfg.default_speed_limit_mps,
        "num_particles": cfg.num_particles,
        "process_noise_pos_m": cfg.process_noise_pos_m,
        "process_noise_yaw_rad": cfg.process_noise_yaw_rad,
        "observation_sigma_m": cfg.observation_sigma_m,
        "lateral_convergence_threshold_m": cfg.lateral_convergence_threshold_m,
        "heading_convergence_threshold_deg": cfg.heading_convergence_threshold_deg,
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    table = Table(title="High-Definition (HD) Map & Localization Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Standard", payload["standard"])
    table.add_row("Tile Size", f"{cfg.tile_size_m} m")
    table.add_row("Douglas-Peucker Epsilon", f"{cfg.douglas_peucker_epsilon_m} m")
    table.add_row("Lane Width Envelope", f"{cfg.min_lane_width_m} - {cfg.max_lane_width_m} m")
    table.add_row("Default Speed Limit", f"{cfg.default_speed_limit_mps} m/s")
    table.add_row("Particle Filter Size (N)", str(cfg.num_particles))
    table.add_row("Position Process Noise", f"{cfg.process_noise_pos_m} m")
    table.add_row("Observation Sigma", f"{cfg.observation_sigma_m} m")
    table.add_row("Lateral Convergence Limit", f"{cfg.lateral_convergence_threshold_m} m")
    table.add_row("Heading Convergence Limit", f"{cfg.heading_convergence_threshold_deg} deg")
    console.print(table)


@hdmap_app.command("vectorize")
def hdmap_vectorize(
    context: typer.Context,
    lane_id: Annotated[str, typer.Option("--id", help="Lanelet ID.")] = "L-DEMO-01",
    speed: Annotated[float, typer.Option("--speed", help="Speed limit m/s.")] = 3.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Vectorize boundary coordinates and extract simplified metric Lanelet corridor."""
    state = get_state(context)
    vectorizer = BEVVectorizer(config=state.config.hdmap)

    left_raw = [(0.0, 2.0), (10.0, 2.05), (20.0, 1.95), (30.0, 2.0)]
    right_raw = [(0.0, -2.0), (10.0, -1.95), (20.0, -2.05), (30.0, -2.0)]

    lanelet = vectorizer.build_lanelet(
        lanelet_id=lane_id,
        left_pts=left_raw,
        right_pts=right_raw,
        speed_limit_mps=speed,
    )

    payload = lanelet.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[green]Lanelet Vectorized: {lanelet.lanelet_id}[/green]")
    console.print(f"Centerline Length: {lanelet.centerline.length_m:.2f} m")
    console.print(
        f"Speed Limit: {lanelet.speed_limit_mps:.1f} m/s | Type: {lanelet.lanelet_type.value}"
    )
    console.print(
        f"Left Vertices: {len(lanelet.left_boundary.points)} | "
        f"Right Vertices: {len(lanelet.right_boundary.points)}"
    )


@hdmap_app.command("tile")
def hdmap_tile(
    context: typer.Context,
    facility: Annotated[str, typer.Option("--facility", help="Facility ID.")] = "FAC-AIRPORT",
    level: Annotated[str, typer.Option("--level", help="Level ID.")] = "L2",
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Partition HD vector map features into spatial metric grid tiles."""
    state = get_state(context)
    tiler = MapTiler(config=state.config.hdmap)

    # 3 sequential lanes spanning 120m
    left1 = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=45.0, y=2.0)])
    right1 = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=45.0, y=-2.0)])
    center1 = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=45.0, y=0.0)])
    lane1 = Lanelet(lanelet_id="L1", left_boundary=left1, right_boundary=right1, centerline=center1)

    left2 = Polyline2D(points=[Point2D(x=45.0, y=2.0), Point2D(x=95.0, y=2.0)])
    right2 = Polyline2D(points=[Point2D(x=45.0, y=-2.0), Point2D(x=95.0, y=-2.0)])
    center2 = Polyline2D(points=[Point2D(x=45.0, y=0.0), Point2D(x=95.0, y=0.0)])
    lane2 = Lanelet(lanelet_id="L2", left_boundary=left2, right_boundary=right2, centerline=center2)

    hd_map = tiler.tile_map(
        "MAP-TILE-RUN", lanelets=[lane1, lane2], facility_id=facility, level_id=level
    )

    payload = hd_map.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[green]HD Map Partitioned into {len(hd_map.tiles)} Metric Tiles[/green]")
    console.print(f"Facility: {facility} | Level: {level} | Tile Step: {tiler.tile_size}m")
    for t in hd_map.tiles:
        console.print(
            f"  - Tile {t.tile_id}: bounds [{t.x_min:.0f}, {t.y_min:.0f}] "
            f"-> [{t.x_max:.0f}, {t.y_max:.0f}]"
        )


@hdmap_app.command("localize")
def hdmap_localize(
    context: typer.Context,
    init_x: Annotated[float, typer.Option("--x", help="Initial X prior m.")] = 15.2,
    init_y: Annotated[float, typer.Option("--y", help="Initial Y prior m.")] = 0.2,
    init_yaw: Annotated[float, typer.Option("--yaw", help="Initial heading prior rad.")] = 0.05,
    steps: Annotated[int, typer.Option("--steps", help="Number of recursive filter updates.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Execute multi-hypothesis fisheye feature-map matching Monte Carlo particle filter."""
    state = get_state(context)
    tiler, _ = _build_demo_map()

    pf = FisheyeParticleFilter(tiler=tiler, config=state.config.hdmap, seed=42)
    pf.initialize(init_x=init_x, init_y=init_y, init_yaw_rad=init_yaw)

    # Simulated observations in vehicle frame
    obs = [(0.0, 2.0), (0.0, -2.0), (5.0, 0.0), (10.0, 0.0)]

    for _ in range(max(1, steps)):
        pf.predict(delta_s_m=0.5, delta_yaw_rad=0.0)
        pf.update(obs)

    loc = pf.get_state(timestamp_s=1.0, associated_features_count=len(obs))

    payload = loc.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    col = "green" if loc.status.value in ("CONVERGED", "TRACKING") else "yellow"
    console.print(f"[{col}]Localization State: {loc.status.value}[/{col}]")
    console.print(
        f"Estimated Pose: X={loc.estimated_x:.3f}m | Y={loc.estimated_y:.3f}m | "
        f"Yaw={loc.estimated_yaw_rad:.3f}rad"
    )
    console.print(
        f"Uncertainty: sigma_x={loc.std_x * 100:.1f}cm | sigma_y={loc.std_y * 100:.1f}cm | "
        f"Tile={loc.active_tile_id}"
    )


@hdmap_app.command("export")
def hdmap_export(
    context: typer.Context,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Target format (geojson, lanelet2, opendrive).")
    ] = "geojson",
    output: OutputOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON if GeoJSON format.")] = False,
) -> None:
    """Export HD vector map to OpenDRIVE XML, Lanelet2 OSM XML, or GeoJSON."""
    _, hd_map = _build_demo_map()

    target_fmt = fmt.lower()
    if target_fmt == "geojson":
        res = MapFormatExporter.to_geojson(hd_map)
        if output is not None:
            write_json(output, res, overwrite=True)
        if as_json or output is None:
            console.print_json(data=res)
        else:
            console.print(f"[green]GeoJSON exported to {output}[/green]")
    elif target_fmt in ("lanelet2", "osm"):
        osm_str = MapFormatExporter.to_lanelet2_osm(hd_map)
        if output is not None:
            output.write_text(osm_str, encoding="utf-8")
            console.print(f"[green]Lanelet2 OSM XML exported to {output}[/green]")
        else:
            console.print(osm_str[:400] + "\n...")
    elif target_fmt in ("opendrive", "xodr"):
        od_str = MapFormatExporter.to_opendrive_xml(hd_map)
        if output is not None:
            output.write_text(od_str, encoding="utf-8")
            console.print(f"[green]OpenDRIVE XML exported to {output}[/green]")
        else:
            console.print(od_str[:400] + "\n...")
    else:
        console.print(f"[red]Unknown format '{fmt}'. Choose geojson, lanelet2, or opendrive.[/red]")
        raise typer.Exit(code=1)


@hdmap_app.command("render")
def hdmap_render(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", resolve_path=True, help="Path for rendered PNG dashboard."),
    ] = Path("reports/hdmap_dashboard.png"),
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON summary.")] = False,
) -> None:
    """Generate tactical 1280x720 HUD dashboard visualizing HD map and particle filter state."""
    state = get_state(context)
    tiler, hd_map = _build_demo_map()

    pf = FisheyeParticleFilter(tiler=tiler, config=state.config.hdmap, seed=42)
    pf.initialize(init_x=20.0, init_y=0.0, init_yaw_rad=0.0)

    obs = [(0.0, 2.0), (0.0, -2.0), (5.0, 0.0)]
    for _ in range(4):
        pf.predict(0.2, 0.0)
        pf.update(obs)

    loc = pf.get_state(timestamp_s=1.0, associated_features_count=len(obs))
    canvas = render_hdmap_dashboard(
        vector_map=hd_map,
        localization=loc,
        particles=pf.particles,
        observed_points_vehicle=obs,
        output_path=output,
    )

    payload = {
        "rendered_image_path": str(output),
        "canvas_shape": list(canvas.shape),
        "estimated_x": loc.estimated_x,
        "estimated_y": loc.estimated_y,
        "status": loc.status.value,
    }

    if as_json:
        console.print_json(data=payload)
    else:
        console.print(f"[green]Tactical HD Map Dashboard rendered successfully to {output}[/green]")
        console.print(f"Canvas: {canvas.shape[1]}x{canvas.shape[0]} | State: {loc.status.value}")


@hdmap_app.command("audit")
def hdmap_audit(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Audit HD map vector tiling and localization readiness against ISO 23374 & Lanelet2."""
    state = get_state(context)
    cfg = state.config.hdmap

    checks = [
        ("Metric Spatial Tile Size Bounded (10m - 250m)", 10.0 <= cfg.tile_size_m <= 250.0),
        (
            "Douglas-Peucker Simplification Sensitivity (0.01m - 1.0m)",
            0.01 <= cfg.douglas_peucker_epsilon_m <= 1.0,
        ),
        (
            "Lane Width Tolerance Envelope Valid (1.5m - 8.0m)",
            1.5 <= cfg.min_lane_width_m < cfg.max_lane_width_m <= 8.0,
        ),
        (
            "Facility Default Speed Limit Valid (0.5 - 15.0 m/s)",
            0.5 <= cfg.default_speed_limit_mps <= 15.0,
        ),
        ("Particle Cloud Allocation Sufficient (>= 50 particles)", cfg.num_particles >= 50),
        (
            "Lateral Convergence Safety Margin (<= 0.15 m)",
            cfg.lateral_convergence_threshold_m <= 0.151,
        ),
        (
            "Heading Convergence Safety Margin (<= 3.0 deg)",
            cfg.heading_convergence_threshold_deg <= 3.01,
        ),
        ("Observation Noise Model Defined (0.01m - 1.0m)", 0.01 <= cfg.observation_sigma_m <= 1.0),
    ]

    all_passed = all(status for _, status in checks)

    payload = {
        "audit_target": "High-Definition (HD) Map Vector Tiling & Localization Subsystem",
        "compliant": all_passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for _, status in checks if status),
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
    }

    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        console.print_json(data=payload)
        return

    status_str = "[green]COMPLIANT[/green]" if all_passed else "[red]NON-COMPLIANT[/red]"
    console.print(f"HD Map & Localization Audit: {status_str}")
    for name, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  [{icon}] {name}")
