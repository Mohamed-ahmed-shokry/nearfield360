"""Unit tests for Fleet Visualization and coordination telemetry rendering."""

from __future__ import annotations

from pathlib import Path

from nearfield360.fleet.coordinator import FleetCoordinator
from nearfield360.fleet.viz import render_fleet_coordination_dashboard
from nearfield360.mapping.builder import build_benchmark_garage


def test_render_fleet_coordination_dashboard(tmp_path: Path) -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=())
    coordinator = FleetCoordinator(facility)

    coordinator.register_vehicle("v1", (0.0, 0.0, 0.0))
    coordinator.register_vehicle("v2", (15.0, 5.0, 1.57))

    coordinator.dispatch_vehicle("v1")
    coordinator.dispatch_vehicle("v2")

    # Step to generate motion
    coordinator.step(dt_s=1.0)
    report = coordinator.generate_report()

    out_png = tmp_path / "fleet_dashboard.png"
    canvas = render_fleet_coordination_dashboard(
        facility=facility,
        vehicles=list(coordinator.vehicles.values()),
        conflicts=coordinator.active_conflicts,
        report=report,
        canvas_size=600,
        output_path=out_png,
    )

    assert canvas.shape[0] == 600
    assert canvas.shape[1] == 600 + 380
    assert canvas.shape[2] == 3
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
