"""Unit tests for operations dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.config import OperationsConfig
from nearfield360.operations.demand import generate_arrivals
from nearfield360.operations.kpi import aggregate_daily_kpis, compute_hourly_snapshots
from nearfield360.operations.simulator import simulate_journeys
from nearfield360.operations.viz import render_operations_dashboard


def test_render_operations_dashboard(tmp_path: Path) -> None:
    config = OperationsConfig(day_duration_hours=24.0)
    arrivals = generate_arrivals(config, seed=123)
    slot_cap = 60
    journeys = simulate_journeys(arrivals, config, slot_capacity=slot_cap)
    snaps = compute_hourly_snapshots(journeys, config)
    kpis = aggregate_daily_kpis(journeys, snaps, config, slot_capacity=slot_cap)

    png_path = tmp_path / "ops_dashboard.png"
    canvas = render_operations_dashboard(kpis, snaps, slot_capacity=slot_cap, output_path=png_path)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 5000
