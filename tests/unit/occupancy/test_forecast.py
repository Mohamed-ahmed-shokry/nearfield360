from __future__ import annotations

import numpy as np
import pytest

from nearfield360.config import OccupancyForecastConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.health import CameraHealthMetrics, CameraHealthReport, CameraHealthStatus
from nearfield360.occupancy import (
    OccupancyEvidence,
    OccupancyForecastGrid,
    TemporalOccupancyForecaster,
    fuse_cross_attention_occupancy,
)
from nearfield360.tracking.models import TrackedObstacle, TrackState


@pytest.fixture
def bev_grid() -> BevGrid:
    return BevGrid(x_min=-4.0, x_max=4.0, y_min=-4.0, y_max=4.0, resolution=0.5)


def make_evidence(
    grid: BevGrid,
    occ_coords: list[tuple[int, int]] | None = None,
    free_coords: list[tuple[int, int]] | None = None,
    observed_all: bool = False,
) -> OccupancyEvidence:
    shape = grid.shape
    occ = np.zeros(shape, dtype=np.float64)
    free = np.zeros(shape, dtype=np.float64)
    obs = np.zeros(shape, dtype=np.int64)

    if observed_all:
        obs.fill(1)
        free.fill(1.0)

    if occ_coords:
        for r, c in occ_coords:
            occ[r, c] = 5.0
            obs[r, c] += 5

    if free_coords:
        for r, c in free_coords:
            free[r, c] = 5.0
            obs[r, c] += 5

    return OccupancyEvidence(grid=grid, occupied=occ, free=free, observed=obs)


def test_forecaster_init(bev_grid: BevGrid) -> None:
    cfg = OccupancyForecastConfig(horizon_seconds=2.0, step_seconds=0.5, hidden_channels=8)
    forecaster = TemporalOccupancyForecaster(grid=bev_grid, config=cfg)

    assert forecaster.grid == bev_grid
    assert forecaster.config == cfg
    assert forecaster.current_state is None

    with pytest.raises(ValueError, match="grid must be a BevGrid instance"):
        TemporalOccupancyForecaster(grid="invalid")  # type: ignore[arg-type]


def test_fuse_cross_attention_occupancy(bev_grid: BevGrid) -> None:
    ev1 = make_evidence(bev_grid, occ_coords=[(2, 2)])
    ev2 = make_evidence(bev_grid, occ_coords=[(2, 2)], free_coords=[(4, 4)])

    with pytest.raises(ValueError, match="requires at least one layer"):
        fuse_cross_attention_occupancy(bev_grid, [])

    fused = fuse_cross_attention_occupancy(bev_grid, [ev1, ev2])
    assert fused.grid == bev_grid
    assert fused.observed[2, 2] == 10
    assert fused.occupied[2, 2] > 0.0

    # With camera health discount
    report_healthy = CameraHealthReport(
        camera="FV",
        status=CameraHealthStatus.HEALTHY,
        metrics=CameraHealthMetrics(
            soiling_score=0.1,
            blur_score=150.0,
            mean_brightness=100.0,
            contrast=50.0,
            blockage_ratio=0.0,
            confidence=1.0,
        ),
        discount_weight=1.0,
    )
    report_soiled = CameraHealthReport(
        camera="RV",
        status=CameraHealthStatus.DEGRADED,
        metrics=CameraHealthMetrics(
            soiling_score=0.8,
            blur_score=120.0,
            mean_brightness=90.0,
            contrast=40.0,
            blockage_ratio=0.1,
            confidence=0.2,
        ),
        discount_weight=0.2,
    )

    fused_health = fuse_cross_attention_occupancy(
        bev_grid, [ev1, ev2], health_reports=[report_healthy, report_soiled]
    )
    assert fused_health.observed[2, 2] == 10


def test_forecaster_update_single_frame(bev_grid: BevGrid) -> None:
    forecaster = TemporalOccupancyForecaster(bev_grid)
    ev = make_evidence(bev_grid, occ_coords=[(3, 3)], free_coords=[(1, 1)])

    state = forecaster.update(ev, timestamp=1.0)
    assert state.step_index == 0
    assert state.timestamp == 1.0
    assert state.hidden_state.shape == (16, *bev_grid.shape)
    assert state.occupancy.shape == bev_grid.shape
    assert state.velocity_field.shape == (2, *bev_grid.shape)

    # Occupancy at (3, 3) should be high
    assert state.occupancy[3, 3] > 0.5
    # Occupancy at (1, 1) should be low (free)
    assert state.occupancy[1, 1] < 0.5
    # Unobserved cell should be NaN
    assert np.isnan(state.occupancy[7, 7])


def test_forecaster_update_temporal_sequence_and_flow(bev_grid: BevGrid) -> None:
    forecaster = TemporalOccupancyForecaster(bev_grid)

    # Frame 1: obstacle at (4, 4)
    ev1 = make_evidence(bev_grid, occ_coords=[(4, 4)], observed_all=True)
    state1 = forecaster.update(ev1, timestamp=0.0)
    assert state1.step_index == 0

    # Frame 2: obstacle moved to (4, 5) with tracked obstacle anchor
    ev2 = make_evidence(bev_grid, occ_coords=[(4, 5)], observed_all=True)
    tracked_obs = TrackedObstacle(
        track_id=1,
        class_id=3,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(0.0, 0.0),
        velocity=(2.0, 0.0),  # moving forward at 2 m/s
        speed=2.0,
        hits=5,
        age=5,
        time_since_update=0,
    )
    state2 = forecaster.update(ev2, timestamp=0.1, obstacles=[tracked_obs])
    assert state2.step_index == 1
    assert state2.timestamp == 0.1

    # Velocity field should have positive vx in obstacle region
    center_idx = bev_grid.world_to_grid(np.array([[0.0, 0.0]])).indices[0]
    cr, cc = center_idx[0], center_idx[1]
    assert state2.velocity_field[0, cr, cc] == 2.0
    assert state2.dynamic_mask[cr, cc] is np.True_ or state2.dynamic_mask[cr, cc] is True


def test_forecaster_forecast_multi_step(bev_grid: BevGrid) -> None:
    forecaster = TemporalOccupancyForecaster(bev_grid)
    ev = make_evidence(bev_grid, occ_coords=[(4, 4)], observed_all=True)
    forecaster.update(ev, timestamp=0.0)

    forecast_grid = forecaster.forecast(horizon_seconds=2.0, step_seconds=0.5)
    assert isinstance(forecast_grid, OccupancyForecastGrid)
    assert forecast_grid.horizon_seconds == 2.0
    assert forecast_grid.step_seconds == 0.5
    assert forecast_grid.num_steps == 4

    # Check uncertainty increases / diffuses forward across horizon
    first_step = forecast_grid.steps[0]
    last_step = forecast_grid.steps[-1]
    assert np.mean(last_step.uncertainty) >= np.mean(first_step.uncertainty)

    # Static occupancy persists with memory decay
    assert last_step.occupancy[4, 4] <= first_step.occupancy[4, 4]


def test_forecaster_reset(bev_grid: BevGrid) -> None:
    forecaster = TemporalOccupancyForecaster(bev_grid)
    ev = make_evidence(bev_grid, occ_coords=[(2, 2)])
    state = forecaster.update(ev, timestamp=1.0)
    assert state.step_index == 0

    forecaster.reset()
    assert forecaster.current_state is None


def test_forecast_without_update_raises(bev_grid: BevGrid) -> None:
    forecaster = TemporalOccupancyForecaster(bev_grid)
    with pytest.raises(RuntimeError, match="Cannot forecast occupancy without prior state"):
        forecaster.forecast()
