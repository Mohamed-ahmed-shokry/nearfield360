"""Unit tests for temporal occupancy forecasting domain models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy import (
    OccupancyForecastGrid,
    OccupancyForecastStep,
    RiskZone,
    TemporalForecastSummary,
    TemporalOccupancyState,
    ZoneForecastRisk,
)


@pytest.fixture
def sample_grid() -> BevGrid:
    return BevGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolution=1.0)


def test_temporal_occupancy_state_valid(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape  # (10, 10)
    hidden = np.zeros((4, *shape), dtype=np.float64)
    occ = np.zeros(shape, dtype=np.float64)
    unc = np.ones(shape, dtype=np.float64) * 0.1
    vel = np.zeros((2, *shape), dtype=np.float64)
    dyn = np.zeros(shape, dtype=bool)

    state = TemporalOccupancyState(
        timestamp=1.5,
        step_index=3,
        grid=sample_grid,
        hidden_state=hidden,
        occupancy=occ,
        uncertainty=unc,
        velocity_field=vel,
        dynamic_mask=dyn,
    )

    assert state.timestamp == 1.5
    assert state.step_index == 3
    assert state.grid == sample_grid
    assert state.hidden_state.shape == (4, *shape)
    assert state.velocity_field.shape == (2, *shape)


def test_temporal_occupancy_state_invalid_inputs(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape
    hidden = np.zeros((4, *shape), dtype=np.float64)
    occ = np.zeros(shape, dtype=np.float64)
    unc = np.ones(shape, dtype=np.float64)
    vel = np.zeros((2, *shape), dtype=np.float64)
    dyn = np.zeros(shape, dtype=bool)

    with pytest.raises(ValueError, match="timestamp must be a finite non-negative float"):
        TemporalOccupancyState(
            timestamp=-1.0,
            step_index=0,
            grid=sample_grid,
            hidden_state=hidden,
            occupancy=occ,
            uncertainty=unc,
            velocity_field=vel,
            dynamic_mask=dyn,
        )

    with pytest.raises(ValueError, match="step_index must be non-negative"):
        TemporalOccupancyState(
            timestamp=0.0,
            step_index=-1,
            grid=sample_grid,
            hidden_state=hidden,
            occupancy=occ,
            uncertainty=unc,
            velocity_field=vel,
            dynamic_mask=dyn,
        )

    with pytest.raises(ValueError, match="hidden_state must have shape"):
        TemporalOccupancyState(
            timestamp=0.0,
            step_index=0,
            grid=sample_grid,
            hidden_state=np.zeros((shape[0], shape[1]), dtype=np.float64),
            occupancy=occ,
            uncertainty=unc,
            velocity_field=vel,
            dynamic_mask=dyn,
        )

    with pytest.raises(ValueError, match="velocity_field must have shape"):
        TemporalOccupancyState(
            timestamp=0.0,
            step_index=0,
            grid=sample_grid,
            hidden_state=hidden,
            occupancy=occ,
            uncertainty=unc,
            velocity_field=np.zeros((3, *shape), dtype=np.float64),
            dynamic_mask=dyn,
        )


def test_occupancy_forecast_step_valid(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape
    step = OccupancyForecastStep(
        time_offset_s=0.5,
        timestamp=2.5,
        occupancy=np.full(shape, 0.4),
        uncertainty=np.full(shape, 0.05),
        free=np.full(shape, 0.6),
        dynamic_mask=np.zeros(shape, dtype=bool),
    )

    assert step.time_offset_s == 0.5
    assert step.timestamp == 2.5
    assert step.occupancy.shape == shape


def test_occupancy_forecast_step_invalid(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape
    with pytest.raises(ValueError, match="time_offset_s must be a finite positive float"):
        OccupancyForecastStep(
            time_offset_s=0.0,
            timestamp=1.0,
            occupancy=np.zeros(shape),
            uncertainty=np.zeros(shape),
            free=np.zeros(shape),
            dynamic_mask=np.zeros(shape, dtype=bool),
        )

    with pytest.raises(ValueError, match="uncertainty shape"):
        OccupancyForecastStep(
            time_offset_s=0.5,
            timestamp=1.0,
            occupancy=np.zeros(shape),
            uncertainty=np.zeros((5, 5)),
            free=np.zeros(shape),
            dynamic_mask=np.zeros(shape, dtype=bool),
        )


def test_occupancy_forecast_grid_rollout_and_queries(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape
    step1_occ = np.zeros(shape)
    step1_occ[2, 2] = 0.3
    step1 = OccupancyForecastStep(
        time_offset_s=0.5,
        timestamp=1.5,
        occupancy=step1_occ,
        uncertainty=np.full(shape, 0.1),
        free=np.full(shape, 0.7),
        dynamic_mask=np.zeros(shape, dtype=bool),
    )

    step2_occ = np.zeros(shape)
    step2_occ[2, 2] = 0.8
    step2_occ[3, 3] = 0.6
    dyn2 = np.zeros(shape, dtype=bool)
    dyn2[3, 3] = True
    step2 = OccupancyForecastStep(
        time_offset_s=1.0,
        timestamp=2.0,
        occupancy=step2_occ,
        uncertainty=np.full(shape, 0.15),
        free=np.full(shape, 0.4),
        dynamic_mask=dyn2,
    )

    grid = OccupancyForecastGrid(
        grid=sample_grid,
        base_timestamp=1.0,
        steps=(step1, step2),
    )

    assert grid.horizon_seconds == 1.0
    assert grid.step_seconds == 0.5
    assert grid.num_steps == 2

    # Query closest horizon
    at_06 = grid.at_horizon(0.6)
    assert at_06.time_offset_s == 0.5
    at_09 = grid.at_horizon(0.9)
    assert at_09.time_offset_s == 1.0

    # Max occupancy envelope
    max_env = grid.max_occupancy_over_horizon()
    assert max_env[2, 2] == 0.8
    assert max_env[3, 3] == 0.6
    assert max_env[0, 0] == 0.0

    # Mean occupancy over horizon
    mean_env = grid.mean_occupancy_over_horizon()
    assert np.isclose(mean_env[2, 2], 0.55)

    # Zone risks
    zone_mask = np.zeros(shape, dtype=bool)
    zone_mask[2, 2] = True
    test_zone = RiskZone(name="test_front", mask=zone_mask)

    zone_risks = grid.zone_forecasts([test_zone], danger_occupancy=0.5)
    assert len(zone_risks) == 1
    zr = zone_risks[0]
    assert zr.zone_name == "test_front"
    assert zr.peak_occupancy == 0.8
    assert zr.is_threat is True
    assert zr.time_to_intrusion_s == 1.0  # exceeded 0.5 at step 2 (t=1.0s)

    # Summary
    summary = grid.summary([test_zone], danger_occupancy=0.5, observed_cells=10, dynamic_cells=1)
    assert isinstance(summary, TemporalForecastSummary)
    assert summary.horizon_seconds == 1.0
    assert summary.min_time_to_intrusion_s == 1.0
    assert summary.critical_zone == "test_front"
    assert summary.dynamic_cells == 1

    # to_dict
    d = grid.to_dict()
    assert d["horizon_seconds"] == 1.0
    assert len(d["steps"]) == 2
    assert d["steps"][1]["dynamic_cells"] == 1


def test_occupancy_forecast_grid_rejects_non_increasing_offsets(sample_grid: BevGrid) -> None:
    shape = sample_grid.shape
    step1 = OccupancyForecastStep(
        time_offset_s=1.0,
        timestamp=2.0,
        occupancy=np.zeros(shape),
        uncertainty=np.zeros(shape),
        free=np.zeros(shape),
        dynamic_mask=np.zeros(shape, dtype=bool),
    )
    step2 = OccupancyForecastStep(
        time_offset_s=0.5,
        timestamp=1.5,
        occupancy=np.zeros(shape),
        uncertainty=np.zeros(shape),
        free=np.zeros(shape),
        dynamic_mask=np.zeros(shape, dtype=bool),
    )

    with pytest.raises(ValueError, match="strictly increasing time offsets"):
        OccupancyForecastGrid(grid=sample_grid, base_timestamp=1.0, steps=(step1, step2))


def test_zone_forecast_risk_validation() -> None:
    zr = ZoneForecastRisk(
        zone_name="rear",
        peak_occupancy=0.75,
        mean_occupancies=(0.1, 0.4, 0.75),
        max_occupancies=(0.2, 0.5, 0.75),
        time_to_intrusion_s=1.5,
        is_threat=True,
    )
    assert zr.is_threat is True
    assert zr.peak_occupancy == 0.75

    with pytest.raises(ValidationError):
        # extra field forbidden
        ZoneForecastRisk(  # type: ignore[call-arg]
            zone_name="rear",
            peak_occupancy=0.5,
            mean_occupancies=(0.5,),
            max_occupancies=(0.5,),
            extra_field="invalid",
        )
