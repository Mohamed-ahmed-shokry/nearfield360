"""Unit tests for dynamic obstacle risk and trajectory forecasting."""

from __future__ import annotations

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.risk import RiskZone, surround_parking_zones
from nearfield360.tracking.models import TrackedObstacle, TrackState
from nearfield360.tracking.risk import (
    forecast_all_trajectories,
    forecast_obstacle_trajectory,
)


def _setup_grid_and_zones() -> tuple[BevGrid, list[RiskZone]]:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.05)
    zones = list(
        surround_parking_zones(
            grid,
            front_length=3.0,
            rear_length=3.0,
            half_width=0.9,
            start_x=0.0,
            rear_start_x=0.0,
            lateral_width=0.8,
            vehicle_x_min=-2.0,
            vehicle_x_max=2.0,
            near_radius=0.5,
            warning_radius=1.5,
        )
    )
    return grid, zones


def test_forecast_approaching_obstacle_detects_collision_ttc() -> None:
    grid, zones = _setup_grid_and_zones()
    # Obstacle at x=6.0, y=0.0 approaching front at vx=-2.0 m/s
    # Forward corridor starts at x=3.0 down to x=0.0.
    # Distance to boundary is 6.0 - 3.0 = 3.0 m. At 2 m/s, arrival should be ~1.5s!
    obstacle = TrackedObstacle(
        track_id=101,
        class_id=0,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(6.0, 0.0),
        velocity=(-2.0, 0.0),
        speed=2.0,
        hits=5,
        age=5,
        time_since_update=0,
        history=((6.2, 0.0), (6.0, 0.0)),
    )

    forecast = forecast_obstacle_trajectory(
        obstacle,
        grid=grid,
        zones=zones,
        horizon_s=3.0,
        step_s=0.1,
    )

    assert forecast.track_id == 101
    assert forecast.class_name == "vehicles"
    assert len(forecast.predicted_positions) > 0
    assert forecast.min_ttc_s is not None
    # Boundary is at 3.0m -> arrival around 1.5s
    assert 1.4 <= forecast.min_ttc_s <= 1.6
    intrusions = dict(forecast.zone_intrusions)
    assert "forward_corridor" in intrusions
    assert 1.4 <= intrusions["forward_corridor"] <= 1.6


def test_forecast_departing_obstacle_has_no_intrusions() -> None:
    grid, zones = _setup_grid_and_zones()
    # Obstacle at x=4.0, y=0.0 moving forward away at vx=+2.0 m/s
    obstacle = TrackedObstacle(
        track_id=102,
        class_id=1,
        class_name="person",
        state=TrackState.CONFIRMED,
        position=(4.0, 0.0),
        velocity=(2.0, 0.0),
        speed=2.0,
        hits=5,
        age=5,
        time_since_update=0,
    )

    forecast = forecast_obstacle_trajectory(
        obstacle,
        grid=grid,
        zones=zones,
        horizon_s=3.0,
    )

    assert forecast.min_ttc_s is None
    assert len(forecast.zone_intrusions) == 0


def test_forecast_obstacle_already_inside_zone() -> None:
    grid, zones = _setup_grid_and_zones()
    # Obstacle inside forward corridor at (1.5, 0.0)
    obstacle = TrackedObstacle(
        track_id=103,
        class_id=0,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(1.5, 0.0),
        velocity=(-0.5, 0.0),
        speed=0.5,
        hits=3,
        age=3,
        time_since_update=0,
    )

    forecast = forecast_obstacle_trajectory(
        obstacle,
        grid=grid,
        zones=zones,
        horizon_s=2.0,
    )

    assert forecast.min_ttc_s == 0.0
    intrusions = dict(forecast.zone_intrusions)
    assert intrusions.get("forward_corridor") == 0.0


def test_forecast_all_trajectories_filters_confirmed() -> None:
    grid, zones = _setup_grid_and_zones()
    obs_confirmed = TrackedObstacle(
        track_id=1,
        class_id=0,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(5.0, 0.0),
        velocity=(-1.0, 0.0),
        speed=1.0,
        hits=3,
        age=3,
        time_since_update=0,
    )
    obs_tentative = TrackedObstacle(
        track_id=2,
        class_id=1,
        class_name="person",
        state=TrackState.TENTATIVE,
        position=(5.0, 1.0),
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=1,
        age=1,
        time_since_update=0,
    )

    all_obs = [obs_confirmed, obs_tentative]

    # Confirmed only
    forecasts = forecast_all_trajectories(all_obs, grid=grid, zones=zones, confirmed_only=True)
    assert len(forecasts) == 1
    assert forecasts[0].track_id == 1

    # Include all
    all_forecasts = forecast_all_trajectories(
        all_obs, grid=grid, zones=zones, confirmed_only=False
    )
    assert len(all_forecasts) == 2
