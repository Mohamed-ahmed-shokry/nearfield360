"""Dynamic obstacle risk assessment, trajectory forecasting, and zone intrusion detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.tracking.models import (
    TrackedObstacle,
    TrackState,
    TrajectoryForecast,
)

if TYPE_CHECKING:
    from nearfield360.geometry.bev import BevGrid
    from nearfield360.occupancy.risk import RiskZone


def forecast_obstacle_trajectory(
    obstacle: TrackedObstacle,
    *,
    grid: BevGrid,
    zones: Sequence[RiskZone],
    horizon_s: float = 3.0,
    step_s: float = 0.1,
) -> TrajectoryForecast:
    """Forecast future positions of a tracked obstacle and detect safety zone ingress.

    Args:
        obstacle: TrackedObstacle containing position and velocity estimates.
        grid: BevGrid defining the vehicle-centric spatial frame.
        zones: Sequence of RiskZone safety zones to monitor.
        horizon_s: Total time horizon in seconds for forward trajectory propagation.
        step_s: Discrete time step in seconds between forecasted waypoints.

    Returns:
        TrajectoryForecast with predicted waypoints, zone intrusions, and minimum TTC.
    """
    x0, y0 = obstacle.position
    vx, vy = obstacle.velocity

    n_steps = max(1, int(np.ceil(horizon_s / max(step_s, 1e-4))))
    times = np.linspace(0.0, horizon_s, n_steps + 1)

    px = x0 + vx * times
    py = y0 + vy * times

    predicted_positions: list[tuple[float, float, float]] = [
        (round(float(t), 2), round(float(x), 3), round(float(y), 3))
        for t, x, y in zip(times, px, py, strict=True)
    ]

    # Map trajectory waypoints to BEV grid indices
    pts = np.column_stack((px, py))
    indexed = grid.world_to_grid(pts)
    valid_mask = indexed.valid
    indices = indexed.indices

    zone_intrusions: list[tuple[str, float]] = []
    earliest_ttc: float | None = None

    for zone in zones:
        first_intrusion_t: float | None = None
        for i, t_val in enumerate(times):
            if not valid_mask[i]:
                continue
            r, c = int(indices[i, 0]), int(indices[i, 1])
            if zone.mask[r, c]:
                first_intrusion_t = float(t_val)
                break

        if first_intrusion_t is not None:
            zone_intrusions.append((zone.name, round(first_intrusion_t, 2)))
            if earliest_ttc is None or first_intrusion_t < earliest_ttc:
                earliest_ttc = first_intrusion_t

    return TrajectoryForecast(
        track_id=obstacle.track_id,
        class_name=obstacle.class_name,
        predicted_positions=tuple(predicted_positions),
        zone_intrusions=tuple(zone_intrusions),
        min_ttc_s=round(earliest_ttc, 2) if earliest_ttc is not None else None,
    )


def forecast_all_trajectories(
    obstacles: Sequence[TrackedObstacle],
    *,
    grid: BevGrid,
    zones: Sequence[RiskZone],
    horizon_s: float = 3.0,
    step_s: float = 0.1,
    confirmed_only: bool = True,
) -> list[TrajectoryForecast]:
    """Evaluate trajectory forecasts and dynamic collision risk for all tracked obstacles."""
    forecasts: list[TrajectoryForecast] = []
    for obs in obstacles:
        if confirmed_only and obs.state != TrackState.CONFIRMED:
            continue
        forecast = forecast_obstacle_trajectory(
            obs,
            grid=grid,
            zones=zones,
            horizon_s=horizon_s,
            step_s=step_s,
        )
        forecasts.append(forecast)
    return forecasts


__all__ = [
    "forecast_all_trajectories",
    "forecast_obstacle_trajectory",
]
