"""Seeded stochastic arrival-demand generator for facility day-runs."""

from __future__ import annotations

import math

import numpy as np

from nearfield360.config import OperationsConfig
from nearfield360.operations.models import VehicleArrival

_PEAK_SIGMA_HOURS = 1.5


def arrival_rate_per_hour(hour: float, config: OperationsConfig) -> float:
    """Return the expected arrivals per hour at ``hour`` of the day.

    The profile is a base rate with two Gaussian demand peaks (morning and
    evening). The wrap-around ``hour % 24`` keeps multi-day horizons periodic.
    """
    wrapped = hour % 24.0

    def _bump(center: float) -> float:
        delta = min(abs(wrapped - center), 24.0 - abs(wrapped - center))
        return math.exp(-0.5 * (delta / _PEAK_SIGMA_HOURS) ** 2)

    shape = max(
        _bump(config.peak_morning_arrival_hour),
        _bump(config.peak_evening_arrival_hour),
    )
    peak = max(config.peak_arrivals_per_hour, config.base_arrivals_per_hour)
    return config.base_arrivals_per_hour + (peak - config.base_arrivals_per_hour) * shape


def generate_arrivals(config: OperationsConfig, seed: int) -> list[VehicleArrival]:
    """Draw a deterministic arrival list for one day-run.

    Uses Lewis-Shedler thinning of a non-homogeneous Poisson process. Dwell time is
    exponential around ``parking_duration_mean_hours`` (clipped to 0.25-24 h).
    """
    rng = np.random.default_rng(seed)
    horizon_s = config.day_duration_hours * 3600.0
    rate_max_per_s = max(config.peak_arrivals_per_hour, config.base_arrivals_per_hour) / 3600.0
    arrivals: list[VehicleArrival] = []
    if rate_max_per_s <= 0.0:
        return arrivals

    time_s = 0.0
    while True:
        time_s += float(rng.exponential(1.0 / rate_max_per_s))
        if time_s >= horizon_s:
            break
        accept_probability = (
            arrival_rate_per_hour(time_s / 3600.0, config) / 3600.0 / rate_max_per_s
        )
        if float(rng.random()) > accept_probability:
            continue
        dwell = float(np.clip(rng.exponential(config.parking_duration_mean_hours), 0.25, 24.0))
        arrivals.append(
            VehicleArrival(
                vehicle_id=f"EV-{len(arrivals):05d}",
                arrival_time_s=time_s,
                plate_confidence=float(rng.uniform(0.80, 0.99)),
                requested_dwell_hours=dwell,
                requests_charging=bool(rng.random() < config.charging_activation_share),
            )
        )
    return arrivals
