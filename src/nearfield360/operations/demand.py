"""Seeded stochastic arrival-demand generator for facility day-runs."""

from __future__ import annotations

import math

from nearfield360.config import OperationsConfig

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
