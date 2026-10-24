"""Unit tests for the facility arrival-demand generator."""

from __future__ import annotations

import pytest

from nearfield360.config import OperationsConfig
from nearfield360.operations.demand import arrival_rate_per_hour


def test_rate_peaks_at_configured_hours() -> None:
    config = OperationsConfig()
    morning = arrival_rate_per_hour(config.peak_morning_arrival_hour, config)
    evening = arrival_rate_per_hour(config.peak_evening_arrival_hour, config)
    assert morning == pytest.approx(config.peak_arrivals_per_hour)
    assert evening == pytest.approx(config.peak_arrivals_per_hour)


def test_rate_returns_to_base_far_from_peaks() -> None:
    config = OperationsConfig()
    night = arrival_rate_per_hour(2.0, config)
    assert config.base_arrivals_per_hour <= night < config.base_arrivals_per_hour + 1.0


def test_rate_is_periodic_over_days() -> None:
    config = OperationsConfig()
    assert arrival_rate_per_hour(8.5, config) == pytest.approx(arrival_rate_per_hour(32.5, config))
