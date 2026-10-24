"""Unit tests for the facility arrival-demand generator."""

from __future__ import annotations

import pytest

from nearfield360.config import OperationsConfig
from nearfield360.operations.demand import arrival_rate_per_hour, generate_arrivals


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


def test_generate_arrivals_is_deterministic_per_seed() -> None:
    config = OperationsConfig()
    first = generate_arrivals(config, seed=3)
    second = generate_arrivals(config, seed=3)
    other = generate_arrivals(config, seed=4)
    assert first == second
    assert first != other


def test_generate_arrivals_sorted_unique_and_in_horizon() -> None:
    config = OperationsConfig()
    arrivals = generate_arrivals(config, seed=11)
    times = [a.arrival_time_s for a in arrivals]
    assert times == sorted(times)
    assert all(0.0 <= t < config.day_duration_hours * 3600.0 for t in times)
    assert len({a.vehicle_id for a in arrivals}) == len(arrivals)
    assert all(0.25 <= a.requested_dwell_hours <= 24.0 for a in arrivals)


def test_generate_arrivals_shows_morning_and_evening_peaks() -> None:
    config = OperationsConfig()
    arrivals = generate_arrivals(config, seed=5)
    by_hour = [0] * 24
    for a in arrivals:
        by_hour[int(a.arrival_time_s // 3600)] += 1
    assert by_hour[8] > 2 * by_hour[2]
    assert by_hour[17] > 2 * by_hour[2]
    assert by_hour[8] > by_hour[12] or by_hour[17] > by_hour[12]


def test_generate_arrivals_charging_share_tracks_config() -> None:
    config = OperationsConfig(charging_activation_share=1.0)
    assert all(a.requests_charging for a in generate_arrivals(config, seed=1))
    config = OperationsConfig(charging_activation_share=0.0)
    assert not any(a.requests_charging for a in generate_arrivals(config, seed=1))


def test_generate_arrivals_zero_demand_is_empty() -> None:
    config = OperationsConfig(base_arrivals_per_hour=0.0, peak_arrivals_per_hour=0.5)
    config = config.model_copy(update={"peak_arrivals_per_hour": 0.0})
    assert generate_arrivals(config, seed=1) == []
