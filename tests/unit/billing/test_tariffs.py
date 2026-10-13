"""Unit tests for dynamic tariff engine, congestion pricing multipliers, and grace period."""

from __future__ import annotations

from nearfield360.billing.models import TariffTier
from nearfield360.billing.tariffs import TariffEngine
from nearfield360.config import BillingConfig


def test_tariff_engine_grace_period() -> None:
    config = BillingConfig(grace_period_minutes=15.0)
    engine = TariffEngine(config)

    assert engine.evaluate_billable_minutes(10.0) == 0.0
    assert engine.evaluate_billable_minutes(15.0) == 0.0
    assert engine.evaluate_billable_minutes(16.0) == 16.0
    assert engine.evaluate_billable_minutes(60.0) == 60.0


def test_tariff_engine_pricing_tiers() -> None:
    config = BillingConfig(base_hourly_rate=10.0, peak_multiplier=1.5)
    engine = TariffEngine(config)

    # Standard rate (low occupancy)
    rate, tier = engine.calculate_effective_hourly_rate(occupancy_ratio=0.50)
    assert rate == 10.0
    assert tier == TariffTier.STANDARD

    # Moderate congestion (75% occupancy)
    rate, tier = engine.calculate_effective_hourly_rate(occupancy_ratio=0.75)
    assert rate == 15.0  # 10.0 * 1.5
    assert tier == TariffTier.PEAK_CONGESTION

    # Extreme congestion (90% occupancy) -> 1.5 * 1.25 = 1.875 -> 18.75
    rate, tier = engine.calculate_effective_hourly_rate(occupancy_ratio=0.92)
    assert rate == 18.75
    assert tier == TariffTier.PEAK_CONGESTION

    # Overnight discount
    rate, tier = engine.calculate_effective_hourly_rate(occupancy_ratio=0.90, is_overnight=True)
    assert rate == 5.0  # 50% discount off base rate
    assert tier == TariffTier.OVERNIGHT_DISCOUNT
