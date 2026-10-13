"""Dynamic tariff scheduling, congestion pricing, and grace period evaluation engine."""

from __future__ import annotations

from nearfield360.billing.models import TariffTier
from nearfield360.config import BillingConfig


class TariffEngine:
    """Calculates real-time dynamic pricing rates and tariff tiers for parking facilities."""

    def __init__(self, config: BillingConfig | None = None) -> None:
        self.config = config or BillingConfig()

    def determine_tariff_tier(
        self,
        occupancy_ratio: float,
        is_peak_hours: bool = False,
        is_overnight: bool = False,
        is_ev_premium: bool = False,
    ) -> TariffTier:
        """Determine appropriate tariff tier based on facility conditions."""
        if is_overnight:
            return TariffTier.OVERNIGHT_DISCOUNT
        if is_ev_premium:
            return TariffTier.EV_PREMIUM
        if is_peak_hours or occupancy_ratio >= 0.85:
            return TariffTier.PEAK_CONGESTION
        return TariffTier.STANDARD

    def calculate_effective_hourly_rate(
        self,
        occupancy_ratio: float,
        is_peak_hours: bool = False,
        is_overnight: bool = False,
    ) -> tuple[float, TariffTier]:
        """Compute effective hourly parking rate and corresponding tariff tier."""
        base_rate = self.config.base_hourly_rate

        if is_overnight:
            effective_rate = round(base_rate * 0.50, 2)
            return effective_rate, TariffTier.OVERNIGHT_DISCOUNT

        # Congestion multiplier based on facility occupancy load factor
        if occupancy_ratio >= 0.90:
            congestion_multiplier = self.config.peak_multiplier * 1.25
        elif occupancy_ratio >= 0.75 or is_peak_hours:
            congestion_multiplier = self.config.peak_multiplier
        else:
            congestion_multiplier = 1.0

        effective_rate = round(base_rate * congestion_multiplier, 2)
        tier = TariffTier.PEAK_CONGESTION if congestion_multiplier > 1.0 else TariffTier.STANDARD
        return effective_rate, tier

    def evaluate_billable_minutes(self, duration_minutes: float) -> float:
        """Apply facility grace period deduction."""
        if duration_minutes <= self.config.grace_period_minutes:
            return 0.0
        return duration_minutes
