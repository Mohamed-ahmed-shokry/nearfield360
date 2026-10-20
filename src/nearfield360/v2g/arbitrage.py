"""Dynamic time-of-use economic arbitrage evaluation and battery degradation models."""

from __future__ import annotations

from nearfield360.v2g.models import GridTariffWindow


def get_default_tariff_schedule() -> list[GridTariffWindow]:
    """Return standard 4-tier commercial Time-of-Use tariff windows."""
    return [
        GridTariffWindow(
            window_id="OFF-PEAK-NIGHT",
            start_hour=22,
            end_hour=24,
            import_price_per_kwh=0.12,
            export_price_per_kwh=0.06,
            carbon_intensity_g_per_kwh=120.0,
        ),
        GridTariffWindow(
            window_id="OFF-PEAK-MORNING",
            start_hour=0,
            end_hour=6,
            import_price_per_kwh=0.12,
            export_price_per_kwh=0.06,
            carbon_intensity_g_per_kwh=120.0,
        ),
        GridTariffWindow(
            window_id="MID-PEAK-DAY",
            start_hour=6,
            end_hour=14,
            import_price_per_kwh=0.22,
            export_price_per_kwh=0.15,
            carbon_intensity_g_per_kwh=220.0,
        ),
        GridTariffWindow(
            window_id="ON-PEAK-EVENING",
            start_hour=14,
            end_hour=20,
            import_price_per_kwh=0.42,
            export_price_per_kwh=0.36,
            carbon_intensity_g_per_kwh=380.0,
        ),
        GridTariffWindow(
            window_id="SHOULDER-NIGHT",
            start_hour=20,
            end_hour=22,
            import_price_per_kwh=0.24,
            export_price_per_kwh=0.16,
            carbon_intensity_g_per_kwh=250.0,
        ),
    ]


def find_tariff_window(hour: float, schedule: list[GridTariffWindow]) -> GridTariffWindow:
    """Find the tariff window matching the given hour of the day (modulo 24)."""
    norm_hour = int(hour) % 24
    for window in schedule:
        if window.start_hour <= norm_hour < window.end_hour:
            return window
    # Fallback to first window if boundary
    return schedule[0]


def is_discharge_arbitrage_profitable(
    export_price: float,
    future_recharge_price: float,
    degradation_cost_per_kwh: float,
    round_trip_efficiency: float = 0.92,
) -> bool:
    """Assess whether discharging now and recharging later yields positive net economic margin.

    Net margin = export_price - degradation_cost - (future_recharge_price / round_trip_efficiency).
    """
    recharge_cost_effective = future_recharge_price / max(0.50, round_trip_efficiency)
    net_spread = export_price - degradation_cost_per_kwh - recharge_cost_effective
    return net_spread > 0.01  # Minimum 1 cent/kWh profit hurdle
