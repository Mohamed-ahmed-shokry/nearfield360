"""Dynamic time-of-use economic arbitrage evaluation and battery degradation models."""

from __future__ import annotations

from nearfield360.config import V2GConfig
from nearfield360.v2g.models import (
    GridTariffWindow,
    V2GArbitrageSchedule,
    V2GPowerMode,
    V2GTimeSlotDispatch,
    VehicleBPTProfile,
)


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


class ArbitrageScheduler:
    """Multi-interval economic optimizer generating degradation-constrained V2G dispatch plans."""

    def __init__(
        self,
        config: V2GConfig | None = None,
        tariff_schedule: list[GridTariffWindow] | None = None,
    ) -> None:
        self.config = config or V2GConfig()
        self.tariff_schedule = tariff_schedule or get_default_tariff_schedule()
        self.interval_minutes = self.config.arbitrage_interval_minutes
        self.base_degradation_penalty = self.config.battery_degradation_penalty_per_kwh
        self.min_departure_reserve_soc = self.config.min_departure_reserve_soc

    def schedule_vehicle(
        self,
        profile: VehicleBPTProfile,
        max_inverter_kw: float | None = None,
    ) -> V2GArbitrageSchedule:
        """Generate an optimal multi-interval charging and discharging dispatch schedule."""
        dt_hours = self.interval_minutes / 60.0
        dwell_hours = max(0.25, profile.departure_hour - profile.arrival_hour)
        num_slots = max(1, round(dwell_hours / dt_hours))

        rated_chg_kw = min(profile.max_charge_kw, max_inverter_kw or profile.max_charge_kw)
        rated_dis_kw = min(profile.max_discharge_kw, max_inverter_kw or profile.max_discharge_kw)
        deg_cost = self.base_degradation_penalty * profile.degradation_cost_factor

        # Find min import tariff available during dwell
        dwell_tariffs = [
            find_tariff_window(profile.arrival_hour + k * dt_hours, self.tariff_schedule)
            for k in range(num_slots)
        ]
        min_import_price = min(t.import_price_per_kwh for t in dwell_tariffs)

        target_departure_soc = max(profile.min_departure_soc, self.min_departure_reserve_soc)
        cap = profile.battery_capacity_kwh
        soc = profile.current_soc

        dispatches: list[V2GTimeSlotDispatch] = []
        tot_charged_kwh = 0.0
        tot_discharged_kwh = 0.0
        tot_cost_dollars = 0.0
        tot_revenue_dollars = 0.0
        tot_deg_dollars = 0.0

        for k in range(num_slots):
            hour_k = profile.arrival_hour + k * dt_hours
            tariff = dwell_tariffs[k]

            # Calculate slots remaining including this slot
            slots_left = num_slots - k
            max_recharge_in_slots = slots_left * rated_chg_kw * dt_hours * 0.96 / max(1.0, cap)
            deficit_soc = max(0.0, target_departure_soc - soc)
            must_charge = deficit_soc >= (max_recharge_in_slots * 0.80)

            # Check if profitable to discharge for peak arbitrage
            profitable_to_discharge = is_discharge_arbitrage_profitable(
                export_price=tariff.export_price_per_kwh,
                future_recharge_price=min_import_price,
                degradation_cost_per_kwh=deg_cost,
            )

            # Determine dispatch mode
            # Priority 1: Charge if below target departure SoC
            if soc < target_departure_soc:
                should_charge = must_charge or (
                    tariff.import_price_per_kwh <= min_import_price * 1.50
                )
                if should_charge:
                    target_fill = (
                        1.0
                        if tariff.import_price_per_kwh <= min_import_price * 1.15
                        else target_departure_soc
                    )
                    room_in_batt_kwh = max(0.0, (target_fill - soc) * cap)
                    p_ac = min(rated_chg_kw, room_in_batt_kwh / (dt_hours * 0.96))
                    if p_ac > 0.1:
                        energy_ac = p_ac * dt_hours
                        energy_batt = energy_ac * 0.96
                        soc += energy_batt / cap
                        soc = min(1.0, soc)

                        cost = energy_ac * tariff.import_price_per_kwh
                        tot_charged_kwh += energy_batt
                        tot_cost_dollars += cost

                        dispatches.append(
                            V2GTimeSlotDispatch(
                                slot_index=k,
                                hour=round(hour_k, 2),
                                power_mode=V2GPowerMode.CHARGE,
                                active_power_kw=round(p_ac, 2),
                                soc_end=round(soc, 4),
                                import_tariff=tariff.import_price_per_kwh,
                                export_tariff=tariff.export_price_per_kwh,
                                cost_or_revenue_delta=-round(cost, 3),
                            )
                        )
                        continue

            # Priority 2: Discharge for peak arbitrage only if holding surplus above departure
            can_discharge = (
                profitable_to_discharge and (soc > (target_departure_soc + 0.03)) and soc > 0.20
            )
            if can_discharge:
                surplus_soc = soc - target_departure_soc
                max_dis_energy_kwh = surplus_soc * cap * 0.96
                p_dis_ac = min(rated_dis_kw, max_dis_energy_kwh / dt_hours)
                if p_dis_ac > 0.1:
                    energy_ac = p_dis_ac * dt_hours
                    energy_batt = energy_ac / 0.96
                    soc -= energy_batt / cap
                    soc = max(0.10, soc)

                    revenue = energy_ac * tariff.export_price_per_kwh
                    cell_deg = energy_ac * deg_cost
                    tot_discharged_kwh += energy_ac
                    tot_revenue_dollars += revenue
                    tot_deg_dollars += cell_deg

                    dispatches.append(
                        V2GTimeSlotDispatch(
                            slot_index=k,
                            hour=round(hour_k, 2),
                            power_mode=V2GPowerMode.DISCHARGE_PEAK_SHAVE,
                            active_power_kw=-round(p_dis_ac, 2),
                            soc_end=round(soc, 4),
                            import_tariff=tariff.import_price_per_kwh,
                            export_tariff=tariff.export_price_per_kwh,
                            cost_or_revenue_delta=round(revenue, 3),
                        )
                    )
                    continue

            # Standby mode
            dispatches.append(
                V2GTimeSlotDispatch(
                    slot_index=k,
                    hour=round(hour_k, 2),
                    power_mode=V2GPowerMode.STANDBY,
                    active_power_kw=0.0,
                    soc_end=round(soc, 4),
                    import_tariff=tariff.import_price_per_kwh,
                    export_tariff=tariff.export_price_per_kwh,
                    cost_or_revenue_delta=0.0,
                )
            )

        net_profit = tot_revenue_dollars - tot_cost_dollars - tot_deg_dollars
        final_soc = round(soc, 4)
        min_met = final_soc >= (profile.min_departure_soc - 1e-4)

        return V2GArbitrageSchedule(
            vehicle_id=profile.vehicle_id,
            dispatches=dispatches,
            total_energy_charged_kwh=round(tot_charged_kwh, 2),
            total_energy_discharged_kwh=round(tot_discharged_kwh, 2),
            net_cost_dollars=round(tot_cost_dollars, 2),
            net_revenue_dollars=round(tot_revenue_dollars, 2),
            net_profit_dollars=round(net_profit, 2),
            final_soc=final_soc,
            min_departure_soc_met=min_met,
        )
