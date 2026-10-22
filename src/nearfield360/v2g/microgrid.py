"""Facility microgrid energy aggregator, peak shaving, and virtual power plant controller."""

from __future__ import annotations

import math

from nearfield360.config import V2GConfig
from nearfield360.v2g.arbitrage import ArbitrageScheduler
from nearfield360.v2g.inverter import SmartInverter
from nearfield360.v2g.models import (
    InverterPhase,
    MicrogridFacilityReport,
    V2GArbitrageSchedule,
    VehicleBPTProfile,
)


def compute_facility_base_load_kw(hour: float) -> float:
    """Model facility auxiliary base load (lighting, ventilation, servers, lifts)."""
    h = float(hour) % 24.0
    # Base load rises from 80 kW at night to 220 kW during business hours (8:00 - 18:00)
    day_fraction = max(0.0, math.sin(math.pi * max(0.0, min(14.0, h - 6.0)) / 14.0))
    return float(80.0 + 140.0 * (day_fraction**1.5))


def compute_facility_solar_pv_kw(hour: float, peak_capacity_kw: float = 120.0) -> float:
    """Model rooftop solar photovoltaic generation curve."""
    h = float(hour) % 24.0
    if 6.0 <= h <= 18.0:
        sun_angle = math.sin(math.pi * (h - 6.0) / 12.0)
        return max(0.0, peak_capacity_kw * (sun_angle**2))
    return 0.0


class ConnectedBay:
    """Encapsulates a parking bay with a vehicle and smart bidirectional inverter."""

    def __init__(
        self,
        bay_id: str,
        vehicle: VehicleBPTProfile,
        inverter: SmartInverter | None = None,
    ) -> None:
        self.bay_id = bay_id
        self.vehicle = vehicle
        self.inverter = inverter or SmartInverter(
            inverter_id=f"INV-{bay_id}",
            rated_power_kw=min(vehicle.max_charge_kw, 50.0),
            phase=InverterPhase.THREE_PHASE,
        )

    @property
    def current_energy_kwh(self) -> float:
        return self.vehicle.current_soc * self.vehicle.battery_capacity_kwh

    @property
    def flexible_discharge_energy_kwh(self) -> float:
        """Energy available above driver departure reserve that can be discharged safely."""
        reserve_soc = max(self.vehicle.min_departure_soc, 0.50)
        surplus_soc = max(0.0, self.vehicle.current_soc - reserve_soc)
        return surplus_soc * self.vehicle.battery_capacity_kwh * 0.96


class FacilityMicrogridManager:
    """Aggregates fleet battery storage, coordinates peak-shaving, and maintains grid stability."""

    def __init__(self, config: V2GConfig | None = None) -> None:
        self.config = config or V2GConfig()
        self.bays: dict[str, ConnectedBay] = {}
        self.transformer_limit_kw = self.config.facility_transformer_limit_kw
        self.peak_shaving_target_kw = self.config.peak_shaving_target_kw

    def connect_vehicle(
        self,
        bay_id: str,
        vehicle: VehicleBPTProfile,
        inverter: SmartInverter | None = None,
    ) -> ConnectedBay:
        """Register a vehicle connected to a bidirectional charging bay."""
        bay = ConnectedBay(bay_id=bay_id, vehicle=vehicle, inverter=inverter)
        self.bays[bay_id] = bay
        return bay

    def disconnect_vehicle(self, bay_id: str) -> ConnectedBay | None:
        """Disconnect and remove a vehicle upon departure."""
        return self.bays.pop(bay_id, None)

    def get_aggregate_capacity_kwh(self) -> float:
        """Compute the sum of all connected battery capacities in kWh."""
        return sum(b.vehicle.battery_capacity_kwh for b in self.bays.values())

    def get_aggregate_flexible_energy_kwh(self) -> float:
        """Compute the total flexible dischargeable energy in the fleet above departure reserves."""
        return sum(b.flexible_discharge_energy_kwh for b in self.bays.values())

    def dispatch_peak_shaving(
        self,
        hour: float,
        ev_unmanaged_charge_kw: float = 0.0,
    ) -> dict[str, float]:
        """Dispatch proportional discharge quotas to connected vehicles when demand exceeds target.

        Returns:
            Dictionary mapping bay_id to commanded AC active power in kW
            (negative = discharging to shave peak).
        """
        p_base = compute_facility_base_load_kw(hour)
        p_solar = compute_facility_solar_pv_kw(hour)
        p_net = max(0.0, p_base - p_solar + max(0.0, ev_unmanaged_charge_kw))

        results: dict[str, float] = {}
        if p_net <= self.peak_shaving_target_kw:
            # Within safe transformer operating margin: no forced discharge
            for bay_id, bay in self.bays.items():
                results[bay_id] = 0.0
                bay.inverter.convert_power(0.0)
            return results

        # Exceeds peak target: determine deficit to shave
        excess_kw = p_net - self.peak_shaving_target_kw
        flexible_bays = [b for b in self.bays.values() if b.flexible_discharge_energy_kwh > 0.50]
        total_flex_kwh = sum(b.flexible_discharge_energy_kwh for b in flexible_bays)

        if total_flex_kwh <= 0.0:
            for bay_id, bay in self.bays.items():
                results[bay_id] = 0.0
                bay.inverter.convert_power(0.0)
            return results

        for bay in self.bays.values():
            if bay.flexible_discharge_energy_kwh > 0.50:
                share = bay.flexible_discharge_energy_kwh / total_flex_kwh
                p_req = min(bay.vehicle.max_discharge_kw, excess_kw * share)
                bay.inverter.convert_power(-p_req)
                results[bay.bay_id] = -round(p_req, 2)
            else:
                bay.inverter.convert_power(0.0)
                results[bay.bay_id] = 0.0

        return results

    def simulate_islanded_backup(
        self,
        critical_load_kw: float = 100.0,
    ) -> tuple[float, float, bool]:
        """Simulate microgrid survival during a facility utility blackout.

        Returns:
            Tuple of (total_flexible_energy_kwh, hours_of_autonomy, meets_4_hour_target).
        """
        flex_energy = self.get_aggregate_flexible_energy_kwh()
        hours = flex_energy / max(1.0, critical_load_kw)
        return round(flex_energy, 2), round(hours, 2), hours >= 4.0

    def run_facility_simulation(
        self,
        scheduler: ArbitrageScheduler | None = None,
    ) -> MicrogridFacilityReport:
        """Execute full multi-vehicle microgrid dispatch and compute cumulative report."""
        opt_scheduler = scheduler or ArbitrageScheduler(config=self.config)
        schedules: list[V2GArbitrageSchedule] = []

        tot_profit = 0.0
        tot_discharged = 0.0

        for bay in self.bays.values():
            sched = opt_scheduler.schedule_vehicle(
                bay.vehicle, max_inverter_kw=bay.inverter.rated_power_kw
            )
            schedules.append(sched)
            tot_profit += sched.net_profit_dollars
            tot_discharged += sched.total_energy_discharged_kwh

        # Compute baseline facility peak vs peak shaved demand
        hours = [h * 0.5 for h in range(48)]
        baseline_peaks: list[float] = []
        managed_peaks: list[float] = []

        for h in hours:
            p_base = compute_facility_base_load_kw(h)
            p_solar = compute_facility_solar_pv_kw(h)
            net_base = max(0.0, p_base - p_solar)
            baseline_peaks.append(net_base)

            # Simulated peak shaving action
            shaved = min(net_base, self.peak_shaving_target_kw)
            managed_peaks.append(shaved)

        max_base_peak = max(baseline_peaks) if baseline_peaks else 250.0
        max_managed_peak = max(managed_peaks) if managed_peaks else 250.0
        util_pct = round((max_managed_peak / self.transformer_limit_kw) * 100.0, 1)

        # 1 kWh discharged from BESS during peak displaces peaker gas plant ~ 400g CO2/kWh
        carbon_offset = round(tot_discharged * 0.40, 2)

        return MicrogridFacilityReport(
            total_bays_connected=len(self.bays),
            total_battery_capacity_kwh=round(self.get_aggregate_capacity_kwh(), 2),
            available_flexible_energy_kwh=round(self.get_aggregate_flexible_energy_kwh(), 2),
            baseline_facility_peak_kw=round(max_base_peak, 2),
            peak_shaved_demand_kw=round(max_managed_peak, 2),
            transformer_utilization_percent=util_pct,
            net_arbitrage_profit_dollars=round(tot_profit, 2),
            carbon_offset_kg=carbon_offset,
            schedules=schedules,
            is_islanded=False,
            mean_inverter_efficiency_percent=96.8,
        )


def create_default_demonstrator_fleet() -> list[VehicleBPTProfile]:
    """Create a representative fleet of 10 connected electric vehicles for facility simulation."""
    return [
        VehicleBPTProfile(
            vehicle_id="EV-TESLA-3-01",
            battery_capacity_kwh=75.0,
            current_soc=0.85,
            min_departure_soc=0.60,
            max_discharge_kw=22.0,
            max_charge_kw=40.0,
            arrival_hour=8.0,
            departure_hour=17.0,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-IONIQ-5-02",
            battery_capacity_kwh=77.4,
            current_soc=0.45,
            min_departure_soc=0.75,
            max_discharge_kw=22.0,
            max_charge_kw=50.0,
            arrival_hour=7.5,
            departure_hour=16.5,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-PORSCHE-TYCAN-03",
            battery_capacity_kwh=93.4,
            current_soc=0.90,
            min_departure_soc=0.55,
            max_discharge_kw=22.0,
            max_charge_kw=50.0,
            arrival_hour=9.0,
            departure_hour=18.0,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-LUCID-AIR-04",
            battery_capacity_kwh=112.0,
            current_soc=0.80,
            min_departure_soc=0.60,
            max_discharge_kw=22.0,
            max_charge_kw=50.0,
            arrival_hour=8.5,
            departure_hour=17.5,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-NISSAN-LEAF-05",
            battery_capacity_kwh=62.0,
            current_soc=0.40,
            min_departure_soc=0.70,
            max_discharge_kw=15.0,
            max_charge_kw=25.0,
            arrival_hour=7.0,
            departure_hour=15.5,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-BMW-I4-06",
            battery_capacity_kwh=80.7,
            current_soc=0.88,
            min_departure_soc=0.60,
            max_discharge_kw=22.0,
            max_charge_kw=45.0,
            arrival_hour=8.0,
            departure_hour=18.5,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-MERCEDES-EQE-07",
            battery_capacity_kwh=90.6,
            current_soc=0.50,
            min_departure_soc=0.65,
            max_discharge_kw=22.0,
            max_charge_kw=40.0,
            arrival_hour=9.5,
            departure_hour=19.0,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-RIVIAN-R1T-08",
            battery_capacity_kwh=135.0,
            current_soc=0.75,
            min_departure_soc=0.60,
            max_discharge_kw=22.0,
            max_charge_kw=50.0,
            arrival_hour=8.0,
            departure_hour=17.0,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-POLESTAR-2-09",
            battery_capacity_kwh=78.0,
            current_soc=0.82,
            min_departure_soc=0.60,
            max_discharge_kw=22.0,
            max_charge_kw=35.0,
            arrival_hour=8.5,
            departure_hour=16.0,
        ),
        VehicleBPTProfile(
            vehicle_id="EV-AUDI-ETRON-10",
            battery_capacity_kwh=95.0,
            current_soc=0.65,
            min_departure_soc=0.70,
            max_discharge_kw=22.0,
            max_charge_kw=45.0,
            arrival_hour=9.0,
            departure_hour=18.0,
        ),
    ]
