"""Facility microgrid energy aggregator, peak shaving, and virtual power plant controller."""

from __future__ import annotations

import math

from nearfield360.config import V2GConfig
from nearfield360.v2g.inverter import SmartInverter
from nearfield360.v2g.models import InverterPhase, VehicleBPTProfile


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
