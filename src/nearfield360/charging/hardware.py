"""Automated charging hardware simulation and non-linear CC/CV battery curve modeling."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.charging.models import (
    ChargerBay,
    ChargingSession,
    ChargingSessionStatus,
    EVBatteryState,
)


class ChargingHardwareSimulator:
    """Simulates robotic docking, connector handshakes, and non-linear CC/CV curves.

    In lithium-ion battery charging:
    - Constant Current (CC) Phase (0% to ~80% SoC): Battery accepts full allocated power.
    - Constant Voltage (CV) Phase (80% to 100% SoC): Acceptance power tapers exponentially:
      $P(SoC) = P_{\max} \cdot \exp(-k \cdot (SoC - 0.80))$ to avoid cell over-voltage.
    """

    def __init__(self, taper_factor: float = 6.0) -> None:
        self.taper_factor = taper_factor

    def compute_instantaneous_power_acceptance(
        self,
        battery: EVBatteryState,
        allocated_power_kw: float,
    ) -> float:
        """Calculate actual power absorbed by the battery considering CC/CV saturation."""
        if battery.current_soc >= battery.target_soc:
            return 0.0

        # Maximum acceptance capped by battery and grid allocation
        max_kw = min(battery.max_charge_acceptance_kw, allocated_power_kw)

        # CC phase below 80% SoC
        if battery.current_soc <= 0.80:
            return max_kw

        # CV tapering above 80% SoC
        excess_soc = battery.current_soc - 0.80
        taper_ratio = math.exp(-self.taper_factor * excess_soc)
        return max(5.0, max_kw * taper_ratio)

    def step_session(
        self,
        session: ChargingSession,
        battery: EVBatteryState,
        bay: ChargerBay,
        dt_s: float,
    ) -> float:
        """Advance a charging session by dt_s seconds.

        Returns energy delivered in this step in kWh.
        """
        if session.status == ChargingSessionStatus.PENDING_DOCKING:
            session.idle_dwell_s += dt_s
            # Once docking dwell threshold is exceeded, advance to active charging
            if session.idle_dwell_s >= bay.docking_dwell_s:
                session.status = ChargingSessionStatus.ACTIVE_CHARGING
            return 0.0

        if session.status in (ChargingSessionStatus.COMPLETED, ChargingSessionStatus.ABORTED):
            session.idle_dwell_s += dt_s
            return 0.0

        # Calculate power acceptance with CC/CV tapering
        power_kw = self.compute_instantaneous_power_acceptance(
            battery=battery,
            allocated_power_kw=session.allocated_power_kw,
        )

        # Apply charger efficiency (e.g. 95%)
        effective_power_kw = power_kw * bay.efficiency_rating

        # Energy delivered: kWh = kW * (dt_s / 3600.0)
        energy_kwh = effective_power_kw * (dt_s / 3600.0)

        # Update battery SoC
        if battery.battery_capacity_kwh > 0.0:
            delta_soc = energy_kwh / (battery.battery_capacity_kwh * battery.degradation_factor)
            new_soc = min(1.0, battery.current_soc + delta_soc)
            object.__setattr__(battery, "current_soc", new_soc)
            session.current_soc = new_soc

        session.energy_delivered_kwh += energy_kwh
        session.last_update_time_s += dt_s

        # Update topping off status
        if session.current_soc >= 0.80 and session.current_soc < session.target_soc:
            session.status = ChargingSessionStatus.TOPPING_OFF

        # Estimate remaining time to target SoC
        if effective_power_kw > 0.1 and session.current_soc < session.target_soc:
            remaining_kwh = battery.energy_needed_kwh
            session.estimated_time_to_target_s = (remaining_kwh / effective_power_kw) * 3600.0
        else:
            session.estimated_time_to_target_s = 0.0

        return energy_kwh

    def step_facility(
        self,
        sessions: Sequence[ChargingSession],
        batteries: Sequence[EVBatteryState],
        bays: Sequence[ChargerBay],
        dt_s: float,
    ) -> float:
        """Advance all active charging sessions across the facility."""
        battery_map = {b.vehicle_id: b for b in batteries}
        bay_map = {b.bay_id: b for b in bays}

        total_step_energy_kwh = 0.0
        for sess in sessions:
            bat = battery_map.get(sess.vehicle_id)
            bay = bay_map.get(sess.bay_id)
            if bat and bay:
                step_energy = self.step_session(sess, bat, bay, dt_s)
                total_step_energy_kwh += step_energy

        return total_step_energy_kwh
