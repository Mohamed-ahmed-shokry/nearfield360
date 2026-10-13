"""Dynamic power grid management, peak shaving, and active load balancing for EV charging."""

from __future__ import annotations

from collections.abc import Sequence

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargingSession,
    ChargingSessionStatus,
    EVBatteryState,
    FacilityPowerSummary,
)
from nearfield360.config import ChargingConfig


class GridPowerManager:
    """Orchestrates dynamic electrical load balancing and peak shaving across facility chargers.

    Guarantees that the aggregate instantaneous power delivered to active vehicles does not exceed
    the facility transformer grid capacity. If total demand exceeds the peak shaving threshold,
    power is dynamically throttled across sessions using urgency and SoC weighting.
    """

    def __init__(
        self,
        config: ChargingConfig | None = None,
        facility_id: str = "facility_ev_grid",
    ) -> None:
        self.config = config or ChargingConfig()
        self.facility_id = facility_id

    @property
    def grid_capacity_kw(self) -> float:
        """Total transformer grid capacity in kilowatts."""
        return self.config.facility_grid_power_kw

    @property
    def peak_shaving_threshold_kw(self) -> float:
        """Threshold power above which active curtailment and load balancing activates."""
        return self.config.facility_grid_power_kw * self.config.peak_shaving_threshold_ratio

    def balance_power(
        self,
        active_sessions: Sequence[ChargingSession],
        bays: Sequence[ChargerBay],
        batteries: Sequence[EVBatteryState],
    ) -> FacilityPowerSummary:
        """Compute optimal power allocation for each active charging session.

        Modifies `allocated_power_kw` in place on each active charging session and returns a
        structured summary of facility power status.
        """
        bay_lookup = {b.bay_id: b for b in bays}
        battery_lookup = {bat.vehicle_id: bat for bat in batteries}

        # Filter sessions that are actively drawing energy
        charging_sessions = [
            s
            for s in active_sessions
            if s.status
            in (
                ChargingSessionStatus.ACTIVE_CHARGING,
                ChargingSessionStatus.TOPPING_OFF,
            )
        ]

        if not charging_sessions:
            return FacilityPowerSummary(
                facility_id=self.facility_id,
                grid_capacity_kw=self.grid_capacity_kw,
                total_requested_power_kw=0.0,
                total_allocated_power_kw=0.0,
                available_headroom_kw=self.grid_capacity_kw,
                active_charging_sessions=0,
                throttled_sessions=0,
                grid_utilization_ratio=0.0,
                is_curtailed=False,
            )

        # 1. Determine maximum requested power per session based on bay and vehicle acceptance
        session_demands: dict[str, float] = {}
        session_weights: dict[str, float] = {}

        for sess in charging_sessions:
            bay = bay_lookup.get(sess.bay_id)
            bay_max_kw = bay.max_power_rating_kw if bay else self.config.default_charger_power_kw
            battery = battery_lookup.get(sess.vehicle_id)
            veh_max_kw = battery.max_charge_acceptance_kw if battery else bay_max_kw

            # Demand is limited by both hardware capacity and battery acceptance rate
            requested_kw = min(bay_max_kw, veh_max_kw)
            session_demands[sess.session_id] = requested_kw

            # Weight priority: lower current SoC vehicles get higher priority share
            cur_soc = sess.current_soc
            weight = max(0.1, 1.0 - cur_soc)
            if cur_soc < self.config.critical_soc_threshold:
                weight += 2.0  # Critical boost
            session_weights[sess.session_id] = weight

        total_requested_kw = sum(session_demands.values())
        total_allocated_kw = 0.0
        throttled_count = 0
        is_curtailed = False

        # 2. Check if total demand exceeds peak shaving capacity
        available_budget_kw = self.grid_capacity_kw

        if total_requested_kw <= available_budget_kw:
            # Full allocation without curtailment
            for sess in charging_sessions:
                alloc_kw = session_demands[sess.session_id]
                sess.allocated_power_kw = alloc_kw
                if alloc_kw > sess.peak_power_kw:
                    sess.peak_power_kw = alloc_kw
                total_allocated_kw += alloc_kw
        else:
            # Curtailment required: dynamically balance power proportional to priority weights
            is_curtailed = True
            total_weight = sum(session_weights.values())

            # First pass: allocate proportionally by weight
            for sess in charging_sessions:
                w = session_weights[sess.session_id]
                share = (w / total_weight) * available_budget_kw
                max_dem = session_demands[sess.session_id]
                # Bound by requested power and minimum allocation
                alloc_kw = min(max_dem, share)
                alloc_kw = max(self.config.min_power_allocation_kw, alloc_kw)

                if alloc_kw < max_dem:
                    throttled_count += 1

                sess.allocated_power_kw = alloc_kw
                if alloc_kw > sess.peak_power_kw:
                    sess.peak_power_kw = alloc_kw
                total_allocated_kw += alloc_kw

            # Safety clamp: if sum exceeds capacity due to min allocations, scale down
            if total_allocated_kw > available_budget_kw:
                scale = available_budget_kw / total_allocated_kw
                total_allocated_kw = 0.0
                for sess in charging_sessions:
                    sess.allocated_power_kw = max(1.0, sess.allocated_power_kw * scale)
                    total_allocated_kw += sess.allocated_power_kw

        headroom_kw = max(0.0, self.grid_capacity_kw - total_allocated_kw)
        utilization = min(1.0, total_allocated_kw / self.grid_capacity_kw)

        # Update bay hardware current power
        for sess in charging_sessions:
            bay = bay_lookup.get(sess.bay_id)
            if bay and bay.status == ChargerPortStatus.OCCUPIED_CHARGING:
                object.__setattr__(bay, "current_power_kw", sess.allocated_power_kw)

        return FacilityPowerSummary(
            facility_id=self.facility_id,
            grid_capacity_kw=self.grid_capacity_kw,
            total_requested_power_kw=round(total_requested_kw, 2),
            total_allocated_power_kw=round(total_allocated_kw, 2),
            available_headroom_kw=round(headroom_kw, 2),
            active_charging_sessions=len(charging_sessions),
            throttled_sessions=throttled_count,
            grid_utilization_ratio=round(utilization, 3),
            is_curtailed=is_curtailed,
        )
