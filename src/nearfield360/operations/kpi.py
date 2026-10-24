"""KPI and snapshot aggregation for facility operations control tower."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nearfield360.config import OperationsConfig
from nearfield360.operations.models import (
    DailyKPIReport,
    HourlySnapshot,
    JourneyStage,
    ServiceLevel,
    VehicleJourneyRecord,
)

BASE_FACILITY_LOAD_KW = 45.0


def compute_hourly_snapshots(
    journeys: Sequence[VehicleJourneyRecord],
    config: OperationsConfig,
) -> list[HourlySnapshot]:
    """Compute per-hour facility activity snapshots across the simulated day."""
    hours = int(config.day_duration_hours)
    snapshots: list[HourlySnapshot] = []

    for h in range(hours):
        t_start = h * 3600.0
        t_end = (h + 1) * 3600.0

        arrivals = 0
        departures = 0
        blocked = 0
        occupied_slots = 0
        queue_count = 0
        charging_count = 0
        hourly_energy_charged_kwh = 0.0
        hourly_energy_exported_kwh = 0.0
        revenue_usd = 0.0

        for j in journeys:
            # Arrivals in this hour
            if t_start <= j.arrival_time_s < t_end:
                arrivals += 1
                if j.service_level is ServiceLevel.BLOCKED:
                    blocked += 1

            # Departures in this hour
            if (
                j.departed_time_s is not None
                and t_start <= j.departed_time_s < t_end
                and j.stage is JourneyStage.DEPARTED
                and j.service_level is not ServiceLevel.BLOCKED
            ):
                departures += 1
                revenue_usd += j.parking_fee_usd + j.energy_fee_usd

            # Active occupancy at midpoint of hour
            t_mid = t_start + 1800.0
            if j.parked_time_s is not None and j.service_level is not ServiceLevel.BLOCKED:
                park_end = j.departed_time_s if j.departed_time_s is not None else float("inf")
                if j.parked_time_s <= t_mid < park_end:
                    occupied_slots += 1
                    if j.charging_windows > 0 and j.energy_charged_kwh > 0.0:
                        charging_count += 1

            # Queue / curbside wait active during this hour
            wait_end = j.arrival_time_s + j.dispatch_wait_s
            is_delayed = j.service_level in (
                ServiceLevel.CURBSIDE_HANDOVER,
                ServiceLevel.DEGRADED_DISPATCH,
            )
            if (
                j.dispatch_wait_s > 0.0
                and j.service_level is not ServiceLevel.BLOCKED
                and is_delayed
                and max(t_start, j.arrival_time_s) < min(t_end, wait_end)
            ):
                queue_count += 1

            # Energy attribution to this hour
            if (
                j.parked_time_s is not None
                and j.unpark_time_s is not None
                and j.service_level is not ServiceLevel.BLOCKED
            ):
                dwell_dur = max(j.unpark_time_s - j.parked_time_s, 1.0)
                overlap = max(0.0, min(t_end, j.unpark_time_s) - max(t_start, j.parked_time_s))
                if overlap > 0.0:
                    frac = overlap / dwell_dur
                    hourly_energy_charged_kwh += j.energy_charged_kwh * frac
                    hourly_energy_exported_kwh += j.energy_exported_kwh * frac

        load_kw = BASE_FACILITY_LOAD_KW + hourly_energy_charged_kwh
        export_kw = hourly_energy_exported_kwh

        snapshots.append(
            HourlySnapshot(
                hour=h,
                arrivals=arrivals,
                departures=departures,
                blocked=blocked,
                occupied_slots=occupied_slots,
                queue_length=queue_count,
                charging_vehicles=charging_count,
                facility_load_kw=round(load_kw, 2),
                export_kw=round(export_kw, 2),
                revenue_usd=round(revenue_usd, 2),
            )
        )

    return snapshots


def aggregate_daily_kpis(
    journeys: Sequence[VehicleJourneyRecord],
    snapshots: Sequence[HourlySnapshot],
    config: OperationsConfig,
    slot_capacity: int,
) -> DailyKPIReport:
    """Aggregate high-level KPIs across vehicle journeys and hourly snapshots."""
    total_arrivals = len(journeys)
    completed_journeys = sum(
        1
        for j in journeys
        if j.stage is JourneyStage.DEPARTED and j.service_level is not ServiceLevel.BLOCKED
    )
    blocked_vehicles = sum(1 for j in journeys if j.service_level is ServiceLevel.BLOCKED)

    duration_h = config.day_duration_hours
    throughput_per_hour = round(completed_journeys / duration_h, 2) if duration_h > 0 else 0.0

    peak_occupancy = max((s.occupied_slots for s in snapshots), default=0)
    peak_ratio = round(peak_occupancy / slot_capacity, 4) if slot_capacity > 0 else 0.0

    waits = [j.dispatch_wait_s for j in journeys if j.service_level is not ServiceLevel.BLOCKED]
    mean_wait = round(float(np.mean(waits)), 2) if waits else 0.0
    p95_wait = round(float(np.percentile(waits, 95)), 2) if waits else 0.0

    energy_charged = round(sum(j.energy_charged_kwh for j in journeys), 2)
    energy_exported = round(sum(j.energy_exported_kwh for j in journeys), 2)
    net_grid_energy = round(energy_charged - energy_exported, 2)

    parking_rev = round(
        sum(j.parking_fee_usd for j in journeys if j.service_level is not ServiceLevel.BLOCKED), 2
    )
    energy_rev = round(
        sum(j.energy_fee_usd for j in journeys if j.service_level is not ServiceLevel.BLOCKED), 2
    )
    total_rev = round(parking_rev + energy_rev, 2)

    service_counts: dict[str, int] = {}
    for j in journeys:
        service_counts[j.service_level.value] = service_counts.get(j.service_level.value, 0) + 1

    nominal_count = service_counts.get(ServiceLevel.NOMINAL.value, 0)
    sla_ratio = round(nominal_count / total_arrivals, 4) if total_arrivals > 0 else 1.0

    return DailyKPIReport(
        total_arrivals=total_arrivals,
        completed_journeys=completed_journeys,
        blocked_vehicles=blocked_vehicles,
        throughput_per_hour=throughput_per_hour,
        peak_occupancy=peak_occupancy,
        peak_occupancy_ratio=peak_ratio,
        mean_dispatch_wait_s=mean_wait,
        p95_dispatch_wait_s=p95_wait,
        energy_charged_kwh=energy_charged,
        energy_exported_kwh=energy_exported,
        net_grid_energy_kwh=net_grid_energy,
        parking_revenue_usd=parking_rev,
        energy_revenue_usd=energy_rev,
        total_revenue_usd=total_rev,
        service_level_counts=service_counts,
        sla_compliance_ratio=sla_ratio,
    )
