"""Coordinated day-run simulation of gate, dispatch, parking, energy and billing."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from nearfield360.config import OperationsConfig
from nearfield360.operations.models import (
    JourneyStage,
    ServiceLevel,
    VehicleArrival,
    VehicleJourneyRecord,
)

DEFAULT_SLOT_CAPACITY = 120
DISPATCH_LATENCY_S = 20.0
MANUAL_PLATE_REVIEW_S = 40.0
LOW_PLATE_CONFIDENCE = 0.85
QUEUE_PATIENCE_S = 900.0
RETRIEVAL_LATENCY_S = 120.0
PARKING_RATE_USD_PER_HOUR = 2.50
ENERGY_PRICE_USD_PER_KWH = 0.30
V2G_CREDIT_USD_PER_KWH = 0.12
CHARGER_POWER_KW = 11.0
CHARGER_EFFICIENCY = 0.90
MAX_CHARGE_WINDOW_HOURS = 3.0
MAX_ENERGY_PER_SESSION_KWH = 60.0
V2G_EXPORT_FRACTION = 0.15
V2G_WINDOW_HOURS = (16.0, 20.0)


def parking_fee_usd(dwell_hours: float, grace_minutes: int) -> float:
    """Return the parking fee after subtracting the free grace period."""
    billable_hours = max(0.0, dwell_hours - grace_minutes / 60.0)
    return round(billable_hours * PARKING_RATE_USD_PER_HOUR, 2)


def energy_fee_usd(charged_kwh: float, exported_kwh: float) -> float:
    """Return the energy fee: imported energy price minus V2G export credit (never negative)."""
    net = charged_kwh * ENERGY_PRICE_USD_PER_KWH - exported_kwh * V2G_CREDIT_USD_PER_KWH
    return round(max(0.0, net), 2)


def simulate_journeys(
    arrivals: Sequence[VehicleArrival],
    config: OperationsConfig,
    slot_capacity: int = DEFAULT_SLOT_CAPACITY,
) -> list[VehicleJourneyRecord]:
    """Run every arrival through gate review, slot dispatch and retrieval.

    Arrivals are served FIFO. When all slots are occupied a vehicle waits for the
    earliest slot release; if that wait exceeds ``QUEUE_PATIENCE_S`` it balks and is
    recorded as ``BLOCKED``. Vehicles still inside the facility at the end of the
    day stay in the ``PARKED`` stage.
    """
    if slot_capacity < 1:
        raise ValueError("slot_capacity must be at least 1")

    horizon_s = config.day_duration_hours * 3600.0
    releases: list[tuple[float, int]] = []
    free_slots = list(range(slot_capacity))
    journeys: list[VehicleJourneyRecord] = []

    for arrival in sorted(arrivals, key=lambda a: a.arrival_time_s):
        now = arrival.arrival_time_s
        while releases and releases[0][0] <= now:
            heapq.heappush(free_slots, heapq.heappop(releases)[1])

        queue_wait_s = 0.0
        if not free_slots:
            queue_wait_s = releases[0][0] - now
            if queue_wait_s > QUEUE_PATIENCE_S:
                journeys.append(
                    VehicleJourneyRecord(
                        vehicle_id=arrival.vehicle_id,
                        arrival_time_s=now,
                        requested_dwell_hours=arrival.requested_dwell_hours,
                        stage=JourneyStage.DEPARTED,
                        service_level=ServiceLevel.BLOCKED,
                        dispatch_wait_s=QUEUE_PATIENCE_S,
                        departed_time_s=now + QUEUE_PATIENCE_S,
                    )
                )
                continue
            heapq.heappush(free_slots, heapq.heappop(releases)[1])

        low_confidence = arrival.plate_confidence < LOW_PLATE_CONFIDENCE
        gate_s = DISPATCH_LATENCY_S + (MANUAL_PLATE_REVIEW_S if low_confidence else 0.0)
        dispatch_wait_s = queue_wait_s + gate_s
        parked_s = now + dispatch_wait_s
        unpark_s = parked_s + arrival.requested_dwell_hours * 3600.0
        departed_s = unpark_s + RETRIEVAL_LATENCY_S
        slot_index = heapq.heappop(free_slots)
        heapq.heappush(releases, (departed_s, slot_index))

        if queue_wait_s > 0.0:
            level = ServiceLevel.CURBSIDE_HANDOVER
        elif low_confidence:
            level = ServiceLevel.DEGRADED_DISPATCH
        else:
            level = ServiceLevel.NOMINAL

        inside = departed_s <= horizon_s

        # Energy and billing computation for non-blocked vehicles
        charging_windows = 0
        charged_kwh = 0.0
        exported_kwh = 0.0
        if arrival.requests_charging:
            charge_duration_hours = min(arrival.requested_dwell_hours, MAX_CHARGE_WINDOW_HOURS)
            raw_energy = charge_duration_hours * CHARGER_POWER_KW * CHARGER_EFFICIENCY
            charged_kwh = round(min(raw_energy, MAX_ENERGY_PER_SESSION_KWH), 2)
            charging_windows = 1

            # Check if vehicle dwell intersects V2G export window
            parked_hour = (parked_s / 3600.0) % 24.0
            unpark_hour = (unpark_s / 3600.0) % 24.0
            v2g_start, v2g_end = V2G_WINDOW_HOURS
            if max(parked_hour, v2g_start) < min(unpark_hour, v2g_end):
                exported_kwh = round(charged_kwh * V2G_EXPORT_FRACTION, 2)

        p_fee = parking_fee_usd(arrival.requested_dwell_hours, config.billing_grace_minutes)
        e_fee = energy_fee_usd(charged_kwh, exported_kwh)

        journeys.append(
            VehicleJourneyRecord(
                vehicle_id=arrival.vehicle_id,
                arrival_time_s=now,
                requested_dwell_hours=arrival.requested_dwell_hours,
                stage=JourneyStage.DEPARTED if inside else JourneyStage.PARKED,
                service_level=level,
                dispatched_slot_id=f"S-{slot_index + 1:04d}",
                dispatch_wait_s=dispatch_wait_s,
                parked_time_s=parked_s,
                unpark_time_s=unpark_s,
                departed_time_s=departed_s if inside else None,
                charging_windows=charging_windows,
                energy_charged_kwh=charged_kwh,
                energy_exported_kwh=exported_kwh,
                parking_fee_usd=p_fee,
                energy_fee_usd=e_fee,
            )
        )
    return journeys
