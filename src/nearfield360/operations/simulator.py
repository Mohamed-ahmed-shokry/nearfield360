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
            )
        )
    return journeys
