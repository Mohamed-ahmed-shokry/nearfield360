"""Unit tests for facility operations domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.operations.models import (
    JourneyStage,
    ServiceLevel,
    VehicleArrival,
    VehicleJourneyRecord,
)


def test_enums_have_stable_wire_values() -> None:
    assert ServiceLevel.NOMINAL == "nominal"
    assert ServiceLevel.DEGRADED_DISPATCH == "degraded_dispatch"
    assert ServiceLevel.BLOCKED == "blocked"
    assert JourneyStage.ARRIVED == "arrived"
    assert JourneyStage.DEPARTED == "departed"


def test_vehicle_arrival_defaults_and_frozen() -> None:
    arrival = VehicleArrival(vehicle_id="EV-1", arrival_time_s=10.0)
    assert arrival.requested_dwell_hours == 3.0
    assert arrival.requests_charging is False
    with pytest.raises(ValidationError):
        arrival.arrival_time_s = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vehicle_id": "  ", "arrival_time_s": 0.0},
        {"vehicle_id": "bad id!", "arrival_time_s": 0.0},
        {"vehicle_id": "EV-1", "arrival_time_s": -1.0},
        {"vehicle_id": "EV-1", "arrival_time_s": float("nan")},
        {"vehicle_id": "EV-1", "arrival_time_s": 0.0, "requested_dwell_hours": 0.0},
        {"vehicle_id": "EV-1", "arrival_time_s": 0.0, "plate_confidence": 1.5},
    ],
)
def test_vehicle_arrival_rejects_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        VehicleArrival(**kwargs)  # type: ignore[arg-type]


def test_journey_record_json_round_trip() -> None:
    record = VehicleJourneyRecord(
        vehicle_id="EV-2",
        arrival_time_s=100.0,
        requested_dwell_hours=2.0,
        stage=JourneyStage.DEPARTED,
        service_level=ServiceLevel.RELOCATED,
        energy_charged_kwh=12.5,
    )
    restored = VehicleJourneyRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_journey_record_rejects_negative_energy() -> None:
    with pytest.raises(ValidationError):
        VehicleJourneyRecord(
            vehicle_id="EV-3",
            arrival_time_s=0.0,
            requested_dwell_hours=1.0,
            energy_charged_kwh=-1.0,
        )
