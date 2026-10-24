"""Unit tests for facility operations domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.operations.models import (
    DailyKPIReport,
    HourlySnapshot,
    JourneyStage,
    OperationsDayReport,
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


def test_hourly_snapshot_round_trip_and_validation() -> None:
    snap = HourlySnapshot(
        hour=8,
        arrivals=40,
        departures=5,
        blocked=1,
        occupied_slots=120,
        queue_length=2,
        charging_vehicles=30,
        facility_load_kw=210.5,
    )
    assert HourlySnapshot.model_validate_json(snap.model_dump_json()) == snap
    with pytest.raises(ValidationError):
        HourlySnapshot(
            hour=-1,
            arrivals=0,
            departures=0,
            blocked=0,
            occupied_slots=0,
            queue_length=0,
            charging_vehicles=0,
            facility_load_kw=0.0,
        )


def _kpis(**overrides: object) -> DailyKPIReport:
    data: dict[str, object] = {
        "total_arrivals": 10,
        "completed_journeys": 9,
        "blocked_vehicles": 1,
        "throughput_per_hour": 0.4,
        "peak_occupancy": 5,
        "peak_occupancy_ratio": 0.5,
        "mean_dispatch_wait_s": 12.0,
        "p95_dispatch_wait_s": 30.0,
        "energy_charged_kwh": 100.0,
        "energy_exported_kwh": 10.0,
        "net_grid_energy_kwh": 90.0,
        "parking_revenue_usd": 50.0,
        "energy_revenue_usd": 20.0,
        "total_revenue_usd": 70.0,
        "sla_compliance_ratio": 0.9,
    }
    data.update(overrides)
    return DailyKPIReport(**data)  # type: ignore[arg-type]


def test_kpi_report_rejects_ratio_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _kpis(peak_occupancy_ratio=1.2)
    with pytest.raises(ValidationError):
        _kpis(sla_compliance_ratio=-0.1)


def test_day_report_round_trip() -> None:
    report = OperationsDayReport(seed=7, slot_capacity=10, day_duration_hours=24.0, kpis=_kpis())
    restored = OperationsDayReport.model_validate_json(report.model_dump_json())
    assert restored == report
    with pytest.raises(ValidationError):
        OperationsDayReport(seed=1, slot_capacity=0, day_duration_hours=24.0, kpis=_kpis())
