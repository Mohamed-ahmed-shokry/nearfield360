"""Unit tests for curbside domain models, serialization, and validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nearfield360.curbside.models import (
    BayStatus,
    CurbsideBay,
    CurbsideMission,
    CurbsideReport,
    HandoverType,
    PedestrianDetection,
    SafetyInterlockStatus,
)


def test_curbside_enums() -> None:
    assert HandoverType.DROPOFF.value == "dropoff"
    assert HandoverType.PICKUP.value == "pickup"

    assert BayStatus.VACANT.value == "vacant"
    assert BayStatus.OCCUPIED_UNLOADING.value == "occupied_unloading"
    assert BayStatus.CLEARING.value == "clearing"

    assert SafetyInterlockStatus.DISENGAGED_CLEAR.value == "disengaged_clear"
    assert SafetyInterlockStatus.ENGAGED_DOOR_OPEN.value == "engaged_door_open"


def test_pedestrian_detection_model() -> None:
    ped = PedestrianDetection(
        pedestrian_id="ped-01",
        distance_m=1.2,
        relative_bearing_rad=0.35,
        in_critical_zone=True,
    )
    assert ped.pedestrian_id == "ped-01"
    assert ped.distance_m == 1.2
    assert ped.in_critical_zone is True

    with pytest.raises(ValidationError):
        PedestrianDetection(pedestrian_id="", distance_m=-1.0)


def test_curbside_bay_model() -> None:
    bay = CurbsideBay(
        bay_id="dopu-bay-01",
        bay_type=HandoverType.DROPOFF,
        status=BayStatus.VACANT,
    )
    assert bay.bay_id == "dopu-bay-01"
    assert bay.is_occupied is False

    occupied = bay.model_copy(
        update={
            "status": BayStatus.OCCUPIED_UNLOADING,
            "occupied_vehicle_id": "veh-42",
            "interlock_status": SafetyInterlockStatus.ENGAGED_DOOR_OPEN,
        }
    )
    assert occupied.is_occupied is True
    assert occupied.occupied_vehicle_id == "veh-42"
    assert occupied.interlock_status == SafetyInterlockStatus.ENGAGED_DOOR_OPEN


def test_curbside_mission_and_report_serialization() -> None:
    mission = CurbsideMission(
        mission_id="mission-101",
        vehicle_id="veh-42",
        handover_type=HandoverType.DROPOFF,
        bay_id="dopu-bay-01",
        requested_at_s=100.0,
        completed_at_s=220.0,
        dwell_duration_s=120.0,
        interlock_triggers_count=2,
        overstay_penalty_usd=0.0,
        is_completed=True,
    )
    assert mission.is_completed is True
    assert mission.dwell_duration_s == 120.0

    bay = CurbsideBay(bay_id="dopu-bay-01", status=BayStatus.VACANT)
    report = CurbsideReport(
        facility_id="facility-avp-01",
        timestamp_s=250.0,
        total_bays_count=1,
        vacant_bays_count=1,
        bays=[bay],
        active_missions=[],
        completed_missions_count=1,
        active_interlocks_count=0,
        total_overstay_penalties_usd=0.0,
        average_dwell_duration_s=120.004,
    )
    assert report.average_dwell_duration_s == 120.00

    # JSON roundtrip
    dumped = report.model_dump_json()
    loaded = CurbsideReport.model_validate(json.loads(dumped))
    assert loaded.facility_id == "facility-avp-01"
    assert len(loaded.bays) == 1
    assert loaded.bays[0].bay_id == "dopu-bay-01"
