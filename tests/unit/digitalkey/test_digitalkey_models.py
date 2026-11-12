"""Unit tests for CCC Digital Key 3.0 domain models, enums, and schemas."""

import json

import pytest
from pydantic import ValidationError

from nearfield360.digitalkey.models import (
    CCCDelegationToken,
    DelegationScope,
    DevicePosition3D,
    DigitalKeyReport,
    HandoverState,
    ProximityInterlockDecision,
    ProximityZone,
    UWBAnchor,
    UWBRangeMeasurement,
)


def test_proximity_zone_and_handover_state_enums() -> None:
    assert ProximityZone.TOUCH == "TOUCH"
    assert ProximityZone.IMMEDIATE == "IMMEDIATE"
    assert ProximityZone.NEAR == "NEAR"
    assert ProximityZone.FAR == "FAR"
    assert ProximityZone.INTERIOR == "INTERIOR"
    assert ProximityZone.OUT_OF_RANGE == "OUT_OF_RANGE"

    assert HandoverState.DISCONNECTED == "DISCONNECTED"
    assert HandoverState.AVP_DELEGATED == "AVP_DELEGATED"
    assert HandoverState.TAKEOVER_IN_PROGRESS == "TAKEOVER_IN_PROGRESS"
    assert HandoverState.MANUAL_CONTROL == "MANUAL_CONTROL"


def test_uwb_anchor_validation() -> None:
    anchor = UWBAnchor(anchor_id="FL", x_m=1.8, y_m=0.8, z_m=0.6, channel=5)
    assert anchor.anchor_id == "FL"
    assert anchor.x_m == 1.8
    assert anchor.channel == 5
    assert anchor.is_aoa_capable is True

    with pytest.raises(ValidationError):
        UWBAnchor(anchor_id="", x_m=1.8, y_m=0.8)


def test_uwb_range_measurement_serialization() -> None:
    meas = UWBRangeMeasurement(
        anchor_id="FR",
        distance_m=2.45,
        timestamp_ns=1700000000000000,
        sts_valid=True,
        tof_ns=8.17,
        snr_db=26.5,
        first_path_power_dbm=-72.0,
        aoa_azimuth_deg=15.4,
        aoa_elevation_deg=-3.2,
    )
    assert meas.anchor_id == "FR"
    assert meas.distance_m == 2.45
    assert meas.aoa_azimuth_deg == 15.4

    dump = meas.model_dump(mode="json")
    assert dump["sts_valid"] is True
    assert dump["tof_ns"] == 8.17

    with pytest.raises(ValidationError):
        UWBRangeMeasurement(
            anchor_id="FR",
            distance_m=-1.0,  # Negative distance invalid
            timestamp_ns=100,
            tof_ns=2.0,
        )


def test_device_position_3d() -> None:
    pos = DevicePosition3D(
        x_m=0.5,
        y_m=0.2,
        z_m=1.1,
        distance_m=0.54,
        zone=ProximityZone.INTERIOR,
        is_interior=True,
        residual_error_m=0.04,
        confidence=0.98,
        timestamp_ns=1700000000100000,
    )
    assert pos.is_interior is True
    assert pos.zone == ProximityZone.INTERIOR
    assert pos.residual_error_m == 0.04


def test_ccc_delegation_token() -> None:
    token = CCCDelegationToken(
        token_id="tok-avp-9876",  # noqa: S106
        vin="WP0ZZZ99ZTS392001",
        user_id="usr-alice",
        device_id="iphone-16-pro-uwb",
        scopes=[DelegationScope.AVP_OPERATE, DelegationScope.CABIN_LOCK],
        issued_at_utc="2026-11-13T00:00:00Z",
        expires_at_utc="2026-11-13T02:00:00Z",
        nonce="a1b2c3d4e5f67890",
        signature="3045022100abcd1234ef5678901234abcd",
        is_revoked=False,
    )
    assert len(token.scopes) == 2
    assert token.is_revoked is False

    dump = token.model_dump(mode="json")
    assert "AVP_OPERATE" in dump["scopes"]


def test_proximity_interlock_decision() -> None:
    decision = ProximityInterlockDecision(
        interlock_active=True,
        standstill_command=True,
        detected_distance_m=1.2,
        proximity_zone=ProximityZone.TOUCH,
        handover_state=HandoverState.AVP_DELEGATED,
        reason="Pedestrian within 1.5m safety boundary during autonomous maneuvering",
    )
    assert decision.interlock_active is True
    assert decision.standstill_command is True
    assert decision.proximity_zone == ProximityZone.TOUCH


def test_digital_key_report_round_trip() -> None:
    pos = DevicePosition3D(
        x_m=2.5,
        y_m=1.2,
        z_m=1.0,
        distance_m=2.77,
        zone=ProximityZone.IMMEDIATE,
        is_interior=False,
        residual_error_m=0.08,
        confidence=0.95,
        timestamp_ns=1700000000500000,
    )
    interlock = ProximityInterlockDecision(
        interlock_active=False,
        standstill_command=False,
        detected_distance_m=2.77,
        proximity_zone=ProximityZone.IMMEDIATE,
        handover_state=HandoverState.APPROACHING,
        reason="Normal approach detected outside touch zone",
    )
    meas = UWBRangeMeasurement(
        anchor_id="FL",
        distance_m=2.8,
        timestamp_ns=1700000000500000,
        sts_valid=True,
        tof_ns=9.34,
    )
    report = DigitalKeyReport(
        device_id="dev-samsung-s25",
        vin="WP0ZZZ99ZTS392001",
        handover_state=HandoverState.APPROACHING,
        proximity_zone=ProximityZone.IMMEDIATE,
        device_position=pos,
        interlock=interlock,
        uwb_measurements=[meas],
        relay_attack_detected=False,
        relay_discrepancy_ns=0.12,
    )

    data = report.to_dict()
    assert data["device_id"] == "dev-samsung-s25"
    assert data["proximity_zone"] == "IMMEDIATE"

    # Verify JSON serializability
    json_str = json.dumps(data)
    assert "WP0ZZZ99ZTS392001" in json_str
