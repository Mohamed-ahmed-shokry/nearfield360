"""Unit tests for proximity safety standstill interlock and AVP handover supervisor."""

import math

from nearfield360.digitalkey.crypto import DigitalKeyCrypto
from nearfield360.digitalkey.interlock import HandoverSupervisor
from nearfield360.digitalkey.models import (
    DelegationScope,
    HandoverState,
    ProximityZone,
    UWBRangeMeasurement,
)


def _make_measurements(
    tx: float, ty: float, tz: float, sts_valid: bool = True
) -> list[UWBRangeMeasurement]:
    """Helper to generate consistent 4-anchor range measurements."""
    anchors = {
        "FL": (1.8, 0.8, 0.6),
        "FR": (1.8, -0.8, 1.2),
        "RL": (-1.8, 0.8, 1.2),
        "RR": (-1.8, -0.8, 0.6),
    }
    meas = []
    ts = 1700000000000000
    for a_id, (ax, ay, az) in anchors.items():
        dist = math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2)
        meas.append(
            UWBRangeMeasurement(
                anchor_id=a_id,
                distance_m=dist,
                timestamp_ns=ts,
                sts_valid=sts_valid,
                tof_ns=dist / 0.299792458,
            )
        )
    return meas


def test_handover_supervisor_initialization() -> None:
    supervisor = HandoverSupervisor()
    assert supervisor.state == HandoverState.DISCONNECTED
    assert supervisor.active_token is None


def test_delegate_to_avp_success_and_failure() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    supervisor = HandoverSupervisor(vin=vin, crypto=crypto)

    # Valid token
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-alice",
        device_id="iphone-16",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    ok, _msg = supervisor.delegate_to_avp(token)
    assert ok is True
    assert supervisor.state == HandoverState.AVP_DELEGATED
    assert supervisor.active_token == token

    # Invalid token (wrong VIN)
    supervisor2 = HandoverSupervisor(vin="WP0ZZZ99ZTS999999", crypto=crypto)
    bad_token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-alice",
        device_id="iphone-16",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    ok2, msg2 = supervisor2.delegate_to_avp(bad_token)
    assert ok2 is False
    assert "rejected" in msg2


def test_standstill_interlock_in_touch_zone_during_avp() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    supervisor = HandoverSupervisor(vin=vin, crypto=crypto)
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-alice",
        device_id="iphone-16",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    supervisor.delegate_to_avp(token)

    # Device positioned outside driver door in TOUCH zone (y=1.2m, distance=1.2m <= 1.5m)
    meas = _make_measurements(0.0, 1.2, 1.0)
    report = supervisor.evaluate(meas)

    assert report.proximity_zone == ProximityZone.TOUCH
    assert report.interlock.interlock_active is True
    assert report.interlock.standstill_command is True
    assert "STANDSTILL INTERLOCK" in report.interlock.reason


def test_standstill_interlock_on_relay_attack() -> None:
    supervisor = HandoverSupervisor()
    # Measurements with sts_valid=False
    attack_meas = _make_measurements(4.0, 0.0, 1.0, sts_valid=False)
    report = supervisor.evaluate(attack_meas)

    assert report.relay_attack_detected is True
    assert report.interlock.interlock_active is True
    assert report.interlock.standstill_command is True
    assert "Relay attack detected" in report.interlock.reason


def test_pickup_handover_sequencing() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    supervisor = HandoverSupervisor(vin=vin, crypto=crypto)
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-alice",
        device_id="iphone-16",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    supervisor.delegate_to_avp(token)
    assert supervisor.state == HandoverState.AVP_DELEGATED

    # Step 1: Vehicle arrives at pick-up berth
    supervisor.mark_pickup_ready()
    assert supervisor.state == HandoverState.PICKUP_READY

    # Step 2: Driver approaches to IMMEDIATE zone (e.g. x=2.2m)
    approach_meas = _make_measurements(2.2, 0.0, 1.0)
    report2 = supervisor.evaluate(approach_meas)
    assert supervisor.state == HandoverState.TAKEOVER_IN_PROGRESS
    assert report2.interlock.standstill_command is False

    # Step 3: Driver enters cabin interior (e.g. x=0.2, y=0.3, z=1.0)
    interior_meas = _make_measurements(0.2, 0.3, 1.0)
    report3 = supervisor.evaluate(interior_meas)
    assert report3.device_position.is_interior is True
    assert supervisor.state == HandoverState.MANUAL_CONTROL


def test_supervisor_reset() -> None:
    supervisor = HandoverSupervisor()
    supervisor.reset()
    assert supervisor.state == HandoverState.DISCONNECTED
    assert supervisor.active_token is None
