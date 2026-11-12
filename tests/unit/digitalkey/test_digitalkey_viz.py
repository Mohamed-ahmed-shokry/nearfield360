"""Unit tests for tactical 1280x720 Digital Key visualizer."""

from pathlib import Path

import cv2
import numpy as np

from nearfield360.digitalkey.crypto import DigitalKeyCrypto
from nearfield360.digitalkey.models import (
    DelegationScope,
    DevicePosition3D,
    DigitalKeyReport,
    HandoverState,
    ProximityInterlockDecision,
    ProximityZone,
    UWBRangeMeasurement,
)
from nearfield360.digitalkey.viz import render_digitalkey_dashboard


def _sample_report(relay_attack: bool = False, with_token: bool = True) -> DigitalKeyReport:
    pos = DevicePosition3D(
        x_m=2.0,
        y_m=1.2,
        z_m=1.0,
        distance_m=2.33,
        zone=ProximityZone.IMMEDIATE,
        is_interior=False,
        residual_error_m=0.04,
        confidence=0.96,
        timestamp_ns=1700000000000000,
    )
    interlock = ProximityInterlockDecision(
        interlock_active=relay_attack,
        standstill_command=relay_attack,
        detected_distance_m=2.33,
        proximity_zone=ProximityZone.IMMEDIATE,
        handover_state=HandoverState.AVP_DELEGATED,
        reason="Standstill triggered: Relay attack detected"
        if relay_attack
        else "Normal operation",
    )
    meas = [
        UWBRangeMeasurement(
            anchor_id=a,
            distance_m=2.5,
            timestamp_ns=1700000000000000,
            sts_valid=not relay_attack,
            tof_ns=8.34,
            snr_db=25.0,
            aoa_azimuth_deg=18.5,
        )
        for a in ("FL", "FR", "RL", "RR")
    ]
    token = None
    if with_token:
        crypto = DigitalKeyCrypto()
        token = crypto.issue_delegation_token(
            vin="WP0ZZZ99ZTS392001",
            user_id="usr-alice",
            device_id="iphone-16",
            scopes=[DelegationScope.AVP_OPERATE, DelegationScope.CABIN_LOCK],
        )

    return DigitalKeyReport(
        device_id="iphone-16",
        vin="WP0ZZZ99ZTS392001",
        handover_state=HandoverState.AVP_DELEGATED,
        proximity_zone=ProximityZone.IMMEDIATE,
        device_position=pos,
        interlock=interlock,
        uwb_measurements=meas,
        delegation_token=token,
        relay_attack_detected=relay_attack,
        relay_discrepancy_ns=12.5 if relay_attack else 0.05,
    )


def test_render_digitalkey_dashboard_dimensions() -> None:
    report = _sample_report()
    canvas = render_digitalkey_dashboard(report)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8


def test_render_digitalkey_dashboard_to_disk(tmp_path: Path) -> None:
    report = _sample_report()
    out_png = tmp_path / "digitalkey_dashboard.png"

    canvas = render_digitalkey_dashboard(report, output_path=out_png)
    assert out_png.is_file()
    assert out_png.stat().st_size > 5000

    loaded = cv2.imread(str(out_png))
    assert loaded is not None
    assert loaded.shape == canvas.shape


def test_render_digitalkey_dashboard_relay_attack_and_no_token() -> None:
    report = _sample_report(relay_attack=True, with_token=False)
    canvas = render_digitalkey_dashboard(report)

    assert canvas.shape == (720, 1280, 3)
