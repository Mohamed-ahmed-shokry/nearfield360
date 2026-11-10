"""Unit tests for DSSADExporter JSON, CSV, and UDS ISO 14229 readouts."""

from __future__ import annotations

import csv
import io
import json

from nearfield360.dssad.exporter import DSSADExporter
from nearfield360.dssad.models import (
    DSSADEventRecord,
    DSSADEventType,
    DSSADReport,
    EDRIncidentRecord,
    EventSeverity,
    TelemetrySnapshot,
)


def _make_dummy_incident() -> EDRIncidentRecord:
    snap = TelemetrySnapshot(
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        speed_mps=4.5,
        longitudinal_accel_mps2=-15.2,
        lateral_accel_mps2=1.2,
        yaw_rate_rads=0.05,
        steering_angle_rad=-0.10,
        brake_pressure_bar=65.0,
        throttle_pct=0.0,
        gear="D",
        x_m=10.0,
        y_m=5.0,
        yaw_rad=0.785,
        odd_state="ODD_BREACH_MRM",
        aeb_active=True,
        active_dtcs=["C1201", "U0100"],
    )
    evt = DSSADEventRecord(
        record_id="EVT-20261110-00001",
        sequence_id=0,
        timestamp_ns=snap.timestamp_ns,
        timestamp_iso=snap.timestamp_iso,
        event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
        severity=EventSeverity.FATAL,
        trigger_subsystem="chassis_imu",
        description="Severe collision impact",
        snapshot=snap,
        active_dtcs=snap.active_dtcs,
        block_hash="abcd" * 16,
    )
    return EDRIncidentRecord(
        incident_id="INC-20261110-0001",
        freeze_timestamp_iso="2026-11-10T12:00:00.050Z",
        trigger_event=evt,
        pre_trigger_telemetry=[snap],
        post_trigger_telemetry=[snap],
        delta_v_mps=5.4,
        peak_accel_g=3.25,
        tamper_chain_digest="1234567890abcdef" * 4,
        is_locked=True,
    )


def test_export_json_validates() -> None:
    incident = _make_dummy_incident()
    report = DSSADReport(
        report_id="REP-001",
        vehicle_id="AVP-TEST-01",
        generated_at="2026-11-10T12:00:00Z",
        total_events_recorded=1,
        total_incidents_frozen=1,
        chain_verified=True,
        events=[incident.trigger_event],
        incidents=[incident],
    )

    json_str = DSSADExporter.export_json(report)
    parsed = json.loads(json_str)

    assert parsed["report_id"] == "REP-001"
    assert parsed["total_events_recorded"] == 1
    assert len(parsed["incidents"]) == 1

    # Roundtrip to DSSADReport
    reconstructed = DSSADReport.model_validate(parsed)
    assert reconstructed.report_id == "REP-001"


def test_export_telemetry_csv() -> None:
    incident = _make_dummy_incident()
    csv_str = DSSADExporter.export_telemetry_csv(incident.pre_trigger_telemetry)

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["speed_mps"] == "4.500"
    assert row["longitudinal_accel_mps2"] == "-15.200"
    assert row["aeb_active"] == "1"
    assert "C1201" in row["active_dtcs"]


def test_export_uds_did_frame() -> None:
    incident = _make_dummy_incident()
    uds_frame = DSSADExporter.export_uds_did_frame(incident, did=0x19ED)

    assert uds_frame["service_id"] == "0x62"
    assert uds_frame["did"] == "0x19ED"
    assert uds_frame["incident_id"] == incident.incident_id
    assert uds_frame["is_locked"] is True

    raw_hex = uds_frame["raw_hex"]
    raw_bytes = bytes.fromhex(raw_hex)

    # Positive response byte
    assert raw_bytes[0] == 0x62
    # DID bytes
    assert raw_bytes[1] == 0x19
    assert raw_bytes[2] == 0xED

    # Delta-V (5.4 m/s * 10 = 54 = 0x0036)
    delta_v_unpacked = int.from_bytes(raw_bytes[3:5], byteorder="big") / 10.0
    assert abs(delta_v_unpacked - 5.4) < 0.1

    # Peak G (3.25 g * 100 = 325 = 0x0145)
    peak_g_unpacked = int.from_bytes(raw_bytes[5:7], byteorder="big") / 100.0
    assert abs(peak_g_unpacked - 3.25) < 0.02
