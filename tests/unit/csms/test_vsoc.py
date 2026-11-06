"""Unit tests for vSOC telemetry manager and RFC 5424 incident logger."""

from __future__ import annotations

from nearfield360.config import CSMSConfig
from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    SecurityEvent,
)
from nearfield360.csms.quarantine import QuarantineFilter
from nearfield360.csms.vsoc import VSOCManager


def test_vsoc_record_and_query_events() -> None:
    vsoc = VSOCManager()

    evt_low = SecurityEvent(
        event_id="EVT-01",
        timestamp_s=10.0,
        category=CyberThreatCategory.CAN_INJECTION,
        severity=IncidentSeverity.LOW,
        source_node="ecu-aux",
        anomaly_score=0.25,
        defense_action=DefenseAction.MONITOR_ONLY,
        message="Low severity anomalous bit",
    )
    evt_crit = SecurityEvent(
        event_id="EVT-02",
        timestamp_s=11.0,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.CRITICAL,
        source_node="ecu-gateway",
        anomaly_score=0.98,
        defense_action=DefenseAction.REQUEST_SAFE_STOP,
        message="Critical rate flood",
    )
    vsoc.record_event(evt_low)
    vsoc.record_event(evt_crit)

    # Query all
    assert len(vsoc.get_events()) == 2

    # Query filtered by min_severity
    high_or_crit = vsoc.get_events(min_severity=IncidentSeverity.HIGH)
    assert len(high_or_crit) == 1
    assert high_or_crit[0].event_id == "EVT-02"

    # Query filtered by category
    dos_events = vsoc.get_events(category="can_dos_flood")
    assert len(dos_events) == 1
    assert dos_events[0].category == CyberThreatCategory.CAN_DOS_FLOOD


def test_vsoc_rfc5424_syslog_formatting() -> None:
    vsoc = VSOCManager()
    evt = SecurityEvent(
        event_id="SEC-EVT-00042",
        timestamp_s=1730000000.0,  # Valid POSIX timestamp
        category=CyberThreatCategory.UDS_UNAUTHORIZED_DIAG,
        severity=IncidentSeverity.CRITICAL,
        source_node="diag_0x7DF",
        anomaly_score=0.95,
        defense_action=DefenseAction.LOCKOUT_DIAGNOSTICS,
        message="Unauthorized diagnostic access attempt",
    )
    syslog = vsoc.format_rfc5424(evt, hostname="vehicle-ap-01")

    # PRI for local0 (16) and Critical (2): 16 * 8 + 2 = 130
    assert syslog.startswith("<130>1 ")
    assert "vehicle-ap-01 NF360-CSMS diag_0x7DF SEC-EVT-00042" in syslog
    assert 'category="uds_unauthorized_diag"' in syslog
    assert 'action="lockout_diagnostics"' in syslog
    assert "Unauthorized diagnostic access attempt" in syslog


def test_vsoc_report_generation() -> None:
    qf = QuarantineFilter()
    qf.quarantine_entity("0x120", reason="DoS flood", current_time_s=10.0)

    vsoc = VSOCManager(quarantine=qf)
    evt = SecurityEvent(
        event_id="EVT-REP-01",
        timestamp_s=10.0,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.HIGH,
        source_node="arb_0x120",
        arbitration_id=0x120,
        anomaly_score=0.80,
        defense_action=DefenseAction.QUARANTINE_NODE,
        message="High rate DoS",
    )
    vsoc.record_event(evt)

    report = vsoc.generate_report(
        report_id="VSOC-REP-FINAL",
        total_frames_evaluated=5000,
        current_time_s=12.0,
    )
    assert report.report_id == "VSOC-REP-FINAL"
    assert report.threat_assessment.total_frames_evaluated == 5000
    assert report.threat_assessment.anomalous_frames_detected == 1
    assert report.threat_assessment.active_quarantines_count == 1
    assert report.threat_assessment.max_severity == IncidentSeverity.HIGH
    assert report.threat_assessment.mean_anomaly_score == 0.80
    assert len(report.quarantined_entities) == 1


def test_vsoc_ring_buffer_capacity() -> None:
    config = CSMSConfig(max_active_security_incidents=10)
    vsoc = VSOCManager(config=config)

    for i in range(15):
        evt = SecurityEvent(
            event_id=f"EVT-{i}",
            timestamp_s=float(i),
            category=CyberThreatCategory.CAN_INJECTION,
            severity=IncidentSeverity.LOW,
            source_node="ecu",
            anomaly_score=0.3,
            defense_action=DefenseAction.MONITOR_ONLY,
            message="Test",
        )
        vsoc.record_event(evt)

    events = vsoc.get_events()
    assert len(events) == 10
    assert events[0].event_id == "EVT-5"
    assert events[-1].event_id == "EVT-14"
