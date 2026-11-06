"""Unit tests for UN R155 CSMS domain data models and schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    NodeQuarantineRecord,
    SecurityEvent,
    ThreatAssessment,
    VSOCReport,
)


def test_cyber_threat_enums() -> None:
    assert CyberThreatCategory.CAN_DOS_FLOOD == "can_dos_flood"
    assert CyberThreatCategory.REPLAY_ATTACK == "replay_attack"
    assert CyberThreatCategory.UDS_UNAUTHORIZED_DIAG == "uds_unauthorized_diag"
    assert CyberThreatCategory.ANOMALOUS_PAYLOAD_FUZZ == "anomalous_payload_fuzz"

    assert IncidentSeverity.CRITICAL == "critical"
    assert IncidentSeverity.HIGH == "high"

    assert DefenseAction.DROP_FRAME == "drop_frame"
    assert DefenseAction.QUARANTINE_NODE == "quarantine_node"
    assert DefenseAction.REQUEST_SAFE_STOP == "request_safe_stop"


def test_security_event_valid() -> None:
    event = SecurityEvent(
        event_id="SEC-2026-0001",
        timestamp_s=12.345,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.HIGH,
        source_node="ecu-actuator-gateway",
        arbitration_id=0x120,
        anomaly_score=0.88,
        defense_action=DefenseAction.QUARANTINE_NODE,
        message="Excessive CAN transmission rate detected (450 fps > 250 fps)",
        raw_payload_hex="0102030405060708",
    )
    assert event.event_id == "SEC-2026-0001"
    assert event.arbitration_id == 0x120
    assert event.anomaly_score == 0.88

    # Immutability
    with pytest.raises(ValidationError):
        event.anomaly_score = 0.50  # type: ignore[misc]


def test_security_event_rejections() -> None:
    # Invalid anomaly score (> 1.0)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        SecurityEvent(
            event_id="SEC-BAD",
            timestamp_s=1.0,
            category=CyberThreatCategory.CAN_INJECTION,
            severity=IncidentSeverity.LOW,
            source_node="ecu-telematics",
            anomaly_score=1.5,
            defense_action=DefenseAction.MONITOR_ONLY,
            message="Test",
        )

    # Negative timestamp
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SecurityEvent(
            event_id="SEC-BAD",
            timestamp_s=-1.0,
            category=CyberThreatCategory.CAN_INJECTION,
            severity=IncidentSeverity.LOW,
            source_node="ecu-telematics",
            anomaly_score=0.5,
            defense_action=DefenseAction.MONITOR_ONLY,
            message="Test",
        )


def test_quarantine_record() -> None:
    rec = NodeQuarantineRecord(
        entity_id="0x120",
        quarantined_at_s=100.0,
        expires_at_s=115.0,
        reason="Repeated CAN DoS flood",
    )
    assert rec.entity_id == "0x120"
    assert rec.is_active is True
    assert rec.expires_at_s - rec.quarantined_at_s == 15.0


def test_vsoc_report_serialization() -> None:
    assessment = ThreatAssessment(
        total_frames_evaluated=1000,
        anomalous_frames_detected=5,
        active_quarantines_count=1,
        mean_anomaly_score=0.12,
        max_severity=IncidentSeverity.HIGH,
        mitigation_active=True,
    )
    event = SecurityEvent(
        event_id="SEC-SAMPLE-01",
        timestamp_s=50.0,
        category=CyberThreatCategory.REPLAY_ATTACK,
        severity=IncidentSeverity.MEDIUM,
        source_node="ecu-gateway",
        arbitration_id=0x140,
        anomaly_score=0.72,
        defense_action=DefenseAction.DROP_FRAME,
        message="Stale identical payload replayed",
    )
    report = VSOCReport(
        report_id="VSOC-REP-001",
        threat_assessment=assessment,
        events=[event],
        quarantined_entities=[
            NodeQuarantineRecord(
                entity_id="0x140",
                quarantined_at_s=50.0,
                expires_at_s=65.0,
                reason="Replay",
            )
        ],
    )
    data = report.model_dump(mode="json")
    assert data["report_id"] == "VSOC-REP-001"
    assert data["standard"] == "UN R155 / ISO/SAE 21434"
    assert len(data["events"]) == 1

    roundtrip = VSOCReport.model_validate(data)
    assert roundtrip.report_id == report.report_id
    assert roundtrip.threat_assessment.max_severity == IncidentSeverity.HIGH
