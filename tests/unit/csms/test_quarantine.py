"""Unit tests for defensive quarantine filter and ASIL-D safety interlock."""

from __future__ import annotations

from nearfield360.config import CSMSConfig
from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    SecurityEvent,
)
from nearfield360.csms.quarantine import QuarantineFilter
from nearfield360.safety.models import SafetyState
from nearfield360.safety.supervisor import SafetySupervisor


def test_quarantine_entity_and_filtering() -> None:
    qf = QuarantineFilter(config=CSMSConfig(quarantine_duration_s=10.0))
    record = qf.quarantine_entity("0x120", reason="Rate flood", current_time_s=5.0)

    assert record.entity_id == "0x120"
    assert record.expires_at_s == 15.0
    assert qf.is_quarantined("0x120", current_time_s=6.0) is True

    # Check frame filtering
    assert qf.should_filter_frame(0x120, current_time_s=6.0) is True
    assert qf.should_filter_frame(0x130, current_time_s=6.0) is False


def test_quarantine_expiration_and_pruning() -> None:
    qf = QuarantineFilter(config=CSMSConfig(quarantine_duration_s=5.0))
    qf.quarantine_entity("0x200", reason="Replay", current_time_s=10.0)

    assert qf.is_quarantined("0x200", current_time_s=12.0) is True
    assert qf.is_quarantined("0x200", current_time_s=15.5) is False

    # Second entity
    qf.quarantine_entity("0x300", reason="Fuzzing", current_time_s=10.0)
    expired = qf.prune_expired(current_time_s=16.0)
    assert "0x300" in expired
    assert len(qf.active_quarantines(current_time_s=16.0)) == 0


def test_manual_release_quarantine() -> None:
    qf = QuarantineFilter()
    qf.quarantine_entity("ecu-telematics", reason="Test", current_time_s=0.0)
    assert qf.is_quarantined("ecu-telematics", current_time_s=1.0) is True

    released = qf.release_quarantine("ecu-telematics")
    assert released is True
    assert qf.is_quarantined("ecu-telematics", current_time_s=1.0) is False
    assert qf.release_quarantine("non-existent") is False


def test_handle_security_event_quarantine_action() -> None:
    qf = QuarantineFilter()
    evt = SecurityEvent(
        event_id="SEC-001",
        timestamp_s=20.0,
        category=CyberThreatCategory.ANOMALOUS_PAYLOAD_FUZZ,
        severity=IncidentSeverity.HIGH,
        source_node="ecu-sensor-fusion",
        arbitration_id=0x180,
        anomaly_score=0.85,
        defense_action=DefenseAction.QUARANTINE_NODE,
        message="Fuzzed sensor CAN frame",
    )
    action = qf.handle_security_event(evt, current_time_s=20.0)
    assert action == DefenseAction.QUARANTINE_NODE
    assert qf.is_quarantined("0x180", current_time_s=21.0) is True


def test_handle_security_event_safety_escalation() -> None:
    supervisor = SafetySupervisor()
    # Transition supervisor to NORMAL_OPERATION
    supervisor.transition_to(
        target_state=SafetyState.NORMAL_OPERATION,
        reason="Self test passed",
        timestamp_s=1.0,
    )
    assert supervisor.current_state == SafetyState.NORMAL_OPERATION

    qf = QuarantineFilter(safety_supervisor=supervisor)
    evt = SecurityEvent(
        event_id="SEC-CRITICAL-001",
        timestamp_s=25.0,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.CRITICAL,
        source_node="ecu-actuator-can",
        arbitration_id=0x120,
        anomaly_score=0.99,
        defense_action=DefenseAction.REQUEST_SAFE_STOP,
        message="Safety-critical bus under heavy DoS flood",
    )
    action = qf.handle_security_event(evt, current_time_s=25.0)
    assert action == DefenseAction.REQUEST_SAFE_STOP
    assert qf.is_quarantined("0x120", current_time_s=25.0) is True
    # Verify supervisor state escalated to EMERGENCY_SAFE_STOP
    assert supervisor.current_state == SafetyState.EMERGENCY_SAFE_STOP


def test_defensive_mitigation_disabled() -> None:
    config = CSMSConfig(enable_defensive_mitigation=False)
    qf = QuarantineFilter(config=config)
    qf.quarantine_entity("0x120", reason="Test", current_time_s=0.0)

    # Frame filtering disabled
    assert qf.should_filter_frame(0x120, current_time_s=1.0) is False

    evt = SecurityEvent(
        event_id="SEC-002",
        timestamp_s=5.0,
        category=CyberThreatCategory.CAN_DOS_FLOOD,
        severity=IncidentSeverity.CRITICAL,
        source_node="ecu",
        arbitration_id=0x120,
        anomaly_score=0.9,
        defense_action=DefenseAction.REQUEST_SAFE_STOP,
        message="Flood",
    )
    action = qf.handle_security_event(evt, current_time_s=5.0)
    assert action == DefenseAction.MONITOR_ONLY
