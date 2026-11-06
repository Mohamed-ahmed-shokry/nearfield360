"""Unit tests for tactical CSMS / IDS visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    NodeQuarantineRecord,
    SecurityEvent,
    ThreatAssessment,
    VSOCReport,
)
from nearfield360.csms.viz import render_csms_dashboard


def test_render_csms_dashboard_nominal(tmp_path: Path) -> None:
    assessment = ThreatAssessment(
        total_frames_evaluated=2500,
        anomalous_frames_detected=0,
        active_quarantines_count=0,
        mean_anomaly_score=0.0,
        max_severity=IncidentSeverity.INFO,
        mitigation_active=True,
    )
    report = VSOCReport(
        report_id="VSOC-NOMINAL-01",
        threat_assessment=assessment,
        events=[],
        quarantined_entities=[],
    )

    out_png = tmp_path / "csms_nominal.png"
    canvas = render_csms_dashboard(report, output_path=out_png)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000


def test_render_csms_dashboard_active_threats(tmp_path: Path) -> None:
    events = [
        SecurityEvent(
            event_id="EVT-001",
            timestamp_s=10.0,
            category=CyberThreatCategory.CAN_DOS_FLOOD,
            severity=IncidentSeverity.CRITICAL,
            source_node="arb_0x120",
            arbitration_id=0x120,
            anomaly_score=0.95,
            defense_action=DefenseAction.REQUEST_SAFE_STOP,
            message="CAN DoS flood detected on powertrain bus",
        ),
        SecurityEvent(
            event_id="EVT-002",
            timestamp_s=11.2,
            category=CyberThreatCategory.REPLAY_ATTACK,
            severity=IncidentSeverity.MEDIUM,
            source_node="arb_0x150",
            arbitration_id=0x150,
            anomaly_score=0.75,
            defense_action=DefenseAction.DROP_FRAME,
            message="Replay attack detected on radar telemetry",
        ),
        SecurityEvent(
            event_id="EVT-003",
            timestamp_s=12.5,
            category=CyberThreatCategory.UDS_UNAUTHORIZED_DIAG,
            severity=IncidentSeverity.HIGH,
            source_node="diag_0x7DF",
            arbitration_id=0x7DF,
            anomaly_score=0.88,
            defense_action=DefenseAction.LOCKOUT_DIAGNOSTICS,
            message="Unauthorized diagnostic access attempt",
        ),
    ]

    quarantines = [
        NodeQuarantineRecord(
            entity_id="0x120",
            quarantined_at_s=10.0,
            expires_at_s=25.0,
            reason="CAN DoS flood on powertrain",
        )
    ]

    assessment = ThreatAssessment(
        total_frames_evaluated=5000,
        anomalous_frames_detected=3,
        active_quarantines_count=1,
        mean_anomaly_score=0.86,
        max_severity=IncidentSeverity.CRITICAL,
        mitigation_active=True,
    )
    report = VSOCReport(
        report_id="VSOC-THREAT-01",
        threat_assessment=assessment,
        events=events,
        quarantined_entities=quarantines,
    )

    out_png = tmp_path / "csms_threats.png"
    canvas = render_csms_dashboard(report, output_path=out_png)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000
