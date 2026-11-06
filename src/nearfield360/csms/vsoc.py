"""Vehicle Security Operations Center (vSOC) telemetry manager and incident reporter."""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime

from nearfield360.config import CSMSConfig
from nearfield360.csms.ids import IntrusionDetectionSystem
from nearfield360.csms.models import (
    IncidentSeverity,
    SecurityEvent,
    ThreatAssessment,
    VSOCReport,
)
from nearfield360.csms.quarantine import QuarantineFilter

logger = logging.getLogger(__name__)

SEVERITY_ORDER: dict[IncidentSeverity, int] = {
    IncidentSeverity.INFO: 0,
    IncidentSeverity.LOW: 1,
    IncidentSeverity.MEDIUM: 2,
    IncidentSeverity.HIGH: 3,
    IncidentSeverity.CRITICAL: 4,
}

RFC5424_SEVERITY: dict[IncidentSeverity, int] = {
    IncidentSeverity.CRITICAL: 2,  # Critical: critical conditions
    IncidentSeverity.HIGH: 3,  # Error: error conditions
    IncidentSeverity.MEDIUM: 4,  # Warning: warning conditions
    IncidentSeverity.LOW: 5,  # Notice: normal but significant condition
    IncidentSeverity.INFO: 6,  # Informational: informational messages
}


class VSOCManager:
    """Manages telemetry aggregation, RFC 5424 formatting, and vSOC compliance reports."""

    def __init__(
        self,
        config: CSMSConfig | None = None,
        ids: IntrusionDetectionSystem | None = None,
        quarantine: QuarantineFilter | None = None,
    ) -> None:
        self.config = config or CSMSConfig()
        self.ids = ids or IntrusionDetectionSystem(self.config)
        self.quarantine = quarantine or QuarantineFilter(self.config)
        self._events: deque[SecurityEvent] = deque(maxlen=self.config.max_active_security_incidents)
        self._report_counter = 0

    def record_event(self, event: SecurityEvent) -> None:
        """Record an inbound security event into internal telemetry buffer."""
        self._events.append(event)
        logger.info(
            "vSOC recorded event %s [%s/%s]: %s",
            event.event_id,
            event.category.value,
            event.severity.value,
            event.message,
        )

    def format_rfc5424(self, event: SecurityEvent, hostname: str = "vehicle-ecu-gateway") -> str:
        """Format a SecurityEvent into standard RFC 5424 syslog string.

        Format: <PRI>1 TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [STRUCTURED-DATA] MSG
        """
        facility = 16  # local0 (common for automotive embedded sub-networks)
        sec_code = RFC5424_SEVERITY[event.severity]
        pri = facility * 8 + sec_code

        # Format ISO timestamp
        dt = datetime.fromtimestamp(event.timestamp_s, tz=UTC)
        ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        sd = (
            f'[csms@nearfield360 category="{event.category.value}" '
            f'anomaly="{event.anomaly_score:.2f}" action="{event.defense_action.value}"]'
        )
        msg_id = event.event_id
        app_name = "NF360-CSMS"
        proc_id = event.source_node

        return f"<{pri}>1 {ts_iso} {hostname} {app_name} {proc_id} {msg_id} {sd} {event.message}"

    def get_events(
        self,
        min_severity: IncidentSeverity | None = None,
        category: str | None = None,
    ) -> list[SecurityEvent]:
        """Query stored events with optional severity and category filtering."""
        results: list[SecurityEvent] = []
        min_rank = SEVERITY_ORDER[min_severity] if min_severity else 0

        for evt in self._events:
            if SEVERITY_ORDER[evt.severity] < min_rank:
                continue
            if category and evt.category.value != category:
                continue
            results.append(evt)

        return results

    def generate_report(
        self,
        report_id: str | None = None,
        total_frames_evaluated: int = 0,
        current_time_s: float = 0.0,
    ) -> VSOCReport:
        """Generate a consolidated UN R155 / ISO/SAE 21434 compliance report."""
        self._report_counter += 1
        rep_id = report_id or f"VSOC-REP-{self._report_counter:04d}"

        events_list = list(self._events)
        anomaly_count = len(events_list)
        mean_score = (
            sum(e.anomaly_score for e in events_list) / anomaly_count if anomaly_count > 0 else 0.0
        )

        max_sev = IncidentSeverity.INFO
        for e in events_list:
            if SEVERITY_ORDER[e.severity] > SEVERITY_ORDER[max_sev]:
                max_sev = e.severity

        active_quars = self.quarantine.active_quarantines(current_time_s)

        assessment = ThreatAssessment(
            total_frames_evaluated=max(total_frames_evaluated, anomaly_count),
            anomalous_frames_detected=anomaly_count,
            active_quarantines_count=len(active_quars),
            mean_anomaly_score=round(mean_score, 4),
            max_severity=max_sev,
            mitigation_active=self.config.enable_defensive_mitigation,
        )

        return VSOCReport(
            report_id=rep_id,
            generated_at_iso=datetime.now(UTC).isoformat(),
            standard="UN R155 / ISO/SAE 21434",
            threat_assessment=assessment,
            events=events_list,
            quarantined_entities=active_quars,
        )
