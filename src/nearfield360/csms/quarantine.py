"""Defensive containment, entity quarantining, and safety supervisor interlock."""

from __future__ import annotations

import logging
from typing import Any

from nearfield360.config import CSMSConfig
from nearfield360.csms.models import (
    DefenseAction,
    IncidentSeverity,
    NodeQuarantineRecord,
    SecurityEvent,
)

logger = logging.getLogger(__name__)


class QuarantineFilter:
    """Manages active network blacklists, frame filtering, and safety escalation."""

    def __init__(
        self,
        config: CSMSConfig | None = None,
        safety_supervisor: Any | None = None,
    ) -> None:
        self.config = config or CSMSConfig()
        self.safety_supervisor = safety_supervisor
        # Map of entity_id -> NodeQuarantineRecord
        self._quarantines: dict[str, NodeQuarantineRecord] = {}

    def quarantine_entity(
        self,
        entity_id: str,
        reason: str,
        duration_s: float | None = None,
        current_time_s: float = 0.0,
    ) -> NodeQuarantineRecord:
        """Place an arbitration ID or ECU node into active quarantine blacklist."""
        dur = duration_s if duration_s is not None else self.config.quarantine_duration_s
        expires_at = current_time_s + dur
        record = NodeQuarantineRecord(
            entity_id=entity_id,
            quarantined_at_s=current_time_s,
            expires_at_s=expires_at,
            reason=reason,
            is_active=True,
        )
        self._quarantines[entity_id] = record
        logger.warning(
            "CSMS Quarantine engaged for entity '%s' until t=%.2fs (Reason: %s)",
            entity_id,
            expires_at,
            reason,
        )
        return record

    def release_quarantine(self, entity_id: str) -> bool:
        """Manually release an entity from active quarantine."""
        if entity_id in self._quarantines:
            del self._quarantines[entity_id]
            logger.info("CSMS Quarantine manually released for entity '%s'", entity_id)
            return True
        return False

    def prune_expired(self, current_time_s: float) -> list[str]:
        """Remove and return entity IDs whose quarantine duration has elapsed."""
        expired = [
            eid
            for eid, record in self._quarantines.items()
            if current_time_s >= record.expires_at_s
        ]
        for eid in expired:
            del self._quarantines[eid]
            logger.info("CSMS Quarantine expired and lifted for entity '%s'", eid)
        return expired

    def is_quarantined(self, entity_id: str, current_time_s: float) -> bool:
        """Check if an entity is currently quarantined, pruning if expired."""
        if entity_id not in self._quarantines:
            return False

        record = self._quarantines[entity_id]
        if current_time_s >= record.expires_at_s:
            del self._quarantines[entity_id]
            return False

        return True

    def should_filter_frame(
        self,
        arbitration_id: int,
        source_node: str | None = None,
        current_time_s: float = 0.0,
    ) -> bool:
        """Determine whether inbound frame must be blocked due to an active quarantine."""
        if not self.config.enable_defensive_mitigation:
            return False

        # Check hex arbitration ID format e.g. "0x120"
        hex_id = f"0x{arbitration_id:03X}"
        if self.is_quarantined(hex_id, current_time_s):
            return True

        # Check raw integer string representation
        if self.is_quarantined(str(arbitration_id), current_time_s):
            return True

        # Check source node identifier if provided
        return bool(source_node and self.is_quarantined(source_node, current_time_s))

    def active_quarantines(self, current_time_s: float) -> list[NodeQuarantineRecord]:
        """Return list of all currently active quarantine records."""
        self.prune_expired(current_time_s)
        return list(self._quarantines.values())

    def handle_security_event(
        self,
        event: SecurityEvent,
        current_time_s: float,
    ) -> DefenseAction:
        """Apply defensive mitigation commanded by a SecurityEvent.

        Quarantines offending entities and signals the ISO 26262 ASIL-D safety supervisor
        if emergency safe stop is requested.
        """
        if not self.config.enable_defensive_mitigation:
            return DefenseAction.MONITOR_ONLY

        target_entity = (
            f"0x{event.arbitration_id:03X}"
            if event.arbitration_id is not None
            else event.source_node
        )

        if event.defense_action == DefenseAction.QUARANTINE_NODE:
            self.quarantine_entity(
                entity_id=target_entity,
                reason=event.message,
                current_time_s=current_time_s,
            )
            return DefenseAction.QUARANTINE_NODE

        if event.defense_action == DefenseAction.REQUEST_SAFE_STOP:
            self.quarantine_entity(
                entity_id=target_entity,
                reason=f"CRITICAL CYBER THREAT: {event.message}",
                current_time_s=current_time_s,
            )
            if self.safety_supervisor is not None and hasattr(
                self.safety_supervisor, "trigger_emergency_stop"
            ):
                self.safety_supervisor.trigger_emergency_stop(
                    reason=f"Cyber intrusion critical: {event.message}",
                    timestamp_s=current_time_s,
                )
            return DefenseAction.REQUEST_SAFE_STOP

        if event.severity in (IncidentSeverity.HIGH, IncidentSeverity.CRITICAL):
            # Auto-quarantine entity on severe uncontained attacks
            self.quarantine_entity(
                entity_id=target_entity,
                reason=event.message,
                current_time_s=current_time_s,
            )

        return event.defense_action
