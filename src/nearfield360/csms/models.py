"""Domain data models and schemas for UN R155 CSMS and in-vehicle intrusion detection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CyberThreatCategory(StrEnum):
    """Classification of cyber threats per UN R155 and ISO/SAE 21434 Annex 5."""

    CAN_INJECTION = "can_injection"
    CAN_DOS_FLOOD = "can_dos_flood"
    REPLAY_ATTACK = "replay_attack"
    UDS_UNAUTHORIZED_DIAG = "uds_unauthorized_diag"
    ANOMALOUS_PAYLOAD_FUZZ = "anomalous_payload_fuzz"
    ECU_MASQUERADE = "ecu_masquerade"
    SENSOR_SPOOF = "sensor_spoof"


class IncidentSeverity(StrEnum):
    """Incident severity classification for in-vehicle cybersecurity threats."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseAction(StrEnum):
    """Defensive mitigation action commanded by the In-Vehicle IDS and CSMS."""

    MONITOR_ONLY = "monitor_only"
    DROP_FRAME = "drop_frame"
    QUARANTINE_NODE = "quarantine_node"
    REQUEST_SAFE_STOP = "request_safe_stop"
    LOCKOUT_DIAGNOSTICS = "lockout_diagnostics"


class SecurityEvent(BaseModel):
    """Immutable audit record of a detected cybersecurity event or attack attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(..., min_length=2)
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    category: CyberThreatCategory = Field(...)
    severity: IncidentSeverity = Field(...)
    source_node: str = Field(..., min_length=2)
    arbitration_id: int | None = Field(default=None, ge=0, le=0x1FFFFFFF)
    anomaly_score: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    defense_action: DefenseAction = Field(...)
    message: str = Field(..., min_length=1)
    raw_payload_hex: str | None = Field(default=None)


class NodeQuarantineRecord(BaseModel):
    """Record of an active or expired network quarantine on an ECU or arbitration ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(..., min_length=2)
    quarantined_at_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    expires_at_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    reason: str = Field(..., min_length=1)
    is_active: bool = Field(default=True)


class ThreatAssessment(BaseModel):
    """Aggregated threat state and anomaly metrics across the in-vehicle network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_frames_evaluated: int = Field(default=0, ge=0)
    anomalous_frames_detected: int = Field(default=0, ge=0)
    active_quarantines_count: int = Field(default=0, ge=0)
    mean_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    max_severity: IncidentSeverity = Field(default=IncidentSeverity.INFO)
    mitigation_active: bool = Field(default=True)


class VSOCReport(BaseModel):
    """Consolidated UN R155 / ISO/SAE 21434 telemetry report for Vehicle Security Operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(..., min_length=2)
    generated_at_iso: str = Field(default="2026-11-06T00:00:00Z")
    standard: str = Field(default="UN R155 / ISO/SAE 21434")
    threat_assessment: ThreatAssessment = Field(...)
    events: list[SecurityEvent] = Field(default_factory=list)
    quarantined_entities: list[NodeQuarantineRecord] = Field(default_factory=list)
