"""Domain models and schemas for UN R157 DSSAD & UN R160 EDR Incident Recording."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DSSADEventType(StrEnum):
    """UN R157 & UN R160 standardized automated driving and incident event taxonomy."""

    SYSTEM_ACTIVATION = "SYSTEM_ACTIVATION"
    SYSTEM_DEACTIVATION = "SYSTEM_DEACTIVATION"
    DRIVER_TAKEOVER_REQUEST = "DRIVER_TAKEOVER_REQUEST"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    MINIMUM_RISK_MANEUVER_INITIATED = "MINIMUM_RISK_MANEUVER_INITIATED"
    EMERGENCY_BRAKING_TRIGGERED = "EMERGENCY_BRAKING_TRIGGERED"
    ODD_BOUNDARY_BREACH = "ODD_BOUNDARY_BREACH"
    SENSOR_FAILURE_OCCLUSION = "SENSOR_FAILURE_OCCLUSION"
    COLLISION_IMPACT_DETECTED = "COLLISION_IMPACT_DETECTED"
    SECURITY_INTRUSION_ALERT = "SECURITY_INTRUSION_ALERT"
    ACOUSTIC_SIREN_YIELD = "ACOUSTIC_SIREN_YIELD"
    WATCHDOG_HEARTBEAT_TIMEOUT = "WATCHDOG_HEARTBEAT_TIMEOUT"


class EventSeverity(StrEnum):
    """Regulatory severity classification of logged events."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


def event_severity_rank(severity: EventSeverity) -> int:
    """Return numeric severity ranking for monotonic filtering and prioritization."""
    ranks = {
        EventSeverity.INFO: 1,
        EventSeverity.WARNING: 2,
        EventSeverity.CRITICAL: 3,
        EventSeverity.FATAL: 4,
    }
    return ranks[severity]


class TelemetrySnapshot(BaseModel):
    """High-frequency synchronized vehicle kinematic and state sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_ns: int = Field(ge=0, description="Monotonic timestamp in nanoseconds.")
    timestamp_iso: str = Field(description="ISO 8601 UTC timestamp string.")
    speed_mps: float = Field(ge=0.0, allow_inf_nan=False, description="Vehicle ground speed.")
    longitudinal_accel_mps2: float = Field(
        allow_inf_nan=False, description="Longitudinal acceleration (+forward, -braking)."
    )
    lateral_accel_mps2: float = Field(
        allow_inf_nan=False, description="Lateral acceleration (+left, -right)."
    )
    yaw_rate_rads: float = Field(allow_inf_nan=False, description="Yaw rate in radians per second.")
    steering_angle_rad: float = Field(allow_inf_nan=False, description="Road wheel steering angle.")
    brake_pressure_bar: float = Field(
        ge=0.0, allow_inf_nan=False, description="Hydraulic brake circuit pressure."
    )
    throttle_pct: float = Field(
        ge=0.0, le=100.0, allow_inf_nan=False, description="Throttle actuator command percentage."
    )
    gear: str = Field(default="D", description="Transmission gear selection (P, R, N, D).")
    x_m: float = Field(default=0.0, allow_inf_nan=False, description="UTM / Local coordinate X.")
    y_m: float = Field(default=0.0, allow_inf_nan=False, description="UTM / Local coordinate Y.")
    yaw_rad: float = Field(default=0.0, allow_inf_nan=False, description="Vehicle heading angle.")
    odd_state: str = Field(
        default="NOMINAL", description="Dynamic ODD state at time of telemetry sample."
    )
    aeb_active: bool = Field(default=False, description="Autonomous Emergency Braking flag.")
    active_dtcs: list[str] = Field(
        default_factory=list, description="Active Diagnostic Trouble Codes (DTCs)."
    )


class TamperProofBlock(BaseModel):
    """Cryptographic chained block establishing an immutable forensic audit trail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_id: int = Field(ge=0, description="Strictly monotonic sequence identifier.")
    prev_hash: str = Field(description="SHA-256 hexadecimal digest of previous block.")
    timestamp_ns: int = Field(ge=0, description="Creation timestamp in nanoseconds.")
    payload_digest: str = Field(description="SHA-256 digest of record payload data.")
    block_hash: str = Field(description="SHA-256 composite block digest.")
    hmac_signature: str = Field(description="HMAC-SHA256 signature authenticating block.")


class DSSADEventRecord(BaseModel):
    """UN R157 discrete event record captured in non-volatile storage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(description="Unique record identifier (e.g. EVT-2026-00001).")
    sequence_id: int = Field(ge=0, description="Monotonic sequence number.")
    timestamp_ns: int = Field(ge=0, description="Timestamp in nanoseconds.")
    timestamp_iso: str = Field(description="ISO 8601 UTC timestamp string.")
    event_type: DSSADEventType = Field(description="Classified DSSAD event category.")
    severity: EventSeverity = Field(description="Event severity level.")
    trigger_subsystem: str = Field(description="Subsystem origin of the event trigger.")
    description: str = Field(description="Human and machine readable event detail.")
    snapshot: TelemetrySnapshot = Field(description="Kinematic telemetry at moment of event.")
    active_dtcs: list[str] = Field(
        default_factory=list, description="Diagnostic trouble codes associated with event."
    )
    block_hash: str = Field(default="", description="Cryptographic chained block hash.")


class EDRIncidentRecord(BaseModel):
    """UN R160 frozen crash or safety incident record with synchronized pre/post telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(description="Unique incident identifier (e.g. INC-20261110-001).")
    freeze_timestamp_iso: str = Field(description="ISO 8601 UTC freeze timestamp.")
    trigger_event: DSSADEventRecord = Field(description="DSSAD event triggering the freeze.")
    pre_trigger_telemetry: list[TelemetrySnapshot] = Field(
        description="High-frequency samples preceding trigger (nominal 5.0 s)."
    )
    post_trigger_telemetry: list[TelemetrySnapshot] = Field(
        description="High-frequency samples following trigger (nominal 2.0 s)."
    )
    delta_v_mps: float = Field(
        default=0.0, allow_inf_nan=False, description="Cumulative impact velocity reduction."
    )
    peak_accel_g: float = Field(
        default=0.0, allow_inf_nan=False, description="Peak acceleration magnitude in Gs."
    )
    tamper_chain_digest: str = Field(
        default="", description="Root cryptographic digest certifying incident integrity."
    )
    is_locked: bool = Field(default=True, description="Write-once lock preventing overwrites.")


class ChainVerificationResult(BaseModel):
    """Cryptographic audit trail verification result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool = Field(description="True if cryptographic chain is completely unbroken.")
    total_blocks: int = Field(ge=0, description="Total number of chained blocks inspected.")
    corrupted_index: int | None = Field(
        default=None, description="Index of first corrupted block if validation failed."
    )
    error_message: str | None = Field(
        default=None, description="Descriptive error explanation if chain is invalid."
    )


class DSSADReport(BaseModel):
    """Comprehensive regulatory forensic package containing event logs and incident records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(description="Unique forensic report identifier.")
    vehicle_id: str = Field(default="AVP-TEST-01", description="Vehicle identification number.")
    generated_at: str = Field(description="ISO 8601 timestamp of report creation.")
    total_events_recorded: int = Field(ge=0, description="Total discrete DSSAD events in archive.")
    total_incidents_frozen: int = Field(ge=0, description="Total frozen EDR crash incidents.")
    chain_verified: bool = Field(description="Cryptographic chain validity status.")
    events: list[DSSADEventRecord] = Field(default_factory=list, description="List of events.")
    incidents: list[EDRIncidentRecord] = Field(
        default_factory=list, description="List of frozen incidents."
    )
