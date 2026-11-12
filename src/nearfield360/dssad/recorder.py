"""Crash freeze manager, incident recorder, and multi-source trigger interlock engine."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

from nearfield360.config import DSSADConfig
from nearfield360.dssad.buffer import TelemetryRingBuffer
from nearfield360.dssad.crypto import HashChainer
from nearfield360.dssad.models import (
    ChainVerificationResult,
    DSSADEventRecord,
    DSSADEventType,
    DSSADReport,
    EDRIncidentRecord,
    EventSeverity,
    TelemetrySnapshot,
    event_severity_rank,
)

GRAVITY_MPS2 = 9.80665


class DSSADRecorder:
    """Automotive-grade UN R157 DSSAD & UN R160 EDR recorder and crash freeze supervisor."""

    def __init__(self, config: DSSADConfig | None = None) -> None:
        self.config = config or DSSADConfig()
        buffer_capacity = int(
            self.config.sampling_rate_hz
            * (self.config.pre_trigger_duration_s + self.config.post_trigger_duration_s + 5.0)
        )
        self.ring_buffer = TelemetryRingBuffer(capacity=max(buffer_capacity, 500))
        self.chainer = HashChainer(signing_key_salt=self.config.signing_key_salt)
        self._events: list[DSSADEventRecord] = []
        self._incidents: list[EDRIncidentRecord] = []
        self._last_frozen_trigger_ns: int = -1

    @property
    def total_events(self) -> int:
        """Total number of logged discrete DSSAD events."""
        return len(self._events)

    @property
    def total_incidents(self) -> int:
        """Total number of locked EDR crash/safety incidents."""
        return len(self._incidents)

    def record_telemetry(self, snapshot: TelemetrySnapshot) -> DSSADEventRecord | None:
        """Ingest a continuous high-frequency telemetry sample and evaluate triggers."""
        self.ring_buffer.append(snapshot)

        # Automatic impact detection via accelerometer thresholding (UN R160)
        long_g = abs(snapshot.longitudinal_accel_mps2) / GRAVITY_MPS2
        lat_g = abs(snapshot.lateral_accel_mps2) / GRAVITY_MPS2
        total_g = math.hypot(long_g, lat_g)

        if total_g >= self.config.impact_accel_threshold_g and (
            self._last_frozen_trigger_ns < 0
            or snapshot.timestamp_ns - self._last_frozen_trigger_ns > int(1.0 * 1e9)
        ):
            return self.log_event(
                event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
                severity=EventSeverity.FATAL,
                trigger_subsystem="chassis_imu",
                description=(
                    f"Collision impact detected: {total_g:.2f}g exceeded threshold "
                    f"{self.config.impact_accel_threshold_g:.2f}g."
                ),
                snapshot=snapshot,
                freeze_incident=True,
            )

        # Automatic MRM emergency braking interlock trigger
        if (
            self.config.auto_freeze_on_mrm
            and snapshot.odd_state == "ODD_BREACH_MRM"
            and snapshot.aeb_active
            and (
                self._last_frozen_trigger_ns < 0
                or snapshot.timestamp_ns - self._last_frozen_trigger_ns > int(2.0 * 1e9)
            )
        ):
            return self.log_event(
                event_type=DSSADEventType.MINIMUM_RISK_MANEUVER_INITIATED,
                severity=EventSeverity.CRITICAL,
                trigger_subsystem="odd_supervisor",
                description="Autonomous ASIL-D MRM emergency stop initiated upon ODD breach.",
                snapshot=snapshot,
                freeze_incident=True,
            )

        return None

    def log_event(
        self,
        event_type: DSSADEventType,
        severity: EventSeverity,
        trigger_subsystem: str,
        description: str,
        snapshot: TelemetrySnapshot | None = None,
        active_dtcs: list[str] | None = None,
        freeze_incident: bool = False,
    ) -> DSSADEventRecord:
        """Record a discrete UN R157 automated driving event with cryptographic hash chaining."""
        now_dt = datetime.now(UTC)
        now_iso = now_dt.isoformat()
        now_ns = int(now_dt.timestamp() * 1e9)

        if snapshot is None:
            latest = self.ring_buffer.latest()
            if latest is not None:
                snapshot = latest
            else:
                snapshot = TelemetrySnapshot(
                    timestamp_ns=now_ns,
                    timestamp_iso=now_iso,
                    speed_mps=0.0,
                    longitudinal_accel_mps2=0.0,
                    lateral_accel_mps2=0.0,
                    yaw_rate_rads=0.0,
                    steering_angle_rad=0.0,
                    brake_pressure_bar=0.0,
                    throttle_pct=0.0,
                )

        seq_id = len(self._events)
        record_id = f"EVT-{now_dt.strftime('%Y%m%d')}-{seq_id:05d}"
        dtcs = active_dtcs or list(snapshot.active_dtcs)

        block_hash = ""
        if self.config.enable_tamper_proof_chain:
            payload = {
                "record_id": record_id,
                "event_type": event_type.value,
                "severity": severity.value,
                "trigger_subsystem": trigger_subsystem,
                "description": description,
                "snapshot": snapshot.model_dump(),
                "active_dtcs": dtcs,
            }
            block = self.chainer.create_block(payload=payload, timestamp_ns=snapshot.timestamp_ns)
            block_hash = block.block_hash

        event = DSSADEventRecord(
            record_id=record_id,
            sequence_id=seq_id,
            timestamp_ns=snapshot.timestamp_ns,
            timestamp_iso=snapshot.timestamp_iso,
            event_type=event_type,
            severity=severity,
            trigger_subsystem=trigger_subsystem,
            description=description,
            snapshot=snapshot,
            active_dtcs=dtcs,
            block_hash=block_hash,
        )
        self._events.append(event)

        should_freeze = (
            freeze_incident
            or severity in (EventSeverity.CRITICAL, EventSeverity.FATAL)
            or event_type
            in (
                DSSADEventType.COLLISION_IMPACT_DETECTED,
                DSSADEventType.MINIMUM_RISK_MANEUVER_INITIATED,
            )
        )
        if should_freeze:
            self._freeze_incident(trigger_event=event)

        return event

    def _freeze_incident(self, trigger_event: DSSADEventRecord) -> EDRIncidentRecord:
        """Lock pre/post crash telemetry into an immutable UN R160 forensic incident record."""
        self._last_frozen_trigger_ns = trigger_event.timestamp_ns

        pre_telemetry, post_telemetry = self.ring_buffer.get_pre_post_window(
            trigger_ns=trigger_event.timestamp_ns,
            pre_duration_s=self.config.pre_trigger_duration_s,
            post_duration_s=self.config.post_trigger_duration_s,
        )

        all_samples = pre_telemetry + post_telemetry
        if all_samples:
            speeds = [s.speed_mps for s in all_samples]
            delta_v = max(speeds) - min(speeds)
            accels = [
                math.hypot(
                    abs(s.longitudinal_accel_mps2) / GRAVITY_MPS2,
                    abs(s.lateral_accel_mps2) / GRAVITY_MPS2,
                )
                for s in all_samples
            ]
            peak_g = max(accels)
        else:
            delta_v = 0.0
            peak_g = math.hypot(
                abs(trigger_event.snapshot.longitudinal_accel_mps2) / GRAVITY_MPS2,
                abs(trigger_event.snapshot.lateral_accel_mps2) / GRAVITY_MPS2,
            )

        now_dt = datetime.now(UTC)
        incident_id = f"INC-{now_dt.strftime('%Y%m%d')}-{len(self._incidents) + 1:04d}"

        incident = EDRIncidentRecord(
            incident_id=incident_id,
            freeze_timestamp_iso=now_dt.isoformat(),
            trigger_event=trigger_event,
            pre_trigger_telemetry=pre_telemetry,
            post_trigger_telemetry=post_telemetry,
            delta_v_mps=float(delta_v),
            peak_accel_g=float(peak_g),
            tamper_chain_digest=self.chainer.latest_digest(),
            is_locked=True,
        )

        # Enforce maximum retained incidents FIFO retention
        if len(self._incidents) >= self.config.max_retained_incidents:
            self._incidents.pop(0)

        self._incidents.append(incident)
        return incident

    def query_events(
        self,
        event_type: DSSADEventType | None = None,
        min_severity: EventSeverity | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[DSSADEventRecord]:
        """Query stored events by classification, severity threshold, or time window."""
        results = list(self._events)
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if min_severity is not None:
            min_rank = event_severity_rank(min_severity)
            results = [e for e in results if event_severity_rank(e.severity) >= min_rank]
        if start_ns is not None:
            results = [e for e in results if e.timestamp_ns >= start_ns]
        if end_ns is not None:
            results = [e for e in results if e.timestamp_ns <= end_ns]
        return results

    def query_incidents(self, incident_id: str | None = None) -> list[EDRIncidentRecord]:
        """Query locked incident records by incident ID."""
        if incident_id is None:
            return list(self._incidents)
        return [inc for inc in self._incidents if inc.incident_id == incident_id]

    def verify_integrity(self) -> ChainVerificationResult:
        """Verify unbroken cryptographic hash chain continuity across all records."""
        return self.chainer.verify_chain()

    def get_report(self, vehicle_id: str = "AVP-TEST-01") -> DSSADReport:
        """Generate comprehensive forensic package compliant with UN R157 & UN R160."""
        verification = self.verify_integrity()
        now_iso = datetime.now(UTC).isoformat()
        return DSSADReport(
            report_id=f"REP-{uuid.uuid4().hex[:12].upper()}",
            vehicle_id=vehicle_id,
            generated_at=now_iso,
            total_events_recorded=len(self._events),
            total_incidents_frozen=len(self._incidents),
            chain_verified=verification.valid,
            events=list(self._events),
            incidents=list(self._incidents),
        )
