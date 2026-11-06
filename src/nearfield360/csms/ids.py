"""In-Vehicle Intrusion Detection System (IDS) for UN R155 / ISO/SAE 21434."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from nearfield360.can.models import CANFrame
from nearfield360.config import CSMSConfig
from nearfield360.csms.models import (
    CyberThreatCategory,
    DefenseAction,
    IncidentSeverity,
    SecurityEvent,
)


def compute_shannon_entropy(payload: bytes) -> float:
    """Calculate the Shannon entropy in bits per byte for payload bytes.

    Formula: H(X) = -sum(p(x) * log2(p(x))) for all unique bytes in payload.
    """
    if not payload:
        return 0.0

    counts: dict[int, int] = defaultdict(int)
    for b in payload:
        counts[b] += 1

    length = len(payload)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


class IntrusionDetectionSystem:
    """In-Vehicle Intrusion Detection System monitoring CAN frames and diagnostic telemetry."""

    def __init__(self, config: CSMSConfig | None = None) -> None:
        self.config = config or CSMSConfig()
        # Sliding history of frame arrival timestamps per arbitration ID
        self._frame_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=2000))
        # Last observed payload and timestamp per arbitration ID for replay detection
        self._last_payload: dict[int, tuple[bytes, float]] = {}
        # Counter for unique event IDs
        self._event_counter = 0

    def reset(self) -> None:
        """Clear all sliding history and state counters."""
        self._frame_history.clear()
        self._last_payload.clear()
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"SEC-EVT-{self._event_counter:06d}"

    def inspect_frame(
        self,
        frame: CANFrame | dict[str, Any],
        current_time_s: float | None = None,
        vehicle_speed_mps: float = 0.0,
    ) -> SecurityEvent | None:
        """Evaluate a CAN frame for rate floods, replay, payload fuzzing, and UDS anomalies.

        Returns a SecurityEvent if an anomaly or intrusion is detected, or None if nominal.
        """
        if isinstance(frame, CANFrame):
            arb_id = frame.arbitration_id
            payload = frame.payload
            t_s = current_time_s if current_time_s is not None else frame.timestamp_us / 1_000_000.0
        else:
            arb_id = int(frame["arbitration_id"])
            raw_payload = frame.get("payload", b"")
            payload = (
                raw_payload if isinstance(raw_payload, bytes) else bytes.fromhex(str(raw_payload))
            )
            t_s = (
                current_time_s
                if current_time_s is not None
                else float(frame.get("timestamp_s", 0.0))
            )

        # 1. DoS / Message Rate Flood Inspection
        history = self._frame_history[arb_id]
        history.append(t_s)

        # Prune timestamps older than 1.0 second from current time
        window_start = t_s - 1.0
        while history and history[0] < window_start:
            history.popleft()

        rate_in_window = float(len(history))
        if rate_in_window > self.config.max_can_frame_rate_per_sec:
            score = min(1.0, rate_in_window / (self.config.max_can_frame_rate_per_sec * 1.5))
            severity = (
                IncidentSeverity.CRITICAL
                if arb_id in (0x120, 0x130, 0x010)
                else IncidentSeverity.HIGH
            )
            defense = (
                DefenseAction.REQUEST_SAFE_STOP
                if severity == IncidentSeverity.CRITICAL
                else DefenseAction.DROP_FRAME
            )
            return SecurityEvent(
                event_id=self._next_event_id(),
                timestamp_s=t_s,
                category=CyberThreatCategory.CAN_DOS_FLOOD,
                severity=severity,
                source_node=f"arb_0x{arb_id:03X}",
                arbitration_id=arb_id,
                anomaly_score=score,
                defense_action=defense,
                message=(
                    f"CAN DoS flood on 0x{arb_id:03X}: {rate_in_window:.1f} fps "
                    f"exceeds limit of {self.config.max_can_frame_rate_per_sec:.1f} fps"
                ),
                raw_payload_hex=payload.hex(),
            )

        # 2. Replay Attack Inspection
        if arb_id in self._last_payload:
            last_p, last_t = self._last_payload[arb_id]
            dt_s = t_s - last_t
            tolerance_s = self.config.replay_window_tolerance_ms / 1000.0

            # Retrograde or zero timestamp with identical payload indicates replayed traffic
            if dt_s <= 0.0 or (dt_s < tolerance_s and payload == last_p and len(payload) > 0):
                score = 0.82
                return SecurityEvent(
                    event_id=self._next_event_id(),
                    timestamp_s=t_s,
                    category=CyberThreatCategory.REPLAY_ATTACK,
                    severity=IncidentSeverity.MEDIUM,
                    source_node=f"arb_0x{arb_id:03X}",
                    arbitration_id=arb_id,
                    anomaly_score=score,
                    defense_action=DefenseAction.DROP_FRAME,
                    message=(
                        f"Replay attack detected on 0x{arb_id:03X}: "
                        f"identical payload retransmitted within {dt_s * 1000.0:.2f} ms"
                    ),
                    raw_payload_hex=payload.hex(),
                )

        self._last_payload[arb_id] = (payload, t_s)

        # 3. UDS Diagnostic Access Violation (ISO 14229)
        # Diagnostic requests typically use 0x7DF (broadcast) or 0x7E0-0x7E8 (physical ECUs)
        if 0x7DF <= arb_id <= 0x7E8 and len(payload) >= 2:
            service_id = payload[1] if payload[0] <= 7 else payload[0]
            # Service 0x27: SecurityAccess, Service 0x2E: WriteDataByIdentifier
            if service_id == 0x27 and vehicle_speed_mps > 0.05:
                return SecurityEvent(
                    event_id=self._next_event_id(),
                    timestamp_s=t_s,
                    category=CyberThreatCategory.UDS_UNAUTHORIZED_DIAG,
                    severity=IncidentSeverity.CRITICAL,
                    source_node=f"diag_0x{arb_id:03X}",
                    arbitration_id=arb_id,
                    anomaly_score=0.95,
                    defense_action=DefenseAction.LOCKOUT_DIAGNOSTICS,
                    message=(
                        f"Unauthorized UDS SecurityAccess (0x27) requested on 0x{arb_id:03X} "
                        f"while vehicle in motion ({vehicle_speed_mps:.2f} m/s)"
                    ),
                    raw_payload_hex=payload.hex(),
                )

        # 4. Shannon Payload Entropy / Fuzzing Detection
        if len(payload) >= 4:
            entropy = compute_shannon_entropy(payload)
            if entropy > self.config.payload_entropy_max:
                return SecurityEvent(
                    event_id=self._next_event_id(),
                    timestamp_s=t_s,
                    category=CyberThreatCategory.ANOMALOUS_PAYLOAD_FUZZ,
                    severity=IncidentSeverity.HIGH,
                    source_node=f"arb_0x{arb_id:03X}",
                    arbitration_id=arb_id,
                    anomaly_score=0.78,
                    defense_action=DefenseAction.DROP_FRAME,
                    message=(
                        f"Payload fuzzing anomaly on 0x{arb_id:03X}: "
                        f"entropy {entropy:.2f} exceeds limit {self.config.payload_entropy_max:.2f}"
                    ),
                    raw_payload_hex=payload.hex(),
                )

        return None
