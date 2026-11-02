"""Cyclic hardware heartbeat watchdog and deadline monitoring engine (ISO 26262 ASIL-D)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nearfield360.safety.models import (
    ECUNodeId,
    HeartbeatPing,
    HeartbeatStatus,
    NodeHealthStatus,
)

if TYPE_CHECKING:
    from nearfield360.config import SafetyConfig

# Default nominal periods (ms) for automotive ECUs
DEFAULT_NODE_PERIODS_MS: dict[ECUNodeId, float] = {
    ECUNodeId.PERCEPTION_SURROUND: 100.0,  # 10 Hz
    ECUNodeId.RADAR_TRANSCEIVER: 50.0,  # 20 Hz
    ECUNodeId.ULTRASONIC_ARRAY: 50.0,  # 20 Hz
    ECUNodeId.TRAJECTORY_PLANNER: 100.0,  # 10 Hz
    ECUNodeId.CAN_GATEWAY: 20.0,  # 50 Hz
    ECUNodeId.VEHICLE_CONTROLLER: 20.0,  # 50 Hz
    ECUNodeId.TELEOP_SUPERVISOR: 250.0,  # 4 Hz
}

CRITICAL_SAFETY_NODES: set[ECUNodeId] = {
    ECUNodeId.VEHICLE_CONTROLLER,
    ECUNodeId.CAN_GATEWAY,
    ECUNodeId.TRAJECTORY_PLANNER,
}


class NodeTracker:
    """Internal state tracker for a single ECU node."""

    def __init__(self, node_id: ECUNodeId, expected_period_ms: float) -> None:
        self.node_id = node_id
        self.expected_period_s = expected_period_ms / 1000.0
        self.last_heartbeat_s = 0.0
        self.last_sequence_num = -1
        self.latency_ms = 0.0
        self.jitter_ms = 0.0
        self.consecutive_misses = 0
        self.status = HeartbeatStatus.HEALTHY
        self.has_received_first = False

    def to_status(self) -> NodeHealthStatus:
        return NodeHealthStatus(
            node_id=self.node_id,
            last_heartbeat_s=round(self.last_heartbeat_s, 4),
            sequence_num=max(0, self.last_sequence_num),
            status=self.status,
            latency_ms=round(self.latency_ms, 2),
            jitter_ms=round(self.jitter_ms, 2),
            consecutive_misses=self.consecutive_misses,
        )


class HeartbeatWatchdog:
    """Deterministic cyclic heartbeat watchdog with microsecond deadline monitoring."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        timeout_ms: float | None = None,
        max_consecutive_misses: int | None = None,
    ) -> None:
        if config is None:
            from nearfield360.config import SafetyConfig

            config = SafetyConfig()

        self.timeout_s: float = (
            (timeout_ms / 1000.0)
            if timeout_ms is not None
            else (config.heartbeat_timeout_ms / 1000.0)
        )
        self.max_misses: int = (
            max_consecutive_misses
            if max_consecutive_misses is not None
            else config.max_consecutive_misses
        )

        self.trackers: dict[ECUNodeId, NodeTracker] = {}
        for node_id, period_ms in DEFAULT_NODE_PERIODS_MS.items():
            self.trackers[node_id] = NodeTracker(node_id, period_ms)

    def register_node(self, node_id: ECUNodeId, expected_period_ms: float = 50.0) -> None:
        """Register an ECU node with an expected periodic broadcast interval."""
        self.trackers[node_id] = NodeTracker(node_id, expected_period_ms)

    def record_heartbeat(self, ping: HeartbeatPing) -> NodeHealthStatus:
        """Process incoming heartbeat ping and update jitter and latency metrics."""
        tracker = self.trackers.get(ping.node_id)
        if tracker is None:
            tracker = NodeTracker(ping.node_id, 50.0)
            self.trackers[ping.node_id] = tracker

        if tracker.has_received_first:
            delta_s = max(0.0, ping.timestamp_s - tracker.last_heartbeat_s)
            tracker.latency_ms = delta_s * 1000.0
            jitter_s = abs(delta_s - tracker.expected_period_s)
            tracker.jitter_ms = jitter_s * 1000.0

            # Jitter warning if deviation exceeds 50% of expected cycle
            if tracker.jitter_ms > (tracker.expected_period_s * 500.0):
                tracker.status = HeartbeatStatus.WARN_JITTER
            else:
                tracker.status = HeartbeatStatus.HEALTHY
        else:
            tracker.has_received_first = True
            tracker.status = HeartbeatStatus.HEALTHY

        tracker.last_heartbeat_s = ping.timestamp_s
        tracker.last_sequence_num = ping.sequence_num
        tracker.consecutive_misses = 0

        return tracker.to_status()

    def evaluate(self, current_time_s: float) -> dict[str, NodeHealthStatus]:
        """Evaluate deadline compliance for all registered nodes at current timestamp."""
        results: dict[str, NodeHealthStatus] = {}

        for node_id, tracker in self.trackers.items():
            if not tracker.has_received_first:
                # Node has not yet initialized
                tracker.status = HeartbeatStatus.MISSED
                results[node_id.value] = tracker.to_status()
                continue

            elapsed_s = max(0.0, current_time_s - tracker.last_heartbeat_s)
            tracker.latency_ms = elapsed_s * 1000.0

            if elapsed_s > self.timeout_s:
                # Deadline missed
                tracker.consecutive_misses += 1
                if tracker.consecutive_misses >= self.max_misses:
                    tracker.status = HeartbeatStatus.EXPIRED
                else:
                    tracker.status = HeartbeatStatus.MISSED
            elif tracker.status == HeartbeatStatus.MISSED:
                tracker.status = HeartbeatStatus.HEALTHY

            results[node_id.value] = tracker.to_status()

        return results

    def check_critical_expiration(
        self,
        statuses: dict[str, NodeHealthStatus],
    ) -> tuple[bool, str]:
        """Check if any critical ECU node has exceeded maximum allowable consecutive misses."""
        for node in CRITICAL_SAFETY_NODES:
            health = statuses.get(node.value)
            if health is not None and health.status == HeartbeatStatus.EXPIRED:
                return True, f"Critical ECU heartbeat expired: {node.value}"
        return False, ""
