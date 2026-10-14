"""Bandwidth-adaptive network monitoring, RTT tracking, and video fallback throttling."""

from __future__ import annotations

from typing import Any

from nearfield360.config import TeleopConfig
from nearfield360.teleop.models import NetworkQuality, TeleopHeartbeat


class NetworkMonitor:
    """Real-time link quality estimator and adaptive streaming controller."""

    def __init__(self, config: TeleopConfig | None = None) -> None:
        self.config = config or TeleopConfig()
        self.last_heartbeat_time_s: float | None = None
        self.last_sequence_number: int | None = None
        self.total_packets_received: int = 0
        self.total_packets_lost: int = 0
        self.ema_rtt_ms: float = 0.0
        self.jitter_ms: float = 0.0
        self._alpha_ema: float = 0.2

    def record_heartbeat(self, heartbeat: TeleopHeartbeat) -> NetworkQuality:
        """Process an incoming heartbeat and return updated network link quality."""
        current_time_s = heartbeat.timestamp_s
        self.total_packets_received += 1

        # Check sequence continuity to count lost packets
        if self.last_sequence_number is not None:
            expected_seq = self.last_sequence_number + 1
            if heartbeat.sequence_number > expected_seq:
                lost = heartbeat.sequence_number - expected_seq
                self.total_packets_lost += lost

        self.last_sequence_number = heartbeat.sequence_number
        self.last_heartbeat_time_s = current_time_s

        # Update smoothed RTT (EMA) and jitter
        if self.total_packets_received == 1:
            self.ema_rtt_ms = heartbeat.rtt_ms
            self.jitter_ms = 0.0
        else:
            diff = abs(heartbeat.rtt_ms - self.ema_rtt_ms)
            self.jitter_ms = (1.0 - self._alpha_ema) * self.jitter_ms + self._alpha_ema * diff
            self.ema_rtt_ms = (
                1.0 - self._alpha_ema
            ) * self.ema_rtt_ms + self._alpha_ema * heartbeat.rtt_ms

        return self.evaluate_quality(current_time_s)

    def evaluate_quality(self, current_time_s: float) -> NetworkQuality:
        """Determine network link classification at the specified timestamp."""
        if self.last_heartbeat_time_s is None:
            return NetworkQuality.SEVERED

        time_since_hb = current_time_s - self.last_heartbeat_time_s
        if time_since_hb > self.config.heartbeat_timeout_s:
            return NetworkQuality.SEVERED

        loss_ratio = self.packet_loss_ratio

        if self.ema_rtt_ms >= self.config.critical_rtt_ms or loss_ratio >= 0.20:
            return NetworkQuality.CRITICAL
        if self.ema_rtt_ms >= self.config.max_rtt_ms or loss_ratio >= 0.05:
            return NetworkQuality.DEGRADED
        return NetworkQuality.OPTIMAL

    @property
    def packet_loss_ratio(self) -> float:
        """Ratio of dropped packets over total expected transmissions."""
        total = self.total_packets_received + self.total_packets_lost
        if total == 0:
            return 0.0
        return self.total_packets_lost / total

    def compute_streaming_recommendation(self, current_time_s: float) -> dict[str, Any]:
        """Compute recommended FPS and scale factor based on network conditions."""
        quality = self.evaluate_quality(current_time_s)

        if quality == NetworkQuality.OPTIMAL:
            fps = self.config.video_target_fps
            scale_factor = 1.0
            direct_driving_permitted = True
        elif quality == NetworkQuality.DEGRADED:
            # Intermediate fallback
            mid_fps = 0.5 * (self.config.video_target_fps + self.config.video_fallback_fps)
            fps = round(mid_fps, 1)
            scale_factor = 0.75
            direct_driving_permitted = False
        elif quality == NetworkQuality.CRITICAL:
            fps = self.config.video_fallback_fps
            scale_factor = 0.5
            direct_driving_permitted = False
        else:  # SEVERED
            fps = 0.0
            scale_factor = 0.25
            direct_driving_permitted = False

        return {
            "network_quality": quality.value,
            "recommended_fps": fps,
            "scale_factor": scale_factor,
            "direct_driving_permitted": direct_driving_permitted,
            "ema_rtt_ms": round(self.ema_rtt_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "packet_loss_ratio": round(self.packet_loss_ratio, 4),
        }
