"""Unit tests for teleoperation network monitoring and adaptive streaming."""

from __future__ import annotations

from nearfield360.config import TeleopConfig
from nearfield360.teleop.models import NetworkQuality, TeleopHeartbeat
from nearfield360.teleop.network import NetworkMonitor


def test_network_monitor_initial_state() -> None:
    monitor = NetworkMonitor()
    assert monitor.evaluate_quality(current_time_s=10.0) == NetworkQuality.SEVERED
    rec = monitor.compute_streaming_recommendation(current_time_s=10.0)
    assert rec["network_quality"] == "severed"
    assert rec["recommended_fps"] == 0.0
    assert rec["direct_driving_permitted"] is False


def test_network_monitor_optimal_link() -> None:
    cfg = TeleopConfig(
        max_rtt_ms=150.0,
        critical_rtt_ms=350.0,
        video_target_fps=20.0,
        video_fallback_fps=5.0,
    )
    monitor = NetworkMonitor(config=cfg)

    # Send 3 sequential heartbeats with low latency (50ms)
    for i in range(3):
        hb = TeleopHeartbeat(
            vehicle_id="veh-01",
            sequence_number=i,
            timestamp_s=1.0 + i * 0.1,
            rtt_ms=50.0,
        )
        quality = monitor.record_heartbeat(hb)
        assert quality == NetworkQuality.OPTIMAL

    rec = monitor.compute_streaming_recommendation(current_time_s=1.3)
    assert rec["network_quality"] == "optimal"
    assert rec["recommended_fps"] == 20.0
    assert rec["direct_driving_permitted"] is True
    assert rec["packet_loss_ratio"] == 0.0


def test_network_monitor_degraded_and_critical_link() -> None:
    cfg = TeleopConfig(
        max_rtt_ms=100.0,
        critical_rtt_ms=250.0,
        video_target_fps=20.0,
        video_fallback_fps=5.0,
    )
    monitor = NetworkMonitor(config=cfg)

    # 1. Degraded due to elevated RTT (180ms)
    hb_degraded = TeleopHeartbeat(
        vehicle_id="veh-01",
        sequence_number=1,
        timestamp_s=1.0,
        rtt_ms=180.0,
    )
    assert monitor.record_heartbeat(hb_degraded) == NetworkQuality.DEGRADED
    rec_deg = monitor.compute_streaming_recommendation(current_time_s=1.0)
    assert rec_deg["network_quality"] == "degraded"
    assert rec_deg["direct_driving_permitted"] is False
    assert rec_deg["recommended_fps"] == 12.5

    # 2. Critical due to very high RTT (300ms)
    hb_crit = TeleopHeartbeat(
        vehicle_id="veh-01",
        sequence_number=2,
        timestamp_s=1.2,
        rtt_ms=300.0,
    )
    assert monitor.record_heartbeat(hb_crit) == NetworkQuality.CRITICAL
    rec_crit = monitor.compute_streaming_recommendation(current_time_s=1.2)
    assert rec_crit["network_quality"] == "critical"
    assert rec_crit["recommended_fps"] == 5.0
    assert rec_crit["scale_factor"] == 0.5


def test_network_monitor_packet_loss_detection() -> None:
    monitor = NetworkMonitor()

    # Packet 0
    monitor.record_heartbeat(
        TeleopHeartbeat(vehicle_id="v1", sequence_number=0, timestamp_s=0.0, rtt_ms=40.0)
    )
    # Packet 5 (dropped 1, 2, 3, 4 -> 4 lost out of 6 total = 66.7% loss)
    monitor.record_heartbeat(
        TeleopHeartbeat(vehicle_id="v1", sequence_number=5, timestamp_s=0.5, rtt_ms=40.0)
    )

    assert monitor.total_packets_lost == 4
    assert monitor.packet_loss_ratio > 0.5
    assert monitor.evaluate_quality(current_time_s=0.5) == NetworkQuality.CRITICAL


def test_network_monitor_heartbeat_timeout_severance() -> None:
    cfg = TeleopConfig(heartbeat_timeout_s=1.0)
    monitor = NetworkMonitor(config=cfg)

    monitor.record_heartbeat(
        TeleopHeartbeat(vehicle_id="v1", sequence_number=0, timestamp_s=5.0, rtt_ms=30.0)
    )
    assert monitor.evaluate_quality(current_time_s=5.5) == NetworkQuality.OPTIMAL
    # Exceed timeout (5.0 + 1.1s = 6.1s)
    assert monitor.evaluate_quality(current_time_s=6.1) == NetworkQuality.SEVERED
