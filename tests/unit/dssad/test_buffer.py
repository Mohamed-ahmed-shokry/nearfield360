"""Unit tests for TelemetryRingBuffer capacity, FIFO eviction, and temporal slicing."""

from __future__ import annotations

import pytest

from nearfield360.dssad.buffer import TelemetryRingBuffer
from nearfield360.dssad.models import TelemetrySnapshot


def _make_snapshot(t_ns: int, speed: float = 5.0) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp_ns=t_ns,
        timestamp_iso=f"2026-11-10T12:00:{t_ns // 1_000_000_000:02d}Z",
        speed_mps=speed,
        longitudinal_accel_mps2=0.0,
        lateral_accel_mps2=0.0,
        yaw_rate_rads=0.0,
        steering_angle_rad=0.0,
        brake_pressure_bar=0.0,
        throttle_pct=15.0,
    )


def test_ring_buffer_init_and_capacity() -> None:
    buf = TelemetryRingBuffer(capacity=100)
    assert buf.capacity == 100
    assert len(buf) == 0
    assert buf.latest() is None
    assert buf.oldest() is None

    with pytest.raises(ValueError, match="strictly positive"):
        TelemetryRingBuffer(capacity=0)


def test_ring_buffer_fifo_rollover() -> None:
    buf = TelemetryRingBuffer(capacity=3)

    s1 = _make_snapshot(1_000_000_000, 1.0)
    s2 = _make_snapshot(2_000_000_000, 2.0)
    s3 = _make_snapshot(3_000_000_000, 3.0)
    s4 = _make_snapshot(4_000_000_000, 4.0)

    buf.append(s1)
    buf.append(s2)
    buf.append(s3)
    assert len(buf) == 3
    assert buf.oldest() == s1
    assert buf.latest() == s3

    # Adding s4 should evict s1
    buf.append(s4)
    assert len(buf) == 3
    assert buf.oldest() == s2
    assert buf.latest() == s4
    assert [s.speed_mps for s in buf.all_samples()] == [2.0, 3.0, 4.0]


def test_ring_buffer_window_slicing() -> None:
    buf = TelemetryRingBuffer(capacity=10)
    snapshots = [_make_snapshot(i * 1_000_000_000, float(i)) for i in range(1, 6)]
    buf.extend(snapshots)

    # Window from 2s to 4s
    window = buf.get_window(2_000_000_000, 4_000_000_000)
    assert len(window) == 3
    assert [s.speed_mps for s in window] == [2.0, 3.0, 4.0]

    # Inverted window returns empty
    assert buf.get_window(4_000_000_000, 2_000_000_000) == []


def test_ring_buffer_pre_post_trigger_extraction() -> None:
    # 50 Hz sampling (dt = 20 ms = 20_000_000 ns) over 10 seconds (500 samples)
    buf = TelemetryRingBuffer(capacity=1000)
    dt_ns = 20_000_000
    base_ns = 1_000_000_000_000

    for i in range(500):
        t_ns = base_ns + i * dt_ns
        buf.append(_make_snapshot(t_ns, speed=float(i % 20)))

    # Trigger at index 300 (t = base_ns + 300 * 20ms = base_ns + 6.0s)
    trigger_ns = base_ns + 300 * dt_ns

    # Pre-trigger 5.0 s (250 samples), Post-trigger 2.0 s (100 samples)
    pre_window, post_window = buf.get_pre_post_window(
        trigger_ns=trigger_ns, pre_duration_s=5.0, post_duration_s=2.0
    )

    # Pre-window includes trigger sample: [trigger_ns - 5s, trigger_ns] -> 251 samples
    assert len(pre_window) == 251
    assert pre_window[0].timestamp_ns == trigger_ns - int(5.0 * 1e9)
    assert pre_window[-1].timestamp_ns == trigger_ns

    # Post-window strictly following trigger: (trigger_ns, trigger_ns + 2s] -> 100 samples
    assert len(post_window) == 100
    assert post_window[0].timestamp_ns == trigger_ns + dt_ns
    assert post_window[-1].timestamp_ns == trigger_ns + int(2.0 * 1e9)


def test_ring_buffer_clear() -> None:
    buf = TelemetryRingBuffer(capacity=5)
    buf.extend([_make_snapshot(i * 1_000_000_000) for i in range(3)])
    assert len(buf) == 3
    buf.clear()
    assert len(buf) == 0
    assert buf.latest() is None
