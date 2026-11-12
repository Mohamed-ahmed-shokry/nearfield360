"""Unit tests for DSSADRecorder crash freeze manager and trigger interlocks."""

from __future__ import annotations

from nearfield360.config import DSSADConfig
from nearfield360.dssad.models import DSSADEventType, EventSeverity, TelemetrySnapshot
from nearfield360.dssad.recorder import DSSADRecorder


def _sample(
    t_ns: int,
    speed: float = 4.0,
    long_accel: float = 0.0,
    lat_accel: float = 0.0,
    odd_state: str = "NOMINAL",
    aeb: bool = False,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp_ns=t_ns,
        timestamp_iso=f"2026-11-10T12:00:{t_ns // 1_000_000_000:02d}Z",
        speed_mps=speed,
        longitudinal_accel_mps2=long_accel,
        lateral_accel_mps2=lat_accel,
        yaw_rate_rads=0.0,
        steering_angle_rad=0.0,
        brake_pressure_bar=30.0 if aeb else 0.0,
        throttle_pct=0.0 if aeb else 20.0,
        odd_state=odd_state,
        aeb_active=aeb,
    )


def test_recorder_init_and_empty_state() -> None:
    rec = DSSADRecorder()
    assert rec.total_events == 0
    assert rec.total_incidents == 0

    report = rec.get_report()
    assert report.total_events_recorded == 0
    assert report.total_incidents_frozen == 0
    assert report.chain_verified is True


def test_recorder_manual_event_logging() -> None:
    rec = DSSADRecorder()
    snapshot = _sample(1_000_000_000, speed=3.0)

    evt1 = rec.log_event(
        event_type=DSSADEventType.SYSTEM_ACTIVATION,
        severity=EventSeverity.INFO,
        trigger_subsystem="avp_mission_executive",
        description="Autonomous valet parking engaged at drop-off zone.",
        snapshot=snapshot,
    )

    assert rec.total_events == 1
    assert evt1.event_type == DSSADEventType.SYSTEM_ACTIVATION
    assert evt1.sequence_id == 0
    assert len(evt1.block_hash) == 64

    # Verification passes
    res = rec.verify_integrity()
    assert res.valid is True
    assert res.total_blocks == 1


def test_recorder_auto_impact_trigger() -> None:
    cfg = DSSADConfig(
        sampling_rate_hz=50,
        pre_trigger_duration_s=2.0,
        post_trigger_duration_s=1.0,
        impact_accel_threshold_g=2.5,
    )
    rec = DSSADRecorder(config=cfg)
    base_ns = 1_000_000_000_000
    dt_ns = 20_000_000  # 50 Hz

    # Ingest 100 normal samples (2.0s)
    for i in range(100):
        t = base_ns + i * dt_ns
        rec.record_telemetry(_sample(t, speed=6.0, long_accel=0.0))

    # Impact sample: -28.0 m/s^2 (~2.85 g deceleration)
    impact_t = base_ns + 100 * dt_ns
    impact_evt = rec.record_telemetry(_sample(impact_t, speed=2.0, long_accel=-28.0))

    assert impact_evt is not None
    assert impact_evt.event_type == DSSADEventType.COLLISION_IMPACT_DETECTED
    assert impact_evt.severity == EventSeverity.FATAL
    assert rec.total_incidents == 1

    # Ingest post-impact samples (1.0s = 50 samples)
    for i in range(101, 151):
        t = base_ns + i * dt_ns
        rec.record_telemetry(_sample(t, speed=0.0, long_accel=0.0))

    incidents = rec.query_incidents()
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.trigger_event.event_type == DSSADEventType.COLLISION_IMPACT_DETECTED
    assert inc.is_locked is True
    assert inc.peak_accel_g >= 2.5
    assert inc.delta_v_mps >= 4.0


def test_recorder_mrm_interlock_trigger() -> None:
    cfg = DSSADConfig(auto_freeze_on_mrm=True)
    rec = DSSADRecorder(config=cfg)

    # Ingest sample with ODD_BREACH_MRM and aeb_active
    s = _sample(
        1_000_000_000,
        speed=1.5,
        long_accel=-4.0,
        odd_state="ODD_BREACH_MRM",
        aeb=True,
    )
    evt = rec.record_telemetry(s)

    assert evt is not None
    assert evt.event_type == DSSADEventType.MINIMUM_RISK_MANEUVER_INITIATED
    assert evt.severity == EventSeverity.CRITICAL
    assert rec.total_incidents == 1


def test_recorder_query_filtering() -> None:
    rec = DSSADRecorder()
    rec.log_event(
        event_type=DSSADEventType.SYSTEM_ACTIVATION,
        severity=EventSeverity.INFO,
        trigger_subsystem="app",
        description="Activation",
        snapshot=_sample(1_000_000_000),
    )
    rec.log_event(
        event_type=DSSADEventType.ODD_BOUNDARY_BREACH,
        severity=EventSeverity.WARNING,
        trigger_subsystem="odd",
        description="Advisory speed restriction",
        snapshot=_sample(2_000_000_000),
    )
    rec.log_event(
        event_type=DSSADEventType.SECURITY_INTRUSION_ALERT,
        severity=EventSeverity.CRITICAL,
        trigger_subsystem="csms",
        description="CAN flood alert",
        snapshot=_sample(3_000_000_000),
    )

    assert len(rec.query_events()) == 3
    assert len(rec.query_events(event_type=DSSADEventType.SECURITY_INTRUSION_ALERT)) == 1
    assert len(rec.query_events(min_severity=EventSeverity.WARNING)) == 2
    assert len(rec.query_events(min_severity=EventSeverity.CRITICAL)) == 1
    assert len(rec.query_events(start_ns=1_500_000_000, end_ns=2_500_000_000)) == 1


def test_recorder_max_retained_incidents_fifo() -> None:
    cfg = DSSADConfig(max_retained_incidents=2)
    rec = DSSADRecorder(config=cfg)

    for i in range(3):
        rec.log_event(
            event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
            severity=EventSeverity.FATAL,
            trigger_subsystem="sensor",
            description=f"Impact {i}",
            snapshot=_sample((i + 1) * 1_000_000_000),
            freeze_incident=True,
        )

    # Only 2 retained
    assert rec.total_incidents == 2
    incidents = rec.query_incidents()
    assert incidents[0].trigger_event.description == "Impact 1"
    assert incidents[1].trigger_event.description == "Impact 2"
