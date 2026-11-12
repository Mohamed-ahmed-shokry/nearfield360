"""Unit tests for tactical 1280x720 forensic HUD visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.config import DSSADConfig
from nearfield360.dssad.models import DSSADReport, TelemetrySnapshot
from nearfield360.dssad.recorder import DSSADRecorder
from nearfield360.dssad.viz import render_dssad_dashboard


def _sample(t_ns: int, speed: float, accel: float) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp_ns=t_ns,
        timestamp_iso="2026-11-10T12:00:00Z",
        speed_mps=speed,
        longitudinal_accel_mps2=accel,
        lateral_accel_mps2=0.5,
        yaw_rate_rads=0.01,
        steering_angle_rad=0.02,
        brake_pressure_bar=40.0 if accel < -10 else 0.0,
        throttle_pct=0.0 if accel < -10 else 25.0,
        odd_state="NOMINAL" if accel > -10 else "ODD_BREACH_MRM",
        aeb_active=accel < -10,
    )


def test_render_dssad_dashboard_with_incident(tmp_path: Path) -> None:
    rec = DSSADRecorder(config=DSSADConfig())
    base_ns = 1_000_000_000_000
    dt_ns = 20_000_000

    # Feed 100 pre-crash samples
    for i in range(100):
        t = base_ns + i * dt_ns
        rec.record_telemetry(_sample(t, speed=6.0, accel=-0.5))

    # Crash sample: -28 m/s^2 (~2.85 g)
    crash_t = base_ns + 100 * dt_ns
    rec.record_telemetry(_sample(crash_t, speed=2.0, accel=-28.0))

    # Post crash samples
    for i in range(101, 150):
        t = base_ns + i * dt_ns
        rec.record_telemetry(_sample(t, speed=0.0, accel=0.0))

    report = rec.get_report()
    assert report.total_incidents_frozen == 1

    png_path = tmp_path / "dssad_dashboard.png"
    img = render_dssad_dashboard(report, output_path=png_path)

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert img.dtype == np.uint8
    assert png_path.is_file()
    assert png_path.stat().st_size > 15_000  # valid compressed PNG artifact


def test_render_dssad_dashboard_empty_report(tmp_path: Path) -> None:
    report = DSSADReport(
        report_id="REP-EMPTY",
        vehicle_id="AVP-TEST-02",
        generated_at="2026-11-10T12:00:00Z",
        total_events_recorded=0,
        total_incidents_frozen=0,
        chain_verified=True,
    )

    png_path = tmp_path / "empty_dashboard.png"
    img = render_dssad_dashboard(report, output_path=png_path)

    assert img.shape == (720, 1280, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 10_000
