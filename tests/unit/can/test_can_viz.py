"""Unit tests for tactical HIL oscilloscope and CAN telemetry visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.can.gateway import CANGateway
from nearfield360.can.viz import render_hil_dashboard
from nearfield360.config import CANConfig


def test_render_hil_dashboard(tmp_path: Path) -> None:
    config = CANConfig(bus_bitrate_kbps=500, fault_injection_rate=0.15)
    gateway = CANGateway(config)
    report = gateway.run_simulation(duration_s=0.5)

    png_path = tmp_path / "hil_oscilloscope.png"
    canvas = render_hil_dashboard(report, output_path=png_path)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert png_path.is_file()
    assert png_path.stat().st_size > 5000
