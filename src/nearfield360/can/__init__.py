"""Hardware-in-the-Loop (HIL) automated test harness and vehicle CAN bus gateway (ISO 11898)."""

from __future__ import annotations

from nearfield360.can.codec import (
    ARBITRATION_ID_CONTROL,
    ARBITRATION_ID_FEEDBACK,
    CANCodec,
    compute_crc8_j1850,
)
from nearfield360.can.faults import FaultInjector
from nearfield360.can.gateway import BITS_PER_CAN_FRAME_8BYTE, CANGateway
from nearfield360.can.models import (
    CANBusTelemetry,
    CANFaultType,
    CANFrame,
    CANFrameType,
    HILSimulationReport,
    VehicleControlSignal,
    VehicleFeedbackSignal,
)
from nearfield360.can.viz import render_hil_dashboard

__all__ = [
    "ARBITRATION_ID_CONTROL",
    "ARBITRATION_ID_FEEDBACK",
    "BITS_PER_CAN_FRAME_8BYTE",
    "CANBusTelemetry",
    "CANCodec",
    "CANFaultType",
    "CANFrame",
    "CANFrameType",
    "CANGateway",
    "FaultInjector",
    "HILSimulationReport",
    "VehicleControlSignal",
    "VehicleFeedbackSignal",
    "compute_crc8_j1850",
    "render_hil_dashboard",
]
