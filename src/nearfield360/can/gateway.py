"""Discrete-event HIL CAN bus gateway and real-time cyclic scheduler (ISO 11898)."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from nearfield360.can.codec import CANCodec
from nearfield360.can.faults import FaultInjector
from nearfield360.can.models import (
    CANBusTelemetry,
    CANFaultType,
    CANFrame,
    HILSimulationReport,
    VehicleControlSignal,
    VehicleFeedbackSignal,
)
from nearfield360.config import CANConfig

# Typical standard frame length in bit times (SOF, ID, Control, Data, CRC, ACK, EOF + stuffing)
# Standard 8-byte frame: ~108 to 128 bit times (approx 120 bits avg)
BITS_PER_CAN_FRAME_8BYTE = 128


class ScheduledMessage(NamedTuple):
    timestamp_us: float
    sequence_id: int
    ecu_name: str
    frame: CANFrame


class CANGateway:
    """Simulates real-time cyclic CAN scheduling, bus load estimation, and fault processing."""

    def __init__(self, config: CANConfig, fault_injector: FaultInjector | None = None) -> None:
        self.config = config
        self.fault_injector = fault_injector or FaultInjector(
            fault_rate=config.fault_injection_rate,
            max_jitter_us=config.clock_jitter_max_us,
        )

    def run_simulation(
        self,
        duration_s: float = 1.0,
        control_signal: VehicleControlSignal | None = None,
        feedback_signal: VehicleFeedbackSignal | None = None,
    ) -> HILSimulationReport:
        """Execute a discrete-time HIL simulation of vehicle ECU cyclic broadcasts."""
        duration_us = duration_s * 1_000_000.0

        ctrl = control_signal or VehicleControlSignal(gear="D", speed_mps=5.0)
        fb = feedback_signal or VehicleFeedbackSignal()

        def make_ctrl(t: float, hb: int) -> CANFrame:
            sig = ctrl.model_copy(update={"heartbeat_counter": hb})
            return CANCodec.encode_control_signal(sig, timestamp_us=t)

        def make_chassis(t: float, hb: int) -> CANFrame:
            sig = fb.model_copy(update={"heartbeat_counter": hb})
            return CANCodec.encode_feedback_signal(sig, timestamp_us=t)

        def make_sonar(t: float, hb: int) -> CANFrame:
            sig = fb.model_copy(update={"heartbeat_counter": hb})
            return CANCodec.encode_feedback_signal(sig, timestamp_us=t)

        # Define cyclic transmission periods in microseconds
        # - ECM/Control: 20 ms (50 Hz)
        # - WheelSpeed/Chassis: 10 ms (100 Hz)
        # - Sonar/ParkAssist: 50 ms (20 Hz)
        ecu_streams: list[tuple[str, float, Callable[[float, int], CANFrame]]] = [
            ("ECM_CTRL", 20_000.0, make_ctrl),
            ("CHASSIS_FB", 10_000.0, make_chassis),
            ("BODY_SONAR", 50_000.0, make_sonar),
        ]

        event_queue: list[ScheduledMessage] = []
        seq = 0
        for ecu_name, period_us, generator in ecu_streams:
            curr_t = 0.0
            hb = 0
            while curr_t < duration_us:
                frame = generator(curr_t, hb)
                heapq.heappush(event_queue, ScheduledMessage(curr_t, seq, ecu_name, frame))
                curr_t += period_us
                hb = (hb + 1) % 16
                seq += 1

        frames_sent = 0
        frames_received = 0
        dropped_frames = 0
        corrupted_frames = 0
        jitters_us: list[float] = []
        fault_log: list[dict[str, str | float | int]] = []

        total_bus_bits = 0

        while event_queue:
            msg = heapq.heappop(event_queue)
            frames_sent += 1

            # Process frame through physical bus / fault injector
            result = self.fault_injector.process_frame(msg.frame)
            if result is None:
                dropped_frames += 1
                fault_log.append(
                    {
                        "time_s": round(msg.timestamp_us / 1e6, 4),
                        "ecu": msg.ecu_name,
                        "fault": "frame_drop",
                        "id": hex(msg.frame.arbitration_id),
                    }
                )
                continue

            frames_received += 1
            total_bus_bits += BITS_PER_CAN_FRAME_8BYTE

            jitter = abs(result.timestamp_us - msg.timestamp_us)
            jitters_us.append(jitter)

            if result.fault_type != CANFaultType.NONE:
                corrupted_frames += 1
                fault_log.append(
                    {
                        "time_s": round(msg.timestamp_us / 1e6, 4),
                        "ecu": msg.ecu_name,
                        "fault": result.fault_type.value,
                        "id": hex(msg.frame.arbitration_id),
                    }
                )

        # Calculate bus bandwidth utilization (bus load %)
        # max capacity in bits = duration_s * bitrate_kbps * 1000
        capacity_bits = duration_s * (self.config.bus_bitrate_kbps * 1000.0)
        bus_load_pct = min(100.0, (total_bus_bits / max(1.0, capacity_bits)) * 100.0)

        mean_jitter = float(np.mean(jitters_us)) if jitters_us else 0.0
        peak_jitter = float(np.max(jitters_us)) if jitters_us else 0.0
        is_overloaded = bus_load_pct > self.config.max_bus_load_percent

        telemetry = CANBusTelemetry(
            frames_sent=frames_sent,
            frames_received=frames_received,
            dropped_frames=dropped_frames,
            corrupted_frames=corrupted_frames,
            bus_load_percent=round(bus_load_pct, 2),
            mean_jitter_us=round(mean_jitter, 2),
            peak_jitter_us=round(peak_jitter, 2),
            is_overloaded=is_overloaded,
        )

        # Evaluate target ECU health
        ecu_health: dict[str, str] = {}
        target_ecus = [e[0] for e in ecu_streams]
        for ecu in target_ecus:
            ecu_faults = sum(1 for f in fault_log if f.get("ecu") == ecu)
            if ecu_faults == 0:
                ecu_health[ecu] = "nominal"
            elif ecu_faults <= 3:
                ecu_health[ecu] = "degraded"
            else:
                ecu_health[ecu] = "fault_present"

        return HILSimulationReport(
            duration_s=duration_s,
            target_ecus=target_ecus,
            telemetry=telemetry,
            fault_injection_log=fault_log,
            ecu_health_status=ecu_health,
        )
