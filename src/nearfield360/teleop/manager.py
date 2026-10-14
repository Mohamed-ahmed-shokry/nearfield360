"""Facility-level teleoperation assistance manager, session coordinator, and safety dispatcher."""

from __future__ import annotations

from nearfield360.config import TeleopConfig
from nearfield360.teleop.models import (
    MRMStage,
    NetworkQuality,
    TeleopCommand,
    TeleopHeartbeat,
    TeleopMode,
    TeleopReport,
    TeleopSession,
)
from nearfield360.teleop.mrm import MRMEngine
from nearfield360.teleop.network import NetworkMonitor


class TeleopManager:
    """Coordinates remote assistance sessions, network watchdog, and safety command routing."""

    def __init__(self, facility_id: str, config: TeleopConfig | None = None) -> None:
        self.facility_id = facility_id
        self.config = config or TeleopConfig()
        self.active_sessions: dict[str, TeleopSession] = {}
        self.archived_sessions: list[TeleopSession] = []
        self.network_monitors: dict[str, NetworkMonitor] = {}
        self.mrm_engines: dict[str, MRMEngine] = {}
        self._session_counter: int = 1
        self._total_interventions: int = 0
        self._mrm_executions: int = 0

    def start_session(
        self,
        vehicle_id: str,
        operator_id: str,
        mode: TeleopMode = TeleopMode.TELE_GUIDANCE,
        current_time_s: float = 0.0,
    ) -> TeleopSession:
        """Initiate a new teleoperation engagement for a vehicle."""
        if vehicle_id in self.active_sessions:
            self.end_session(vehicle_id, current_time_s)

        session_id = f"teleop-{self._session_counter:04d}"
        self._session_counter += 1

        session = TeleopSession(
            session_id=session_id,
            vehicle_id=vehicle_id,
            operator_id=operator_id,
            active_mode=mode,
            started_at_s=current_time_s,
            is_active=True,
        )

        self.active_sessions[vehicle_id] = session
        self.network_monitors[vehicle_id] = NetworkMonitor(config=self.config)
        self.mrm_engines[vehicle_id] = MRMEngine(vehicle_id=vehicle_id, config=self.config)
        self._total_interventions += 1

        return session

    def end_session(self, vehicle_id: str, current_time_s: float) -> TeleopSession | None:
        """Conclude an active teleoperation session."""
        session = self.active_sessions.pop(vehicle_id, None)
        if session is None:
            return None

        ended_session = session.model_copy(
            update={"ended_at_s": current_time_s, "is_active": False}
        )
        self.archived_sessions.append(ended_session)
        return ended_session

    def process_heartbeat(self, heartbeat: TeleopHeartbeat) -> NetworkQuality:
        """Record a telemetry heartbeat from a vehicle and enforce failsafe watchdog."""
        vehicle_id = heartbeat.vehicle_id
        if vehicle_id not in self.network_monitors:
            self.network_monitors[vehicle_id] = NetworkMonitor(config=self.config)
        if vehicle_id not in self.mrm_engines:
            self.mrm_engines[vehicle_id] = MRMEngine(vehicle_id=vehicle_id, config=self.config)

        monitor = self.network_monitors[vehicle_id]
        quality = monitor.record_heartbeat(heartbeat)

        # Heartbeat watchdog check
        if quality == NetworkQuality.SEVERED and self.config.auto_mrm_on_heartbeat_loss:
            mrm = self.mrm_engines[vehicle_id]
            if not mrm.is_active:
                mrm.trigger_mrm(
                    "heartbeat_loss_timeout",
                    heartbeat.timestamp_s,
                    heartbeat.current_speed_mps,
                )
                self._mrm_executions += 1
                if vehicle_id in self.active_sessions:
                    session = self.active_sessions[vehicle_id]
                    self.active_sessions[vehicle_id] = session.model_copy(
                        update={"active_mode": TeleopMode.MRM_SAFE_STOP, "mrm_stage": mrm.stage}
                    )

        return quality

    def dispatch_command(
        self,
        command: TeleopCommand,
        current_time_s: float,
        current_speed_mps: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate and dispatch an operator command to the target vehicle."""
        vehicle_id = command.vehicle_id
        session = self.active_sessions.get(vehicle_id)
        if session is None or not session.is_active:
            return False, f"No active teleoperation session for vehicle '{vehicle_id}'"

        mrm = self.mrm_engines.get(vehicle_id)
        if mrm is None:
            mrm = MRMEngine(vehicle_id=vehicle_id, config=self.config)
            self.mrm_engines[vehicle_id] = mrm

        # 1. Immediate E-stop override
        if command.emergency_stop:
            mrm.trigger_mrm("operator_e_stop_command", current_time_s, current_speed_mps)
            self._mrm_executions += 1
            self.active_sessions[vehicle_id] = session.model_copy(
                update={
                    "active_mode": TeleopMode.MRM_SAFE_STOP,
                    "mrm_stage": mrm.stage,
                    "commands_processed_count": session.commands_processed_count + 1,
                }
            )
            return True, "Emergency stop executed: vehicle decelerating to safe stop"

        # 2. Check if vehicle is already in MRM lockdown
        if mrm.is_active and mrm.stage in (MRMStage.STANDSTILL_LOCKED, MRMStage.EMERGENCY_BEACON):
            return False, "Command rejected: vehicle is locked in Minimum Risk Maneuver standstill"

        # 3. Direct driving latency check
        monitor = self.network_monitors.get(vehicle_id)
        if command.mode == TeleopMode.DIRECT_DRIVING:
            quality = (
                monitor.evaluate_quality(current_time_s) if monitor else NetworkQuality.SEVERED
            )
            if quality != NetworkQuality.OPTIMAL:
                return (
                    False,
                    f"Direct driving rejected: link quality is {quality.value} (requires optimal)",
                )

        # 4. Successful command routing
        self.active_sessions[vehicle_id] = session.model_copy(
            update={
                "active_mode": command.mode,
                "commands_processed_count": session.commands_processed_count + 1,
            }
        )
        return True, f"Command {command.command_id} dispatched successfully"

    def step(self, current_time_s: float) -> None:
        """Advance time step across all active sessions, MRMs, and watchdog checks."""
        for vehicle_id, session in list(self.active_sessions.items()):
            monitor = self.network_monitors.get(vehicle_id)
            mrm = self.mrm_engines.get(vehicle_id)

            if monitor and self.config.auto_mrm_on_heartbeat_loss:
                quality = monitor.evaluate_quality(current_time_s)
                if quality == NetworkQuality.SEVERED and mrm and not mrm.is_active:
                    mrm.trigger_mrm("heartbeat_loss_timeout", current_time_s, current_speed_mps=1.0)
                    self._mrm_executions += 1
                    self.active_sessions[vehicle_id] = session.model_copy(
                        update={"active_mode": TeleopMode.MRM_SAFE_STOP, "mrm_stage": mrm.stage}
                    )

            if mrm and mrm.is_active:
                mrm.step(current_time_s)
                self.active_sessions[vehicle_id] = session.model_copy(
                    update={"mrm_stage": mrm.stage}
                )

    def generate_report(self, current_time_s: float) -> TeleopReport:
        """Compile a facility-wide teleoperation diagnostic and audit report."""
        all_sessions = list(self.archived_sessions) + list(self.active_sessions.values())

        rtt_values = [
            m.ema_rtt_ms for m in self.network_monitors.values() if m.total_packets_received > 0
        ]
        avg_rtt = sum(rtt_values) / len(rtt_values) if rtt_values else 0.0

        return TeleopReport(
            facility_id=self.facility_id,
            timestamp_s=current_time_s,
            active_sessions_count=len(self.active_sessions),
            completed_sessions_count=len(self.archived_sessions),
            total_interventions_count=self._total_interventions,
            average_rtt_ms=avg_rtt,
            mrm_executions_count=self._mrm_executions,
            sessions=all_sessions,
        )
