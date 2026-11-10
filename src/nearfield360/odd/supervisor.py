"""ODD Supervisor, speed governor, and ASIL-D MRM fallback safety interlock."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any

from nearfield360.config import ODDConfig
from nearfield360.odd.evaluator import ODDStateMachine
from nearfield360.odd.models import (
    ODDEvaluationResult,
    ODDMetrics,
    ODDReport,
    ODDState,
    SensorHealthStatus,
)

if TYPE_CHECKING:
    from nearfield360.safety.supervisor import SafetySupervisor

logger = logging.getLogger(__name__)


class ODDSupervisor:
    """Real-time supervisory monitor enforcing ODD speed governance and MRM interlocks."""

    def __init__(
        self,
        config: ODDConfig | None = None,
        state_machine: ODDStateMachine | None = None,
        safety_supervisor: SafetySupervisor | Any | None = None,
    ) -> None:
        self.config = config or ODDConfig()
        self.state_machine = state_machine or ODDStateMachine(self.config)
        self.safety_supervisor = safety_supervisor
        self.latest_evaluation: ODDEvaluationResult = ODDEvaluationResult()
        self.sensor_statuses: dict[str, SensorHealthStatus] = {}
        self.mrm_trigger_count: int = 0
        self.total_evaluations: int = 0

    @property
    def current_state(self) -> ODDState:
        """Active operational ODD tier."""
        return self.state_machine.current_state

    def update(
        self,
        metrics: ODDMetrics,
        sensors: dict[str, SensorHealthStatus] | None = None,
        timestamp_s: float = 0.0,
    ) -> ODDEvaluationResult:
        """Evaluate latest telemetry, update state machine, and apply MRM interlock if required."""
        if sensors:
            self.sensor_statuses.update(sensors)

        evaluation = self.state_machine.step(metrics, sensors, timestamp_s)
        self.latest_evaluation = evaluation
        self.total_evaluations += 1

        if evaluation.mrm_required and self.config.enable_mrm_interlock:
            self.mrm_trigger_count += 1
            reason = f"ODD boundary breach: {', '.join(evaluation.active_limiters)}"
            logger.critical("ODD Supervisor triggering ASIL-D MRM safe stop: %s", reason)
            if self.safety_supervisor is not None and hasattr(
                self.safety_supervisor, "trigger_emergency_stop"
            ):
                self.safety_supervisor.trigger_emergency_stop(reason, timestamp_s)

        return evaluation

    def govern_actuation(
        self,
        target_speed_mps: float,
        target_headway_m: float,
    ) -> tuple[float, float, ODDEvaluationResult]:
        """Apply ODD speed governor and safety headway scaling to vehicle actuation commands.

        Returns:
            (governed_speed_mps, governed_headway_m, latest_evaluation)
        """
        eval_res = self.latest_evaluation
        governed_speed = min(max(0.0, target_speed_mps), eval_res.governor_speed_limit_mps)
        governed_headway = target_headway_m * eval_res.safety_headway_multiplier
        return governed_speed, governed_headway, eval_res

    def generate_report(
        self,
        report_id: str = "ODD-REPORT-LIVE",
        vehicle_id: str = "AVP_EGO_01",
    ) -> ODDReport:
        """Generate comprehensive diagnostic and audit report."""
        timestamp_iso = datetime.datetime.now(datetime.UTC).isoformat()
        return ODDReport(
            report_id=report_id,
            vehicle_id=vehicle_id,
            timestamp_iso=timestamp_iso,
            latest_evaluation=self.latest_evaluation,
            sensor_statuses=dict(self.sensor_statuses),
            history=list(self.state_machine.history),
            total_samples=self.total_evaluations,
            mrm_triggered_count=self.mrm_trigger_count,
        )
