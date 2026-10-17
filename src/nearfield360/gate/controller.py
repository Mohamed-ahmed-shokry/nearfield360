"""Inductive loop sequencing, barrier gate kinematics, and safety interlock controller."""

from __future__ import annotations

from nearfield360.config import GateConfig
from nearfield360.gate.envelope import VehicleEnvelopeProfiler
from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    BarrierKinematicsState,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    GateDirection,
    GateTransitEvent,
    InductiveLoopId,
    InductiveLoopState,
)
from nearfield360.gate.validator import AccessCredentialValidator


class BarrierInterlockController:
    """Orchestrates barrier gate kinematics, inductive safety loops, and access decisions."""

    def __init__(
        self,
        config: GateConfig,
        validator: AccessCredentialValidator,
        envelope_profiler: VehicleEnvelopeProfiler,
        gate_id: str = "GATE_01",
        direction: GateDirection = GateDirection.ENTRY,
    ) -> None:
        self._config = config
        self._validator = validator
        self._envelope_profiler = envelope_profiler
        self._gate_id = gate_id
        self._direction = direction

        self._barrier = BarrierKinematicsState(
            barrier_angle_deg=0.0,
            angular_velocity_deg_s=0.0,
            state=BarrierArmState.LOWERED_CLOSED,
            safety_interlock_active=False,
            obstruction_detected=False,
        )

        self._loops: dict[InductiveLoopId, InductiveLoopState] = {
            loop_id: InductiveLoopState(loop_id=loop_id, occupied=False)
            for loop_id in InductiveLoopId
        }

        self._authorized_vehicle_id: str | None = None
        self._active_decision: GateAccessDecision | None = None
        self._authorized_timestamp_s: float | None = None
        self._departure_cleared_at_s: float | None = None

        self._tailgating_alarm_active: bool = False
        self._tailgating_incidents_count: int = 0
        self._clearance_violations_count: int = 0
        self._transits: list[GateTransitEvent] = []

    @property
    def config(self) -> GateConfig:
        return self._config

    @property
    def gate_id(self) -> str:
        return self._gate_id

    @property
    def direction(self) -> GateDirection:
        return self._direction

    @property
    def barrier(self) -> BarrierKinematicsState:
        return self._barrier

    @property
    def loops(self) -> dict[InductiveLoopId, InductiveLoopState]:
        return self._loops

    @property
    def tailgating_alarm_active(self) -> bool:
        return self._tailgating_alarm_active

    @property
    def clearance_violations_count(self) -> int:
        return self._clearance_violations_count

    @property
    def tailgating_incidents_count(self) -> int:
        return self._tailgating_incidents_count

    def get_loop(self, loop_id: InductiveLoopId) -> InductiveLoopState:
        return self._loops[loop_id]

    def reset_tailgating_alarm(self) -> None:
        """Clear active tailgating warning latch."""
        self._tailgating_alarm_active = False

    def set_loop_state(
        self, loop_id: InductiveLoopId, occupied: bool, timestamp_s: float = 0.0
    ) -> None:
        """Update occupancy state and trigger safety interlocks for an inductive roadway loop."""
        loop = self._loops[loop_id]
        prev_occupied = loop.occupied
        loop.occupied = occupied

        if occupied and not prev_occupied:
            loop.activated_at_s = timestamp_s
            loop.dwell_duration_s = 0.0
        elif prev_occupied:
            loop.dwell_duration_s = max(0.0, timestamp_s - loop.activated_at_s)

        # Safety Loop C Interlock: immediate reversal if downward closing arm obstructed
        if (
            loop_id == InductiveLoopId.LOOP_C_SAFETY
            and occupied
            and self._barrier.state in (BarrierArmState.CLOSING, BarrierArmState.LOWERED_CLOSED)
        ):
            self._barrier.state = BarrierArmState.SAFETY_REVERSING
            self._barrier.safety_interlock_active = True
            self._barrier.obstruction_detected = True
            open_rate = 90.0 / max(0.1, self._config.barrier_opening_time_s)
            self._barrier.angular_velocity_deg_s = open_rate * 1.5

        # Anti-tailgating check: Loop B or C activated without active authorized passage
        if (
            loop_id in (InductiveLoopId.LOOP_B_PRESENCE, InductiveLoopId.LOOP_C_SAFETY)
            and occupied
            and self._barrier.state in (BarrierArmState.RAISED_OPEN, BarrierArmState.OPENING)
            and self._authorized_vehicle_id is None
        ):
            self._tailgating_alarm_active = True
            self._tailgating_incidents_count += 1

    def command_open(self) -> None:
        """Initiate upward actuation of the barrier boom arm."""
        self._barrier.state = BarrierArmState.OPENING
        self._barrier.angular_velocity_deg_s = 90.0 / max(0.1, self._config.barrier_opening_time_s)
        self._barrier.safety_interlock_active = False
        self._barrier.obstruction_detected = False

    def command_close(self, force: bool = False) -> bool:
        """Initiate downward descent of barrier boom arm if safety loop is clear."""
        if self._loops[InductiveLoopId.LOOP_C_SAFETY].occupied and not force:
            self._barrier.safety_interlock_active = True
            self._barrier.obstruction_detected = True
            return False

        self._barrier.state = BarrierArmState.CLOSING
        close_rate = -90.0 / max(0.1, self._config.barrier_closing_time_s)
        self._barrier.angular_velocity_deg_s = close_rate
        return True

    def step_kinematics(self, dt_s: float, timestamp_s: float = 0.0) -> BarrierKinematicsState:
        """Advance barrier boom arm kinematics by time delta dt_s."""
        # Safety Loop C check during descent
        if (
            self._barrier.state == BarrierArmState.CLOSING
            and self._loops[InductiveLoopId.LOOP_C_SAFETY].occupied
        ):
            self._barrier.state = BarrierArmState.SAFETY_REVERSING
            self._barrier.safety_interlock_active = True
            self._barrier.obstruction_detected = True
            open_rate = 90.0 / max(0.1, self._config.barrier_opening_time_s)
            self._barrier.angular_velocity_deg_s = open_rate * 1.5

        if self._barrier.state == BarrierArmState.OPENING:
            open_rate = 90.0 / max(0.1, self._config.barrier_opening_time_s)
            self._barrier.angular_velocity_deg_s = open_rate
            self._barrier.barrier_angle_deg += open_rate * dt_s
            if self._barrier.barrier_angle_deg >= 90.0:
                self._barrier.barrier_angle_deg = 90.0
                self._barrier.angular_velocity_deg_s = 0.0
                self._barrier.state = BarrierArmState.RAISED_OPEN

        elif self._barrier.state == BarrierArmState.SAFETY_REVERSING:
            reverse_rate = (90.0 / max(0.1, self._config.barrier_opening_time_s)) * 1.5
            self._barrier.angular_velocity_deg_s = reverse_rate
            self._barrier.barrier_angle_deg += reverse_rate * dt_s
            if self._barrier.barrier_angle_deg >= 90.0:
                self._barrier.barrier_angle_deg = 90.0
                self._barrier.angular_velocity_deg_s = 0.0
                self._barrier.state = BarrierArmState.RAISED_OPEN
                self._barrier.safety_interlock_active = False

        elif self._barrier.state == BarrierArmState.CLOSING:
            close_rate = 90.0 / max(0.1, self._config.barrier_closing_time_s)
            self._barrier.angular_velocity_deg_s = -close_rate
            self._barrier.barrier_angle_deg -= close_rate * dt_s
            if self._barrier.barrier_angle_deg <= 0.0:
                self._barrier.barrier_angle_deg = 0.0
                self._barrier.angular_velocity_deg_s = 0.0
                self._barrier.state = BarrierArmState.LOWERED_CLOSED

        return self._barrier

    def process_access_request(self, request: GateAccessRequest) -> GateAccessDecision:
        """Validate dimensional clearance and credential authority, commanding barrier actuation."""
        # 1. Dimensional envelope clearance verification
        clearance_decision = self._envelope_profiler.evaluate_clearance(
            request.envelope, self._gate_id, request.request_id
        )
        if not clearance_decision.granted:
            self._clearance_violations_count += 1
            return clearance_decision

        # 2. Access credential and reservation authorization
        cred_decision = self._validator.validate_request(request)
        if not cred_decision.granted:
            return cred_decision

        # 3. Both clearance and credentials granted
        combined_decision = GateAccessDecision(
            decision_id=f"dec-{request.request_id}-approved",
            request_id=request.request_id,
            status=AccessDecisionStatus.GRANTED,
            granted=True,
            message=(
                f"{cred_decision.message} | "
                f"clearance[head={clearance_decision.headroom_margin_m:+.2f}m, "
                f"width={clearance_decision.width_margin_m:+.2f}m]"
            )[:256],
            authorized_at_s=request.timestamp_s,
            headroom_margin_m=clearance_decision.headroom_margin_m,
            width_margin_m=clearance_decision.width_margin_m,
            issued_pass_id=cred_decision.issued_pass_id,
        )

        self._authorized_vehicle_id = request.vehicle_id
        self._active_decision = combined_decision
        self._authorized_timestamp_s = request.timestamp_s
        self.command_open()
        return combined_decision

    def simulate_transit(
        self, request: GateAccessRequest, step_dt_s: float = 0.1
    ) -> GateTransitEvent:
        """Simulate an end-to-end discrete gateway approach, clearance, and barrier cycle."""
        sim_time = request.timestamp_s

        # Step 1: Vehicle triggers Loop A and Loop B
        self.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, True, sim_time)
        self.set_loop_state(InductiveLoopId.LOOP_B_PRESENCE, True, sim_time)

        decision = self.process_access_request(request)
        if not decision.granted:
            # Transit blocked
            self.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, False, sim_time + 1.0)
            self.set_loop_state(InductiveLoopId.LOOP_B_PRESENCE, False, sim_time + 1.0)
            event = GateTransitEvent(
                event_id=f"evt-{request.request_id}",
                gate_id=self._gate_id,
                vehicle_id=request.vehicle_id,
                decision=decision,
                duration_s=0.0,
                tailgating_detected=False,
                timestamp_s=sim_time,
            )
            self._transits.append(event)
            return event

        # Step 2: Actuate barrier opening
        for _ in range(500):
            if self._barrier.state == BarrierArmState.RAISED_OPEN:
                break
            sim_time += step_dt_s
            self.step_kinematics(step_dt_s, sim_time)

        # Step 3: Vehicle moves onto safety loop C and leaves loop A/B
        sim_time += 0.5
        self.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, False, sim_time)
        self.set_loop_state(InductiveLoopId.LOOP_B_PRESENCE, False, sim_time)
        self.set_loop_state(InductiveLoopId.LOOP_C_SAFETY, True, sim_time)

        # Step 4: Vehicle clears loop C and enters departure loop D
        sim_time += 1.0
        self.set_loop_state(InductiveLoopId.LOOP_C_SAFETY, False, sim_time)
        self.set_loop_state(InductiveLoopId.LOOP_D_DEPARTURE, True, sim_time)

        # Step 5: Vehicle clears departure loop D
        sim_time += 1.0
        self.set_loop_state(InductiveLoopId.LOOP_D_DEPARTURE, False, sim_time)
        self._authorized_vehicle_id = None  # Authorization completed

        # Step 6: Close barrier
        self.command_close()
        for _ in range(500):
            if self._barrier.state == BarrierArmState.LOWERED_CLOSED:
                break
            sim_time += step_dt_s
            self.step_kinematics(step_dt_s, sim_time)

        total_duration = round(sim_time - request.timestamp_s, 2)
        event = GateTransitEvent(
            event_id=f"evt-{request.request_id}",
            gate_id=self._gate_id,
            vehicle_id=request.vehicle_id,
            decision=decision,
            duration_s=total_duration,
            tailgating_detected=self._tailgating_alarm_active,
            timestamp_s=request.timestamp_s,
        )
        self._transits.append(event)
        return event

    def generate_audit_report(self) -> GateAuditReport:
        """Synthesize operational statistics across all logged transit cycles."""
        total = len(self._transits)
        granted = sum(1 for e in self._transits if e.decision.granted)
        rejected = total - granted
        tailgating = self._tailgating_incidents_count
        clearance = self._clearance_violations_count
        granted_durations = [e.duration_s for e in self._transits if e.decision.granted]
        avg_dur = (
            round(sum(granted_durations) / len(granted_durations), 2) if granted_durations else 0.0
        )

        return GateAuditReport(
            gate_id=self._gate_id,
            total_transits=total,
            granted_count=granted,
            rejected_count=rejected,
            tailgating_incidents=tailgating,
            clearance_violations=clearance,
            average_transit_duration_s=avg_dur,
        )
