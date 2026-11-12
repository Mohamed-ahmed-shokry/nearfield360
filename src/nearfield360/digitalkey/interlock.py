"""Proximity safety standstill interlock and AVP drop-off / pick-up handover supervisor."""

from collections.abc import Sequence
from datetime import UTC, datetime

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.crypto import DigitalKeyCrypto
from nearfield360.digitalkey.models import (
    CCCDelegationToken,
    DelegationScope,
    DevicePosition3D,
    DigitalKeyReport,
    HandoverState,
    ProximityInterlockDecision,
    ProximityZone,
    UWBRangeMeasurement,
)
from nearfield360.digitalkey.solver import MultilaterationSolver
from nearfield360.digitalkey.uwb import UWBEngine


class HandoverSupervisor:
    """Coordinates drop-off delegation, standstill safety interlocks, and pick-up handover."""

    def __init__(
        self,
        vin: str = "WP0ZZZ99ZTS392001",
        config: DigitalKeyConfig | None = None,
        crypto: DigitalKeyCrypto | None = None,
        uwb_engine: UWBEngine | None = None,
        solver: MultilaterationSolver | None = None,
    ) -> None:
        self.vin = vin
        self.config = config or DigitalKeyConfig()
        self.crypto = crypto or DigitalKeyCrypto(self.config)
        self.uwb = uwb_engine or UWBEngine(self.config)
        self.solver = solver or MultilaterationSolver(config=self.config)

        self._state = HandoverState.DISCONNECTED
        self._active_token: CCCDelegationToken | None = None
        self._last_position: DevicePosition3D | None = None
        self._last_interlock: ProximityInterlockDecision | None = None

    @property
    def state(self) -> HandoverState:
        """Current operational handover lifecycle state."""
        return self._state

    @property
    def active_token(self) -> CCCDelegationToken | None:
        """Currently verified and active delegation token."""
        return self._active_token

    def delegate_to_avp(
        self,
        token: CCCDelegationToken,
        consume_nonce: bool = True,
    ) -> tuple[bool, str]:
        """Drop-off delegation: Hand over vehicle from driver to autonomous valet parking."""
        valid, reason = self.crypto.verify_token(
            token,
            expected_vin=self.vin,
            required_scope=DelegationScope.AVP_OPERATE,
            consume_nonce=consume_nonce,
        )
        if not valid:
            return False, f"Drop-off delegation rejected: {reason}"

        self._active_token = token
        self._state = HandoverState.AVP_DELEGATED
        return True, "AVP delegation confirmed: Vehicle locked and autonomous navigation engaged"

    def mark_pickup_ready(self) -> None:
        """AVP vehicle arrived at pick-up berth and awaits driver return."""
        if self._state == HandoverState.AVP_DELEGATED:
            self._state = HandoverState.PICKUP_READY

    def evaluate(
        self,
        measurements: Sequence[UWBRangeMeasurement],
        timestamp_ns: int | None = None,
    ) -> DigitalKeyReport:
        """Evaluate UWB measurements, detect relay attacks, solve position, and enforce interlocks.

        Returns:
            DigitalKeyReport containing diagnostic telemetry and safety decision.
        """
        # 1. Process physical UWB layer and anti-relay checks
        valid_meas, any_relay, max_disc, attack_reasons = self.uwb.process_measurements(
            list(measurements)
        )

        # 2. Solve 3D spatial position from valid measurements
        position = self.solver.solve(valid_meas, timestamp_ns=timestamp_ns)
        self._last_position = position

        # 3. Determine safety standstill interlock
        standstill = False
        interlock_active = False
        reason = "Normal operation"

        if any_relay:
            interlock_active = True
            standstill = True
            reason = f"EMERGENCY STANDSTILL: Relay attack detected ({'; '.join(attack_reasons)})"
        elif (
            self.config.enable_standstill_interlock
            and self._state in (HandoverState.AVP_DELEGATED, HandoverState.PICKUP_READY)
            and (
                position.zone == ProximityZone.TOUCH
                or position.distance_m <= self.config.touch_threshold_m
            )
        ):
            interlock_active = True
            standstill = True
            reason = (
                f"STANDSTILL INTERLOCK: Device in TOUCH zone ({position.distance_m:.2f}m) "
                "during autonomous maneuvering"
            )

        # 4. Advance handover state machine
        if not any_relay:
            if self._state == HandoverState.PICKUP_READY:
                if position.is_interior:
                    self._state = HandoverState.MANUAL_CONTROL
                elif position.zone in (ProximityZone.IMMEDIATE, ProximityZone.TOUCH):
                    self._state = HandoverState.TAKEOVER_IN_PROGRESS
            elif self._state == HandoverState.TAKEOVER_IN_PROGRESS:
                if position.is_interior:
                    self._state = HandoverState.MANUAL_CONTROL
            elif (
                self._state == HandoverState.DISCONNECTED
                and position.zone != ProximityZone.OUT_OF_RANGE
            ):
                self._state = HandoverState.APPROACHING

        interlock_decision = ProximityInterlockDecision(
            interlock_active=interlock_active,
            standstill_command=standstill,
            detected_distance_m=position.distance_m,
            proximity_zone=position.zone,
            handover_state=self._state,
            reason=reason,
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        self._last_interlock = interlock_decision

        device_id = self._active_token.device_id if self._active_token else "unknown-device"

        return DigitalKeyReport(
            device_id=device_id,
            vin=self.vin,
            handover_state=self._state,
            proximity_zone=position.zone,
            device_position=position,
            interlock=interlock_decision,
            uwb_measurements=list(measurements),
            delegation_token=self._active_token,
            relay_attack_detected=any_relay,
            relay_discrepancy_ns=round(max_disc, 2),
        )

    def reset(self) -> None:
        """Reset handover state to DISCONNECTED and clear active token."""
        self._state = HandoverState.DISCONNECTED
        self._active_token = None
        self._last_position = None
        self._last_interlock = None
