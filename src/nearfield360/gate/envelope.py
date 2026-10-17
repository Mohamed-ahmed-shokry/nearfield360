"""Vehicle dimensional envelope profiling and physical gateway clearance verification."""

from __future__ import annotations

from nearfield360.config import GateConfig
from nearfield360.gate.models import (
    AccessDecisionStatus,
    GateAccessDecision,
    VehicleDimensionEnvelope,
)


class VehicleEnvelopeProfiler:
    """Evaluates vehicle 3D bounding geometry against structural gateway portal limits."""

    def __init__(self, config: GateConfig) -> None:
        self._config = config

    @property
    def config(self) -> GateConfig:
        return self._config

    def evaluate_clearance(
        self,
        envelope: VehicleDimensionEnvelope,
        gate_id: str = "GATE_PORTAL",
        request_id: str = "req-envelope",
    ) -> GateAccessDecision:
        """Evaluate dimensional envelope against portal height, width, and ground clearance."""
        headroom_margin_m = round(self._config.portal_clearance_height_m - envelope.height_m, 3)
        width_margin_m = round(self._config.portal_clearance_width_m - envelope.width_m, 3)
        ground_margin_m = round(
            envelope.ground_clearance_m - self._config.ground_clearance_min_m, 3
        )

        # 1. Height check
        if envelope.height_m > self._config.portal_clearance_height_m:
            excess = envelope.height_m - self._config.portal_clearance_height_m
            msg = (
                f"Height {envelope.height_m:.2f}m exceeds portal limit "
                f"{self._config.portal_clearance_height_m:.2f}m by {excess:.2f}m"
            )
            return GateAccessDecision(
                decision_id=f"dec-{request_id}-height",
                request_id=request_id,
                status=AccessDecisionStatus.REJECTED_OVER_HEIGHT,
                granted=False,
                message=msg[:256],
                headroom_margin_m=headroom_margin_m,
                width_margin_m=width_margin_m,
            )

        # 2. Width check
        if envelope.width_m > self._config.portal_clearance_width_m:
            excess = envelope.width_m - self._config.portal_clearance_width_m
            msg = (
                f"Width {envelope.width_m:.2f}m exceeds portal limit "
                f"{self._config.portal_clearance_width_m:.2f}m by {excess:.2f}m"
            )
            return GateAccessDecision(
                decision_id=f"dec-{request_id}-width",
                request_id=request_id,
                status=AccessDecisionStatus.REJECTED_OVER_WIDTH,
                granted=False,
                message=msg[:256],
                headroom_margin_m=headroom_margin_m,
                width_margin_m=width_margin_m,
            )

        # 3. Ground clearance check
        if envelope.ground_clearance_m < self._config.ground_clearance_min_m:
            deficit = self._config.ground_clearance_min_m - envelope.ground_clearance_m
            msg = (
                f"Ground clearance {envelope.ground_clearance_m:.2f}m below required "
                f"{self._config.ground_clearance_min_m:.2f}m by {deficit:.2f}m"
            )
            return GateAccessDecision(
                decision_id=f"dec-{request_id}-clearance",
                request_id=request_id,
                status=AccessDecisionStatus.REJECTED_LOW_GROUND_CLEARANCE,
                granted=False,
                message=msg[:256],
                headroom_margin_m=headroom_margin_m,
                width_margin_m=width_margin_m,
            )

        # 4. Valid envelope
        msg = (
            f"Clearance verified: headroom={headroom_margin_m:+.2f}m, "
            f"width={width_margin_m:+.2f}m, ground={ground_margin_m:+.2f}m"
        )
        return GateAccessDecision(
            decision_id=f"dec-{request_id}-ok",
            request_id=request_id,
            status=AccessDecisionStatus.GRANTED,
            granted=True,
            message=msg[:256],
            headroom_margin_m=headroom_margin_m,
            width_margin_m=width_margin_m,
            issued_pass_id=f"PASS-CLEARANCE-{gate_id}-{request_id}",
        )
