"""Facility Access Control, Barrier Gate Interlocks, ALPR & Vehicle Clearance Enveloping."""

from __future__ import annotations

from nearfield360.gate.controller import BarrierInterlockController
from nearfield360.gate.envelope import VehicleEnvelopeProfiler
from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    BarrierKinematicsState,
    CredentialType,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    GateDirection,
    GateTransitEvent,
    InductiveLoopId,
    InductiveLoopState,
    VehicleCredential,
    VehicleDimensionEnvelope,
)
from nearfield360.gate.validator import AccessCredentialValidator
from nearfield360.gate.viz import render_gate_dashboard

__all__ = [
    "AccessCredentialValidator",
    "AccessDecisionStatus",
    "BarrierArmState",
    "BarrierInterlockController",
    "BarrierKinematicsState",
    "CredentialType",
    "GateAccessDecision",
    "GateAccessRequest",
    "GateAuditReport",
    "GateDirection",
    "GateTransitEvent",
    "InductiveLoopId",
    "InductiveLoopState",
    "VehicleCredential",
    "VehicleDimensionEnvelope",
    "VehicleEnvelopeProfiler",
    "render_gate_dashboard",
]
