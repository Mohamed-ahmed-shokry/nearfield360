"""Pydantic domain models for facility access control, barrier gates, and vehicle enveloping."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GateDirection(StrEnum):
    """Direction of traffic transit allowed through the gateway."""

    ENTRY = "entry"
    EXIT = "exit"
    BIDIRECTIONAL = "bidirectional"


class BarrierArmState(StrEnum):
    """Operational state of the physical barrier boom arm."""

    LOWERED_CLOSED = "lowered_closed"
    OPENING = "opening"
    RAISED_OPEN = "raised_open"
    CLOSING = "closing"
    SAFETY_REVERSING = "safety_reversing"
    FAULT_LOCKED = "fault_locked"


class CredentialType(StrEnum):
    """Type of authentication credential presented at the gateway."""

    ALPR_PLATE = "alpr_plate"
    UWB_DIGITAL_KEY = "uwb_digital_key"
    RFID_BADGE = "rfid_badge"
    QR_BARCODE = "qr_barcode"
    V2X_DIRECT_AUTH = "v2x_direct_auth"


class AccessDecisionStatus(StrEnum):
    """Validation outcome for a gateway access request."""

    GRANTED = "granted"
    REJECTED_UNKNOWN_CREDENTIAL = "rejected_unknown_credential"
    REJECTED_OVER_HEIGHT = "rejected_over_height"
    REJECTED_OVER_WIDTH = "rejected_over_width"
    REJECTED_LOW_GROUND_CLEARANCE = "rejected_low_ground_clearance"
    REJECTED_EXPIRED_RESERVATION = "rejected_expired_reservation"
    TAILGATING_LOCKOUT = "tailgating_lockout"


class VehicleDimensionEnvelope(BaseModel):
    """3D bounding envelope and clearance measurements of a vehicle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    length_m: float = Field(gt=0.0, le=15.0, allow_inf_nan=False)
    width_m: float = Field(gt=0.0, le=4.0, allow_inf_nan=False)
    height_m: float = Field(gt=0.0, le=4.0, allow_inf_nan=False)
    ground_clearance_m: float = Field(default=0.15, ge=0.0, le=1.0, allow_inf_nan=False)
    front_overhang_m: float = Field(default=0.85, ge=0.0, le=3.0, allow_inf_nan=False)
    rear_overhang_m: float = Field(default=0.95, ge=0.0, le=3.0, allow_inf_nan=False)


class VehicleCredential(BaseModel):
    """Access credential associated with an authorized vehicle reservation or fleet account."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_id: str = Field(min_length=1)
    credential_type: CredentialType
    plate_number: str = Field(default="", max_length=16)
    token_hash: str = Field(default="", max_length=128)
    valid_from_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    valid_until_s: float = Field(default=86400.0, ge=0.0, allow_inf_nan=False)
    tenant_id: str = Field(default="PUBLIC", min_length=1)
    vip_status: bool = False
