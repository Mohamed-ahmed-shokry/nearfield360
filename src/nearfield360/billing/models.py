"""Domain data models for autonomous valet parking smart billing, dynamic tariffs, and receipts."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TariffTier(StrEnum):
    """Classification of dynamic billing rate schedules."""

    STANDARD = "standard"
    PEAK_CONGESTION = "peak_congestion"
    EV_PREMIUM = "ev_premium"
    OVERNIGHT_DISCOUNT = "overnight_discount"


class PaymentStatus(StrEnum):
    """Settlement status for an issued parking session bill."""

    PENDING = "pending"
    SETTLED = "settled"
    WAIVED = "waived"
    DISPUTED = "disputed"


class BillingCategory(StrEnum):
    """Line item fee categorization."""

    PARKING_TIME = "parking_time"
    EV_ENERGY = "ev_energy"
    IDLE_OVERSTAY = "idle_overstay"
    VALET_SERVICE = "valet_service"


class BillingItem(BaseModel):
    """Individual line item charge on a parking session receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1)
    category: BillingCategory
    quantity: float = Field(ge=0.0, allow_inf_nan=False)
    unit_price: float = Field(ge=0.0, allow_inf_nan=False)
    amount: float = Field(ge=0.0, allow_inf_nan=False)

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: float) -> float:
        return round(value, 2)


class ParkingReceipt(BaseModel):
    """Cryptographically verifiable receipt for an autonomous valet parking session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    entry_timestamp: float = Field(ge=0.0, allow_inf_nan=False)
    exit_timestamp: float = Field(ge=0.0, allow_inf_nan=False)
    duration_minutes: float = Field(ge=0.0, allow_inf_nan=False)
    items: list[BillingItem] = Field(default_factory=list)
    subtotal: float = Field(ge=0.0, allow_inf_nan=False)
    tax: float = Field(ge=0.0, allow_inf_nan=False)
    total_charged: float = Field(ge=0.0, allow_inf_nan=False)
    tariff_tier: TariffTier = TariffTier.STANDARD
    payment_status: PaymentStatus = PaymentStatus.PENDING
    signature_digest: str = Field(min_length=16)

    @field_validator("subtotal", "tax", "total_charged")
    @classmethod
    def round_currency(cls, value: float) -> float:
        return round(value, 2)

    def to_canonical_json(self) -> str:
        """Produce canonical deterministic JSON payload excluding signature for hashing."""
        payload = {
            "receipt_id": self.receipt_id,
            "vehicle_id": self.vehicle_id,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "duration_minutes": self.duration_minutes,
            "items": [
                {
                    "description": item.description,
                    "category": item.category.value,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "amount": item.amount,
                }
                for item in self.items
            ],
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total_charged": self.total_charged,
            "tariff_tier": self.tariff_tier.value,
            "payment_status": self.payment_status.value,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class FinancialSettlementReport(BaseModel):
    """Comprehensive financial summary and revenue settlement for the parking facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(default="AVP-FACILITY-01", min_length=1)
    settlement_timestamp: float = Field(ge=0.0, allow_inf_nan=False)
    total_receipts_count: int = Field(ge=0)
    total_revenue: float = Field(ge=0.0, allow_inf_nan=False)
    parking_revenue: float = Field(ge=0.0, allow_inf_nan=False)
    energy_revenue: float = Field(ge=0.0, allow_inf_nan=False)
    penalty_revenue: float = Field(ge=0.0, allow_inf_nan=False)
    valet_service_revenue: float = Field(ge=0.0, allow_inf_nan=False)
    tax_collected: float = Field(ge=0.0, allow_inf_nan=False)
    average_dwell_minutes: float = Field(ge=0.0, allow_inf_nan=False)
    settled_receipts_count: int = Field(ge=0)
    pending_receipts_count: int = Field(ge=0)
    receipts: list[ParkingReceipt] = Field(default_factory=list)

    @field_validator(
        "total_revenue",
        "parking_revenue",
        "energy_revenue",
        "penalty_revenue",
        "valet_service_revenue",
        "tax_collected",
    )
    @classmethod
    def round_revenue(cls, value: float) -> float:
        return round(value, 2)
