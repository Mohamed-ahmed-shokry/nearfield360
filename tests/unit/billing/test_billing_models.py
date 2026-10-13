"""Unit tests for billing domain models, enum values, and canonical serialization."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from nearfield360.billing.models import (
    BillingCategory,
    BillingItem,
    FinancialSettlementReport,
    ParkingReceipt,
    PaymentStatus,
    TariffTier,
)


def test_tariff_tier_and_billing_category_enums() -> None:
    assert TariffTier.STANDARD == "standard"
    assert TariffTier.PEAK_CONGESTION == "peak_congestion"
    assert TariffTier.EV_PREMIUM == "ev_premium"
    assert TariffTier.OVERNIGHT_DISCOUNT == "overnight_discount"

    assert BillingCategory.PARKING_TIME == "parking_time"
    assert BillingCategory.EV_ENERGY == "ev_energy"
    assert BillingCategory.IDLE_OVERSTAY == "idle_overstay"
    assert BillingCategory.VALET_SERVICE == "valet_service"

    assert PaymentStatus.PENDING == "pending"
    assert PaymentStatus.SETTLED == "settled"


def test_billing_item_rounding_and_validation() -> None:
    item = BillingItem(
        description="Standard Parking (2.5 hrs)",
        category=BillingCategory.PARKING_TIME,
        quantity=2.5,
        unit_price=5.0,
        amount=12.503,
    )
    assert item.amount == 12.50

    with pytest.raises(ValidationError):
        BillingItem(
            description="",  # min_length=1
            category=BillingCategory.PARKING_TIME,
            quantity=1.0,
            unit_price=5.0,
            amount=5.0,
        )


def test_parking_receipt_and_canonical_json() -> None:
    item1 = BillingItem(
        description="Standard Parking (1.0 hr)",
        category=BillingCategory.PARKING_TIME,
        quantity=1.0,
        unit_price=5.0,
        amount=5.0,
    )
    item2 = BillingItem(
        description="EV Energy Delivered (20 kWh)",
        category=BillingCategory.EV_ENERGY,
        quantity=20.0,
        unit_price=0.35,
        amount=7.0,
    )

    receipt = ParkingReceipt(
        receipt_id="RCP-001",
        vehicle_id="VEH-42",
        entry_timestamp=1000.0,
        exit_timestamp=4600.0,
        duration_minutes=60.0,
        items=[item1, item2],
        subtotal=12.0,
        tax=0.96,
        total_charged=12.96,
        tariff_tier=TariffTier.STANDARD,
        payment_status=PaymentStatus.SETTLED,
        signature_digest="abcdef0123456789abcdef",
    )

    canonical = receipt.to_canonical_json()
    parsed = json.loads(canonical)
    assert parsed["receipt_id"] == "RCP-001"
    assert parsed["vehicle_id"] == "VEH-42"
    assert parsed["total_charged"] == 12.96
    assert len(parsed["items"]) == 2
    # Ensure signature digest is omitted from canonical payload for cryptographic hashing
    assert "signature_digest" not in parsed


def test_financial_settlement_report() -> None:
    report = FinancialSettlementReport(
        facility_id="AVP-FACILITY-01",
        settlement_timestamp=5000.0,
        total_receipts_count=1,
        total_revenue=12.96,
        parking_revenue=5.0,
        energy_revenue=7.0,
        penalty_revenue=0.0,
        valet_service_revenue=0.0,
        tax_collected=0.96,
        average_dwell_minutes=60.0,
        settled_receipts_count=1,
        pending_receipts_count=0,
    )
    assert report.total_revenue == 12.96
    assert report.settled_receipts_count == 1
