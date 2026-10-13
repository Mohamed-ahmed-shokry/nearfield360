"""Unit tests for itemized billing calculator, cryptographic HMAC signing, and tamper detection."""

from __future__ import annotations

from nearfield360.billing.calculator import BillingCalculator
from nearfield360.billing.models import (
    BillingCategory,
    PaymentStatus,
    TariffTier,
)
from nearfield360.config import BillingConfig


def test_billing_calculator_grace_period_session() -> None:
    config = BillingConfig(
        base_hourly_rate=5.0,
        grace_period_minutes=15.0,
        valet_service_fee=10.0,
        tax_rate=0.08,
    )
    calculator = BillingCalculator(config)

    # Short session under 15 mins with valet fee disabled
    receipt = calculator.calculate_session_charges(
        vehicle_id="VEH-GRACE",
        entry_timestamp=1000.0,
        exit_timestamp=1600.0,  # 10 minutes
        include_valet_fee=False,
    )

    assert receipt.duration_minutes == 10.0
    assert len(receipt.items) == 0
    assert receipt.subtotal == 0.0
    assert receipt.total_charged == 0.0
    assert receipt.payment_status == PaymentStatus.SETTLED
    assert calculator.verify_receipt(receipt) is True


def test_billing_calculator_full_session_with_ev_and_overstay() -> None:
    config = BillingConfig(
        base_hourly_rate=10.0,
        peak_multiplier=1.5,
        kwh_tariff=0.40,
        idle_overstay_per_min=0.50,
        valet_service_fee=15.0,
        tax_rate=0.10,
    )
    calculator = BillingCalculator(config)

    # 2 hour session (120 minutes) at high congestion (80% occupancy -> 1.5x)
    receipt = calculator.calculate_session_charges(
        vehicle_id="VEH-EV-01",
        entry_timestamp=0.0,
        exit_timestamp=7200.0,
        kwh_consumed=25.0,
        idle_overstay_minutes=20.0,
        include_valet_fee=True,
        occupancy_ratio=0.80,
        receipt_id="RCP-TEST-001",
    )

    assert receipt.receipt_id == "RCP-TEST-001"
    assert receipt.tariff_tier == TariffTier.PEAK_CONGESTION
    assert len(receipt.items) == 4

    # Parking: 2 hrs @ $15/hr = $30
    parking_item = next(i for i in receipt.items if i.category == BillingCategory.PARKING_TIME)
    assert parking_item.amount == 30.0

    # EV Energy: 25 kWh @ $0.40 = $10
    energy_item = next(i for i in receipt.items if i.category == BillingCategory.EV_ENERGY)
    assert energy_item.amount == 10.0

    # Overstay: 20 mins @ $0.50 = $10
    overstay_item = next(i for i in receipt.items if i.category == BillingCategory.IDLE_OVERSTAY)
    assert overstay_item.amount == 10.0

    # Valet: $15
    valet_item = next(i for i in receipt.items if i.category == BillingCategory.VALET_SERVICE)
    assert valet_item.amount == 15.0

    # Subtotal = 30 + 10 + 10 + 15 = 65.0, Tax 10% = 6.50, Total = 71.50
    assert receipt.subtotal == 65.0
    assert receipt.tax == 6.50
    assert receipt.total_charged == 71.50

    # Cryptographic verification
    assert calculator.verify_receipt(receipt) is True


def test_billing_tamper_detection() -> None:
    config = BillingConfig()
    calculator = BillingCalculator(config)

    receipt = calculator.calculate_session_charges(
        vehicle_id="VEH-TAMPER",
        entry_timestamp=0.0,
        exit_timestamp=3600.0,
    )

    assert calculator.verify_receipt(receipt) is True

    # Tamper with total_charged (e.g. fraudulent discount)
    tampered_receipt = receipt.model_copy(update={"total_charged": 0.01})
    assert calculator.verify_receipt(tampered_receipt) is False

    # Tamper with vehicle_id
    tampered_vehicle = receipt.model_copy(update={"vehicle_id": "VEH-FORGED"})
    assert calculator.verify_receipt(tampered_vehicle) is False
