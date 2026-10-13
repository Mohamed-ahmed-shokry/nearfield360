"""Unit tests for SettlementAggregator and financial report generation."""

from __future__ import annotations

from nearfield360.billing.models import (
    BillingCategory,
    BillingItem,
    ParkingReceipt,
    PaymentStatus,
    TariffTier,
)
from nearfield360.billing.settlement import SettlementAggregator


def test_settlement_aggregator_empty() -> None:
    aggregator = SettlementAggregator("FACILITY-TEST")
    report = aggregator.generate_settlement_report([], settlement_timestamp=1000.0)

    assert report.facility_id == "FACILITY-TEST"
    assert report.total_receipts_count == 0
    assert report.total_revenue == 0.0
    assert report.settled_receipts_count == 0


def test_settlement_aggregator_multi_receipts() -> None:
    aggregator = SettlementAggregator("FACILITY-TEST")

    r1 = ParkingReceipt(
        receipt_id="RCP-1",
        vehicle_id="VEH-1",
        entry_timestamp=0.0,
        exit_timestamp=3600.0,
        duration_minutes=60.0,
        items=[
            BillingItem(
                description="Parking",
                category=BillingCategory.PARKING_TIME,
                quantity=1.0,
                unit_price=10.0,
                amount=10.0,
            ),
            BillingItem(
                description="EV Energy",
                category=BillingCategory.EV_ENERGY,
                quantity=20.0,
                unit_price=0.35,
                amount=7.0,
            ),
        ],
        subtotal=17.0,
        tax=1.36,
        total_charged=18.36,
        tariff_tier=TariffTier.STANDARD,
        payment_status=PaymentStatus.SETTLED,
        signature_digest="a" * 64,
    )

    r2 = ParkingReceipt(
        receipt_id="RCP-2",
        vehicle_id="VEH-2",
        entry_timestamp=100.0,
        exit_timestamp=7300.0,
        duration_minutes=120.0,
        items=[
            BillingItem(
                description="Parking",
                category=BillingCategory.PARKING_TIME,
                quantity=2.0,
                unit_price=10.0,
                amount=20.0,
            ),
            BillingItem(
                description="Overstay Fine",
                category=BillingCategory.IDLE_OVERSTAY,
                quantity=10.0,
                unit_price=0.50,
                amount=5.0,
            ),
            BillingItem(
                description="Valet Service",
                category=BillingCategory.VALET_SERVICE,
                quantity=1.0,
                unit_price=10.0,
                amount=10.0,
            ),
        ],
        subtotal=35.0,
        tax=2.80,
        total_charged=37.80,
        tariff_tier=TariffTier.PEAK_CONGESTION,
        payment_status=PaymentStatus.PENDING,
        signature_digest="b" * 64,
    )

    report = aggregator.generate_settlement_report([r1, r2], settlement_timestamp=8000.0)

    assert report.total_receipts_count == 2
    assert report.total_revenue == round(18.36 + 37.80, 2)
    assert report.parking_revenue == 30.0  # 10 + 20
    assert report.energy_revenue == 7.0
    assert report.penalty_revenue == 5.0
    assert report.valet_service_revenue == 10.0
    assert report.tax_collected == round(1.36 + 2.80, 2)
    assert report.average_dwell_minutes == 90.0  # (60 + 120) / 2
    assert report.settled_receipts_count == 1
    assert report.pending_receipts_count == 1
    assert len(report.receipts) == 2
