"""Unit tests for BEV billing dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

from nearfield360.billing.models import (
    BillingCategory,
    BillingItem,
    ParkingReceipt,
    PaymentStatus,
    TariffTier,
)
from nearfield360.billing.settlement import SettlementAggregator
from nearfield360.billing.viz import render_billing_dashboard
from nearfield360.mapping.builder import build_benchmark_garage


def test_render_billing_dashboard(tmp_path: Path) -> None:
    facility = build_benchmark_garage()
    aggregator = SettlementAggregator()

    receipt = ParkingReceipt(
        receipt_id="RCP-VIZ-01",
        vehicle_id="VEH-VIZ",
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
                quantity=15.0,
                unit_price=0.35,
                amount=5.25,
            ),
        ],
        subtotal=15.25,
        tax=1.22,
        total_charged=16.47,
        tariff_tier=TariffTier.STANDARD,
        payment_status=PaymentStatus.SETTLED,
        signature_digest="a" * 64,
    )

    report = aggregator.generate_settlement_report([receipt], settlement_timestamp=5000.0)

    out_file = tmp_path / "billing_dashboard.png"
    canvas = render_billing_dashboard(
        facility=facility,
        report=report,
        canvas_size=600,
        output_path=out_file,
    )

    assert canvas.shape[0] == 600
    assert canvas.shape[1] == 600 + 420
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
