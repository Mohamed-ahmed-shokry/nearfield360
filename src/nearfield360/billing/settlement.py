"""Financial settlement aggregation and facility revenue analytics engine."""

from __future__ import annotations

import time
from collections.abc import Sequence

from nearfield360.billing.models import (
    BillingCategory,
    FinancialSettlementReport,
    ParkingReceipt,
    PaymentStatus,
)


class SettlementAggregator:
    """Aggregates multi-session receipts into audited financial settlement reports."""

    def __init__(self, facility_id: str = "AVP-FACILITY-01") -> None:
        self.facility_id = facility_id

    def generate_settlement_report(
        self,
        receipts: Sequence[ParkingReceipt],
        settlement_timestamp: float | None = None,
    ) -> FinancialSettlementReport:
        """Aggregate receipts into a unified financial report."""
        timestamp = time.time() if settlement_timestamp is None else settlement_timestamp
        total_receipts = len(receipts)

        if total_receipts == 0:
            return FinancialSettlementReport(
                facility_id=self.facility_id,
                settlement_timestamp=timestamp,
                total_receipts_count=0,
                total_revenue=0.0,
                parking_revenue=0.0,
                energy_revenue=0.0,
                penalty_revenue=0.0,
                valet_service_revenue=0.0,
                tax_collected=0.0,
                average_dwell_minutes=0.0,
                settled_receipts_count=0,
                pending_receipts_count=0,
                receipts=[],
            )

        total_revenue = 0.0
        parking_revenue = 0.0
        energy_revenue = 0.0
        penalty_revenue = 0.0
        valet_revenue = 0.0
        tax_collected = 0.0
        total_dwell_minutes = 0.0
        settled_count = 0
        pending_count = 0

        for r in receipts:
            total_revenue += r.total_charged
            tax_collected += r.tax
            total_dwell_minutes += r.duration_minutes
            if r.payment_status == PaymentStatus.SETTLED:
                settled_count += 1
            elif r.payment_status == PaymentStatus.PENDING:
                pending_count += 1

            for item in r.items:
                if item.category == BillingCategory.PARKING_TIME:
                    parking_revenue += item.amount
                elif item.category == BillingCategory.EV_ENERGY:
                    energy_revenue += item.amount
                elif item.category == BillingCategory.IDLE_OVERSTAY:
                    penalty_revenue += item.amount
                elif item.category == BillingCategory.VALET_SERVICE:
                    valet_revenue += item.amount

        avg_dwell = round(total_dwell_minutes / total_receipts, 1)

        return FinancialSettlementReport(
            facility_id=self.facility_id,
            settlement_timestamp=timestamp,
            total_receipts_count=total_receipts,
            total_revenue=round(total_revenue, 2),
            parking_revenue=round(parking_revenue, 2),
            energy_revenue=round(energy_revenue, 2),
            penalty_revenue=round(penalty_revenue, 2),
            valet_service_revenue=round(valet_revenue, 2),
            tax_collected=round(tax_collected, 2),
            average_dwell_minutes=avg_dwell,
            settled_receipts_count=settled_count,
            pending_receipts_count=pending_count,
            receipts=list(receipts),
        )
