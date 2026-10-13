"""Itemized billing calculator, cryptographic SHA-256 HMAC receipt signing and verification."""

from __future__ import annotations

import hashlib
import hmac
import uuid

from nearfield360.billing.models import (
    BillingCategory,
    BillingItem,
    ParkingReceipt,
    PaymentStatus,
)
from nearfield360.billing.tariffs import TariffEngine
from nearfield360.config import BillingConfig


class BillingCalculator:
    """Computes itemized charges and cryptographically signs parking receipts."""

    def __init__(
        self,
        config: BillingConfig | None = None,
        tariff_engine: TariffEngine | None = None,
    ) -> None:
        self.config = config or BillingConfig()
        self.tariff_engine = tariff_engine or TariffEngine(self.config)

    def calculate_session_charges(
        self,
        vehicle_id: str,
        entry_timestamp: float,
        exit_timestamp: float,
        kwh_consumed: float = 0.0,
        idle_overstay_minutes: float = 0.0,
        include_valet_fee: bool = True,
        occupancy_ratio: float = 0.50,
        is_peak_hours: bool = False,
        is_overnight: bool = False,
        receipt_id: str | None = None,
    ) -> ParkingReceipt:
        """Calculate complete itemized parking receipt."""
        duration_minutes = max(0.0, (exit_timestamp - entry_timestamp) / 60.0)
        billable_minutes = self.tariff_engine.evaluate_billable_minutes(duration_minutes)

        items: list[BillingItem] = []

        # 1. Base parking time
        effective_rate, tier = self.tariff_engine.calculate_effective_hourly_rate(
            occupancy_ratio=occupancy_ratio,
            is_peak_hours=is_peak_hours,
            is_overnight=is_overnight,
        )

        if billable_minutes > 0.0:
            billable_hours = round(billable_minutes / 60.0, 2)
            parking_amount = round(billable_hours * effective_rate, 2)
            desc = f"Parking Dwell ({billable_hours:.2f} hrs @ ${effective_rate:.2f}/hr)"
            items.append(
                BillingItem(
                    description=desc,
                    category=BillingCategory.PARKING_TIME,
                    quantity=billable_hours,
                    unit_price=effective_rate,
                    amount=parking_amount,
                )
            )

        # 2. EV energy consumed
        if kwh_consumed > 0.0:
            energy_rate = self.config.kwh_tariff
            energy_amount = round(kwh_consumed * energy_rate, 2)
            desc = f"EV Charging Energy ({kwh_consumed:.2f} kWh @ ${energy_rate:.2f}/kWh)"
            items.append(
                BillingItem(
                    description=desc,
                    category=BillingCategory.EV_ENERGY,
                    quantity=round(kwh_consumed, 2),
                    unit_price=energy_rate,
                    amount=energy_amount,
                )
            )

        # 3. Charging idle overstay penalty
        if idle_overstay_minutes > 0.0:
            overstay_rate = self.config.idle_overstay_per_min
            overstay_amount = round(idle_overstay_minutes * overstay_rate, 2)
            desc = (
                f"Charging Bay Overstay Fine ({idle_overstay_minutes:.1f} mins "
                f"@ ${overstay_rate:.2f}/min)"
            )
            items.append(
                BillingItem(
                    description=desc,
                    category=BillingCategory.IDLE_OVERSTAY,
                    quantity=round(idle_overstay_minutes, 1),
                    unit_price=overstay_rate,
                    amount=overstay_amount,
                )
            )

        # 4. Valet service fee
        if include_valet_fee:
            valet_fee = self.config.valet_service_fee
            items.append(
                BillingItem(
                    description="Autonomous Valet Facility Concierge Fee",
                    category=BillingCategory.VALET_SERVICE,
                    quantity=1.0,
                    unit_price=valet_fee,
                    amount=valet_fee,
                )
            )

        subtotal = round(sum(item.amount for item in items), 2)
        tax = round(subtotal * self.config.tax_rate, 2)
        total_charged = round(subtotal + tax, 2)

        rid = receipt_id or f"RCP-{uuid.uuid4().hex[:8].upper()}"

        # Temporary receipt without signature to produce canonical JSON
        temp_receipt = ParkingReceipt(
            receipt_id=rid,
            vehicle_id=vehicle_id,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            duration_minutes=round(duration_minutes, 1),
            items=items,
            subtotal=subtotal,
            tax=tax,
            total_charged=total_charged,
            tariff_tier=tier,
            payment_status=PaymentStatus.SETTLED if total_charged == 0.0 else PaymentStatus.PENDING,
            signature_digest="0" * 64,
        )

        signature = self.compute_signature(temp_receipt)

        return ParkingReceipt(
            receipt_id=temp_receipt.receipt_id,
            vehicle_id=temp_receipt.vehicle_id,
            entry_timestamp=temp_receipt.entry_timestamp,
            exit_timestamp=temp_receipt.exit_timestamp,
            duration_minutes=temp_receipt.duration_minutes,
            items=temp_receipt.items,
            subtotal=temp_receipt.subtotal,
            tax=temp_receipt.tax,
            total_charged=temp_receipt.total_charged,
            tariff_tier=temp_receipt.tariff_tier,
            payment_status=temp_receipt.payment_status,
            signature_digest=signature,
        )

    def compute_signature(self, receipt: ParkingReceipt) -> str:
        """Compute HMAC-SHA256 signature for canonical receipt JSON."""
        secret_bytes = self.config.receipt_hmac_secret.encode("utf-8")
        canonical_data = receipt.to_canonical_json().encode("utf-8")
        return hmac.new(secret_bytes, canonical_data, hashlib.sha256).hexdigest()

    def verify_receipt(self, receipt: ParkingReceipt) -> bool:
        """Verify authenticity and cryptographic integrity of a parking receipt."""
        expected = self.compute_signature(receipt)
        return hmac.compare_digest(expected, receipt.signature_digest)
