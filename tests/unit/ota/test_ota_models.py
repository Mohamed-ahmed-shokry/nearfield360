"""Unit tests for UN R156 SUMS and firmware update domain models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nearfield360.ota.models import (
    ECUPartitionState,
    FirmwareManifest,
    OTACampaignReport,
    OTACampaignStepRecord,
    PartitionSlot,
    PreflightCheck,
    PreflightStatus,
    SlotMetadata,
    UpdateState,
)


def test_partition_slot_opposite() -> None:
    assert PartitionSlot.SLOT_A.opposite() == PartitionSlot.SLOT_B
    assert PartitionSlot.SLOT_B.opposite() == PartitionSlot.SLOT_A


def test_firmware_manifest_validation() -> None:
    manifest = FirmwareManifest(
        component_id="ecu_planner",
        version="v2.1.0",
        monotonic_revision=5,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="trajectory_planner",
        payload_size_bytes=1024 * 1024 * 4,
        sha256_hash="a" * 64,
        signature_hex="b" * 64,
    )
    assert manifest.component_id == "ecu_planner"
    assert manifest.monotonic_revision == 5
    assert manifest.rxswin.startswith("RXSWIN")

    # Invalid sha256 hash length
    with pytest.raises(ValidationError):
        FirmwareManifest(
            component_id="ecu_planner",
            version="v2.1.0",
            monotonic_revision=5,
            rxswin="RXSWIN-E4-156R00-NF360-0042",
            target_ecu="trajectory_planner",
            payload_size_bytes=1024,
            sha256_hash="short_hash",
            signature_hex="b" * 64,
        )

    # Monotonic revision < 1 rejection
    with pytest.raises(ValidationError):
        FirmwareManifest(
            component_id="ecu_planner",
            version="v2.1.0",
            monotonic_revision=0,
            rxswin="RXSWIN-E4-156R00-NF360-0042",
            target_ecu="trajectory_planner",
            payload_size_bytes=1024,
            sha256_hash="a" * 64,
            signature_hex="b" * 64,
        )


def test_preflight_check_bounds() -> None:
    check = PreflightCheck(
        gear_is_park=True,
        speed_mps=0.0,
        battery_soc_pct=85.0,
        safety_supervisor_ready=True,
        status=PreflightStatus.PASSED,
        reasons=["All preflight interlocks verified."],
    )
    assert check.status == PreflightStatus.PASSED
    assert check.gear_is_park is True

    # Battery SoC > 100 rejection
    with pytest.raises(ValidationError):
        PreflightCheck(
            gear_is_park=True,
            speed_mps=0.0,
            battery_soc_pct=110.0,
            safety_supervisor_ready=True,
            status=PreflightStatus.PASSED,
        )


def test_ota_campaign_report_json_roundtrip() -> None:
    manifest = FirmwareManifest(
        component_id="ecu_gateway",
        version="v3.0.0",
        monotonic_revision=10,
        rxswin="RXSWIN-E4-156R00-NF360-0099",
        target_ecu="can_gateway",
        payload_size_bytes=512 * 1024,
        sha256_hash="c" * 64,
        signature_hex="d" * 64,
    )
    preflight = PreflightCheck(
        gear_is_park=True,
        speed_mps=0.0,
        battery_soc_pct=65.0,
        safety_supervisor_ready=True,
        status=PreflightStatus.PASSED,
    )
    partition = ECUPartitionState(
        ecu_id="can_gateway",
        active_slot=PartitionSlot.SLOT_A,
        next_boot_slot=PartitionSlot.SLOT_B,
        slots={
            "slot_a": SlotMetadata(slot=PartitionSlot.SLOT_A, is_active=True, version="v2.9.0"),
            "slot_b": SlotMetadata(slot=PartitionSlot.SLOT_B, is_active=False, version="v3.0.0"),
        },
    )
    report = OTACampaignReport(
        campaign_id="CAMP-2026-001",
        target_ecu="can_gateway",
        active_state=UpdateState.COMMITTED,
        manifest=manifest,
        preflight=preflight,
        partition_state=partition,
        audit_trail=[
            OTACampaignStepRecord(
                from_state=UpdateState.IDLE,
                to_state=UpdateState.VERIFYING,
                message="Payload received",
                timestamp_s=1.0,
            ),
            OTACampaignStepRecord(
                from_state=UpdateState.VERIFYING,
                to_state=UpdateState.COMMITTED,
                message="Update verified and committed",
                timestamp_s=5.0,
            ),
        ],
        success=True,
        duration_s=4.0,
    )

    serialized = report.model_dump_json()
    assert "CAMP-2026-001" in serialized
    assert "RXSWIN" in serialized

    restored = OTACampaignReport.model_validate(json.loads(serialized))
    assert restored.campaign_id == report.campaign_id
    assert restored.active_state == UpdateState.COMMITTED
    assert restored.manifest.sha256_hash == report.manifest.sha256_hash
    assert len(restored.audit_trail) == 2
