"""Unit tests for tactical UN R156 OTA dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from nearfield360.ota.viz import render_ota_dashboard


def _build_sample_report(state: UpdateState = UpdateState.COMMITTED) -> OTACampaignReport:
    manifest = FirmwareManifest(
        component_id="ecu_gateway",
        version="v2.5.0",
        monotonic_revision=5,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="can_gateway",
        payload_size_bytes=256 * 1024,
        sha256_hash="a" * 64,
        signature_hex="b" * 64,
    )
    preflight = PreflightCheck(
        gear_is_park=True,
        speed_mps=0.0,
        battery_soc_pct=78.5,
        safety_supervisor_ready=True,
        status=PreflightStatus.PASSED,
        reasons=["Preflight interlocks nominal."],
    )
    partition = ECUPartitionState(
        ecu_id="can_gateway",
        active_slot=PartitionSlot.SLOT_B
        if state == UpdateState.COMMITTED
        else PartitionSlot.SLOT_A,
        next_boot_slot=PartitionSlot.SLOT_B,
        slots={
            "slot_a": SlotMetadata(
                slot=PartitionSlot.SLOT_A,
                is_active=(state != UpdateState.COMMITTED),
                version="v2.4.0",
            ),
            "slot_b": SlotMetadata(
                slot=PartitionSlot.SLOT_B,
                is_active=(state == UpdateState.COMMITTED),
                version="v2.5.0",
            ),
        },
        rollback_count=1 if state == UpdateState.ROLLED_BACK else 0,
    )
    return OTACampaignReport(
        campaign_id="CAMP-2026-TEST",
        target_ecu="can_gateway",
        active_state=state,
        manifest=manifest,
        preflight=preflight,
        partition_state=partition,
        audit_trail=[
            OTACampaignStepRecord(
                from_state=UpdateState.IDLE,
                to_state=UpdateState.DOWNLOADING,
                message="Download initiated",
                timestamp_s=1.0,
            ),
            OTACampaignStepRecord(
                from_state=UpdateState.DOWNLOADING,
                to_state=state,
                message=f"Campaign reached {state.value}",
                timestamp_s=5.0,
            ),
        ],
        success=(state == UpdateState.COMMITTED),
        duration_s=4.2,
    )


def test_render_ota_dashboard_committed(tmp_path: Path) -> None:
    report = _build_sample_report(state=UpdateState.COMMITTED)
    out_file = tmp_path / "ota_dashboard_committed.png"

    img = render_ota_dashboard(report, output_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_render_ota_dashboard_rolled_back(tmp_path: Path) -> None:
    report = _build_sample_report(state=UpdateState.ROLLED_BACK)
    out_file = tmp_path / "ota_dashboard_rolled_back.png"

    img = render_ota_dashboard(report, output_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
