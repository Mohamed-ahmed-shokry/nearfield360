"""Unit tests for BEV multi-tenant partition and SLA telemetry visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2

from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.tenant.models import (
    AccessPermission,
    TenantPartition,
    TenantPartitionReport,
    TenantSLAContract,
    TenantTier,
    VIPExpressCorridor,
)
from nearfield360.tenant.viz import render_tenant_dashboard


def test_render_tenant_dashboard_memory() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    partition = TenantPartition(
        tenant_id="uber-fleet",
        tenant_name="Uber Technologies",
        tier=TenantTier.COMMERCIAL_FLEET,
        assigned_slot_ids=[facility.slots[0].slot_id, facility.slots[1].slot_id],
        dedicated_bay_quota=2,
        current_occupancy=1,
        active_vehicle_ids=["veh-01"],
        access_permission=AccessPermission.PRIORITY_ACCESS,
    )
    sla = TenantSLAContract(
        contract_id="sla-uber",
        tenant_id="uber-fleet",
        max_dispatch_latency_s=120.0,
        recorded_dispatches_count=10,
        breach_count=1,
        total_penalty_incurred=5.0,
    )
    corridor = VIPExpressCorridor(
        corridor_id="express-aisle-1",
        lane_ids=[facility.lanes[0].lane_id],
        active_vehicle_id="vip-exec-01",
        reservation_start_s=0.0,
        duration_s=60.0,
        is_preempting=True,
    )
    report = TenantPartitionReport(
        facility_id=facility.map_id,
        timestamp_s=12.5,
        total_slots_count=len(facility.slots),
        total_partitions_count=1,
        partitions=[partition],
        sla_contracts=[sla],
        express_corridors=[corridor],
        overall_sla_compliance=0.90,
        total_breach_penalties=5.0,
    )

    img = render_tenant_dashboard(facility, report, canvas_size=800)
    assert img is not None
    assert img.shape == (800, 800 + 440, 3)
    assert img.dtype.name == "uint8"


def test_render_tenant_dashboard_file_export(tmp_path: Path) -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2)
    report = TenantPartitionReport(
        facility_id=facility.map_id,
        timestamp_s=0.0,
        total_slots_count=len(facility.slots),
        total_partitions_count=0,
        partitions=[],
        sla_contracts=[],
        express_corridors=[],
        overall_sla_compliance=1.0,
        total_breach_penalties=0.0,
    )

    out_file = tmp_path / "tenant_hud.png"
    img = render_tenant_dashboard(facility, report, canvas_size=600, output_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == img.shape
