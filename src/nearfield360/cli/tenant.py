"""CLI commands for multi-tenant facility partitioning, SLA monitoring, and VIP corridors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nearfield360.cli.state import get_state
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.tenant.corridor import VIPCorridorManager
from nearfield360.tenant.manager import TenantManager
from nearfield360.tenant.models import (
    AccessPermission,
    TenantPartitionReport,
    TenantSLAContract,
    TenantTier,
)
from nearfield360.tenant.sla import SLAMonitor
from nearfield360.tenant.viz import render_tenant_dashboard
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

tenant_app = typer.Typer(
    help="Multi-tenant facility partitioning, SLA monitoring, and VIP corridors.",
    no_args_is_help=True,
)

MapPathOption = Annotated[
    Path | None,
    typer.Option(
        "--map-path",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to facility map JSON file (defaults to benchmark garage if omitted).",
    ),
]

OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        resolve_path=True,
        help="Path to write JSON artifact atomically.",
    ),
]

PngOption = Annotated[
    Path | None,
    typer.Option(
        "--png",
        resolve_path=True,
        help="Path to render multi-tenant BEV telemetry PNG artifact.",
    ),
]


def _load_facility(map_path: Path | None) -> FacilityMap:
    """Load facility map from JSON path or default to benchmark garage."""
    if map_path is not None:
        try:
            data = read_json(map_path)
            return FacilityMap.model_validate(data)
        except (ArtifactError, Exception) as exc:
            typer.echo(f"Error loading facility map from {map_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    return build_benchmark_garage(aisle_count=2, slots_per_aisle=4, occupied_slot_indices=(1, 5))


@tenant_app.command("info")
def tenant_info(
    context: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
) -> None:
    """Display active facility multi-tenant configuration, quotas, and SLA defaults."""
    cfg = get_state(context).config.tenant

    info_payload = {
        "max_tenant_partitions": cfg.max_tenant_partitions,
        "default_quota_ratio": cfg.default_quota_ratio,
        "sla_max_dispatch_wait_s": cfg.sla_max_dispatch_wait_s,
        "sla_breach_penalty_per_min": cfg.sla_breach_penalty_per_min,
        "vip_priority_multiplier": cfg.vip_priority_multiplier,
        "express_preemption_buffer_s": cfg.express_preemption_buffer_s,
    }

    if output is not None:
        write_json(output, info_payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(info_payload, indent=2))
        return

    typer.echo("Multi-Tenant Facility Partitioning & SLA Configuration:")
    typer.echo(f"  Max Partitions:             {cfg.max_tenant_partitions}")
    typer.echo(f"  Default Quota Ratio:        {cfg.default_quota_ratio * 100:.1f}%")
    typer.echo(f"  Guaranteed SLA Latency:     {cfg.sla_max_dispatch_wait_s:.1f} s")
    typer.echo(f"  Breach Penalty Rate:        ${cfg.sla_breach_penalty_per_min:.2f}/min")
    typer.echo(f"  VIP Priority Multiplier:    {cfg.vip_priority_multiplier:.1f}x")
    typer.echo(f"  Preemption Buffer:          {cfg.express_preemption_buffer_s:.1f} s")


@tenant_app.command("partition")
def tenant_partition(
    context: typer.Context,
    tenant_id: Annotated[str, typer.Option("--tenant-id", help="Tenant ID.")] = "fleet-uber",
    tenant_name: Annotated[str, typer.Option("--name", help="Tenant name.")] = "Uber Rides",
    tier: Annotated[TenantTier, typer.Option("--tier", help="Tenant tier.")] = (
        TenantTier.COMMERCIAL_FLEET
    ),
    quota: Annotated[int, typer.Option("--quota", help="Bay quota.")] = 4,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Create or inspect a tenant partition and allocate facility bays."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.tenant
    mgr = TenantManager(facility_map=facility, config=cfg)

    # Allocate first N slots up to quota
    assigned_slots = [s.slot_id for s in facility.slots[:quota]]
    partition = mgr.create_partition(
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        tier=tier,
        slot_ids=assigned_slots,
        quota=quota,
        access_permission=AccessPermission.PRIORITY_ACCESS,
    )

    report = TenantPartitionReport(
        facility_id=facility.map_id,
        timestamp_s=0.0,
        total_slots_count=len(facility.slots),
        total_partitions_count=len(mgr.partitions),
        partitions=mgr.partitions,
        sla_contracts=[],
        express_corridors=[],
        overall_sla_compliance=1.0,
        total_breach_penalties=0.0,
    )

    if png is not None:
        render_tenant_dashboard(facility, report, output_path=png)

    payload = partition.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Tenant Partition Allocated:")
    typer.echo(f"  Tenant ID:   {partition.tenant_id} ({partition.tenant_name})")
    typer.echo(f"  Tier:        {partition.tier.value}")
    typer.echo(f"  Quota:       {partition.dedicated_bay_quota} bays")
    typer.echo(f"  Assigned:    {', '.join(partition.assigned_slot_ids)}")


@tenant_app.command("corridor")
def tenant_corridor(
    corridor_id: Annotated[str, typer.Option("--corridor-id", help="Corridor ID.")] = "vip-lane-1",
    vehicle_id: Annotated[str, typer.Option("--vehicle-id", help="Vehicle ID.")] = "vip-exec-01",
    duration: Annotated[float, typer.Option("--duration", help="Reservation seconds.")] = 60.0,
    priority: Annotated[int, typer.Option("--priority", help="Priority level (1-10).")] = 2,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Reserve a dedicated VIP express corridor through facility driving lanes."""
    facility = _load_facility(map_path)
    corridor_mgr = VIPCorridorManager()

    lane_ids = [lane.lane_id for lane in facility.lanes[:2]] if facility.lanes else ["lane_default"]
    corridor_mgr.register_corridor(corridor_id, lane_ids, default_duration_s=duration)
    reserved = corridor_mgr.reserve_corridor(
        corridor_id=corridor_id,
        vehicle_id=vehicle_id,
        start_time_s=0.0,
        duration_s=duration,
        priority_level=priority,
    )

    report = TenantPartitionReport(
        facility_id=facility.map_id,
        timestamp_s=0.0,
        total_slots_count=len(facility.slots),
        total_partitions_count=0,
        partitions=[],
        sla_contracts=[],
        express_corridors=[reserved],
        overall_sla_compliance=1.0,
        total_breach_penalties=0.0,
    )

    if png is not None:
        render_tenant_dashboard(facility, report, output_path=png)

    payload = reserved.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("VIP Express Corridor Reserved:")
    typer.echo(f"  Corridor ID:   {reserved.corridor_id}")
    typer.echo(f"  Vehicle ID:    {reserved.active_vehicle_id}")
    typer.echo(f"  Priority:      {reserved.priority_level}")
    typer.echo(f"  Lanes:         {', '.join(reserved.lane_ids)}")
    typer.echo(f"  Duration:      {reserved.duration_s:.1f} s")
    typer.echo(f"  Preemption:    {reserved.is_preempting}")


@tenant_app.command("audit")
def tenant_audit(
    context: typer.Context,
    map_path: MapPathOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit raw JSON output.")] = False,
    output: OutputOption = None,
    png: PngOption = None,
) -> None:
    """Perform a comprehensive facility multi-tenant partition and SLA audit."""
    facility = _load_facility(map_path)
    cfg = get_state(context).config.tenant

    mgr = TenantManager(facility_map=facility, config=cfg)
    p1 = mgr.create_partition(
        tenant_id="fleet-uber",
        tenant_name="Uber Technologies",
        tier=TenantTier.COMMERCIAL_FLEET,
        slot_ids=[s.slot_id for s in facility.slots[:4]],
        quota=4,
    )
    p2 = mgr.create_partition(
        tenant_id="vip-guests",
        tenant_name="Grand Hotel VIPs",
        tier=TenantTier.VIP_EXECUTIVE,
        slot_ids=[s.slot_id for s in facility.slots[4:6]],
        quota=2,
        access_permission=AccessPermission.EXCLUSIVE_VIP,
    )

    # SLA contracts and dispatches
    sla1 = TenantSLAContract(
        contract_id="sla-uber-01",
        tenant_id="fleet-uber",
        max_dispatch_latency_s=cfg.sla_max_dispatch_wait_s,
        breach_penalty_per_min=cfg.sla_breach_penalty_per_min,
    )
    sla_mon = SLAMonitor([sla1])
    sla_mon.record_dispatch("fleet-uber", latency_s=cfg.sla_max_dispatch_wait_s * 0.8)
    sla_mon.record_dispatch("fleet-uber", latency_s=cfg.sla_max_dispatch_wait_s * 1.5)

    # Active express corridor
    c_mgr = VIPCorridorManager()
    lanes = [lane.lane_id for lane in facility.lanes[:1]] if facility.lanes else ["lane_1"]
    c_mgr.register_corridor("corridor-vip-aisle", lanes)
    c1 = c_mgr.reserve_corridor(
        "corridor-vip-aisle",
        vehicle_id="vip-guest-limo",
        start_time_s=0.0,
        duration_s=60.0,
        priority_level=2,
    )

    report = TenantPartitionReport(
        facility_id=facility.map_id,
        timestamp_s=10.0,
        total_slots_count=len(facility.slots),
        total_partitions_count=len(mgr.partitions),
        partitions=[p1, p2],
        sla_contracts=sla_mon.contracts,
        express_corridors=[c1],
        overall_sla_compliance=sla_mon.compute_overall_compliance(),
        total_breach_penalties=sla_mon.compute_total_penalties(),
    )

    if png is not None:
        render_tenant_dashboard(facility, report, output_path=png)

    payload = report.model_dump(mode="json")
    if output is not None:
        write_json(output, payload, overwrite=True)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Multi-Tenant Facility SLA & Partition Audit:")
    typer.echo(f"  Facility:        {report.facility_id}")
    typer.echo(f"  Total Slots:     {report.total_slots_count}")
    typer.echo(f"  Partitions:      {report.total_partitions_count}")
    typer.echo(f"  SLA Compliance:  {report.overall_sla_compliance * 100:.1f}%")
    typer.echo(f"  Total Penalties: ${report.total_breach_penalties:.2f}")
    typer.echo(f"  VIP Corridors:   {len(report.express_corridors)}")
