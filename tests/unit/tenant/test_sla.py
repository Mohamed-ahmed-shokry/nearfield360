"""Unit tests for SLAMonitor performance tracking and breach auditing."""

from __future__ import annotations

import pytest

from nearfield360.tenant.models import TenantSLAContract
from nearfield360.tenant.sla import SLAMonitor


@pytest.fixture
def sample_contract() -> TenantSLAContract:
    return TenantSLAContract(
        contract_id="sla-001",
        tenant_id="uber-fleet",
        max_dispatch_latency_s=120.0,
        target_availability_ratio=0.98,
        breach_penalty_per_min=3.0,
    )


def test_sla_monitor_init_and_registration(sample_contract: TenantSLAContract) -> None:
    monitor = SLAMonitor([sample_contract])
    assert len(monitor.contracts) == 1
    assert monitor.get_contract("uber-fleet") is not None
    assert monitor.get_contract("unknown") is None


def test_sla_monitor_compliant_dispatch(sample_contract: TenantSLAContract) -> None:
    monitor = SLAMonitor([sample_contract])
    # Latency 90s <= 120s limit
    is_breach, penalty = monitor.record_dispatch("uber-fleet", latency_s=90.0)
    assert is_breach is False
    assert penalty == 0.0

    c = monitor.get_contract("uber-fleet")
    assert c is not None
    assert c.recorded_dispatches_count == 1
    assert c.breach_count == 0
    assert c.total_penalty_incurred == 0.0
    assert c.compliance_rate == 1.0


def test_sla_monitor_breaching_dispatch(sample_contract: TenantSLAContract) -> None:
    monitor = SLAMonitor([sample_contract])
    # Latency 240s > 120s limit (120s excess = 2.0 min excess * $3.0/min = $6.0)
    is_breach, penalty = monitor.record_dispatch("uber-fleet", latency_s=240.0)
    assert is_breach is True
    assert penalty == 6.0

    c = monitor.get_contract("uber-fleet")
    assert c is not None
    assert c.recorded_dispatches_count == 1
    assert c.breach_count == 1
    assert c.total_penalty_incurred == 6.0
    assert c.compliance_rate == 0.0


def test_sla_monitor_unknown_tenant() -> None:
    monitor = SLAMonitor()
    is_breach, penalty = monitor.record_dispatch("nonexistent", 500.0)
    assert is_breach is False
    assert penalty == 0.0


def test_sla_monitor_aggregate_metrics() -> None:
    c1 = TenantSLAContract(
        contract_id="c1",
        tenant_id="t1",
        max_dispatch_latency_s=100.0,
        breach_penalty_per_min=2.0,
    )
    c2 = TenantSLAContract(
        contract_id="c2",
        tenant_id="t2",
        max_dispatch_latency_s=100.0,
        breach_penalty_per_min=4.0,
    )
    monitor = SLAMonitor([c1, c2])

    # Initial compliance is 1.0, penalties 0.0
    assert monitor.compute_overall_compliance() == 1.0
    assert monitor.compute_total_penalties() == 0.0

    # t1: 1 success (80s), 1 breach (160s -> 60s excess = 1 min * $2 = $2)
    monitor.record_dispatch("t1", 80.0)
    monitor.record_dispatch("t1", 160.0)

    # t2: 2 successes (50s, 70s)
    monitor.record_dispatch("t2", 50.0)
    monitor.record_dispatch("t2", 70.0)

    # Total dispatches = 4, total breaches = 1 -> compliance = 3/4 = 0.75
    assert monitor.compute_overall_compliance() == 0.75
    assert monitor.compute_total_penalties() == 2.0
