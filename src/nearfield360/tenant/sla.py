"""Fleet operator SLA monitor and breach penalty auditor."""

from __future__ import annotations

from nearfield360.tenant.models import TenantSLAContract


class SLAMonitor:
    """Tracks dispatch turnaround latencies, evaluates SLA compliance, and computes penalties."""

    def __init__(self, contracts: list[TenantSLAContract] | None = None) -> None:
        self._contracts: dict[str, TenantSLAContract] = {}
        if contracts:
            for contract in contracts:
                self.register_contract(contract)

    @property
    def contracts(self) -> list[TenantSLAContract]:
        """Return all registered tenant SLA contracts."""
        return list(self._contracts.values())

    def register_contract(self, contract: TenantSLAContract) -> None:
        """Register or update an SLA contract for a tenant."""
        self._contracts[contract.tenant_id] = contract

    def get_contract(self, tenant_id: str) -> TenantSLAContract | None:
        """Retrieve SLA contract for a specific tenant ID."""
        return self._contracts.get(tenant_id)

    def record_dispatch(self, tenant_id: str, latency_s: float) -> tuple[bool, float]:
        """Record a dispatch latency event and calculate breach penalties.

        Returns:
            Tuple of (is_breach, penalty_incurred_usd).
        """
        contract = self._contracts.get(tenant_id)
        if contract is None:
            return False, 0.0

        is_breach = latency_s > contract.max_dispatch_latency_s
        penalty = 0.0

        if is_breach:
            excess_s = latency_s - contract.max_dispatch_latency_s
            excess_min = excess_s / 60.0
            penalty = round(excess_min * contract.breach_penalty_per_min, 2)
            updated = contract.model_copy(
                update={
                    "recorded_dispatches_count": contract.recorded_dispatches_count + 1,
                    "breach_count": contract.breach_count + 1,
                    "total_penalty_incurred": round(
                        contract.total_penalty_incurred + penalty, 2
                    ),
                }
            )
        else:
            updated = contract.model_copy(
                update={
                    "recorded_dispatches_count": contract.recorded_dispatches_count + 1,
                }
            )

        self._contracts[tenant_id] = updated
        return is_breach, penalty

    def compute_overall_compliance(self) -> float:
        """Compute aggregate SLA compliance rate across all registered contracts."""
        if not self._contracts:
            return 1.0

        total_dispatches = sum(c.recorded_dispatches_count for c in self._contracts.values())
        if total_dispatches == 0:
            return 1.0

        total_breaches = sum(c.breach_count for c in self._contracts.values())
        successful = max(0, total_dispatches - total_breaches)
        return round(successful / total_dispatches, 3)

    def compute_total_penalties(self) -> float:
        """Compute aggregate breach penalty charges across all registered contracts."""
        return round(
            sum(c.total_penalty_incurred for c in self._contracts.values()),
            2,
        )
