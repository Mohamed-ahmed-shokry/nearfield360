"""Single-lane bidirectional ramp token arbiter and deadlock avoidance manager."""

from __future__ import annotations

from typing import NamedTuple

from nearfield360.multilevel.models import RampDirection, RampToken, RampTokenPriority


class QueueEntry(NamedTuple):
    """Waiting vehicle entry in a ramp access queue."""

    vehicle_id: str
    direction: RampDirection
    requested_at_s: float
    estimated_duration_s: float


class RampTokenArbiter:
    """Arbitrates exclusive access tokens for single-lane multi-floor transit ramps."""

    def __init__(
        self,
        default_policy: RampTokenPriority = RampTokenPriority.ASCENDING_FIRST,
        default_lease_timeout_s: float = 30.0,
    ) -> None:
        self.default_policy = default_policy
        self.default_lease_timeout = default_lease_timeout_s
        self._active_tokens: dict[str, RampToken] = {}
        self._queues: dict[str, list[QueueEntry]] = {}
        self._policies: dict[str, RampTokenPriority] = {}
        self._total_grants_count: int = 0

    @property
    def total_grants_count(self) -> int:
        """Total number of access tokens granted across all managed portals."""
        return self._total_grants_count

    def register_portal(
        self,
        portal_id: str,
        policy: RampTokenPriority | None = None,
    ) -> None:
        """Register a single-lane ramp portal under token arbitration."""
        if portal_id not in self._queues:
            self._queues[portal_id] = []
            self._policies[portal_id] = policy or self.default_policy

    def request_token(
        self,
        portal_id: str,
        vehicle_id: str,
        direction: RampDirection,
        timestamp_s: float,
        estimated_duration_s: float = 15.0,
    ) -> RampToken | None:
        """Request exclusive passage through a single-lane ramp portal."""
        self.register_portal(portal_id)
        self.clean_expired_tokens(timestamp_s)

        current_token = self._active_tokens.get(portal_id)
        queue = self._queues[portal_id]

        # If vehicle already holds active token
        if (
            current_token is not None
            and current_token.vehicle_id == vehicle_id
            and current_token.is_active
        ):
            return current_token

        # If ramp is occupied, queue vehicle if not already waiting
        if current_token is not None and current_token.is_active:
            if not any(entry.vehicle_id == vehicle_id for entry in queue):
                queue.append(
                    QueueEntry(
                        vehicle_id=vehicle_id,
                        direction=direction,
                        requested_at_s=timestamp_s,
                        estimated_duration_s=estimated_duration_s,
                    )
                )
            return None

        # If ramp is free, check if higher-priority vehicles are waiting
        if queue:
            best_idx = self._select_next_candidate_index(portal_id)
            if best_idx is not None:
                candidate = queue[best_idx]
                if candidate.vehicle_id != vehicle_id:
                    # Someone else has priority; queue this vehicle and dispatch candidate
                    if not any(e.vehicle_id == vehicle_id for e in queue):
                        queue.append(
                            QueueEntry(
                                vehicle_id=vehicle_id,
                                direction=direction,
                                requested_at_s=timestamp_s,
                                estimated_duration_s=estimated_duration_s,
                            )
                        )
                    chosen = queue.pop(best_idx)
                    token = self._issue_token(
                        portal_id=portal_id,
                        vehicle_id=chosen.vehicle_id,
                        direction=chosen.direction,
                        timestamp_s=timestamp_s,
                        duration_s=chosen.estimated_duration_s,
                    )
                    return token if chosen.vehicle_id == vehicle_id else None

        # Ramp is clear and no queue: grant directly
        return self._issue_token(
            portal_id=portal_id,
            vehicle_id=vehicle_id,
            direction=direction,
            timestamp_s=timestamp_s,
            duration_s=estimated_duration_s,
        )

    def release_token(
        self,
        portal_id: str,
        token_id: str,
        timestamp_s: float,
    ) -> RampToken | None:
        """Release an active passage token and dispatch the next queued vehicle."""
        current_token = self._active_tokens.get(portal_id)
        if current_token is not None and current_token.token_id == token_id:
            del self._active_tokens[portal_id]

        queue = self._queues.get(portal_id, [])
        if not queue:
            return None

        best_idx = self._select_next_candidate_index(portal_id)
        if best_idx is None:
            return None

        candidate = queue.pop(best_idx)
        return self._issue_token(
            portal_id=portal_id,
            vehicle_id=candidate.vehicle_id,
            direction=candidate.direction,
            timestamp_s=timestamp_s,
            duration_s=candidate.estimated_duration_s,
        )

    def clean_expired_tokens(self, timestamp_s: float) -> list[str]:
        """Release tokens whose expiration deadline has lapsed."""
        expired: list[str] = []
        for portal_id, token in list(self._active_tokens.items()):
            if timestamp_s >= token.expires_at_s:
                expired.append(portal_id)
                del self._active_tokens[portal_id]

        # Trigger queue dispatch for portals freed by expiration
        for portal_id in expired:
            queue = self._queues.get(portal_id, [])
            if queue:
                best_idx = self._select_next_candidate_index(portal_id)
                if best_idx is not None:
                    candidate = queue.pop(best_idx)
                    self._issue_token(
                        portal_id=portal_id,
                        vehicle_id=candidate.vehicle_id,
                        direction=candidate.direction,
                        timestamp_s=timestamp_s,
                        duration_s=candidate.estimated_duration_s,
                    )

        return expired

    def get_active_token(self, portal_id: str) -> RampToken | None:
        """Retrieve the currently active passage token for a ramp portal."""
        token = self._active_tokens.get(portal_id)
        return token if (token is not None and token.is_active) else None

    def get_queue_length(self, portal_id: str) -> int:
        """Get the count of vehicles awaiting passage on this portal."""
        return len(self._queues.get(portal_id, []))

    def _select_next_candidate_index(self, portal_id: str) -> int | None:
        queue = self._queues.get(portal_id, [])
        if not queue:
            return None

        policy = self._policies.get(portal_id, self.default_policy)
        if policy == RampTokenPriority.ASCENDING_FIRST:
            for idx, entry in enumerate(queue):
                if entry.direction == RampDirection.ASCENDING:
                    return idx
        elif policy == RampTokenPriority.DESCENDING_FIRST:
            for idx, entry in enumerate(queue):
                if entry.direction == RampDirection.DESCENDING:
                    return idx

        # Default fallback: FIFO order
        return 0

    def _issue_token(
        self,
        portal_id: str,
        vehicle_id: str,
        direction: RampDirection,
        timestamp_s: float,
        duration_s: float,
    ) -> RampToken:
        lease = min(max(duration_s, 5.0), self.default_lease_timeout)
        token = RampToken(
            token_id=f"token-{portal_id}-{vehicle_id}",
            portal_id=portal_id,
            vehicle_id=vehicle_id,
            direction=direction,
            granted_at_s=round(timestamp_s, 2),
            expires_at_s=round(timestamp_s + lease, 2),
            is_active=True,
        )
        self._active_tokens[portal_id] = token
        self._total_grants_count += 1
        return token
