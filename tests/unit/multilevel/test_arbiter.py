"""Unit tests for single-lane ramp token arbiter and conflict avoidance."""

from nearfield360.multilevel.arbiter import RampTokenArbiter
from nearfield360.multilevel.models import RampDirection, RampTokenPriority


def test_ramp_token_direct_grant_and_mutual_exclusion() -> None:
    arbiter = RampTokenArbiter(default_lease_timeout_s=20.0)

    # First vehicle requests passage on free ramp
    token1 = arbiter.request_token(
        portal_id="ramp-01",
        vehicle_id="veh-A",
        direction=RampDirection.ASCENDING,
        timestamp_s=100.0,
        estimated_duration_s=15.0,
    )
    assert token1 is not None
    assert token1.vehicle_id == "veh-A"
    assert token1.portal_id == "ramp-01"
    assert token1.direction == RampDirection.ASCENDING
    assert token1.is_active is True
    assert arbiter.get_active_token("ramp-01") == token1

    # Same vehicle requesting again returns existing active token
    same_token = arbiter.request_token(
        portal_id="ramp-01",
        vehicle_id="veh-A",
        direction=RampDirection.ASCENDING,
        timestamp_s=105.0,
    )
    assert same_token == token1

    # Second vehicle requests while ramp is occupied -> queued
    token2 = arbiter.request_token(
        portal_id="ramp-01",
        vehicle_id="veh-B",
        direction=RampDirection.DESCENDING,
        timestamp_s=102.0,
        estimated_duration_s=12.0,
    )
    assert token2 is None
    assert arbiter.get_queue_length("ramp-01") == 1


def test_ramp_token_priority_ascending_first() -> None:
    arbiter = RampTokenArbiter(default_policy=RampTokenPriority.ASCENDING_FIRST)

    # Veh-1 occupies ramp
    t1 = arbiter.request_token("ramp-01", "veh-1", RampDirection.ASCENDING, timestamp_s=0.0)
    assert t1 is not None

    # Veh-2 (descending) arrives first at t=1.0 and is queued
    assert (
        arbiter.request_token("ramp-01", "veh-2", RampDirection.DESCENDING, timestamp_s=1.0)
        is None
    )

    # Veh-3 (ascending) arrives second at t=2.0 and is queued
    assert (
        arbiter.request_token("ramp-01", "veh-3", RampDirection.ASCENDING, timestamp_s=2.0)
        is None
    )

    assert arbiter.get_queue_length("ramp-01") == 2

    # Veh-1 finishes passage and releases token
    next_token = arbiter.release_token("ramp-01", t1.token_id, timestamp_s=15.0)
    assert next_token is not None
    # Because of ASCENDING_FIRST, Veh-3 must be granted before Veh-2!
    assert next_token.vehicle_id == "veh-3"
    assert next_token.direction == RampDirection.ASCENDING


def test_ramp_token_expiration_cleanup() -> None:
    arbiter = RampTokenArbiter(default_lease_timeout_s=20.0)

    token = arbiter.request_token(
        portal_id="ramp-timeout",
        vehicle_id="veh-stalled",
        direction=RampDirection.ASCENDING,
        timestamp_s=100.0,
        estimated_duration_s=10.0,
    )
    assert token is not None

    # Queue waiting vehicle
    assert (
        arbiter.request_token(
            portal_id="ramp-timeout",
            vehicle_id="veh-waiting",
            direction=RampDirection.DESCENDING,
            timestamp_s=105.0,
        )
        is None
    )

    # Clean at t=105 (not expired yet: granted at 100 with duration 10 -> expires at 110)
    expired = arbiter.clean_expired_tokens(timestamp_s=105.0)
    assert len(expired) == 0

    # Clean at t=111 (expired) -> stalled token evicted and waiting vehicle gets token
    expired = arbiter.clean_expired_tokens(timestamp_s=111.0)
    assert "ramp-timeout" in expired
    active = arbiter.get_active_token("ramp-timeout")
    assert active is not None
    assert active.vehicle_id == "veh-waiting"


def test_fifo_policy() -> None:
    arbiter = RampTokenArbiter()
    arbiter.register_portal("ramp-fifo", policy=RampTokenPriority.FIFO)

    t = arbiter.request_token("ramp-fifo", "veh-x", RampDirection.ASCENDING, timestamp_s=0.0)
    assert t is not None

    arbiter.request_token("ramp-fifo", "veh-first-desc", RampDirection.DESCENDING, timestamp_s=1.0)
    arbiter.request_token("ramp-fifo", "veh-second-asc", RampDirection.ASCENDING, timestamp_s=2.0)

    next_tok = arbiter.release_token("ramp-fifo", t.token_id, timestamp_s=10.0)
    assert next_tok is not None
    assert next_tok.vehicle_id == "veh-first-desc"
