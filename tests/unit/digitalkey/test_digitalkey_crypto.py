"""Unit tests for CCC Digital Key 3.0 cryptographic delegation and signature verification."""

from datetime import UTC, datetime, timedelta

from nearfield360.digitalkey.crypto import DigitalKeyCrypto
from nearfield360.digitalkey.models import DelegationScope


def test_issue_and_verify_valid_token() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-alice",
        device_id="dev-phone-123",
        scopes=[DelegationScope.AVP_OPERATE, DelegationScope.CABIN_LOCK],
        validity_duration_s=3600.0,
    )
    assert token.vin == vin
    assert token.is_revoked is False
    assert len(token.signature) == 64

    is_valid, msg = crypto.verify_token(
        token, expected_vin=vin, required_scope=DelegationScope.AVP_OPERATE
    )
    assert is_valid is True
    assert "Authorized" in msg


def test_verify_token_tampered_signature() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-bob",
        device_id="dev-pixel-9",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    # Alter signature
    tampered_token = token.model_copy(update={"signature": "00" * 32})
    is_valid, msg = crypto.verify_token(tampered_token, expected_vin=vin)
    assert is_valid is False
    assert "signature verification failed" in msg


def test_verify_token_expired() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    past = (datetime.now(UTC) - timedelta(seconds=100)).isoformat()
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-charlie",
        device_id="dev-iphone",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    # Set expires_at_utc to past
    expired_token = token.model_copy(update={"expires_at_utc": past})
    # Re-sign to isolate expiration check
    payload = crypto._canonical_payload(
        token_id=expired_token.token_id,
        vin=expired_token.vin,
        user_id=expired_token.user_id,
        device_id=expired_token.device_id,
        scopes=expired_token.scopes,
        issued_at_utc=expired_token.issued_at_utc,
        expires_at_utc=expired_token.expires_at_utc,
        nonce=expired_token.nonce,
    )
    import hashlib
    import hmac

    valid_sig = hmac.new(crypto._signing_key, payload, hashlib.sha256).hexdigest()
    expired_token = expired_token.model_copy(update={"signature": valid_sig})

    is_valid, msg = crypto.verify_token(expired_token, expected_vin=vin)
    assert is_valid is False
    assert "expired" in msg


def test_verify_token_vin_mismatch() -> None:
    crypto = DigitalKeyCrypto()
    token = crypto.issue_delegation_token(
        vin="WP0ZZZ99ZTS392001",
        user_id="usr-david",
        device_id="dev-watch",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    is_valid, msg = crypto.verify_token(token, expected_vin="WP0ZZZ99ZTS392999")
    assert is_valid is False
    assert "VIN mismatch" in msg


def test_verify_token_missing_required_scope() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-emma",
        device_id="dev-samsung",
        scopes=[DelegationScope.TRUNK_ACCESS],  # Only trunk access
    )
    is_valid, msg = crypto.verify_token(
        token, expected_vin=vin, required_scope=DelegationScope.AVP_OPERATE
    )
    assert is_valid is False
    assert "scope 'AVP_OPERATE' not granted" in msg


def test_verify_token_revoked() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-frank",
        device_id="dev-key",
        scopes=[DelegationScope.AVP_OPERATE],
    )
    revoked_token = token.model_copy(update={"is_revoked": True})
    is_valid, msg = crypto.verify_token(revoked_token, expected_vin=vin)
    assert is_valid is False
    assert "revoked" in msg


def test_verify_token_nonce_replay_prevention() -> None:
    crypto = DigitalKeyCrypto()
    vin = "WP0ZZZ99ZTS392001"
    token = crypto.issue_delegation_token(
        vin=vin,
        user_id="usr-grace",
        device_id="dev-mobile",
        scopes=[DelegationScope.AVP_OPERATE],
    )

    # First consumption succeeds
    ok1, _ = crypto.verify_token(token, expected_vin=vin, consume_nonce=True)
    assert ok1 is True

    # Replaying same token with consumed nonce fails
    ok2, msg2 = crypto.verify_token(token, expected_vin=vin, consume_nonce=True)
    assert ok2 is False
    assert "replay detected" in msg2
