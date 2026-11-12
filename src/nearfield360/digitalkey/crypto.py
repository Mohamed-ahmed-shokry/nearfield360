"""CCC Digital Key Release 3.0 token generation, delegation, and signature verification."""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.models import CCCDelegationToken, DelegationScope


class DigitalKeyCrypto:
    """Cryptographic token engine for CCC Digital Key 3.0 session management and delegation."""

    def __init__(self, config: DigitalKeyConfig | None = None) -> None:
        self.config = config or DigitalKeyConfig()
        self._signing_key = self.config.session_salt.encode()
        self._consumed_nonces: set[str] = set()

    def _canonical_payload(
        self,
        token_id: str,
        vin: str,
        user_id: str,
        device_id: str,
        scopes: list[DelegationScope],
        issued_at_utc: str,
        expires_at_utc: str,
        nonce: str,
    ) -> bytes:
        """Create canonical serialization for cryptographic signing."""
        sorted_scopes = ",".join(sorted(s.value for s in scopes))
        payload_str = (
            f"{token_id}|{vin}|{user_id}|{device_id}|{sorted_scopes}|"
            f"{issued_at_utc}|{expires_at_utc}|{nonce}"
        )
        return payload_str.encode()

    def issue_delegation_token(
        self,
        vin: str,
        user_id: str,
        device_id: str,
        scopes: list[DelegationScope],
        validity_duration_s: float | None = None,
    ) -> CCCDelegationToken:
        """Issue a cryptographically signed CCC Digital Key delegation token."""
        duration = validity_duration_s or self.config.token_validity_window_s
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=duration)

        token_id = f"tok-{secrets.token_hex(6)}"
        nonce = secrets.token_hex(8)
        issued_str = now.isoformat()
        expires_str = expires.isoformat()

        payload = self._canonical_payload(
            token_id=token_id,
            vin=vin,
            user_id=user_id,
            device_id=device_id,
            scopes=scopes,
            issued_at_utc=issued_str,
            expires_at_utc=expires_str,
            nonce=nonce,
        )
        signature = hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest()

        return CCCDelegationToken(
            token_id=token_id,
            vin=vin,
            user_id=user_id,
            device_id=device_id,
            scopes=scopes,
            issued_at_utc=issued_str,
            expires_at_utc=expires_str,
            nonce=nonce,
            signature=signature,
            is_revoked=False,
        )

    def verify_token(
        self,
        token: CCCDelegationToken,
        expected_vin: str,
        required_scope: DelegationScope | None = None,
        consume_nonce: bool = False,
    ) -> tuple[bool, str]:
        """Validate signature, expiration, VIN, revocation, and granted scopes.

        Returns:
            (is_valid, reason)
        """
        if token.is_revoked:
            return False, "Delegation token has been revoked by owner"

        if token.vin != expected_vin:
            return False, f"VIN mismatch: token specifies {token.vin}, vehicle is {expected_vin}"

        # Check signature integrity
        expected_payload = self._canonical_payload(
            token_id=token.token_id,
            vin=token.vin,
            user_id=token.user_id,
            device_id=token.device_id,
            scopes=token.scopes,
            issued_at_utc=token.issued_at_utc,
            expires_at_utc=token.expires_at_utc,
            nonce=token.nonce,
        )
        expected_sig = hmac.new(self._signing_key, expected_payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, token.signature):
            return False, "Cryptographic signature verification failed - token tampered"

        # Check expiration
        try:
            expires_dt = datetime.fromisoformat(token.expires_at_utc)
            if datetime.now(UTC) > expires_dt:
                return False, f"Delegation token expired at {token.expires_at_utc}"
        except ValueError:
            return False, "Invalid ISO 8601 expiration timestamp"

        # Check permission scope
        if required_scope and required_scope not in token.scopes:
            return (
                False,
                f"Required permission scope '{required_scope.value}' not granted in token scopes",
            )

        # Anti-replay check
        if consume_nonce:
            if token.nonce in self._consumed_nonces:
                return False, f"Nonce replay detected: {token.nonce} has already been consumed"
            self._consumed_nonces.add(token.nonce)

        return True, "Token verified - Authorized"
