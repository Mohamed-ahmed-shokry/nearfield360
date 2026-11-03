"""Cryptographic firmware verification and monotonic anti-rollback validation (UN R156)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import re

from nearfield360.config import OTAConfig
from nearfield360.ota.models import FirmwareManifest

logger = logging.getLogger(__name__)

# Valid UN R156 RXSWIN format: e.g. "RXSWIN-E4-156R00-NF360-0042"
RXSWIN_PATTERN = re.compile(r"^RXSWIN-[A-Z0-9_-]+$")


class FirmwareVerifier:
    """UN R156 firmware cryptographic verification and anti-rollback security engine."""

    def __init__(self, config: OTAConfig | None = None) -> None:
        self.config = config or OTAConfig()

    def compute_payload_sha256(self, payload: bytes) -> str:
        """Calculate the hex-encoded SHA-256 digest of binary firmware payload."""
        return hashlib.sha256(payload).hexdigest()

    def sign_payload(self, payload: bytes, secret_key: str | None = None) -> str:
        """Generate HMAC-SHA256 authentication tag for the firmware payload."""
        key = (secret_key or self.config.signing_secret_key).encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    def verify_signature(
        self,
        payload: bytes,
        signature_hex: str,
        secret_key: str | None = None,
    ) -> bool:
        """Verify the cryptographic signature of the payload against expected secret."""
        expected = self.sign_payload(payload, secret_key=secret_key)
        return hmac.compare_digest(expected, signature_hex)

    def verify_manifest(
        self,
        manifest: FirmwareManifest,
        payload: bytes,
        secret_key: str | None = None,
    ) -> tuple[bool, str]:
        """Verify firmware manifest integrity, payload size, SHA-256 digest, and signature."""
        # 1. Check payload size
        actual_size = len(payload)
        if actual_size != manifest.payload_size_bytes:
            msg = (
                f"Payload size mismatch: expected {manifest.payload_size_bytes} bytes, "
                f"got {actual_size} bytes"
            )
            logger.warning(msg)
            return False, msg

        # 2. Check maximum package size
        max_bytes = int(self.config.max_package_size_mb * 1024 * 1024)
        if actual_size > max_bytes:
            msg = f"Payload size {actual_size} bytes exceeds configured limit of {max_bytes} bytes"
            logger.warning(msg)
            return False, msg

        # 3. Check SHA-256 digest
        computed_hash = self.compute_payload_sha256(payload)
        if computed_hash != manifest.sha256_hash:
            msg = (
                f"SHA-256 digest mismatch: manifest={manifest.sha256_hash}, "
                f"computed={computed_hash}"
            )
            logger.warning(msg)
            return False, msg

        # 4. Check cryptographic signature
        if not self.verify_signature(payload, manifest.signature_hex, secret_key=secret_key):
            msg = "Cryptographic signature verification failed (payload tampered or invalid key)"
            logger.warning(msg)
            return False, msg

        # 5. Check RXSWIN syntax conformance
        if not RXSWIN_PATTERN.match(manifest.rxswin):
            msg = f"RXSWIN '{manifest.rxswin}' violates UN R156 syntax format"
            logger.warning(msg)
            return False, msg

        return True, "Manifest, SHA-256 digest, and signature verified successfully"

    def verify_anti_rollback(
        self,
        candidate_revision: int,
        current_installed_revision: int,
    ) -> tuple[bool, str]:
        """Ensure monotonic revision counter strictly increases to prevent downgrade attacks."""
        if not self.config.enforce_monotonic_version:
            return True, "Monotonic version enforcement disabled by configuration"

        if candidate_revision <= current_installed_revision:
            msg = (
                f"Anti-rollback violation: candidate revision {candidate_revision} "
                f"must be strictly greater than currently installed {current_installed_revision}"
            )
            logger.error(msg)
            return False, msg

        return (
            True,
            (
                f"Revision monotonic progression confirmed "
                f"({candidate_revision} > {current_installed_revision})"
            ),
        )
