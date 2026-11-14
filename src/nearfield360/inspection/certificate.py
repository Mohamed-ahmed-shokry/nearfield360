"""Cryptographically signed liability handover certificate manager and snapshot comparator."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import (
    LiabilityCertificate,
    LiabilityVerdict,
    OpticalAnomaly,
    VehicleConditionSnapshot,
)


class LiabilityCertificateManager:
    """Manages cryptographic certificates (SHA-256/HMAC) and automated liability arbitration."""

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config or InspectionConfig()

    def canonicalize_snapshot(self, snapshot: VehicleConditionSnapshot) -> str:
        """Produce canonical, deterministic JSON string representation of snapshot."""
        data = snapshot.model_dump(mode="json")
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_sha256(self, canonical_payload: str) -> str:
        """Compute standard hex SHA-256 hash digest over payload."""
        return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    def compute_hmac(self, payload_sha256: str) -> str:
        """Compute HMAC-SHA256 signature using configured signing salt."""
        salt = self.config.certificate_signing_salt.encode("utf-8")
        return hmac.new(salt, payload_sha256.encode("utf-8"), hashlib.sha256).hexdigest()

    def issue_certificate(
        self,
        snapshot: VehicleConditionSnapshot,
        session_id: str,
        issuer_id: str = "avp-facility-gate-01",
        operator_id: str = "operator-autonomic",
        verdict: LiabilityVerdict = LiabilityVerdict.NO_NEW_DAMAGE,
    ) -> LiabilityCertificate:
        """Issue an immutable, cryptographically signed liability certificate."""
        canonical_str = self.canonicalize_snapshot(snapshot)
        sha256_digest = self.compute_sha256(canonical_str)
        hmac_sig = self.compute_hmac(sha256_digest)

        now_utc = datetime.now(UTC)
        cert_id = f"CERT-{now_utc.strftime('%Y%m%d%H%M%S')}-{sha256_digest[:8].upper()}"

        return LiabilityCertificate(
            certificate_id=cert_id,
            vin=snapshot.vin,
            session_id=session_id,
            phase=snapshot.inspection_phase,
            payload_sha256=sha256_digest,
            signature_hmac=hmac_sig,
            issuer_id=issuer_id,
            operator_id=operator_id,
            verdict=verdict,
            created_at_utc=now_utc.isoformat(),
        )

    def verify_certificate(
        self,
        certificate: LiabilityCertificate,
        snapshot: VehicleConditionSnapshot,
    ) -> bool:
        """Verify certificate payload digest and HMAC cryptographic signature authenticity."""
        canonical_str = self.canonicalize_snapshot(snapshot)
        computed_sha = self.compute_sha256(canonical_str)

        if not hmac.compare_digest(computed_sha, certificate.payload_sha256):
            return False

        expected_hmac = self.compute_hmac(certificate.payload_sha256)
        return hmac.compare_digest(expected_hmac, certificate.signature_hmac)

    def compare_snapshots(
        self,
        pre_snapshot: VehicleConditionSnapshot,
        post_snapshot: VehicleConditionSnapshot,
    ) -> dict[str, Any]:
        """Compare pre-dropoff and post-pickup snapshots to assign liability."""
        pre_anomaly_keys = {
            (a.surface_location, a.damage_type.value) for a in pre_snapshot.anomalies
        }

        new_anomalies: list[OpticalAnomaly] = []
        for post_a in post_snapshot.anomalies:
            key = (post_a.surface_location, post_a.damage_type.value)
            if key not in pre_anomaly_keys:
                new_anomalies.append(post_a)

        score_delta = round(
            post_snapshot.overall_condition_score - pre_snapshot.overall_condition_score, 1
        )

        if new_anomalies:
            verdict = LiabilityVerdict.NEW_DAMAGE_DETECTED
        elif pre_snapshot.anomalies:
            verdict = LiabilityVerdict.PRE_EXISTING_DOCUMENTED
        elif score_delta <= -5.0:
            verdict = LiabilityVerdict.DISPUTE_REQUIRES_REVIEW
        else:
            verdict = LiabilityVerdict.NO_NEW_DAMAGE

        return {
            "vin": post_snapshot.vin,
            "pre_phase": pre_snapshot.inspection_phase.value,
            "post_phase": post_snapshot.inspection_phase.value,
            "verdict": verdict,
            "new_damage_count": len(new_anomalies),
            "new_anomalies": [a.model_dump(mode="json") for a in new_anomalies],
            "pre_condition_score": pre_snapshot.overall_condition_score,
            "post_condition_score": post_snapshot.overall_condition_score,
            "score_delta": score_delta,
        }
