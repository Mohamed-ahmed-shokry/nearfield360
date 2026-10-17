"""Access credential validation and Automated License Plate Recognition (ALPR) matching."""

from __future__ import annotations

import re
from collections.abc import Sequence

from nearfield360.config import GateConfig
from nearfield360.gate.models import (
    AccessDecisionStatus,
    CredentialType,
    GateAccessDecision,
    GateAccessRequest,
    VehicleCredential,
)


def normalize_plate_string(plate: str) -> str:
    """Normalize license plate text by stripping non-alphanumeric chars and converting to upper."""
    return re.sub(r"[^A-Za-z0-9]", "", plate).upper()


def compute_plate_similarity(s1: str, s2: str) -> float:
    """Compute normalized character equality similarity between two plate strings."""
    n1 = normalize_plate_string(s1)
    n2 = normalize_plate_string(s2)
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    # Levenshtein distance calculation
    len1, len2 = len(n1), len(n2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if n1[i - 1] == n2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    max_len = max(len1, len2)
    dist = dp[len1][len2]
    return max(0.0, 1.0 - (dist / max_len))


class AccessCredentialValidator:
    """Authenticates vehicle access requests via ALPR and cryptographic credentials."""

    def __init__(
        self,
        config: GateConfig,
        initial_credentials: Sequence[VehicleCredential] | None = None,
    ) -> None:
        self._config = config
        self._credentials: dict[str, VehicleCredential] = {}
        if initial_credentials is not None:
            for cred in initial_credentials:
                self.register_credential(cred)

    @property
    def config(self) -> GateConfig:
        return self._config

    def register_credential(self, credential: VehicleCredential) -> None:
        """Register or update an authorized credential."""
        self._credentials[credential.credential_id] = credential

    def unregister_credential(self, credential_id: str) -> bool:
        """Remove a credential from the authorization registry."""
        return self._credentials.pop(credential_id, None) is not None

    def get_credential(self, credential_id: str) -> VehicleCredential | None:
        """Retrieve a credential by its unique identifier."""
        return self._credentials.get(credential_id)

    def list_credentials(self) -> list[VehicleCredential]:
        """Return all registered credentials."""
        return list(self._credentials.values())

    def find_by_plate(self, plate: str) -> VehicleCredential | None:
        """Look up credential by exact normalized license plate string."""
        target = normalize_plate_string(plate)
        if not target:
            return None
        for cred in self._credentials.values():
            if normalize_plate_string(cred.plate_number) == target:
                return cred
        return None

    def validate_request(self, request: GateAccessRequest) -> GateAccessDecision:
        """Evaluate credential authentication and reservation validity for an access request."""
        # 1. Direct presentation of valid credential token
        if request.credential is not None:
            cred = self.get_credential(request.credential.credential_id)
            if cred is None:
                cred = request.credential

            # Check validity time window
            if request.timestamp_s < cred.valid_from_s or request.timestamp_s > cred.valid_until_s:
                msg = (
                    f"Credential {cred.credential_id} expired: "
                    f"t={request.timestamp_s:.1f}s not in window"
                )
                return GateAccessDecision(
                    decision_id=f"dec-{request.request_id}",
                    request_id=request.request_id,
                    status=AccessDecisionStatus.REJECTED_EXPIRED_RESERVATION,
                    granted=False,
                    message=msg[:256],
                    authorized_at_s=request.timestamp_s,
                )

            # Valid credential
            pass_id = f"PASS-{request.gate_id}-{request.vehicle_id}"
            cred_type_str = cred.credential_type.value
            return GateAccessDecision(
                decision_id=f"dec-{request.request_id}",
                request_id=request.request_id,
                status=AccessDecisionStatus.GRANTED,
                granted=True,
                message=f"Access granted via {cred_type_str} credential {cred.credential_id}"[:256],
                authorized_at_s=request.timestamp_s,
                issued_pass_id=pass_id,
            )

        # 2. Optical ALPR plate matching
        if not request.detected_plate:
            return GateAccessDecision(
                decision_id=f"dec-{request.request_id}",
                request_id=request.request_id,
                status=AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL,
                granted=False,
                message="No license plate detected or credential presented",
                authorized_at_s=request.timestamp_s,
            )

        if request.plate_confidence < self._config.alpr_confidence_threshold:
            return GateAccessDecision(
                decision_id=f"dec-{request.request_id}",
                request_id=request.request_id,
                status=AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL,
                granted=False,
                message=(
                    f"ALPR plate confidence {request.plate_confidence:.2f} "
                    f"below threshold {self._config.alpr_confidence_threshold:.2f}"
                ),
                authorized_at_s=request.timestamp_s,
            )

        norm_detected = normalize_plate_string(request.detected_plate)
        matched_cred: VehicleCredential | None = None
        for cred in self._credentials.values():
            if (
                cred.credential_type == CredentialType.ALPR_PLATE
                and normalize_plate_string(cred.plate_number) == norm_detected
            ):
                matched_cred = cred
                break

        if matched_cred is None:
            return GateAccessDecision(
                decision_id=f"dec-{request.request_id}",
                request_id=request.request_id,
                status=AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL,
                granted=False,
                message=f"Plate '{request.detected_plate}' not in reservation registry"[:256],
                authorized_at_s=request.timestamp_s,
            )

        # Check reservation validity window for matched plate
        if (
            request.timestamp_s < matched_cred.valid_from_s
            or request.timestamp_s > matched_cred.valid_until_s
        ):
            msg = (
                f"Plate '{request.detected_plate}' expired: "
                f"t={request.timestamp_s:.1f}s not in window"
            )
            return GateAccessDecision(
                decision_id=f"dec-{request.request_id}",
                request_id=request.request_id,
                status=AccessDecisionStatus.REJECTED_EXPIRED_RESERVATION,
                granted=False,
                message=msg[:256],
                authorized_at_s=request.timestamp_s,
            )

        pass_id = f"PASS-{request.gate_id}-{request.vehicle_id}"
        granted_msg = f"Access granted for plate '{norm_detected}' ({matched_cred.tenant_id})"
        return GateAccessDecision(
            decision_id=f"dec-{request.request_id}",
            request_id=request.request_id,
            status=AccessDecisionStatus.GRANTED,
            granted=True,
            message=granted_msg[:256],
            authorized_at_s=request.timestamp_s,
            issued_pass_id=pass_id,
        )
