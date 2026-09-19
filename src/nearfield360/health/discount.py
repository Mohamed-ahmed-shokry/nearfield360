"""Health-aware evidence attenuation and discounting for multi-camera fusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nearfield360.health.models import CameraHealthReport
    from nearfield360.occupancy.evidence import OccupancyEvidence


def apply_health_discount(
    evidence: OccupancyEvidence,
    health: CameraHealthReport | float,
) -> OccupancyEvidence:
    """Discount occupancy evidence based on camera health assessment.

    Args:
        evidence: Raw occupancy evidence layer from a camera.
        health: CameraHealthReport instance or explicit float discount weight in [0.0, 1.0].

    Returns:
        New discounted OccupancyEvidence layer.
    """
    weight = float(health) if isinstance(health, (int, float)) else float(health.discount_weight)
    return evidence.scale(weight)


__all__ = ["apply_health_discount"]
