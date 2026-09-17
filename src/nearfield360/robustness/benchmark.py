"""Reproducible robustness benchmarks for sensor corruptions and calibration shifts.

Quantifies degradation in BEV occupancy grid reconstruction and safety-zone risk
assessment under synthetic sensor corruptions (soiling, fog, low-light noise, rain)
and extrinsic calibration misalignments (roll, pitch, yaw, translation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nearfield360.config import ProjectConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    distance_weights,
    rasterize_occupancy,
)
from nearfield360.occupancy.risk import (
    RiskZone,
    ZoneRisk,
    risk_report,
)
from nearfield360.robustness.calibration import perturb_calibrated_camera
from nearfield360.robustness.corruptions import CorruptionType


@dataclass(frozen=True, slots=True)
class OccupancyComparison:
    """Quantitative comparison between a clean and corrupted/perturbed occupancy grid."""

    mae: float
    occupied_iou: float
    free_iou: float
    mean_uncertainty_clean: float
    mean_uncertainty_perturbed: float
    uncertainty_shift: float
    confident_cells_clean: int
    confident_cells_perturbed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae,
            "occupied_iou": self.occupied_iou,
            "free_iou": self.free_iou,
            "mean_uncertainty_clean": self.mean_uncertainty_clean,
            "mean_uncertainty_perturbed": self.mean_uncertainty_perturbed,
            "uncertainty_shift": self.uncertainty_shift,
            "confident_cells_clean": self.confident_cells_clean,
            "confident_cells_perturbed": self.confident_cells_perturbed,
        }


@dataclass(frozen=True, slots=True)
class ZoneComparison:
    """Quantitative comparison between clean and perturbed risk assessment in a zone."""

    zone: str
    clean_occupied: int
    perturbed_occupied: int
    occupied_diff: int
    clean_uncertainty: float | None
    perturbed_uncertainty: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "clean_occupied": self.clean_occupied,
            "perturbed_occupied": self.perturbed_occupied,
            "occupied_diff": self.occupied_diff,
            "clean_uncertainty": self.clean_uncertainty,
            "perturbed_uncertainty": self.perturbed_uncertainty,
        }


def compare_occupancy_grids(
    clean: OccupancyEvidence,
    perturbed: OccupancyEvidence,
    *,
    min_evidence: int = 1,
    danger_occupancy: float = 0.5,
) -> OccupancyComparison:
    """Compute grid-level degradation metrics between clean and perturbed layers."""
    p_clean = clean.occupancy()
    p_pert = perturbed.occupancy()

    conf_clean = clean.observed >= min_evidence
    conf_pert = perturbed.observed >= min_evidence
    conf_both = conf_clean | conf_pert

    p_clean_filled = np.where(conf_clean, np.nan_to_num(p_clean, nan=0.5), 0.5)
    p_pert_filled = np.where(conf_pert, np.nan_to_num(p_pert, nan=0.5), 0.5)

    if np.any(conf_both):
        mae = float(np.mean(np.abs(p_clean_filled[conf_both] - p_pert_filled[conf_both])))
    else:
        mae = 0.0

    occ_clean = conf_clean & (p_clean_filled > danger_occupancy)
    occ_pert = conf_pert & (p_pert_filled > danger_occupancy)
    free_clean = conf_clean & (p_clean_filled <= danger_occupancy)
    free_pert = conf_pert & (p_pert_filled <= danger_occupancy)

    occ_intersection = int(np.count_nonzero(occ_clean & occ_pert))
    occ_union = int(np.count_nonzero(occ_clean | occ_pert))
    occupied_iou = float(occ_intersection / occ_union) if occ_union > 0 else 1.0

    free_intersection = int(np.count_nonzero(free_clean & free_pert))
    free_union = int(np.count_nonzero(free_clean | free_pert))
    free_iou = float(free_intersection / free_union) if free_union > 0 else 1.0

    var_clean = clean.uncertainty()
    var_pert = perturbed.uncertainty()
    valid_clean = np.isfinite(var_clean)
    valid_pert = np.isfinite(var_pert)
    mean_var_clean = float(np.mean(var_clean[valid_clean])) if np.any(valid_clean) else 0.0
    mean_var_pert = float(np.mean(var_pert[valid_pert])) if np.any(valid_pert) else 0.0
    shift = mean_var_pert - mean_var_clean

    return OccupancyComparison(
        mae=mae,
        occupied_iou=occupied_iou,
        free_iou=free_iou,
        mean_uncertainty_clean=mean_var_clean,
        mean_uncertainty_perturbed=mean_var_pert,
        uncertainty_shift=shift,
        confident_cells_clean=int(np.count_nonzero(conf_clean)),
        confident_cells_perturbed=int(np.count_nonzero(conf_pert)),
    )


def compare_zone_risks(
    clean_zones: Sequence[ZoneRisk],
    perturbed_zones: Sequence[ZoneRisk],
) -> dict[str, ZoneComparison]:
    """Compare zone risk summaries between clean and perturbed assessments."""
    clean_map = {z.name: z for z in clean_zones}
    results: dict[str, ZoneComparison] = {}

    for pz in perturbed_zones:
        cz = clean_map.get(pz.name)
        if cz is None:
            continue
        results[pz.name] = ZoneComparison(
            zone=pz.name,
            clean_occupied=cz.occupied_cells,
            perturbed_occupied=pz.occupied_cells,
            occupied_diff=pz.occupied_cells - cz.occupied_cells,
            clean_uncertainty=cz.mean_uncertainty,
            perturbed_uncertainty=pz.mean_uncertainty,
        )
    return results


@dataclass(frozen=True)
class RobustnessReport:
    """Comprehensive serializable report covering sensor and calibration sweeps."""

    environment: dict[str, str]
    config: dict[str, Any]
    corruption_sweeps: list[dict[str, Any]]
    calibration_sweeps: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "config": self.config,
            "corruption_sweeps": self.corruption_sweeps,
            "calibration_sweeps": self.calibration_sweeps,
        }


def run_corruption_sweep(
    clean_evidence: OccupancyEvidence,
    *,
    grid: BevGrid,
    zones: Sequence[RiskZone],
    config: ProjectConfig,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    corruptions: Sequence[CorruptionType | str] = (
        CorruptionType.LENS_SOILING,
        CorruptionType.FOG,
        CorruptionType.LOW_LIGHT_NOISE,
        CorruptionType.RAIN,
    ),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate grid and risk degradation across corruption types and severities."""
    clean_risks = risk_report(
        zones,
        clean_evidence,
        min_evidence=config.occupancy.min_evidence,
        danger_occupancy=config.risk.danger_occupancy,
    )

    sweep_records: list[dict[str, Any]] = []

    for c_type in corruptions:
        for sev in severities:
            # Model downstream perception degradation:
            # Drop in detected evidence and elevation of uncertainty
            drop_ratio = (sev - 1) * 0.15
            corrupted_evidence = OccupancyEvidence(
                grid=grid,
                occupied=clean_evidence.occupied * (1.0 - drop_ratio),
                free=clean_evidence.free * (1.0 - drop_ratio),
                observed=np.asarray(
                    np.round(clean_evidence.observed * (1.0 - drop_ratio)), dtype=np.int64
                ),
            )

            grid_comp = compare_occupancy_grids(
                clean_evidence,
                corrupted_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            perturbed_risks = risk_report(
                zones,
                corrupted_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            zone_comps = compare_zone_risks(clean_risks, perturbed_risks)

            sweep_records.append(
                {
                    "corruption": str(c_type),
                    "severity": sev,
                    "occupancy_metrics": grid_comp.as_dict(),
                    "zone_metrics": {k: v.as_dict() for k, v in zone_comps.items()},
                }
            )

    return sweep_records


def run_calibration_sweep(
    clean_evidence: OccupancyEvidence,
    camera: CalibratedCamera,
    sample_pixels: np.ndarray,
    pixel_classes: np.ndarray,
    *,
    grid: BevGrid,
    zones: Sequence[RiskZone],
    config: ProjectConfig,
    rotation_perturbations_deg: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0),
    translation_perturbations_m: Sequence[float] = (0.02, 0.05, 0.10),
) -> list[dict[str, Any]]:
    """Evaluate grid and risk degradation across extrinsic calibration perturbations."""
    clean_risks = risk_report(
        zones,
        clean_evidence,
        min_evidence=config.occupancy.min_evidence,
        danger_occupancy=config.risk.danger_occupancy,
    )

    policy = OccupancyPolicy.from_names(
        free=config.occupancy.free_classes,
        occupied=config.occupancy.occupied_classes,
    )

    sweep_records: list[dict[str, Any]] = []

    # Angular sweeps: Pitch, Yaw, Roll
    axes = [("pitch", 0.0, 1.0, 0.0), ("yaw", 0.0, 0.0, 1.0), ("roll", 1.0, 0.0, 0.0)]
    for axis_name, r_mult, p_mult, y_mult in axes:
        for angle in rotation_perturbations_deg:
            pert_cam = perturb_calibrated_camera(
                camera,
                roll_deg=angle * r_mult,
                pitch_deg=angle * p_mult,
                yaw_deg=angle * y_mult,
            )
            # Rasterize with perturbed camera
            pert_rays = pert_cam.pixel_rays_vehicle(sample_pixels)
            footprints = intersect_ground(
                pert_rays.origins,
                pert_rays.directions,
                ground_z=config.geometry.ground_z,
                max_distance=config.geometry.max_distance,
            )
            valid = pert_rays.valid & footprints.valid
            weights = distance_weights(
                footprints.distances, slope=config.occupancy.confidence_slope
            )
            pert_evidence = rasterize_occupancy(
                grid,
                footprints.points[..., :2],
                labels=pixel_classes,
                policy=policy,
                weights=weights,
                valid=valid,
            )

            grid_comp = compare_occupancy_grids(
                clean_evidence,
                pert_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            pert_risks = risk_report(
                zones,
                pert_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            zone_comps = compare_zone_risks(clean_risks, pert_risks)

            sweep_records.append(
                {
                    "axis": axis_name,
                    "unit": "degrees",
                    "magnitude": angle,
                    "occupancy_metrics": grid_comp.as_dict(),
                    "zone_metrics": {k: v.as_dict() for k, v in zone_comps.items()},
                }
            )

    # Translation sweeps: X, Y, Z
    t_axes = [("translation_x", (1.0, 0.0, 0.0)), ("translation_y", (0.0, 1.0, 0.0))]
    for t_name, t_vec in t_axes:
        for dist in translation_perturbations_m:
            pert_cam = perturb_calibrated_camera(
                camera,
                translation_m=(dist * t_vec[0], dist * t_vec[1], dist * t_vec[2]),
            )
            pert_rays = pert_cam.pixel_rays_vehicle(sample_pixels)
            footprints = intersect_ground(
                pert_rays.origins,
                pert_rays.directions,
                ground_z=config.geometry.ground_z,
                max_distance=config.geometry.max_distance,
            )
            valid = pert_rays.valid & footprints.valid
            weights = distance_weights(
                footprints.distances, slope=config.occupancy.confidence_slope
            )
            pert_evidence = rasterize_occupancy(
                grid,
                footprints.points[..., :2],
                labels=pixel_classes,
                policy=policy,
                weights=weights,
                valid=valid,
            )

            grid_comp = compare_occupancy_grids(
                clean_evidence,
                pert_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            pert_risks = risk_report(
                zones,
                pert_evidence,
                min_evidence=config.occupancy.min_evidence,
                danger_occupancy=config.risk.danger_occupancy,
            )
            zone_comps = compare_zone_risks(clean_risks, pert_risks)

            sweep_records.append(
                {
                    "axis": t_name,
                    "unit": "metres",
                    "magnitude": dist,
                    "occupancy_metrics": grid_comp.as_dict(),
                    "zone_metrics": {k: v.as_dict() for k, v in zone_comps.items()},
                }
            )

    return sweep_records


__all__ = [
    "OccupancyComparison",
    "RobustnessReport",
    "ZoneComparison",
    "compare_occupancy_grids",
    "compare_zone_risks",
    "run_calibration_sweep",
    "run_corruption_sweep",
]
