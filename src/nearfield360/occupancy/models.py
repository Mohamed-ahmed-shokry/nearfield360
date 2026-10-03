"""Data models for temporal BEV multi-camera occupancy forecasting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.risk import RiskZone


@dataclass(frozen=True, slots=True)
class TemporalOccupancyState:
    """Internal recurrent state of the temporal BEV perception network.

    Maintains accumulated multi-frame belief, estimated local velocity fields,
    and multi-channel hidden representation on the vehicle-centric BEV grid.
    """

    timestamp: float
    step_index: int
    grid: BevGrid
    hidden_state: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    velocity_field: NDArray[np.float64]
    dynamic_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative float")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(self.grid, BevGrid):
            raise ValueError("grid must be a BevGrid instance")

        shape = self.grid.shape
        if self.hidden_state.ndim != 3 or self.hidden_state.shape[1:] != shape:
            raise ValueError(
                f"hidden_state must have shape (C, {shape[0]}, {shape[1]}), "
                f"got {self.hidden_state.shape}"
            )
        if self.occupancy.shape != shape:
            raise ValueError(f"occupancy must have shape {shape}, got {self.occupancy.shape}")
        if self.uncertainty.shape != shape:
            raise ValueError(f"uncertainty must have shape {shape}, got {self.uncertainty.shape}")
        if self.velocity_field.shape != (2, shape[0], shape[1]):
            raise ValueError(
                f"velocity_field must have shape (2, {shape[0]}, {shape[1]}), "
                f"got {self.velocity_field.shape}"
            )
        if self.dynamic_mask.shape != shape or self.dynamic_mask.dtype.kind != "b":
            raise ValueError(f"dynamic_mask must be a boolean array with shape {shape}")

        object.__setattr__(
            self, "hidden_state", np.ascontiguousarray(self.hidden_state, dtype=np.float64)
        )
        object.__setattr__(
            self, "occupancy", np.ascontiguousarray(self.occupancy, dtype=np.float64)
        )
        object.__setattr__(
            self, "uncertainty", np.ascontiguousarray(self.uncertainty, dtype=np.float64)
        )
        object.__setattr__(
            self, "velocity_field", np.ascontiguousarray(self.velocity_field, dtype=np.float64)
        )
        object.__setattr__(
            self, "dynamic_mask", np.ascontiguousarray(self.dynamic_mask, dtype=np.bool_)
        )


@dataclass(frozen=True, slots=True)
class OccupancyForecastStep:
    """Predicted spatial occupancy grid at a single future time horizon offset."""

    time_offset_s: float
    timestamp: float
    occupancy: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    free: NDArray[np.float64]
    dynamic_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_offset_s) or self.time_offset_s <= 0.0:
            raise ValueError("time_offset_s must be a finite positive float")
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative float")
        if self.occupancy.ndim != 2:
            raise ValueError("occupancy must be a 2D array")
        shape = self.occupancy.shape
        if self.uncertainty.shape != shape:
            raise ValueError(f"uncertainty shape {self.uncertainty.shape} != {shape}")
        if self.free.shape != shape:
            raise ValueError(f"free shape {self.free.shape} != {shape}")
        if self.dynamic_mask.shape != shape or self.dynamic_mask.dtype.kind != "b":
            raise ValueError(f"dynamic_mask must be boolean array with shape {shape}")

        object.__setattr__(
            self, "occupancy", np.ascontiguousarray(self.occupancy, dtype=np.float64)
        )
        object.__setattr__(
            self, "uncertainty", np.ascontiguousarray(self.uncertainty, dtype=np.float64)
        )
        object.__setattr__(self, "free", np.ascontiguousarray(self.free, dtype=np.float64))
        object.__setattr__(
            self, "dynamic_mask", np.ascontiguousarray(self.dynamic_mask, dtype=np.bool_)
        )


class ZoneForecastRisk(BaseModel):
    """Temporal hazard evolution inside a designated safety zone across forecast horizon."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_name: str = Field(description="Name of the evaluated risk zone.")
    peak_occupancy: float = Field(
        ge=0.0, le=1.0, description="Highest occupancy score observed within the zone across time."
    )
    mean_occupancies: tuple[float, ...] = Field(
        description="Mean zone occupancy per forecast time step."
    )
    max_occupancies: tuple[float, ...] = Field(
        description="Maximum zone occupancy per forecast time step."
    )
    time_to_intrusion_s: float | None = Field(
        default=None,
        description="Earliest future horizon seconds where zone occupancy exceeds threshold.",
    )
    is_threat: bool = Field(
        default=False, description="True if occupancy breaches danger threshold within horizon."
    )


class TemporalForecastSummary(BaseModel):
    """High-level summary of multi-step BEV temporal forecast for serialization and reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_timestamp: float = Field(description="Origin timestamp of the forecast horizon.")
    horizon_seconds: float = Field(gt=0.0, description="Total forward prediction duration.")
    step_seconds: float = Field(gt=0.0, description="Discretization step between forecast grids.")
    num_steps: int = Field(ge=1, description="Total number of prediction horizon steps.")
    observed_cells: int = Field(ge=0, description="Number of observed cells at base frame.")
    dynamic_cells: int = Field(
        ge=0, description="Number of cells classified as dynamic flow at base frame."
    )
    max_future_occupancy: float = Field(
        ge=0.0, le=1.0, description="Global maximum occupancy predicted across all horizon steps."
    )
    mean_future_occupancy: float = Field(
        ge=0.0, le=1.0, description="Global mean occupancy predicted across all horizon steps."
    )
    min_time_to_intrusion_s: float | None = Field(
        default=None,
        description="Smallest time to intrusion across any configured safety zone.",
    )
    critical_zone: str | None = Field(
        default=None,
        description="Name of the zone with earliest predicted collision intrusion.",
    )
    zone_risks: tuple[ZoneForecastRisk, ...] = Field(
        default=(),
        description="Per-zone forward trajectory risks.",
    )


@dataclass(frozen=True, slots=True)
class OccupancyForecastGrid:
    """Spatiotemporal 4D occupancy forecast containing multi-step predicted BEV grids."""

    grid: BevGrid
    base_timestamp: float
    steps: tuple[OccupancyForecastStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grid, BevGrid):
            raise ValueError("grid must be a BevGrid instance")
        if not np.isfinite(self.base_timestamp) or self.base_timestamp < 0.0:
            raise ValueError("base_timestamp must be a finite non-negative float")
        if not self.steps:
            raise ValueError("OccupancyForecastGrid requires at least one forecast step")

        shape = self.grid.shape
        last_t = 0.0
        for idx, step in enumerate(self.steps):
            if step.occupancy.shape != shape:
                raise ValueError(
                    f"Step {idx} shape {step.occupancy.shape} does not match grid shape {shape}"
                )
            if step.time_offset_s <= last_t:
                raise ValueError(
                    f"Forecast steps must have strictly increasing time offsets: "
                    f"step {idx} offset {step.time_offset_s} <= previous {last_t}"
                )
            last_t = step.time_offset_s

    @property
    def horizon_seconds(self) -> float:
        """Total duration of the prediction horizon in seconds."""
        return self.steps[-1].time_offset_s

    @property
    def step_seconds(self) -> float:
        """Nominal delta time between consecutive forecast steps."""
        if len(self.steps) == 1:
            return self.steps[0].time_offset_s
        return self.steps[1].time_offset_s - self.steps[0].time_offset_s

    @property
    def num_steps(self) -> int:
        """Total count of forecast steps."""
        return len(self.steps)

    def at_horizon(self, seconds: float) -> OccupancyForecastStep:
        """Return the forecast step closest to the requested seconds offset."""
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("seconds must be a finite non-negative float")
        return min(self.steps, key=lambda s: abs(s.time_offset_s - seconds))

    def max_occupancy_over_horizon(self) -> NDArray[np.float64]:
        """Compute the element-wise maximum occupancy probability envelope across all steps."""
        stacked = np.stack([s.occupancy for s in self.steps], axis=0)
        # Suppress all-NaN warnings by filling NaNs with 0.0 for envelope comparison
        clean = np.where(np.isfinite(stacked), stacked, 0.0)
        envelope = np.max(clean, axis=0)
        # If a cell had no valid predictions across all steps, mark as NaN
        all_nan = np.all(~np.isfinite(stacked), axis=0)
        return np.where(all_nan, np.nan, envelope)

    def mean_occupancy_over_horizon(self) -> NDArray[np.float64]:
        """Compute the element-wise mean occupancy probability across all steps."""
        stacked = np.stack([s.occupancy for s in self.steps], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nanmean(stacked, axis=0)

    def zone_forecasts(
        self,
        zones: Sequence[RiskZone],
        danger_occupancy: float = 0.5,
    ) -> list[ZoneForecastRisk]:
        """Evaluate future hazard evolution for each safety zone over the forecast horizon."""
        results: list[ZoneForecastRisk] = []
        for zone in zones:
            mask = zone.mask
            mean_occs: list[float] = []
            max_occs: list[float] = []
            intrusion_time: float | None = None

            for step in self.steps:
                zone_cells = step.occupancy[mask]
                valid = zone_cells[np.isfinite(zone_cells)]
                if valid.size == 0:
                    mean_val = 0.0
                    max_val = 0.0
                else:
                    mean_val = float(np.mean(valid))
                    max_val = float(np.max(valid))
                mean_occs.append(mean_val)
                max_occs.append(max_val)

                if intrusion_time is None and max_val >= danger_occupancy:
                    intrusion_time = step.time_offset_s

            peak = max(max_occs) if max_occs else 0.0
            results.append(
                ZoneForecastRisk(
                    zone_name=zone.name,
                    peak_occupancy=peak,
                    mean_occupancies=tuple(mean_occs),
                    max_occupancies=tuple(max_occs),
                    time_to_intrusion_s=intrusion_time,
                    is_threat=intrusion_time is not None,
                )
            )
        return results

    def summary(
        self,
        zones: Sequence[RiskZone],
        danger_occupancy: float = 0.5,
        observed_cells: int = 0,
        dynamic_cells: int = 0,
    ) -> TemporalForecastSummary:
        """Construct a validated summary model for serialization."""
        zone_risks = self.zone_forecasts(zones, danger_occupancy=danger_occupancy)
        threats = [z for z in zone_risks if z.time_to_intrusion_s is not None]
        min_tti: float | None = None
        critical_zone: str | None = None
        if threats:
            earliest = min(threats, key=lambda z: (z.time_to_intrusion_s or float("inf")))
            min_tti = earliest.time_to_intrusion_s
            critical_zone = earliest.zone_name

        envelope = self.max_occupancy_over_horizon()
        valid_env = envelope[np.isfinite(envelope)]
        max_future = float(np.max(valid_env)) if valid_env.size else 0.0
        mean_future = float(np.mean(valid_env)) if valid_env.size else 0.0

        return TemporalForecastSummary(
            base_timestamp=self.base_timestamp,
            horizon_seconds=self.horizon_seconds,
            step_seconds=self.step_seconds,
            num_steps=self.num_steps,
            observed_cells=observed_cells,
            dynamic_cells=dynamic_cells,
            max_future_occupancy=min(max(max_future, 0.0), 1.0),
            mean_future_occupancy=min(max(mean_future, 0.0), 1.0),
            min_time_to_intrusion_s=min_tti,
            critical_zone=critical_zone,
            zone_risks=tuple(zone_risks),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert forecast grid to a compact dictionary representation."""
        return {
            "base_timestamp": self.base_timestamp,
            "horizon_seconds": self.horizon_seconds,
            "step_seconds": self.step_seconds,
            "num_steps": self.num_steps,
            "steps": [
                {
                    "time_offset_s": s.time_offset_s,
                    "timestamp": s.timestamp,
                    "dynamic_cells": int(np.sum(s.dynamic_mask)),
                    "mean_occupancy": float(np.nanmean(s.occupancy))
                    if np.any(np.isfinite(s.occupancy))
                    else 0.0,
                    "max_occupancy": float(np.nanmax(s.occupancy))
                    if np.any(np.isfinite(s.occupancy))
                    else 0.0,
                    "mean_uncertainty": float(np.nanmean(s.uncertainty))
                    if np.any(np.isfinite(s.uncertainty))
                    else 0.0,
                }
                for s in self.steps
            ],
        }


__all__ = [
    "OccupancyForecastGrid",
    "OccupancyForecastStep",
    "TemporalForecastSummary",
    "TemporalOccupancyState",
    "ZoneForecastRisk",
]
