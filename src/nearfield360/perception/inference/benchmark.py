"""Benchmarking, latency profiling, and numerical parity verification."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nearfield360.perception.inference.backend import InferenceBackend
from nearfield360.perception.inference.models import BenchmarkSummary


@dataclass(frozen=True, slots=True)
class ParityComparison:
    """Numerical parity comparison metrics between two tensor outputs."""

    is_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    atol: float
    rtol: float


def compare_numerical_parity(
    actual: np.ndarray | tuple[np.ndarray, ...],
    reference: np.ndarray | tuple[np.ndarray, ...],
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ParityComparison:
    """Compute detailed numerical parity metrics between model outputs.

    Args:
        actual: Model output array or tuple of arrays under test.
        reference: Reference output array or tuple of arrays.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        ParityComparison instance with maximum differences and match boolean.
    """
    if isinstance(actual, tuple) and isinstance(reference, tuple):
        if len(actual) != len(reference):
            msg = f"Output tuple length mismatch: {len(actual)} vs {len(reference)}"
            raise ValueError(msg)
        all_matches: list[bool] = []
        max_abs = 0.0
        sum_mean_abs = 0.0
        max_rel = 0.0
        for a, r in zip(actual, reference, strict=True):
            sub_comp = compare_numerical_parity(a, r, atol=atol, rtol=rtol)
            all_matches.append(sub_comp.is_match)
            max_abs = max(max_abs, sub_comp.max_abs_diff)
            sum_mean_abs += sub_comp.mean_abs_diff
            max_rel = max(max_rel, sub_comp.max_rel_diff)
        return ParityComparison(
            is_match=all(all_matches),
            max_abs_diff=max_abs,
            mean_abs_diff=sum_mean_abs / len(actual),
            max_rel_diff=max_rel,
            atol=atol,
            rtol=rtol,
        )

    if not isinstance(actual, np.ndarray) or not isinstance(reference, np.ndarray):
        msg = f"Expected numpy arrays or tuples of arrays, got {type(actual)} and {type(reference)}"
        raise TypeError(msg)

    if actual.shape != reference.shape:
        msg = f"Array shape mismatch: {actual.shape} vs {reference.shape}"
        raise ValueError(msg)

    abs_diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0
    rel_diff = abs_diff / (np.abs(reference.astype(np.float64)) + 1e-8)
    max_rel = float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0

    is_match = bool(np.allclose(actual, reference, rtol=rtol, atol=atol))
    return ParityComparison(
        is_match=is_match,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        max_rel_diff=max_rel,
        atol=atol,
        rtol=rtol,
    )


def verify_numerical_parity(
    actual: np.ndarray | tuple[np.ndarray, ...],
    reference: np.ndarray | tuple[np.ndarray, ...],
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> bool:
    """Verify that model outputs agree within acceptable numerical tolerances."""
    return compare_numerical_parity(actual, reference, atol=atol, rtol=rtol).is_match


def benchmark_inference(
    backend: InferenceBackend,
    input_shape: tuple[int, ...] | None = None,
    iterations: int = 50,
    warmup: int = 10,
    dummy_input: np.ndarray | None = None,
) -> BenchmarkSummary:
    """Profile latency distribution and throughput for an inference backend.

    Args:
        backend: Loaded inference backend.
        input_shape: Input tensor shape. If None and dummy_input is None, uses model metadata.
        iterations: Number of timed inference iterations.
        warmup: Number of untimed warmup iterations.
        dummy_input: Explicit input tensor. If None, random input is generated.

    Returns:
        BenchmarkSummary containing mean, median, p90, p95, p99 latencies and FPS.
    """
    if iterations <= 0:
        msg = f"iterations must be positive, got {iterations}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be non-negative, got {warmup}"
        raise ValueError(msg)

    if dummy_input is not None:
        tensor = dummy_input
        shape = tuple(dummy_input.shape)
    else:
        if input_shape is not None:
            shape = tuple(input_shape)
        elif backend.metadata.input_shapes:
            shape = backend.metadata.input_shapes[0]
        else:
            shape = (1, 3, 480, 640)
        tensor = np.random.randn(*shape).astype(np.float32)

    # Warmup passes
    for _ in range(warmup):
        backend.forward(tensor)

    # Timed passes
    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        backend.forward(tensor)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat = max(1e-6, float(np.mean(lat_arr)))
    median_lat = max(1e-6, float(np.median(lat_arr)))
    p90_lat = max(1e-6, float(np.percentile(lat_arr, 90)))
    p95_lat = max(1e-6, float(np.percentile(lat_arr, 95)))
    p99_lat = max(1e-6, float(np.percentile(lat_arr, 99)))
    min_lat = max(1e-6, float(np.min(lat_arr)))
    max_lat = max(1e-6, float(np.max(lat_arr)))

    batch_size = shape[0] if len(shape) > 0 else 1
    fps = max(1e-6, float((batch_size * 1000.0) / mean_lat))

    return BenchmarkSummary(
        iterations=iterations,
        warmup=warmup,
        backend=backend.backend_type.value,
        device=backend.device.value,
        input_shape=shape,
        mean_latency_ms=round(mean_lat, 4),
        median_latency_ms=round(median_lat, 4),
        p90_latency_ms=round(p90_lat, 4),
        p95_latency_ms=round(p95_lat, 4),
        p99_latency_ms=round(p99_lat, 4),
        min_latency_ms=round(min_lat, 4),
        max_latency_ms=round(max_lat, 4),
        fps=round(fps, 2),
    )


__all__ = [
    "ParityComparison",
    "benchmark_inference",
    "compare_numerical_parity",
    "verify_numerical_parity",
]
