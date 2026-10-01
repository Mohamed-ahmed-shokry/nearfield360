"""Benchmarking, latency profiling, and numerical parity verification."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nearfield360.perception.inference.backend import InferenceBackend
from nearfield360.perception.inference.models import (
    BatchSweepItem,
    BatchSweepSummary,
    BenchmarkSummary,
)


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


def benchmark_batch_sweep(
    backend: InferenceBackend,
    batch_sizes: Sequence[int] = (1, 2, 4, 8),
    iterations: int = 50,
    warmup: int = 10,
    base_shape: tuple[int, ...] | None = None,
) -> BatchSweepSummary:
    """Profile inference throughput and scaling efficiency across a sweep of batch sizes.

    Args:
        backend: Loaded inference backend.
        batch_sizes: Sequence of batch sizes to evaluate.
        iterations: Number of timed inference iterations per batch size.
        warmup: Number of untimed warmup iterations per batch size.
        base_shape: Per-sample tensor shape (e.g. (3, H, W) or (1, 3, H, W)). If None,
            inferred from backend metadata.

    Returns:
        BatchSweepSummary detailing per-batch latency, FPS, speedup, and optimal batch size.
    """
    if not batch_sizes:
        msg = "batch_sizes sequence cannot be empty"
        raise ValueError(msg)
    for b in batch_sizes:
        if b <= 0:
            msg = f"All batch sizes must be positive integers, got {b}"
            raise ValueError(msg)

    c_h_w: tuple[int, ...]
    if base_shape is not None:
        if len(base_shape) == 4:
            c_h_w = base_shape[1:]
        elif len(base_shape) == 3:
            c_h_w = base_shape
        else:
            msg = f"Invalid base_shape {base_shape}, expected 3 or 4 dimensions"
            raise ValueError(msg)
    elif backend.metadata.input_shapes:
        raw_shape = backend.metadata.input_shapes[0]
        c_h_w = raw_shape[1:] if len(raw_shape) == 4 else raw_shape
    else:
        c_h_w = (3, 480, 640)

    summaries: list[tuple[int, BenchmarkSummary]] = []
    for b in batch_sizes:
        full_shape = (b, *c_h_w)
        bm = benchmark_inference(
            backend,
            input_shape=full_shape,
            iterations=iterations,
            warmup=warmup,
        )
        summaries.append((b, bm))

    # Baseline throughput: batch size 1 if present, else normalized per-sample FPS of first batch
    baseline_fps: float | None = None
    for b, bm in summaries:
        if b == 1:
            baseline_fps = bm.fps
            break
    if baseline_fps is None or baseline_fps <= 0.0:
        first_b, first_bm = summaries[0]
        baseline_fps = first_bm.fps / first_b

    items: list[BatchSweepItem] = []
    for b, bm in summaries:
        speedup = round(bm.fps / baseline_fps, 3) if baseline_fps > 0 else 1.0
        scaling_eff = round(speedup / b, 3)
        items.append(
            BatchSweepItem(
                batch_size=b,
                input_shape=bm.input_shape,
                mean_latency_ms=bm.mean_latency_ms,
                median_latency_ms=bm.median_latency_ms,
                p95_latency_ms=bm.p95_latency_ms,
                fps=bm.fps,
                speedup=speedup,
                scaling_efficiency=scaling_eff,
            )
        )

    optimal_item = max(items, key=lambda it: it.fps)

    return BatchSweepSummary(
        backend=backend.backend_type.value,
        device=backend.device.value,
        iterations=iterations,
        warmup=warmup,
        base_input_shape=c_h_w,
        items=tuple(items),
        optimal_batch_size=optimal_item.batch_size,
        max_fps=optimal_item.fps,
    )


__all__ = [
    "BatchSweepItem",
    "BatchSweepSummary",
    "ParityComparison",
    "benchmark_batch_sweep",
    "benchmark_inference",
    "compare_numerical_parity",
    "verify_numerical_parity",
]
