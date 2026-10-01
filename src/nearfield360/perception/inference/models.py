"""Domain data models and enums for neural perception inference and benchmarking."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class InferenceDevice(StrEnum):
    """Execution device hardware target."""

    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"


class InferenceBackendType(StrEnum):
    """Inference execution runtime engine."""

    OPENCV = "opencv"
    ONNXRUNTIME = "onnxruntime"


class ModelMetadata(BaseModel):
    """Structural metadata of an inspected or loaded ONNX neural network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(description="Filesystem path to the model file.")
    backend: str = Field(description="Inference runtime name.")
    device: str = Field(description="Target compute device.")
    input_names: tuple[str, ...] = Field(description="Ordered names of input tensors.")
    input_shapes: tuple[tuple[int, ...], ...] = Field(
        description="Expected tensor shapes for each input."
    )
    output_names: tuple[str, ...] = Field(description="Ordered names of output tensors.")
    output_shapes: tuple[tuple[int, ...], ...] = Field(
        description="Declared or inferred output tensor shapes."
    )


class BenchmarkSummary(BaseModel):
    """Statistical summary of model execution latency and throughput."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int = Field(gt=0, description="Number of timed benchmark passes.")
    warmup: int = Field(ge=0, description="Number of untimed warmup passes.")
    backend: str = Field(description="Runtime engine used.")
    device: str = Field(description="Hardware target utilized.")
    input_shape: tuple[int, ...] = Field(description="Batch tensor input shape.")
    mean_latency_ms: float = Field(gt=0.0, description="Mean inference time in milliseconds.")
    median_latency_ms: float = Field(
        gt=0.0, description="Median inference latency in milliseconds."
    )
    p90_latency_ms: float = Field(gt=0.0, description="90th percentile latency in milliseconds.")
    p95_latency_ms: float = Field(gt=0.0, description="95th percentile latency in milliseconds.")
    p99_latency_ms: float = Field(gt=0.0, description="99th percentile latency in milliseconds.")
    min_latency_ms: float = Field(gt=0.0, description="Minimum latency in milliseconds.")
    max_latency_ms: float = Field(gt=0.0, description="Maximum latency in milliseconds.")
    fps: float = Field(gt=0.0, description="Throughput in frames per second.")


class InferenceResult(BaseModel):
    """Standard container for single-frame inference execution metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_ms: float = Field(ge=0.0, description="Execution time in milliseconds.")
    input_shape: tuple[int, ...] = Field(description="Preprocessed input tensor shape.")


class BatchSweepItem(BaseModel):
    """Metrics for a single evaluated batch size in a throughput sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=0, description="Evaluated batch size.")
    input_shape: tuple[int, ...] = Field(description="Full batch tensor input shape.")
    mean_latency_ms: float = Field(gt=0.0, description="Mean inference time in milliseconds.")
    median_latency_ms: float = Field(gt=0.0, description="Median inference time in milliseconds.")
    p95_latency_ms: float = Field(gt=0.0, description="95th percentile latency in milliseconds.")
    fps: float = Field(gt=0.0, description="Throughput in frames per second.")
    speedup: float = Field(
        ge=0.0, description="Throughput speedup relative to single-sample baseline."
    )
    scaling_efficiency: float = Field(
        ge=0.0, description="Batch scaling efficiency (speedup / batch_size)."
    )


class BatchSweepSummary(BaseModel):
    """Aggregated results of an inference throughput sweep across multiple batch sizes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(description="Inference runtime name.")
    device: str = Field(description="Target compute hardware device.")
    iterations: int = Field(gt=0, description="Timed iterations per batch size.")
    warmup: int = Field(ge=0, description="Warmup passes per batch size.")
    base_input_shape: tuple[int, ...] = Field(
        description="Per-sample base tensor shape (e.g. C, H, W)."
    )
    items: tuple[BatchSweepItem, ...] = Field(description="Ordered sweep results per batch size.")
    optimal_batch_size: int = Field(gt=0, description="Batch size that maximizes FPS throughput.")
    max_fps: float = Field(gt=0.0, description="Maximum achieved throughput in FPS.")


__all__ = [
    "BatchSweepItem",
    "BatchSweepSummary",
    "BenchmarkSummary",
    "InferenceBackendType",
    "InferenceDevice",
    "InferenceResult",
    "ModelMetadata",
]
