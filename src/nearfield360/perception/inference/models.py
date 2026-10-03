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
    TENSORRT = "tensorrt"


class PrecisionType(StrEnum):
    """Supported floating-point and integer numerical precisions."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class QuantizationType(StrEnum):
    """Supported model quantization methods."""

    FP16 = "fp16"
    DYNAMIC_INT8 = "dynamic_int8"
    STATIC_INT8 = "static_int8"


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


class OptimizationSummary(BaseModel):
    """Summary metrics of neural network precision conversion and quantization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str = Field(description="Original input model path.")
    optimized_path: str = Field(description="Target optimized model path.")
    source_precision: str = Field(description="Initial model numerical precision.")
    target_precision: str = Field(description="Optimized target precision (fp16, int8).")
    quantization_type: str = Field(description="Quantization algorithm employed.")
    source_size_bytes: int = Field(gt=0, description="Filesystem byte size of source model.")
    optimized_size_bytes: int = Field(gt=0, description="Filesystem byte size of optimized model.")
    compression_ratio: float = Field(gt=0.0, description="Ratio of source size to optimized size.")
    node_count: int = Field(ge=0, description="Total operator nodes in the graph.")
    quantized_node_count: int = Field(ge=0, description="Number of nodes modified or converted.")
    max_absolute_drift: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum absolute output divergence on test sample.",
    )
    mean_absolute_drift: float | None = Field(
        default=None,
        ge=0.0,
        description="Mean absolute output divergence on test sample.",
    )


class CalibrationSummary(BaseModel):
    """Summary of INT8 calibration dataset generation and dynamic range profiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(description="Calibrated model path.")
    cache_path: str = Field(description="Output TensorRT calibration cache table path.")
    num_samples: int = Field(gt=0, description="Total sample frames used for calibration.")
    tensor_ranges: dict[str, tuple[float, float]] = Field(
        description="Dynamic range [min, max] per tensor name."
    )
    calibration_method: str = Field(
        default="entropy", description="Calibration algorithm (entropy, minmax)."
    )


class ProviderInfo(BaseModel):
    """Information for a single hardware acceleration execution provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Provider identifier (e.g. TensorrtExecutionProvider).")
    available: bool = Field(description="Whether the provider is currently available.")
    priority: int = Field(description="Execution priority order.")
    device: str = Field(description="Associated compute device (cpu, cuda, dla, directml).")
    details: str = Field(default="", description="Diagnostic details or requirement status.")


class ProvidersReport(BaseModel):
    """Consolidated report of all detected execution providers and acceleration hardware."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: tuple[ProviderInfo, ...] = Field(description="Ordered list of execution providers.")
    recommended_backend: str = Field(description="Best available inference backend.")
    recommended_device: str = Field(description="Best available compute device.")
    cuda_available: bool = Field(description="Whether NVIDIA CUDA is detected.")
    tensorrt_available: bool = Field(description="Whether TensorRT acceleration is detected.")


__all__ = [
    "BatchSweepItem",
    "BatchSweepSummary",
    "BenchmarkSummary",
    "CalibrationSummary",
    "InferenceBackendType",
    "InferenceDevice",
    "InferenceResult",
    "ModelMetadata",
    "OptimizationSummary",
    "PrecisionType",
    "ProviderInfo",
    "ProvidersReport",
    "QuantizationType",
]
