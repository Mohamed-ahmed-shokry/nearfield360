"""Inference runtime, preprocessing, and model execution engines."""

from __future__ import annotations

from nearfield360.perception.inference.backend import (
    InferenceBackend,
    InferenceError,
    OnnxRuntimeBackend,
    OpenCVDNNBackend,
    create_backend,
)
from nearfield360.perception.inference.benchmark import (
    BatchSweepItem,
    BatchSweepSummary,
    ParityComparison,
    benchmark_batch_sweep,
    benchmark_inference,
    compare_numerical_parity,
    verify_numerical_parity,
)
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.memory import (
    BufferAllocation,
    CUDAPinnedBufferPool,
    MemoryLayoutPlan,
    PinnedMemoryBuffer,
    is_cuda_pinned_memory_available,
)
from nearfield360.perception.inference.models import (
    BenchmarkSummary,
    CalibrationSummary,
    InferenceBackendType,
    InferenceDevice,
    InferenceResult,
    ModelMetadata,
    OptimizationSummary,
    PrecisionType,
    ProviderInfo,
    ProvidersReport,
    QuantizationType,
)
from nearfield360.perception.inference.optimization import (
    OptimizationError,
    convert_model_to_fp16,
    detect_hardware_providers,
    generate_int8_calibration_table,
    optimize_model_precision,
    quantize_model_dynamic_int8,
)
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
    PreprocessorError,
    PreprocessTransform,
)
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.perception.inference.tensorrt_backend import (
    TensorrtBackend,
    build_tensorrt_provider_options,
)

__all__ = [
    "BatchSweepItem",
    "BatchSweepSummary",
    "BenchmarkSummary",
    "BufferAllocation",
    "CUDAPinnedBufferPool",
    "CalibrationSummary",
    "FisheyeImagePreprocessor",
    "InferenceBackend",
    "InferenceBackendType",
    "InferenceDevice",
    "InferenceError",
    "InferenceResult",
    "MemoryLayoutPlan",
    "ModelMetadata",
    "ObjectDetectionEngine",
    "OnnxRuntimeBackend",
    "OpenCVDNNBackend",
    "OptimizationError",
    "OptimizationSummary",
    "ParityComparison",
    "PinnedMemoryBuffer",
    "PrecisionType",
    "PreprocessTransform",
    "PreprocessorError",
    "ProviderInfo",
    "ProvidersReport",
    "QuantizationType",
    "SemanticSegmentationEngine",
    "TensorrtBackend",
    "benchmark_batch_sweep",
    "benchmark_inference",
    "build_tensorrt_provider_options",
    "compare_numerical_parity",
    "convert_model_to_fp16",
    "create_backend",
    "detect_hardware_providers",
    "generate_int8_calibration_table",
    "is_cuda_pinned_memory_available",
    "optimize_model_precision",
    "quantize_model_dynamic_int8",
    "verify_numerical_parity",
]
