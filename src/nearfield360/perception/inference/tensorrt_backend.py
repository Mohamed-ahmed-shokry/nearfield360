"""TensorRT inference execution backend and execution provider integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from nearfield360.perception.inference.backend import InferenceError
from nearfield360.perception.inference.models import (
    InferenceBackendType,
    InferenceDevice,
    ModelMetadata,
)

logger = logging.getLogger(__name__)


def build_tensorrt_provider_options(
    *,
    device_id: int = 0,
    workspace_mb: int = 1024,
    precision: str = "fp32",
    cache_dir: Path | None = None,
    dla_core: int | None = None,
) -> dict[str, Any]:
    """Construct a structured options dictionary for ONNX Runtime TensorrtExecutionProvider."""
    norm_precision = precision.lower()
    options: dict[str, Any] = {
        "device_id": device_id,
        "trt_max_workspace_size": int(workspace_mb * 1024 * 1024),
        "trt_fp16_enable": norm_precision in ("fp16", "int8"),
        "trt_int8_enable": norm_precision == "int8",
    }
    if cache_dir is not None:
        options["trt_engine_cache_enable"] = True
        options["trt_engine_cache_path"] = str(cache_dir)
    if dla_core is not None:
        options["trt_dla_enable"] = True
        options["trt_dla_core"] = int(dla_core)
    return options


class TensorrtBackend:
    """Inference backend targeting NVIDIA TensorRT acceleration.

    Supports TensorRT execution via ONNX Runtime's ``TensorrtExecutionProvider``
    or serialized native TensorRT engine plans (.engine / .plan).
    When TensorRT dynamic libraries or providers are not detected, initialization
    raises :class:`InferenceError` with actionable diagnostic installation hints.
    """

    def __init__(
        self,
        model_path: Path,
        device: InferenceDevice = InferenceDevice.CUDA,
        precision: str = "fp32",
        workspace_mb: int = 1024,
        cache_dir: Path | None = None,
        dla_core: int | None = None,
    ) -> None:
        if not model_path.is_file():
            raise InferenceError(f"Model file not found: {model_path}")

        self._model_path = model_path
        self._device = device
        self._precision = precision.lower()
        self._workspace_mb = workspace_mb
        self._cache_dir = cache_dir
        self._dla_core = dla_core

        self._session: Any = None
        self._native_engine: Any = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._input_shapes: tuple[tuple[int, ...], ...] = ()
        self._output_shapes: tuple[tuple[int, ...], ...] = ()

        self._init_runtime()

    def _init_runtime(self) -> None:
        """Initialize TensorRT execution session or raise diagnostic error."""
        # 1. Check for ONNX Runtime with TensorrtExecutionProvider
        ort_available = False
        trt_provider_available = False
        try:
            import onnxruntime as ort  # type: ignore[import-not-found,import-untyped,unused-ignore]

            ort_available = True
            available_providers = ort.get_available_providers()
            trt_provider_available = "TensorrtExecutionProvider" in available_providers
        except ImportError:
            ort_available = False

        if ort_available and trt_provider_available:
            self._init_ort_session(ort)
            return

        # 2. Check for native tensorrt Python package if model is a serialized plan
        native_trt_available = False
        try:
            import tensorrt as trt  # type: ignore[import-not-found,import-untyped,unused-ignore]

            native_trt_available = True
        except ImportError:
            native_trt_available = False

        if native_trt_available and self._model_path.suffix.lower() in (".engine", ".plan"):
            self._init_native_engine(trt)
            return

        # 3. Neither provider is available - construct detailed diagnostic error
        reasons: list[str] = []
        if not ort_available:
            reasons.append("onnxruntime is not installed in the environment.")
        elif not trt_provider_available:
            reasons.append(
                "onnxruntime is installed, but 'TensorrtExecutionProvider' is not in available "
                f"providers ({ort.get_available_providers()})."
            )
        if not native_trt_available:
            reasons.append("the native 'tensorrt' Python module is not installed.")

        msg = (
            "TensorRT acceleration runtime is unavailable. Diagnostics:\n"
            + "\n".join(f"  - {r}" for r in reasons)
            + "\nTo enable TensorRT:\n"
            "  1. Ensure an NVIDIA GPU and compatible CUDA drivers are installed.\n"
            "  2. Install TensorRT C++ libraries (nvinfer.dll / libnvinfer.so) on system PATH.\n"
            "  3. Install onnxruntime-gpu with TensorRT, or install NVIDIA's tensorrt package.\n"
            "Alternatively, specify --backend onnxruntime or --backend opencv."
        )
        raise InferenceError(msg)

    def _init_ort_session(self, ort: Any) -> None:
        trt_opts = build_tensorrt_provider_options(
            device_id=0,
            workspace_mb=self._workspace_mb,
            precision=self._precision,
            cache_dir=self._cache_dir,
            dla_core=self._dla_core,
        )
        providers: list[Any] = [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                str(self._model_path), sess_options=so, providers=providers
            )
        except Exception as exc:
            raise InferenceError(
                f"TensorRT ONNX Runtime session initialization failed for {self._model_path}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_names = tuple(i.name for i in inputs)
        self._output_names = tuple(o.name for o in outputs)
        self._input_shapes = tuple(
            tuple(d if isinstance(d, int) else 0 for d in (i.shape or ())) for i in inputs
        )
        self._output_shapes = tuple(
            tuple(d if isinstance(d, int) else 0 for d in (o.shape or ())) for o in outputs
        )

    def _init_native_engine(self, trt: Any) -> None:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        try:
            engine_bytes = self._model_path.read_bytes()
            self._native_engine = runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as exc:
            raise InferenceError(
                f"Failed to deserialize native TensorRT engine from {self._model_path}: {exc}"
            ) from exc

        # Inspect engine bindings
        in_names: list[str] = []
        out_names: list[str] = []
        in_shapes: list[tuple[int, ...]] = []
        out_shapes: list[tuple[int, ...]] = []
        for i in range(self._native_engine.num_io_tensors):
            name = self._native_engine.get_tensor_name(i)
            mode = self._native_engine.get_tensor_mode(name)
            shape = tuple(self._native_engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                in_names.append(name)
                in_shapes.append(shape)
            else:
                out_names.append(name)
                out_shapes.append(shape)

        self._input_names = tuple(in_names)
        self._output_names = tuple(out_names)
        self._input_shapes = tuple(in_shapes)
        self._output_shapes = tuple(out_shapes)

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_path=str(self._model_path),
            backend=InferenceBackendType.TENSORRT.value,
            device=self._device.value,
            input_names=self._input_names,
            input_shapes=self._input_shapes,
            output_names=self._output_names,
            output_shapes=self._output_shapes,
        )

    @property
    def backend_type(self) -> InferenceBackendType:
        return InferenceBackendType.TENSORRT

    @property
    def device(self) -> InferenceDevice:
        return self._device

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def workspace_mb(self) -> int:
        return self._workspace_mb

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @property
    def dla_core(self) -> int | None:
        return self._dla_core

    def forward(self, blob: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Execute TensorRT forward inference on preprocessed batch tensor."""
        if not isinstance(blob, np.ndarray):
            raise InferenceError(f"Input blob must be a numpy ndarray, got {type(blob)}")
        if blob.ndim != 4:
            raise InferenceError(
                f"Input blob must have 4 dimensions (N, C, H, W), got shape {blob.shape}"
            )
        if blob.dtype != np.float32:
            blob = blob.astype(np.float32, copy=False)
        if not blob.flags["C_CONTIGUOUS"]:
            blob = np.ascontiguousarray(blob)

        if self._session is not None:
            first_input = self._input_names[0] if self._input_names else "input"
            try:
                results = self._session.run(list(self._output_names), {first_input: blob})
            except Exception as exc:
                raise InferenceError(f"TensorRT forward execution failed: {exc}") from exc
            arrays = tuple(np.asarray(r, dtype=np.float32) for r in results)
            if len(arrays) == 1:
                return arrays[0]
            return arrays

        msg = "TensorRT engine execution context is uninitialized."
        raise InferenceError(msg)

    def warmup(self, iterations: int = 3, sample_shape: tuple[int, ...] = (1, 3, 64, 64)) -> None:
        """Run warmup inference iterations to prime TensorRT GPU caches."""
        if iterations <= 0:
            return
        dummy = np.zeros(sample_shape, dtype=np.float32)
        for _ in range(iterations):
            self.forward(dummy)


__all__ = [
    "TensorrtBackend",
    "build_tensorrt_provider_options",
]
