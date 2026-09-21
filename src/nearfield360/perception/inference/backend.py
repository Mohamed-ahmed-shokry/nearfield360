"""Inference runtime backend abstractions and OpenCV DNN engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from nearfield360.config import InferenceConfig
from nearfield360.perception.inference.models import (
    InferenceBackendType,
    InferenceDevice,
    ModelMetadata,
)

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when model loading, initialization, or forward execution fails."""


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol representing an execution runtime for neural models."""

    @property
    def metadata(self) -> ModelMetadata:
        """Inspect structural metadata of the loaded model."""
        ...

    @property
    def backend_type(self) -> InferenceBackendType:
        """Type of inference runtime backend."""
        ...

    @property
    def device(self) -> InferenceDevice:
        """Target hardware compute device."""
        ...

    def forward(self, blob: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Execute forward pass on preprocessed batch tensor.

        Args:
            blob: Contiguous float32 input array with shape (N, C, H, W).

        Returns:
            Output ndarray or tuple of ndarrays for multi-head models.
        """
        ...

    def warmup(self, iterations: int = 3, sample_shape: tuple[int, ...] = (1, 3, 64, 64)) -> None:
        """Execute untimed warmup passes to initialize memory and runtime caches."""
        ...


class OpenCVDNNBackend:
    """Inference backend powered by OpenCV's Deep Neural Network (DNN) engine."""

    def __init__(
        self,
        model_path: Path,
        device: InferenceDevice = InferenceDevice.CPU,
        precision: str = "fp32",
    ) -> None:
        if not model_path.is_file():
            raise InferenceError(f"Model file not found: {model_path}")

        self._model_path = model_path
        self._device = device
        self._precision = precision

        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error as exc:
            raise InferenceError(f"OpenCV failed to parse ONNX model {model_path}: {exc}") from exc

        self._configure_hardware(device)
        self._metadata = self._extract_metadata()

    def _configure_hardware(self, device: InferenceDevice) -> None:
        if device == InferenceDevice.CUDA:
            cuda_available = False
            if hasattr(cv2, "cuda"):
                try:
                    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                except Exception:
                    cuda_available = False

            if cuda_available:
                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("OpenCV DNN backend configured for CUDA.")
                    return
                except Exception as exc:
                    logger.warning(
                        "CUDA requested but OpenCV CUDA backend failed (%s). Falling back to CPU.",
                        exc,
                    )
            else:
                logger.warning(
                    "CUDA requested but no CUDA-enabled OpenCV devices found. Falling back to CPU."
                )

            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._device = InferenceDevice.CPU
        else:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _extract_metadata(self) -> ModelMetadata:
        raw_names = self._net.getUnconnectedOutLayersNames()
        out_names = tuple(str(name) for name in raw_names) if raw_names else ("output",)
        return ModelMetadata(
            model_path=str(self._model_path),
            backend=InferenceBackendType.OPENCV.value,
            device=self._device.value,
            input_names=("input",),
            input_shapes=(),
            output_names=out_names,
            output_shapes=(),
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def backend_type(self) -> InferenceBackendType:
        return InferenceBackendType.OPENCV

    @property
    def device(self) -> InferenceDevice:
        return self._device

    def forward(self, blob: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
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

        try:
            self._net.setInput(blob)
            out_names = list(self.metadata.output_names)
            if len(out_names) > 1:
                raw_outputs = self._net.forward(out_names)
                return tuple(np.asarray(out, dtype=np.float32) for out in raw_outputs)
            raw = self._net.forward()
            return np.asarray(raw, dtype=np.float32)
        except cv2.error as exc:
            raise InferenceError(f"OpenCV DNN forward inference failed: {exc}") from exc

    def warmup(self, iterations: int = 3, sample_shape: tuple[int, ...] = (1, 3, 64, 64)) -> None:
        if iterations <= 0:
            return
        dummy = np.zeros(sample_shape, dtype=np.float32)
        for _ in range(iterations):
            self.forward(dummy)


def create_backend(
    model_or_config: InferenceConfig | Path,
    model_path: Path | None = None,
    *,
    backend_type: InferenceBackendType | str = InferenceBackendType.OPENCV,
    device: InferenceDevice | str = InferenceDevice.CPU,
    precision: str = "fp32",
) -> InferenceBackend:
    """Instantiate the appropriate inference backend according to configuration or parameters."""
    prec: str
    if isinstance(model_or_config, InferenceConfig):
        if model_path is None:
            msg = "model_path is required when passing InferenceConfig"
            raise ValueError(msg)
        target_model = model_path
        b_type = model_or_config.backend.lower()
        dev = InferenceDevice(model_or_config.device.lower())
        prec = model_or_config.precision
    else:
        target_model = model_or_config
        b_type = (
            backend_type.value.lower()
            if isinstance(backend_type, InferenceBackendType)
            else str(backend_type).lower()
        )
        dev = (
            device if isinstance(device, InferenceDevice) else InferenceDevice(str(device).lower())
        )
        prec = precision

    if b_type == InferenceBackendType.OPENCV.value:
        return OpenCVDNNBackend(target_model, device=dev, precision=prec)
    if b_type == InferenceBackendType.ONNXRUNTIME.value:
        logger.warning("onnxruntime requested but not installed. Falling back to OpenCV DNN.")
        return OpenCVDNNBackend(target_model, device=dev, precision=prec)
    msg = f"Unsupported inference backend: {b_type}"
    raise InferenceError(msg)


__all__ = [
    "InferenceBackend",
    "InferenceError",
    "OpenCVDNNBackend",
    "create_backend",
]
