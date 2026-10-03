from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nearfield360.config import InferenceConfig
from nearfield360.perception.inference.backend import InferenceError, create_backend
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice
from nearfield360.perception.inference.tensorrt_backend import (
    TensorrtBackend,
    build_tensorrt_provider_options,
)


def test_build_tensorrt_provider_options_defaults() -> None:
    opts = build_tensorrt_provider_options(
        device_id=0,
        workspace_mb=1024,
        precision="fp32",
        cache_dir=None,
        dla_core=None,
    )
    assert opts["device_id"] == 0
    assert opts["trt_max_workspace_size"] == 1024 * 1024 * 1024
    assert opts["trt_fp16_enable"] is False
    assert opts["trt_int8_enable"] is False
    assert "trt_engine_cache_enable" not in opts
    assert "trt_dla_enable" not in opts


def test_build_tensorrt_provider_options_custom(tmp_path: Path) -> None:
    cache = tmp_path / "cache_trt"
    opts = build_tensorrt_provider_options(
        device_id=1,
        workspace_mb=2048,
        precision="int8",
        cache_dir=cache,
        dla_core=0,
    )
    assert opts["device_id"] == 1
    assert opts["trt_max_workspace_size"] == 2048 * 1024 * 1024
    assert opts["trt_fp16_enable"] is True
    assert opts["trt_int8_enable"] is True
    assert opts["trt_engine_cache_enable"] is True
    assert opts["trt_engine_cache_path"] == str(cache)
    assert opts["trt_dla_enable"] is True
    assert opts["trt_dla_core"] == 0


def test_tensorrt_backend_missing_model(tmp_path: Path) -> None:
    missing = tmp_path / "absent.onnx"
    with pytest.raises(InferenceError, match="Model file not found"):
        TensorrtBackend(missing)


def test_tensorrt_backend_unavailable_diagnostics(tmp_path: Path) -> None:
    model_file = tmp_path / "dummy.onnx"
    model_file.write_bytes(b"dummy")

    with pytest.raises(InferenceError, match="TensorRT acceleration runtime is unavailable") as exc:
        TensorrtBackend(model_file)

    msg = str(exc.value)
    assert "To enable TensorRT" in msg
    assert "compatible CUDA drivers" in msg


def test_tensorrt_backend_mocked_ort_execution(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"onnx_content")

    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 480, 640]

    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_output.shape = [1, 10, 480, 640]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.zeros((1, 10, 480, 640), dtype=np.float32)]

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.SessionOptions.return_value = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        backend = TensorrtBackend(
            model_file,
            precision="fp16",
            workspace_mb=512,
            cache_dir=tmp_path / "cache",
            dla_core=None,
        )

        assert backend.backend_type == InferenceBackendType.TENSORRT
        assert backend.device == InferenceDevice.CUDA
        assert backend.precision == "fp16"
        assert backend.workspace_mb == 512
        assert backend.cache_dir == tmp_path / "cache"
        assert backend.dla_core is None

        meta = backend.metadata
        assert meta.backend == "tensorrt"
        assert meta.input_names == ("images",)
        assert meta.output_names == ("output0",)

        # Single output forward pass
        blob = np.ones((1, 3, 480, 640), dtype=np.float32)
        out = backend.forward(blob)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 10, 480, 640)

        # Warmup pass
        backend.warmup(iterations=2, sample_shape=(1, 3, 480, 640))
        assert mock_session.run.call_count >= 3


def test_tensorrt_backend_multi_head_output(tmp_path: Path) -> None:
    model_file = tmp_path / "multihead.onnx"
    model_file.write_bytes(b"onnx_content")

    mock_in = MagicMock()
    mock_in.name = "input"
    mock_in.shape = [1, 3, 64, 64]

    mock_out1 = MagicMock()
    mock_out1.name = "boxes"
    mock_out1.shape = [1, 10, 4]

    mock_out2 = MagicMock()
    mock_out2.name = "scores"
    mock_out2.shape = [1, 10, 8]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_in]
    mock_session.get_outputs.return_value = [mock_out1, mock_out2]
    mock_session.run.return_value = [
        np.zeros((1, 10, 4), dtype=np.float32),
        np.ones((1, 10, 8), dtype=np.float32),
    ]

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["TensorrtExecutionProvider"]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        backend = TensorrtBackend(model_file)
        blob = np.zeros((1, 3, 64, 64), dtype=np.float32)
        outs = backend.forward(blob)
        assert isinstance(outs, tuple)
        assert len(outs) == 2
        assert outs[0].shape == (1, 10, 4)
        assert outs[1].shape == (1, 10, 8)


def test_tensorrt_backend_input_validation(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"content")

    mock_in = MagicMock(name="in")
    mock_in.name = "in"
    mock_in.shape = [1, 3, 32, 32]
    mock_out = MagicMock(name="out")
    mock_out.name = "out"
    mock_out.shape = [1, 5]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_in]
    mock_session.get_outputs.return_value = [mock_out]
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["TensorrtExecutionProvider"]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        backend = TensorrtBackend(model_file)

        with pytest.raises(InferenceError, match="must be a numpy ndarray"):
            backend.forward([1, 2, 3])  # type: ignore[arg-type]

        with pytest.raises(InferenceError, match="must have 4 dimensions"):
            backend.forward(np.zeros((32, 32), dtype=np.float32))


def test_create_backend_tensorrt_routing(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"content")

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["TensorrtExecutionProvider"]
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="i", shape=[1, 3, 32, 32])]
    mock_session.get_outputs.return_value = [MagicMock(name="o", shape=[1, 2])]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        # Direct call
        b1 = create_backend(model_file, backend_type="tensorrt", precision="fp16")
        assert isinstance(b1, TensorrtBackend)
        assert b1.precision == "fp16"

        # Config call
        cfg = InferenceConfig(backend="tensorrt", precision="int8", tensorrt_workspace_mb=256)
        b2 = create_backend(cfg, model_path=model_file)
        assert isinstance(b2, TensorrtBackend)
        assert b2.precision == "int8"
        assert b2.workspace_mb == 256
