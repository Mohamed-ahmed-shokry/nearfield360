"""Synthetic ONNX neural network builders for fast, deterministic testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def create_dummy_segmentation_onnx(
    path: Path,
    *,
    num_classes: int = 10,
    height: int = 64,
    width: int = 64,
    class_biases: list[float] | None = None,
) -> Path:
    """Build a minimal 1x1 Convolution ONNX model outputting class logits.

    The model maps ``(1, 3, height, width)`` to ``(1, num_classes, height, width)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    weights = np.zeros((num_classes, 3, 1, 1), dtype=np.float32)
    # Give slight positive weight to first channel so input variations propagate
    weights[:, 0, 0, 0] = 0.1

    if class_biases is not None:
        biases = np.asarray(class_biases, dtype=np.float32)
    else:
        biases = np.zeros(num_classes, dtype=np.float32)

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_w", "conv_b"],
        outputs=["output"],
        kernel_shape=[1, 1],
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, num_classes, height, width]
    )

    w_init = helper.make_tensor(
        "conv_w", TensorProto.FLOAT, [num_classes, 3, 1, 1], weights.flatten()
    )
    b_init = helper.make_tensor("conv_b", TensorProto.FLOAT, [num_classes], biases.flatten())

    graph = helper.make_graph(
        [conv_node],
        "dummy_segmentation",
        [input_info],
        [output_info],
        [w_init, b_init],
    )
    model = helper.make_model(
        graph,
        producer_name="nearfield360-test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def create_dummy_detection_onnx(
    path: Path,
    *,
    num_classes: int = 5,
    num_boxes: int = 6,
    height: int = 64,
    width: int = 64,
    boxes_and_scores: np.ndarray | None = None,
) -> Path:
    """Build a minimal ONNX model outputting 2D bounding boxes and class logits.

    The model maps ``(1, 3, height, width)`` to ``(1, num_boxes, 4 + num_classes)``.
    Each row contains ``[cx, cy, w, h, class_0_score, ..., class_N_score]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    output_dim = 4 + num_classes
    if boxes_and_scores is not None:
        pred_data = np.asarray(boxes_and_scores, dtype=np.float32)
    else:
        pred_data = np.zeros((1, num_boxes, output_dim), dtype=np.float32)
        # Default box in center of image
        pred_data[0, :, 0] = 0.5  # cx
        pred_data[0, :, 1] = 0.5  # cy
        pred_data[0, :, 2] = 0.2  # w
        pred_data[0, :, 3] = 0.2  # h
        pred_data[0, :, 4] = 0.9  # class 0 confidence high

    pred_init = helper.make_tensor(
        "pred_const",
        TensorProto.FLOAT,
        [1, num_boxes, output_dim],
        pred_data.flatten(),
    )

    mul_zero = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])
    scale_node = helper.make_node("Mul", inputs=["input", "zero_val"], outputs=["zero_grid"])
    reduce_node = helper.make_node(
        "ReduceSum", inputs=["zero_grid"], outputs=["scalar_zero"], keepdims=0
    )
    add_node = helper.make_node("Add", inputs=["pred_const", "scalar_zero"], outputs=["output"])

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, num_boxes, output_dim]
    )

    graph = helper.make_graph(
        [scale_node, reduce_node, add_node],
        "dummy_detection",
        [input_info],
        [output_info],
        [pred_init, mul_zero],
    )
    model = helper.make_model(
        graph,
        producer_name="nearfield360-test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


__all__ = [
    "create_dummy_detection_onnx",
    "create_dummy_segmentation_onnx",
]
