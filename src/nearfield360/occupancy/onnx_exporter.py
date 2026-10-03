"""Export ONNX computation graph for temporal BEV occupancy forecasting network."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def export_temporal_forecaster_onnx(
    path: Path,
    *,
    grid_shape: tuple[int, int] = (160, 160),
    in_channels: int = 4,
    hidden_channels: int = 16,
    horizon_steps: int = 6,
    opset_version: int = 13,
    seed: int = 42,
) -> Path:
    """Export an end-to-end spatiotemporal BEV occupancy forecasting ONNX model.

    The model accepts:
    - ``bev_features``: ``(1, in_channels, height, width)``
    - ``hidden_state``: ``(1, hidden_channels, height, width)``

    And outputs:
    - ``updated_hidden_state``: ``(1, hidden_channels, height, width)``
    - ``forecast_occupancy``: ``(1, horizon_steps, height, width)``
    - ``velocity_field``: ``(1, 2, height, width)``
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = grid_shape
    rng = np.random.default_rng(seed)

    # 1. Tensor Value Infos
    input_bev = helper.make_tensor_value_info(
        "bev_features", TensorProto.FLOAT, [1, in_channels, height, width]
    )
    input_hid = helper.make_tensor_value_info(
        "hidden_state", TensorProto.FLOAT, [1, hidden_channels, height, width]
    )

    out_hid = helper.make_tensor_value_info(
        "updated_hidden_state", TensorProto.FLOAT, [1, hidden_channels, height, width]
    )
    out_occ = helper.make_tensor_value_info(
        "forecast_occupancy", TensorProto.FLOAT, [1, horizon_steps, height, width]
    )
    out_vel = helper.make_tensor_value_info(
        "velocity_field", TensorProto.FLOAT, [1, 2, height, width]
    )

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    def make_init(name: str, array: np.ndarray) -> onnx.TensorProto:
        tensor = helper.make_tensor(
            name,
            TensorProto.FLOAT,
            list(array.shape),
            array.astype(np.float32).flatten(),
        )
        initializers.append(tensor)
        return tensor

    # 2. Recurrent ConvGRU Gates
    scale_in = math.sqrt(2.0 / (in_channels * 9))
    scale_hid = math.sqrt(2.0 / (hidden_channels * 9))

    # Update gate z
    w_z = rng.normal(0.0, scale_in, size=(hidden_channels, in_channels, 3, 3))
    u_z = rng.normal(0.0, scale_hid, size=(hidden_channels, hidden_channels, 3, 3))
    b_z = np.zeros(hidden_channels, dtype=np.float32)
    make_init("W_z", w_z)
    make_init("U_z", u_z)
    make_init("b_z", b_z)

    nodes.append(
        helper.make_node(
            "Conv",
            ["bev_features", "W_z", "b_z"],
            ["conv_w_z"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            ["hidden_state", "U_z"],
            ["conv_u_z"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(helper.make_node("Add", ["conv_w_z", "conv_u_z"], ["z_pre"]))
    nodes.append(helper.make_node("Sigmoid", ["z_pre"], ["z_t"]))

    # Reset gate r
    w_r = rng.normal(0.0, scale_in, size=(hidden_channels, in_channels, 3, 3))
    u_r = rng.normal(0.0, scale_hid, size=(hidden_channels, hidden_channels, 3, 3))
    b_r = np.zeros(hidden_channels, dtype=np.float32)
    make_init("W_r", w_r)
    make_init("U_r", u_r)
    make_init("b_r", b_r)

    nodes.append(
        helper.make_node(
            "Conv",
            ["bev_features", "W_r", "b_r"],
            ["conv_w_r"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            ["hidden_state", "U_r"],
            ["conv_u_r"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(helper.make_node("Add", ["conv_w_r", "conv_u_r"], ["r_pre"]))
    nodes.append(helper.make_node("Sigmoid", ["r_pre"], ["r_t"]))

    # Candidate gate h_cand
    nodes.append(helper.make_node("Mul", ["r_t", "hidden_state"], ["r_hid"]))

    w_h = rng.normal(0.0, scale_in, size=(hidden_channels, in_channels, 3, 3))
    u_h = rng.normal(0.0, scale_hid, size=(hidden_channels, hidden_channels, 3, 3))
    b_h = np.zeros(hidden_channels, dtype=np.float32)
    make_init("W_h", w_h)
    make_init("U_h", u_h)
    make_init("b_h", b_h)

    nodes.append(
        helper.make_node(
            "Conv",
            ["bev_features", "W_h", "b_h"],
            ["conv_w_h"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            ["r_hid", "U_h"],
            ["conv_u_h"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(helper.make_node("Add", ["conv_w_h", "conv_u_h"], ["h_pre"]))
    nodes.append(helper.make_node("Tanh", ["h_pre"], ["h_cand"]))

    # Recurrent combination: h_t = (1 - z) * hidden_state + z * h_cand
    const_one = np.ones((1, hidden_channels, 1, 1), dtype=np.float32)
    make_init("const_one", const_one)
    nodes.append(helper.make_node("Sub", ["const_one", "z_t"], ["one_minus_z"]))
    nodes.append(helper.make_node("Mul", ["one_minus_z", "hidden_state"], ["decayed_hid"]))
    nodes.append(helper.make_node("Mul", ["z_t", "h_cand"], ["new_hid"]))
    nodes.append(
        helper.make_node(
            "Add",
            ["decayed_hid", "new_hid"],
            ["updated_hidden_state"],
        )
    )

    # 3. Output Forecasting Head (Multi-Step Occupancy Grids)
    w_occ = rng.normal(0.0, scale_hid, size=(horizon_steps, hidden_channels, 1, 1))
    b_occ = np.zeros(horizon_steps, dtype=np.float32)
    make_init("W_occ", w_occ)
    make_init("b_occ", b_occ)

    nodes.append(
        helper.make_node(
            "Conv",
            ["updated_hidden_state", "W_occ", "b_occ"],
            ["occ_logits"],
            kernel_shape=[1, 1],
        )
    )
    nodes.append(helper.make_node("Sigmoid", ["occ_logits"], ["forecast_occupancy"]))

    # 4. Output Velocity Field Head (vx, vy in m/s)
    w_vel = rng.normal(0.0, scale_hid, size=(2, hidden_channels, 1, 1))
    b_vel = np.zeros(2, dtype=np.float32)
    make_init("W_vel", w_vel)
    make_init("b_vel", b_vel)

    nodes.append(
        helper.make_node(
            "Conv",
            ["updated_hidden_state", "W_vel", "b_vel"],
            ["velocity_field"],
            kernel_shape=[1, 1],
        )
    )

    # 5. Build and validate ONNX Graph
    graph = helper.make_graph(
        nodes,
        "temporal_bev_occupancy_forecaster",
        [input_bev, input_hid],
        [out_hid, out_occ, out_vel],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="nearfield360-perception",
        opset_imports=[helper.make_opsetid("", opset_version)],
        ir_version=8,
    )

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


__all__ = ["export_temporal_forecaster_onnx"]
