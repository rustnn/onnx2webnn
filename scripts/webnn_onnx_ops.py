# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX operators supported by onnx2webnn convert (mirrors src/onnx/ops/*.rs)."""

from __future__ import annotations

# ai.onnx ops accepted by OpRegistry in webnn-graph.
WEBNN_SUPPORTED_ONNX_OPS: frozenset[str] = frozenset(
    {
        # matmul.rs
        "MatMul",
        "Gemm",
        # conv.rs
        "Conv",
        "ConvTranspose",
        # pool.rs
        "MaxPool",
        "AveragePool",
        "GlobalMaxPool",
        "GlobalAveragePool",
        # elementwise.rs
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Min",
        "Max",
        # comparison.rs
        "Greater",
        "Less",
        "Equal",
        "GreaterOrEqual",
        "LessOrEqual",
        # conditional.rs
        "Where",
        # normalization.rs
        "LayerNormalization",
        "Softmax",
        # reshape.rs
        "Reshape",
        "Transpose",
        "Concat",
        "Split",
        "Unsqueeze",
        "Squeeze",
        "Tile",
        "Expand",
        "Flatten",
        # conversion.rs
        "Cast",
        "Constant",
        # utility.rs
        "Shape",
        "Gather",
        "Slice",
        "ConstantOfShape",
        "Range",
        "Trilu",
        # reduction.rs
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        # activation.rs
        "Relu",
        "Gelu",
        "Tanh",
        "Sigmoid",
        "Sqrt",
        "Exp",
        "Log",
        "Abs",
        "Neg",
        "Erf",
        "Cos",
        "Sin",
        "Identity",
        # scatter.rs
        "ScatterND",
        # pad.rs
        "Pad",
    }
)

# convert.rs MIN/MAX_SUPPORTED_OPSET for domain ai.onnx
WEBNN_MIN_OPSET: int = 11
WEBNN_MAX_OPSET: int = 18


def is_webnn_supported_op(domain: str, op_type: str) -> bool:
    if domain != "ai.onnx":
        return False
    return op_type in WEBNN_SUPPORTED_ONNX_OPS


def is_webnn_supported_opset(domain: str, version: int) -> bool:
    if domain != "ai.onnx":
        return True
    return WEBNN_MIN_OPSET <= version <= WEBNN_MAX_OPSET
