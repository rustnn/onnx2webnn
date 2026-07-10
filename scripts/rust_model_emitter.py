# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit readable Rust source that builds ONNX ModelProto values programmatically."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto, numpy_helper


def _escape_rust_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def _format_i64_list(values: list[int]) -> str:
    inner = ", ".join(str(v) for v in values)
    return f"&[{inner}]"


def _format_f32_list(values: list[float]) -> str:
    parts: list[str] = []
    for value in values:
        if value == int(value) and abs(value) < 1e15:
            parts.append(f"{int(value)}.0")
        else:
            parts.append(repr(float(value)))
    return f"&[{', '.join(parts)}]"


def _format_bool_list(values: list[bool]) -> str:
    inner = ", ".join("true" if v else "false" for v in values)
    return f"&[{inner}]"


def _format_u8_list(values: list[int]) -> str:
    inner = ", ".join(str(int(v)) for v in values)
    return f"&[{inner}]"


def _format_u16_list(values: list[int]) -> str:
    inner = ", ".join(str(int(v)) for v in values)
    return f"&[{inner}]"


def _format_i8_list(values: list[int]) -> str:
    inner = ", ".join(str(int(v)) for v in values)
    return f"&[{inner}]"


def _format_str_list(values: list[str]) -> str:
    inner = ", ".join(f"\"{_escape_rust_str(v)}\"" for v in values)
    return f"&[{inner}]"


def _tensor_dims(tensor: TensorProto) -> list[int]:
    return [int(d) for d in tensor.dims]


def _emit_initializer(init: TensorProto, *, indent: str) -> str:
    name = _escape_rust_str(init.name)
    shape = _format_i64_list(_tensor_dims(init))
    arr = numpy_helper.to_array(init)

    if init.data_type == TensorProto.FLOAT:
        data = _format_f32_list(arr.flatten().astype(np.float32).tolist())
        return f'{indent}f32_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.FLOAT16:
        data = _format_u16_list(arr.flatten().astype(np.float16).view(np.uint16).tolist())
        return f'{indent}f16_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.INT32:
        data = _format_i64_list(arr.flatten().astype(np.int32).tolist())
        return f'{indent}i32_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.INT64:
        data = _format_i64_list(arr.flatten().astype(np.int64).tolist())
        return f'{indent}i64_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.UINT8:
        data = _format_u8_list(arr.flatten().astype(np.uint8).tolist())
        return f'{indent}u8_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.INT8:
        data = _format_i8_list(arr.flatten().astype(np.int8).tolist())
        return f'{indent}i8_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.BOOL:
        data = _format_bool_list(arr.flatten().astype(bool).tolist())
        return f'{indent}bool_init("{name}", {shape}, {data}),'
    if init.data_type == TensorProto.STRING:
        values = [str(x) for x in arr.flatten().tolist()]
        data = _format_str_list(values)
        return f'{indent}string_init("{name}", {shape}, {data}),'

    raise ValueError(f"unsupported initializer dtype {init.data_type} for {init.name}")


def _emit_attr_tensor(tensor: TensorProto, *, indent: str) -> str:
    init_line = _emit_initializer(tensor, indent=indent + "    ").rstrip(",")
    return f"{indent}attr_tensor(\"{_escape_rust_str(tensor.name)}\", {init_line.lstrip()}),"


def _emit_attribute(attr: AttributeProto, *, indent: str) -> str:
    name = _escape_rust_str(attr.name)
    if attr.type == AttributeProto.INT:
        return f'{indent}attr_int("{name}", {int(attr.i)}),'
    if attr.type == AttributeProto.FLOAT:
        return f'{indent}attr_float("{name}", {float(attr.f)}),'
    if attr.type == AttributeProto.STRING:
        value = attr.s.decode("utf-8", errors="replace")
        return f'{indent}attr_string("{name}", "{_escape_rust_str(value)}"),'
    if attr.type == AttributeProto.INTS:
        return f'{indent}attr_ints("{name}", {_format_i64_list(list(attr.ints))}),'
    if attr.type == AttributeProto.FLOATS:
        return f'{indent}attr_floats("{name}", {_format_f32_list(list(attr.floats))}),'
    if attr.type == AttributeProto.TENSOR and attr.HasField("t"):
        tensor = attr.t
        init_line = _emit_initializer(tensor, indent=indent + "    ").rstrip(",")
        return f'{indent}attr_tensor("{name}", {init_line.lstrip()}),'
    if attr.type == AttributeProto.GRAPH and attr.HasField("g"):
        graph_expr = emit_graph(attr.g, indent=indent + "    ").strip()
        return f"{indent}attr_graph(\"{name}\", {graph_expr}),"
    raise ValueError(f"unsupported attribute type {attr.type} for {attr.name}")


def _emit_value_info(vi, *, indent: str, is_output: bool) -> str:
    name = _escape_rust_str(vi.name)
    ty = vi.type
    if ty.HasField("tensor_type"):
        elem_type = int(ty.tensor_type.elem_type)
        shape = [int(d.dim_value) for d in ty.tensor_type.shape.dim]
        shape_lit = _format_i64_list(shape)
        suffix = "output" if is_output else "input"
        mapping = {
            TensorProto.FLOAT: "f32",
            TensorProto.FLOAT16: "f16",
            TensorProto.INT8: "i8",
            TensorProto.INT32: "i32",
            TensorProto.INT64: "i64",
            TensorProto.UINT8: "u8",
            TensorProto.BOOL: "bool",
            TensorProto.STRING: "string",
        }
        if elem_type in mapping:
            helper = f"{mapping[elem_type]}_{suffix}"
            return f'{indent}{helper}("{name}", {shape_lit}),'
        return (
            f'{indent}tensor_{suffix}("{name}", '
            f"TensorProto_DataType::{elem_type} as i32, {shape_lit}),"
        )
    if ty.HasField("sequence_type"):
        elem = ty.sequence_type.elem_type.tensor_type
        shape = [int(d.dim_value) for d in elem.shape.dim]
        if is_output and int(elem.elem_type) == TensorProto.FLOAT:
            return f'{indent}sequence_f32_output("{name}", {_format_i64_list(shape)}),'
        raise ValueError(f"unsupported sequence value info for {name}")
    if ty.HasField("optional_type"):
        elem = ty.optional_type.elem_type.tensor_type
        shape = [int(d.dim_value) for d in elem.shape.dim]
        if is_output and int(elem.elem_type) == TensorProto.FLOAT:
            return f'{indent}optional_f32_output("{name}", {_format_i64_list(shape)}),'
        raise ValueError(f"unsupported optional value info for {name}")
    raise ValueError(f"unsupported value info kind for {name}")


def _emit_node(node, *, indent: str) -> str:
    inputs = ", ".join(f"\"{_escape_rust_str(name)}\"" for name in node.input)
    outputs = ", ".join(f"\"{_escape_rust_str(name)}\"" for name in node.output)
    lines = [
        f"{indent}node(",
        f'{indent}    "{_escape_rust_str(node.op_type)}",',
        f'{indent}    "{_escape_rust_str(node.name)}",',
        f"{indent}    &[{inputs}],",
        f"{indent}    &[{outputs}],",
    ]
    if node.attribute:
        lines.append(f"{indent}    &[")
        for attr in node.attribute:
            lines.append(_emit_attribute(attr, indent=indent + "        "))
        lines.append(f"{indent}    ],")
    else:
        lines.append(f"{indent}    &[],")
    lines.append(f"{indent}),")
    return "\n".join(lines)


def emit_graph(graph: GraphProto, *, indent: str = "    ") -> str:
    lines = [
        "graph(",
        f'{indent}    "{_escape_rust_str(graph.name)}",',
        f"{indent}    vec![",
    ]
    for vi in graph.input:
        lines.append(_emit_value_info(vi, indent=indent + "        ", is_output=False))
    lines.append(f"{indent}    ],")
    lines.append(f"{indent}    vec![")
    for vi in graph.output:
        lines.append(_emit_value_info(vi, indent=indent + "        ", is_output=True))
    lines.append(f"{indent}    ],")
    lines.append(f"{indent}    vec![")
    for node in graph.node:
        lines.append(_emit_node(node, indent=indent + "        "))
    lines.append(f"{indent}    ],")
    lines.append(f"{indent}    vec![")
    for init in graph.initializer:
        lines.append(_emit_initializer(init, indent=indent + "        "))
    lines.append(f"{indent}    ],")
    lines.append(f"{indent})")
    return "\n".join(lines)


def emit_model(model: ModelProto, *, indent: str = "    ") -> str:
    if not model.opset_import:
        raise ValueError("model missing opset_import")
    opset = int(model.opset_import[0].version)
    if model.graph is None:
        raise ValueError("model missing graph")
    graph_expr = emit_graph(model.graph, indent=indent + "    ")
    return "\n".join(
        [
            "model(",
            f"{indent}    {opset},",
            f"{indent}    {graph_expr},",
            f"{indent})",
        ]
    )


def emit_build_function(model: ModelProto, *, fn_name: str = "build_fixture") -> list[str]:
    body = emit_model(model, indent="    ")
    return [
        f"fn {fn_name}() -> ModelProto {{",
        "    use onnx2webnn::test_models::prelude::*;",
        "",
        f"    {body}",
        "}",
        "",
    ]
