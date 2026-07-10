#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build in-memory ONNX ModelProto fixtures for Rust op tests.

Uses the installed onnx schema registry (onnx.defs) at the requested opset version.
Models are validated with onnx.checker before building.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
from onnx import AttributeProto, ModelProto, TensorProto, checker, defs, helper, numpy_helper

from onnx_test_builders import CUSTOM_BUILDERS as _EXTRA_CUSTOM_BUILDERS

ONNX_DOMAIN = ""

DEFAULT_VECTOR_SHAPE = [1, 2]
SPATIAL_SHAPE = [1, 1, 4, 4]

# Operators whose primary tensor input must be NCHW (or similar rank >= 3).
SPATIAL_OPS = frozenset(
    {
        "AveragePool",
        "BatchNormalization",
        "Conv",
        "ConvTranspose",
        "DeformConv",
        "DepthToSpace",
        "GlobalAveragePool",
        "GlobalLpPool",
        "GlobalMaxPool",
        "InstanceNormalization",
        "LRN",
        "LpPool",
        "MaxPool",
        "Pad",
        "RoiAlign",
        "SpaceToDepth",
    }
)

# Primary tensor argument names for spatial / data inputs.
_DATA_INPUT_NAMES = frozenset({"X", "data", "input", "A"})

# Logical / bitwise ops need boolean tensors (ORT rejects float for these).
BOOL_INPUT_OPS = frozenset(
    {
        "And",
        "BitwiseAnd",
        "BitwiseNot",
        "BitwiseOr",
        "BitwiseXor",
        "Not",
        "Or",
        "Xor",
    }
)

# Comparison and predicate ops produce boolean outputs.
BOOL_OUTPUT_OPS = frozenset(
    {
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "IsInf",
        "IsNaN",
        "Less",
        "LessOrEqual",
        "Not",
    }
)

# Outputs that are not float32 in typical test graphs.
INT64_OUTPUT_OPS = frozenset(
    {
        "ArgMax",
        "ArgMin",
        "NonZero",
        "Shape",
        "Size",
    }
)


def _is_optional(param) -> bool:
    return param.option.name == "Optional"


def _is_variadic(param) -> bool:
    return param.option.name == "Variadic"


MIN_SUPPORTED_OPSET = 9
MAX_SUPPORTED_OPSET = 26


def _all_onnx_op_names() -> list[str]:
    names: set[str] = set()
    for schema in defs.get_all_schemas_with_history():
        domain = schema.domain or ""
        if domain not in ("", "ai.onnx"):
            continue
        names.add(schema.name)
    return sorted(names)


def _op_available_at(version: int, name: str, *, include_deprecated: bool = True) -> bool:
    if not defs.has(name, version, ONNX_DOMAIN):
        return False
    if not include_deprecated:
        schema = defs.get_schema(name, version, ONNX_DOMAIN)
        if schema.deprecated:
            return False
    return True


def _ops_at_opset(version: int) -> list[str]:
    return [
        name
        for name in _all_onnx_op_names()
        if _op_available_at(version, name, include_deprecated=False)
    ]


def _schema_fingerprint(schema) -> tuple:
    """Structural signature: inputs, outputs, and attribute names."""
    inputs = tuple((p.name, p.option.name, p.type_str) for p in schema.inputs)
    outputs = tuple((p.name, p.option.name, p.type_str) for p in schema.outputs)
    attrs = tuple(sorted(schema.attributes.keys()))
    return (inputs, outputs, attrs)


def _schema_revisions_for_op(op_type: str) -> list:
    revisions = []
    for schema in defs.get_all_schemas_with_history():
        domain = schema.domain or ""
        if schema.name != op_type or domain not in ("", "ai.onnx"):
            continue
        revisions.append(schema)
    revisions.sort(key=lambda s: s.since_version)
    return revisions


def _schema_structure_bands(
    op_type: str, max_version: int
) -> list[tuple[tuple, int, int]]:
    """Merge consecutive ONNX schema revisions with the same structure into bands."""
    revisions = _schema_revisions_for_op(op_type)
    if not revisions:
        return []
    bands: list[tuple[tuple, int, int]] = []
    for i, rev in enumerate(revisions):
        fp = _schema_fingerprint(rev)
        sv = rev.since_version
        end = revisions[i + 1].since_version - 1 if i + 1 < len(revisions) else max_version
        if bands and bands[-1][0] == fp:
            bands[-1] = (fp, bands[-1][1], end)
        else:
            bands.append((fp, sv, end))
    return bands


def fixture_opsets_for_op(
    op_type: str, min_version: int, max_version: int
) -> list[int]:
    """Return opsets to test: one per distinct ONNX schema structure in range.

    For each structure band (e.g. Pad with ``pads`` attribute vs ``pads`` input), pick the
    highest buildable opset in ``[min_version, max_version]`` that still uses that structure.
    Falls back to the band's ``since_version`` when the high end is not buildable.
    """
    buildable: list[int] = []
    for _fp, since_version, band_end in _schema_structure_bands(op_type, max_version):
        if band_end < min_version or since_version > max_version:
            continue
        lo = max(since_version, min_version)
        hi = min(band_end, max_version)
        if lo > hi:
            continue
        for opset in (hi, lo):
            try:
                build_test_model(op_type, opset)
                buildable.append(opset)
                break
            except Exception:
                continue
    return sorted(set(buildable))


def fixture_opset_for_op(op_type: str, min_version: int, max_version: int) -> int | None:
    """Pick the newest opset in range where ``op_type`` is defined and buildable."""
    opsets = fixture_opsets_for_op(op_type, min_version, max_version)
    return opsets[-1] if opsets else None


def ops_in_opset_range(min_version: int, max_version: int) -> list[str]:
    """Union of operators available at any opset in ``[min_version, max_version]``."""
    result: list[str] = []
    for name in _all_onnx_op_names():
        for version in range(max_version, min_version - 1, -1):
            if _op_available_at(version, name, include_deprecated=True):
                result.append(name)
                break
    return result


def _elem_type_from_type_str(type_str: str, *, default: int = TensorProto.FLOAT) -> int:
    lower = type_str.lower()
    if "bool" in lower or type_str == "B":
        return TensorProto.BOOL
    if "float16" in lower:
        return TensorProto.FLOAT16
    if "double" in lower:
        return TensorProto.DOUBLE
    if "int64" in lower or type_str in ("I", "Tind"):
        return TensorProto.INT64
    if "int32" in lower:
        return TensorProto.INT32
    if "uint8" in lower:
        return TensorProto.UINT8
    if "string" in lower:
        return TensorProto.STRING
    if type_str == "tensor(float)":
        return TensorProto.FLOAT
    if type_str in ("T", "V", "T1", "T2", "T3"):
        return default
    return default


def _default_input_elem_type(op_type: str, *, requested: int = TensorProto.FLOAT) -> int:
    if op_type in BOOL_INPUT_OPS:
        return TensorProto.BOOL
    return requested


def _fp16_eligible_type_str(type_str: str) -> bool:
    """True for schema type variables that should follow the fixture's float dtype."""
    return type_str in ("T", "V", "T1", "T2", "T3") or "float16" in type_str.lower()


def _output_elem_type(op_type: str, out_param, input_elem_type: int) -> int:
    if op_type in BOOL_OUTPUT_OPS or op_type in BOOL_INPUT_OPS:
        return TensorProto.BOOL
    if op_type in INT64_OUTPUT_OPS:
        return TensorProto.INT64
    if op_type == "TopK":
        if out_param.name == "Values":
            return input_elem_type
        if out_param.name == "Indices":
            return TensorProto.INT64
    if op_type == "MaxPool" and out_param.name == "Indices":
        return TensorProto.INT64
    if op_type == "Dropout" and out_param.name == "mask":
        return TensorProto.BOOL
    if op_type in ("QuantizeLinear", "DynamicQuantizeLinear") and out_param.name in (
        "y",
        "y_scale",
        "y_zero_point",
    ):
        return _elem_type_from_type_str(out_param.type_str, default=input_elem_type)
    return _elem_type_from_type_str(out_param.type_str, default=input_elem_type)


def _shape_for_param(
    op_type: str,
    param,
    *,
    data_rank: int,
    input_elem_type: int,
) -> list[int]:
    name = param.name
    if op_type in ("DepthToSpace", "SpaceToDepth") and name in _DATA_INPUT_NAMES:
        # blocksize=2 requires channel count divisible by 4
        return [1, 4, 4, 4]
    if op_type in SPATIAL_OPS and name in _DATA_INPUT_NAMES:
        return list(SPATIAL_SHAPE)
    if op_type in ("Conv", "ConvTranspose", "DeformConv") and name == "W":
        return [1, 1, 3, 3]
    if op_type == "DeformConv" and name == "offset":
        return [1, 18, 2, 2]
    if op_type in ("BatchNormalization", "InstanceNormalization") and name in (
        "scale",
        "B",
        "input_mean",
        "input_var",
    ):
        return [1]
    if op_type == "Gemm":
        if name == "A":
            return [1, 2]
        if name == "B":
            return [2, 2]
        if name == "C":
            return [1, 2]
    if op_type == "MatMul":
        if name == "A":
            return [1, 2]
        if name == "B":
            return [2, 2]
    if op_type == "Range" and name in ("start", "limit", "delta"):
        return []
    if op_type == "Slice" and name in ("starts", "ends", "axes", "steps"):
        return [max(data_rank, 1)]
    if op_type == "ReverseSequence" and name == "sequence_lens":
        return [3]
    if op_type == "ReverseSequence" and name == "input":
        return [3, 2, 4]
    if op_type == "ScatterND" and name == "indices":
        return [1, 1]
    if op_type == "ScatterND" and name == "updates":
        return [1, 2]
    if op_type == "Gather" and name == "indices":
        return [1]
    if op_type == "GatherND" and name == "indices":
        return [1, 1]
    if op_type == "Expand" and name == "shape":
        return [2]
    if op_type == "Reshape" and name == "shape":
        return [2]
    if op_type == "Unsqueeze" and name == "axes":
        return [1]
    if op_type == "ConstantOfShape" and name == "input":
        return [2]
    if param.type_str == "I":
        return []
    if "int64" in param.type_str:
        if op_type == "TopK" and name == "K":
            return []
        return [1]
    if param.type_str in ("B", "tensor(bool)"):
        return list(DEFAULT_VECTOR_SHAPE)
    return list(DEFAULT_VECTOR_SHAPE)


def _required_attr(
    name: str,
    attr,
    op_type: str,
    *,
    data_rank: int = 2,
    input_shape: list[int] | None = None,
):
    attr_type = attr.type
    shape = input_shape or list(DEFAULT_VECTOR_SHAPE)
    if op_type == "Cast" and name == "to":
        return helper.make_attribute("to", TensorProto.FLOAT)
    if op_type == "CastLike" and name == "to":
        return helper.make_attribute("to", TensorProto.FLOAT)
    if name == "axis" and op_type in (
        "Concat",
        "Softmax",
        "LogSoftmax",
        "ReduceMean",
        "Gather",
        "Hardmax",
    ):
        return helper.make_attribute("axis", 0)
    if name == "axis" and op_type == "Split":
        return helper.make_attribute("axis", 1)
    if op_type == "DepthToSpace" and name == "blocksize":
        return helper.make_attribute("blocksize", 2)
    if op_type == "SpaceToDepth" and name == "blocksize":
        return helper.make_attribute("blocksize", 2)
    if op_type == "Einsum" and name == "equation":
        return helper.make_attribute("equation", "ij,jk->ik")
    if op_type == "Constant" and name == "value":
        return helper.make_attribute(
            "value",
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="const"),
        )
    if op_type == "ConstantOfShape" and name == "value":
        return helper.make_attribute(
            "value",
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="value"),
        )
    if op_type == "Mod" and name == "fmod":
        return helper.make_attribute(name, 1)
    if attr_type == AttributeProto.INT:
        return helper.make_attribute(name, 1)
    if attr_type == AttributeProto.INTS:
        if name == "kernel_shape":
            return helper.make_attribute(name, [2, 2])
        if name == "pads" and op_type == "Pad":
            return helper.make_attribute(name, [0] * (2 * data_rank))
        if name == "starts" and op_type == "Slice":
            return helper.make_attribute(name, [0] * data_rank)
        if name == "ends" and op_type == "Slice":
            return helper.make_attribute(name, list(shape))
        if name == "pads":
            return helper.make_attribute(name, [0, 0, 0, 0])
        if name == "strides":
            return helper.make_attribute(name, [1, 1])
        if name == "dilations":
            return helper.make_attribute(name, [1, 1])
        return helper.make_attribute(name, [0])
    if attr_type == AttributeProto.STRING:
        return helper.make_attribute(name, "NOTSET")
    if attr_type == AttributeProto.FLOAT:
        return helper.make_attribute(name, 1.0)
    return None


def _build_attrs(
    schema,
    op_type: str,
    *,
    data_rank: int = 2,
    input_shape: list[int] | None = None,
) -> list:
    skip_cast_attrs = {"saturate", "round_mode"}
    attrs = []
    for name, attr in schema.attributes.items():
        if op_type == "Cast" and name in skip_cast_attrs:
            continue
        if op_type == "Split" and name == "axis":
            attrs.append(helper.make_attribute("axis", 1))
            continue
        if op_type == "ConstantOfShape" and name == "value":
            built = _required_attr(name, attr, op_type, data_rank=data_rank, input_shape=input_shape)
            if built is not None:
                attrs.append(built)
            continue
        if op_type == "Mod" and name == "fmod":
            attrs.append(helper.make_attribute("fmod", 1))
            continue
        if attr.default_value.ByteSize() > 0 and attr.default_value.name:
            attrs.append(attr.default_value)
        elif attr.required:
            built = _required_attr(
                name, attr, op_type, data_rank=data_rank, input_shape=input_shape
            )
            if built is not None:
                attrs.append(built)
    return attrs


def _int64_initializer(name: str, shape: list[int], values: np.ndarray | None = None) -> TensorProto:
    if values is None:
        values = np.zeros(shape, dtype=np.int64)
    else:
        values = np.asarray(values, dtype=np.int64)
    return numpy_helper.from_array(values, name)


def _scalar_initializer(name: str, value: float | int, dtype: int = TensorProto.FLOAT) -> TensorProto:
    if dtype == TensorProto.INT64:
        arr = np.array(value, dtype=np.int64)
    elif dtype == TensorProto.FLOAT16:
        arr = np.array(value, dtype=np.float16)
    else:
        arr = np.array(value, dtype=np.float32)
    return numpy_helper.from_array(arr, name)


def _guess_output_shape(
    op_type: str,
    out_param,
    *,
    input_shape: list[int],
) -> list[int]:
    if op_type in INT64_OUTPUT_OPS:
        if op_type == "Shape":
            return [len(input_shape)]
        if op_type == "Size":
            return []
        if op_type == "NonZero":
            return [len(input_shape), int(np.prod(input_shape))]
        if op_type in ("ArgMax", "ArgMin"):
            if len(input_shape) <= 1:
                return []
            return input_shape[1:]
    if op_type in ("GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool"):
        return [input_shape[0], input_shape[1], 1, 1]
    if op_type in SPATIAL_OPS and out_param.name in ("Y", "output"):
        return list(input_shape)
    if op_type == "Squeeze":
        return [input_shape[-1]] if len(input_shape) > 1 else list(input_shape)
    if op_type == "Unsqueeze":
        return [1, *input_shape]
    if op_type == "Transpose":
        return list(reversed(input_shape))
    if op_type == "Range":
        return [4]
    if op_type.startswith("Reduce") and out_param.name in ("reduced", "Y"):
        # Default test uses axis=0 on [1, 2] -> [2]
        if len(input_shape) > 1:
            return input_shape[1:]
        return []
    if op_type in ("ArgMax", "ArgMin") and out_param.name in ("reduced", "Y"):
        if len(input_shape) > 1:
            return input_shape[1:]
        return []
    if op_type == "Split":
        axis = 1
        if len(input_shape) > axis and input_shape[axis] >= 2:
            out_shape = list(input_shape)
            out_shape[axis] = input_shape[axis] // 2
            return out_shape
    return list(input_shape)


def _pads_initializer(rank: int) -> TensorProto:
    pads = np.zeros(2 * rank, dtype=np.int64)
    if rank >= 2:
        pads[-1] = 1
    return numpy_helper.from_array(pads, "pads")


def _schema_outputs(schema, op_type: str):
    """Graph/node outputs — WebNN handlers may only implement a subset."""
    outputs = list(schema.outputs)
    if op_type == "MaxPool":
        return [o for o in outputs if o.name == "Y"]
    if op_type == "LayerNormalization":
        return [o for o in outputs if o.name == "Y"]
    return outputs


def _build_constant(opset: int) -> ModelProto:
    value = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "value")
    node = helper.make_node("Constant", [], ["output"], value=value, name="test_Constant")
    graph = helper.make_graph(
        [node],
        "test_Constant_graph",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model)
    return model


def _build_generic(
    op_type: str, opset: int, *, input_elem_type: int = TensorProto.FLOAT
) -> ModelProto:
    schema = defs.get_schema(op_type, opset, ONNX_DOMAIN)
    initializers: list[TensorProto] = []
    graph_inputs = []
    node_inputs: list[str] = []
    input_elem_type = _default_input_elem_type(op_type, requested=input_elem_type)
    data_rank = len(DEFAULT_VECTOR_SHAPE)
    input_shape = list(DEFAULT_VECTOR_SHAPE)

    for param in schema.inputs:
        if _is_optional(param):
            continue
        if _is_variadic(param):
            elem_type = input_elem_type
            names = ["in0", "in1"] if op_type == "Concat" else ["in0"]
            for name in names:
                shape = list(DEFAULT_VECTOR_SHAPE)
                graph_inputs.append(helper.make_tensor_value_info(name, elem_type, shape))
                node_inputs.append(name)
            break

        name = param.name
        if op_type == "Pad" and name == "pads":
            initializers.append(_pads_initializer(data_rank))
            node_inputs.append("pads")
            continue

        elem_type = _elem_type_from_type_str(param.type_str, default=input_elem_type)
        shape = _shape_for_param(
            op_type,
            param,
            data_rank=data_rank,
            input_elem_type=input_elem_type,
        )
        if name in _DATA_INPUT_NAMES and shape:
            data_rank = len(shape)
            input_shape = list(shape)

        if op_type == "Range" and name in ("start", "limit", "delta"):
            defaults = {"start": 0.0, "limit": 4.0, "delta": 1.0}
            initializers.append(_scalar_initializer(name, defaults[name], elem_type))
            node_inputs.append(name)
            continue

        if elem_type == TensorProto.INT64:
            init_name = name
            if op_type == "Slice" and name == "starts":
                vals = np.zeros(max(data_rank, 1), dtype=np.int64)
            elif op_type == "Slice" and name == "ends":
                vals = np.array(list(DEFAULT_VECTOR_SHAPE), dtype=np.int64)
            elif op_type == "Slice" and name == "axes":
                vals = np.arange(max(data_rank, 1), dtype=np.int64)
            elif op_type == "ReverseSequence" and name == "sequence_lens":
                vals = np.array([2, 2, 2], dtype=np.int64)
            elif op_type == "ScatterND" and name == "indices":
                vals = np.array([[0]], dtype=np.int64)
            elif op_type == "Expand" and name == "shape":
                vals = np.array([1, 2], dtype=np.int64)
            elif op_type == "Reshape" and name == "shape":
                vals = np.array([1, 2], dtype=np.int64)
            elif op_type == "Unsqueeze" and name == "axes":
                vals = np.array([0], dtype=np.int64)
            elif op_type == "ConstantOfShape" and name == "input":
                vals = np.array([1, 2], dtype=np.int64)
            elif op_type == "TopK" and name == "K":
                vals = np.array(1, dtype=np.int64)
            else:
                vals = None
            initializers.append(_int64_initializer(init_name, shape or [1], vals))
            node_inputs.append(init_name)
            continue

        graph_inputs.append(helper.make_tensor_value_info(name, elem_type, shape))
        node_inputs.append(name)
        if name in _DATA_INPUT_NAMES and shape:
            data_rank = len(shape)
            input_shape = list(shape)

    graph_outputs = [
        helper.make_tensor_value_info(
            out.name,
            _output_elem_type(op_type, out, input_elem_type),
            _guess_output_shape(op_type, out, input_shape=input_shape),
        )
        for out in _schema_outputs(schema, op_type)
    ]

    node = helper.make_node(
        op_type,
        node_inputs,
        [out.name for out in _schema_outputs(schema, op_type)],
        name=f"test_{op_type}",
    )
    node.attribute.extend(
        _build_attrs(schema, op_type, data_rank=data_rank, input_shape=input_shape)
    )

    graph = helper.make_graph(
        [node],
        f"test_{op_type}_graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    checker.check_model(model)
    return model


def _build_sequence_map(opset: int) -> ModelProto:
    elem_type = helper.make_tensor_type_proto(TensorProto.FLOAT, DEFAULT_VECTOR_SHAPE)
    seq_type = helper.make_sequence_type_proto(elem_type)
    body = helper.make_graph(
        [helper.make_node("Identity", ["current"], ["out"], name="id_elem")],
        "body",
        [helper.make_tensor_value_info("current", TensorProto.FLOAT, DEFAULT_VECTOR_SHAPE)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, DEFAULT_VECTOR_SHAPE)],
    )
    tensor_init = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    seq_node = helper.make_node("SequenceConstruct", ["t0"], ["input_sequence"], name="make_seq")
    map_node = helper.make_node(
        "SequenceMap",
        ["input_sequence"],
        ["out_sequence"],
        body=body,
        name="test_SequenceMap",
    )
    graph = helper.make_graph(
        [seq_node, map_node],
        "test_SequenceMap_graph",
        [],
        [helper.make_value_info("out_sequence", seq_type)],
        [tensor_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model)
    return model


CUSTOM_BUILDERS: dict[str, Callable[[int], ModelProto]] = {
    "Constant": _build_constant,
    "SequenceMap": _build_sequence_map,
    **_EXTRA_CUSTOM_BUILDERS,
}


def _fp16_eligible_names(model: ModelProto, op_type: str, opset: int) -> set[str]:
    graph = model.graph
    if graph is None:
        return set()
    eligible: set[str] = set()
    for node in graph.node:
        if node.op_type != op_type:
            continue
        schema = defs.get_schema(node.op_type, opset, ONNX_DOMAIN)
        input_idx = 0
        for param in schema.inputs:
            if _is_optional(param):
                continue
            if _is_variadic(param):
                if _fp16_eligible_type_str(param.type_str):
                    eligible.update(name for name in node.input[input_idx:] if name)
                break
            if input_idx >= len(node.input):
                break
            if _fp16_eligible_type_str(param.type_str) and node.input[input_idx]:
                eligible.add(node.input[input_idx])
            input_idx += 1
        output_idx = 0
        for param in _schema_outputs(schema, node.op_type):
            if _is_variadic(param):
                if _fp16_eligible_type_str(param.type_str):
                    eligible.update(name for name in node.output[output_idx:] if name)
                break
            if output_idx >= len(node.output):
                break
            if _fp16_eligible_type_str(param.type_str) and node.output[output_idx]:
                eligible.add(node.output[output_idx])
            output_idx += 1
    return eligible


def _set_value_info_elem_type(value_info, elem_type: int) -> None:
    if value_info.type.HasField("tensor_type"):
        value_info.type.tensor_type.elem_type = elem_type


def _to_float16_tensor(tensor: TensorProto) -> None:
    arr = numpy_helper.to_array(tensor).astype(np.float16)
    converted = numpy_helper.from_array(arr, tensor.name)
    tensor.CopyFrom(converted)


def _convert_float_fixture_to_float16(model: ModelProto, op_type: str, opset: int) -> ModelProto:
    eligible = _fp16_eligible_names(model, op_type, opset)
    if not eligible:
        raise ValueError(f"{op_type} fixture has no fp16-eligible tensor values")
    graph = model.graph
    if graph is None:
        raise ValueError("model missing graph")

    for value_info in [*graph.input, *graph.output, *graph.value_info]:
        if value_info.name in eligible and value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            _set_value_info_elem_type(value_info, TensorProto.FLOAT16)
    for initializer in graph.initializer:
        if initializer.name in eligible and initializer.data_type == TensorProto.FLOAT:
            _to_float16_tensor(initializer)
    for node in graph.node:
        if any(name in eligible for name in node.output):
            for attr in node.attribute:
                if attr.HasField("t") and attr.t.data_type == TensorProto.FLOAT:
                    _to_float16_tensor(attr.t)

    checker.check_model(model)
    return model


def build_test_model(
    op_type: str, opset: int, *, input_elem_type: int = TensorProto.FLOAT
) -> ModelProto:
    if op_type in CUSTOM_BUILDERS:
        model = CUSTOM_BUILDERS[op_type](opset)
        if input_elem_type == TensorProto.FLOAT16:
            return _convert_float_fixture_to_float16(model, op_type, opset)
        return model
    return _build_generic(op_type, opset, input_elem_type=input_elem_type)



def ops_at_opset(version: int) -> list[str]:
    return _ops_at_opset(version)

