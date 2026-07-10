#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate per-operator Rust integration tests under tests/onnx_ops/."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from onnx_fixture_builders import (
    MAX_SUPPORTED_OPSET,
    MIN_SUPPORTED_OPSET,
    build_test_model,
    fixture_opsets_for_op,
    ops_in_opset_range,
)
from rust_model_emitter import emit_build_function
from webnn_onnx_ops import is_webnn_supported_op

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_OPS_DIR = PROJECT_ROOT / "tests" / "onnx_ops"
ONNX_OPS_ENTRY = PROJECT_ROOT / "tests" / "onnx_op_tests.rs"
DEFAULT_MIN_OPSET = MIN_SUPPORTED_OPSET
DEFAULT_MAX_OPSET = MAX_SUPPORTED_OPSET

_RUST_SPDX_HEADER = """\
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */"""

# Operator → category folder. Unlisted ops fall into `misc`.
_OP_CATEGORIES: dict[str, frozenset[str]] = {
    "activation": frozenset(
        {
            "Abs",
            "Acos",
            "Acosh",
            "Asin",
            "Asinh",
            "Atan",
            "Atanh",
            "Celu",
            "Ceil",
            "Clip",
            "Cos",
            "Cosh",
            "Elu",
            "Erf",
            "Exp",
            "Floor",
            "Gelu",
            "HardSigmoid",
            "HardSwish",
            "Identity",
            "LeakyRelu",
            "Log",
            "Mish",
            "Neg",
            "PRelu",
            "Reciprocal",
            "Relu",
            "Round",
            "Selu",
            "Shrink",
            "Sigmoid",
            "Sign",
            "Sin",
            "Sinh",
            "Softplus",
            "Softsign",
            "Sqrt",
            "Swish",
            "Tan",
            "Tanh",
            "ThresholdedRelu",
        }
    ),
    "elementwise": frozenset({"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mod"}),
    "comparison": frozenset(
        {
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "IsInf",
            "IsNaN",
            "Less",
            "LessOrEqual",
        }
    ),
    "logical": frozenset(
        {
            "And",
            "BitCast",
            "BitShift",
            "BitwiseAnd",
            "BitwiseNot",
            "BitwiseOr",
            "BitwiseXor",
            "Not",
            "Or",
            "Xor",
        }
    ),
    "conv": frozenset(
        {
            "Col2Im",
            "Conv",
            "ConvInteger",
            "ConvTranspose",
            "DeformConv",
            "QLinearConv",
        }
    ),
    "pool": frozenset(
        {
            "AveragePool",
            "GlobalAveragePool",
            "GlobalLpPool",
            "GlobalMaxPool",
            "LpPool",
            "MaxPool",
            "MaxRoiPool",
            "MaxUnpool",
            "RoiAlign",
        }
    ),
    "reshape": frozenset(
        {
            "CenterCropPad",
            "Compress",
            "Concat",
            "DepthToSpace",
            "Expand",
            "Flatten",
            "Gather",
            "GatherElements",
            "GatherND",
            "Reshape",
            "ReverseSequence",
            "SpaceToDepth",
            "Split",
            "Squeeze",
            "Tile",
            "Transpose",
            "Unsqueeze",
        }
    ),
    "reduction": frozenset(
        {
            "ArgMax",
            "ArgMin",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        }
    ),
    "normalization": frozenset(
        {
            "BatchNormalization",
            "GroupNormalization",
            "InstanceNormalization",
            "LayerNormalization",
            "LRN",
            "LpNormalization",
            "MeanVarianceNormalization",
            "RMSNormalization",
        }
    ),
    "utility": frozenset(
        {
            "AffineGrid",
            "Cast",
            "CastLike",
            "Constant",
            "ConstantOfShape",
            "Det",
            "EyeLike",
            "GridSample",
            "ImageDecoder",
            "NonZero",
            "OneHot",
            "Optional",
            "OptionalGetElement",
            "OptionalHasElement",
            "Pad",
            "Range",
            "Resize",
            "ScatterElements",
            "ScatterND",
            "Shape",
            "Size",
            "Slice",
            "TensorScatter",
            "TopK",
            "Trilu",
            "Unique",
            "Where",
        }
    ),
    "matmul": frozenset({"Einsum", "Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"}),
    "control_flow": frozenset({"If", "Loop", "Scan"}),
    "sequence": frozenset(
        {
            "ConcatFromSequence",
            "SequenceAt",
            "SequenceConstruct",
            "SequenceEmpty",
            "SequenceErase",
            "SequenceInsert",
            "SequenceLength",
            "SequenceMap",
            "SplitToSequence",
        }
    ),
    "quantization": frozenset(
        {"DequantizeLinear", "DynamicQuantizeLinear", "QuantizeLinear"}
    ),
    "string": frozenset(
        {
            "RegexFullMatch",
            "StringConcat",
            "StringNormalizer",
            "StringSplit",
            "TfIdfVectorizer",
        }
    ),
    "rnn": frozenset({"Attention", "GRU", "LSTM", "RNN", "RotaryEmbedding"}),
    "random": frozenset(
        {
            "Bernoulli",
            "Dropout",
            "Multinomial",
            "RandomNormal",
            "RandomNormalLike",
            "RandomUniform",
            "RandomUniformLike",
        }
    ),
    "signal": frozenset(
        {
            "BlackmanWindow",
            "DFT",
            "HammingWindow",
            "HannWindow",
            "MelWeightMatrix",
            "STFT",
        }
    ),
    "loss": frozenset({"NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss"}),
    "softmax": frozenset({"Hardmax", "LogSoftmax", "Softmax"}),
    "misc": frozenset({"CumProd", "CumSum", "Mean", "NonMaxSuppression", "Sum"}),
}

_OP_TO_CATEGORY: dict[str, str] = {}
for category, ops in _OP_CATEGORIES.items():
    for op in ops:
        if op in _OP_TO_CATEGORY:
            raise ValueError(f"{op} listed in multiple categories")
        _OP_TO_CATEGORY[op] = category


def _snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


_RUST_KEYWORDS = frozenset(
    {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "unsafe",
        "use",
        "where",
        "while",
    }
)


def _rust_file_stem(snake: str) -> str:
    # `mod.rs` is reserved for the category module index.
    if snake == "mod":
        return "mod_op"
    return snake


def _rust_mod_name(snake: str) -> str:
    stem = _rust_file_stem(snake)
    if stem != snake:
        return stem
    if snake in _RUST_KEYWORDS:
        return f"r#{snake}"
    return snake


def _rust_comment_text(text: str, *, max_len: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.replace("*/", "* /")
    return cleaned[:max_len]


def _rust_ignore_reason(text: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _./:-]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len] or "build failed at codegen"


def _expect_rust(expect_convert_op: str) -> str:
    if expect_convert_op == "success":
        return "ExpectConvertOp::Success"
    if expect_convert_op == "unsupported_op":
        return "ExpectConvertOp::UnsupportedOp"
    raise ValueError(f"unknown expect_convert_op: {expect_convert_op}")


def _category_for(op_type: str) -> str:
    return _OP_TO_CATEGORY.get(op_type, "misc")


def _test_fn_name(opset: int) -> str:
    return f"convert_op_opset_{opset}"


def _build_fn_name(opset: int) -> str:
    return f"build_fixture_opset_{opset}"


def _emit_ignored_test(
    *,
    op_type: str,
    opset: int,
    build_error: str,
) -> list[str]:
    reason = _rust_ignore_reason(build_error)
    return [
        f"// Fixture builder failed at codegen (opset {opset}): {_rust_comment_text(build_error)}",
        "#[test]",
        f'#[ignore = "fixture builder failed at codegen: {reason}"]',
        f"fn {_test_fn_name(opset)}() {{}}",
        "",
    ]


def _emit_op_variant(
    *,
    op_type: str,
    opset: int,
    expect: str,
    model=None,
    build_error: str | None = None,
) -> list[str]:
    if build_error is not None:
        return _emit_ignored_test(op_type=op_type, opset=opset, build_error=build_error)
    assert model is not None
    parts = emit_build_function(model, fn_name=_build_fn_name(opset))
    parts.extend(
        [
            "#[test]",
            f"fn {_test_fn_name(opset)}() {{",
            f"    assert_op_matches_ort({_build_fn_name(opset)}(), {expect}, {opset});",
            "}",
            "",
        ]
    )
    return parts


def _emit_op_file(
    *,
    op_type: str,
    variants: list[dict],
) -> str:
    """Emit one Rust test module with one test per schema-revision opset."""
    opsets = [v["opset"] for v in variants]
    opset_summary = ", ".join(str(o) for o in opsets)
    # Ignored stubs use neither the runner nor `ModelProto`; skip imports when every variant is a
    # stub to avoid unused-import warnings.
    has_buildable = any(v.get("build_error") is None for v in variants)
    parts = [
        _RUST_SPDX_HEADER,
        f"//! ONNX `{op_type}` operator conversion tests (fixture opsets: {opset_summary}).",
        "// Auto-generated by scripts/generate_rust_op_conversion_tests.py — do not edit.",
        "",
    ]
    if has_buildable:
        parts.extend(
            [
                "use crate::common::{assert_op_matches_ort, ExpectConvertOp};",
                "use onnx2webnn::protos::onnx::ModelProto;",
                "",
            ]
        )
    for variant in variants:
        parts.extend(
            _emit_op_variant(
                op_type=op_type,
                opset=variant["opset"],
                expect=variant["expect"],
                model=variant.get("model"),
                build_error=variant.get("build_error"),
            )
        )
    return "\n".join(parts)


def _emit_unbuildable_op_file(*, op_type: str) -> str:
    return _emit_op_file(
        op_type=op_type,
        variants=[
            {
                "opset": MIN_SUPPORTED_OPSET,
                "expect": "ExpectConvertOp::UnsupportedOp",
                "build_error": "no buildable fixture opset in supported range",
            }
        ],
    )


def _write_mod_rs(path: Path, modules: list[str]) -> None:
    lines = [
        _RUST_SPDX_HEADER,
        "// Auto-generated by scripts/generate_rust_op_conversion_tests.py — do not edit.",
        "",
    ]
    for module in sorted(modules):
        lines.append(f"pub mod {_rust_mod_name(module)};")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(*, min_opset: int, max_opset: int) -> tuple[int, int, int]:
    if ONNX_OPS_DIR.exists():
        shutil.rmtree(ONNX_OPS_DIR)
    ONNX_OPS_DIR.mkdir(parents=True)

    categories: dict[str, list[str]] = {}
    built = 0
    skipped = 0
    unbuildable = 0

    for op_type in ops_in_opset_range(min_opset, max_opset):
        category = _category_for(op_type)
        snake = _snake_case(op_type)
        file_stem = _rust_file_stem(snake)
        category_dir = ONNX_OPS_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        out_file = category_dir / f"{file_stem}.rs"

        expect = _expect_rust(
            "success" if is_webnn_supported_op("ai.onnx", op_type) else "unsupported_op"
        )
        fixture_opsets = fixture_opsets_for_op(op_type, min_opset, max_opset)

        if not fixture_opsets:
            content = _emit_unbuildable_op_file(op_type=op_type)
            skipped += 1
            unbuildable += 1
            out_file.write_text(content, encoding="utf-8")
            categories.setdefault(category, []).append(file_stem)
            continue

        variants: list[dict] = []
        for opset in fixture_opsets:
            try:
                model = build_test_model(op_type, opset)
                variants.append({"opset": opset, "expect": expect, "model": model})
                built += 1
            except Exception as exc:  # noqa: BLE001
                variants.append(
                    {"opset": opset, "expect": expect, "build_error": str(exc)}
                )
                skipped += 1

        content = _emit_op_file(op_type=op_type, variants=variants)
        out_file.write_text(content, encoding="utf-8")
        categories.setdefault(category, []).append(file_stem)

    for category, modules in sorted(categories.items()):
        _write_mod_rs(ONNX_OPS_DIR / category / "mod.rs", modules)

    _write_mod_rs(ONNX_OPS_DIR / "mod.rs", sorted(categories.keys()))

    ONNX_OPS_ENTRY.write_text(
        "\n".join(
            [
                _RUST_SPDX_HEADER,
                "// Integration tests for ONNX operator conversion.",
                "// Auto-generated entry point — individual ops live under tests/onnx_ops/.",
                "",
                "// Generated fixtures embed full-precision op attributes and use category/op module",
                "// nesting (e.g. onnx_ops::conv::conv); both are intentional in generated code.",
                "#![allow(clippy::excessive_precision, clippy::module_inception)]",
                "",
                "mod common;",
                "mod onnx_ops;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return built, skipped, unbuildable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-opset",
        type=int,
        default=DEFAULT_MIN_OPSET,
        help="lowest ai.onnx opset to include when discovering operators",
    )
    parser.add_argument(
        "--max-opset",
        type=int,
        default=DEFAULT_MAX_OPSET,
        help="highest ai.onnx opset to include when discovering operators",
    )
    args = parser.parse_args()

    built, skipped, unbuildable = generate(min_opset=args.min_opset, max_opset=args.max_opset)
    print(
        f"Wrote tests/onnx_ops/: {built} test(s), {skipped} ignored "
        f"({unbuildable} op(s) with no buildable fixture)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
