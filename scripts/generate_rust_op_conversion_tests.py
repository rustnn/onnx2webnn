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

from onnx_fixture_builders import build_test_model, ops_at_opset
from rust_model_emitter import emit_build_function
from webnn_onnx_ops import is_webnn_supported_op

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_OPS_DIR = PROJECT_ROOT / "tests" / "onnx_ops"
ONNX_OPS_ENTRY = PROJECT_ROOT / "tests" / "onnx_op_tests.rs"
DEFAULT_FIXTURE_OPSET = 26
TEST_OPSET = 18

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


def _emit_op_file(
    *,
    op_type: str,
    opset: int,
    test_opset: int,
    expect: str,
    model=None,
    build_error: str | None = None,
) -> str:
    if build_error is not None:
        reason = _rust_ignore_reason(build_error)
        return "\n".join(
            [
                _RUST_SPDX_HEADER,
                f"//! ONNX `{op_type}` operator conversion test (fixture opset {opset}).",
                "// Auto-generated by scripts/generate_rust_op_conversion_tests.py — do not edit.",
                "",
                f"// Fixture builder failed at codegen: {_rust_comment_text(build_error)}",
                "#[test]",
                f'#[ignore = "fixture builder failed at codegen: {reason}"]',
                "fn opset26() {}",
                "",
            ]
        )
    assert model is not None
    parts = [
        _RUST_SPDX_HEADER,
        f"//! ONNX `{op_type}` operator conversion test (fixture opset {opset}).",
        "// Auto-generated by scripts/generate_rust_op_conversion_tests.py — do not edit.",
        "",
        "use crate::common::{assert_op_matches_ort, ExpectConvertOp};",
        "use onnx2webnn::protos::onnx::ModelProto;",
        "",
    ] + emit_build_function(model)
    parts.extend(
        [
            "#[test]",
            "fn opset26() {",
            f"    assert_op_matches_ort(build_fixture(), {expect}, {test_opset});",
            "}",
            "",
        ]
    )
    return "\n".join(parts)


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


def generate(*, fixture_opset: int, test_opset: int) -> tuple[int, int]:
    cases = [
        {
            "op_type": op_type,
            "expect_convert_op": (
                "success" if is_webnn_supported_op("ai.onnx", op_type) else "unsupported_op"
            ),
        }
        for op_type in ops_at_opset(fixture_opset)
    ]

    if ONNX_OPS_DIR.exists():
        shutil.rmtree(ONNX_OPS_DIR)
    ONNX_OPS_DIR.mkdir(parents=True)

    categories: dict[str, list[str]] = {}
    built = 0
    skipped = 0

    for case in cases:
        op_type = case["op_type"]
        category = _category_for(op_type)
        snake = _snake_case(op_type)
        expect = _expect_rust(case["expect_convert_op"])
        opset = fixture_opset

        category_dir = ONNX_OPS_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        file_stem = _rust_file_stem(snake)
        out_file = category_dir / f"{file_stem}.rs"

        try:
            model = build_test_model(op_type, opset)
            content = _emit_op_file(
                op_type=op_type,
                opset=opset,
                test_opset=test_opset,
                expect=expect,
                model=model,
            )
            built += 1
        except Exception as exc:  # noqa: BLE001
            content = _emit_op_file(
                op_type=op_type,
                opset=opset,
                test_opset=test_opset,
                expect=expect,
                build_error=str(exc),
            )
            skipped += 1

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
                "mod common;",
                "mod onnx_ops;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return built, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-opset", type=int, default=DEFAULT_FIXTURE_OPSET)
    parser.add_argument("--test-opset", type=int, default=TEST_OPSET)
    args = parser.parse_args()

    built, skipped = generate(fixture_opset=args.fixture_opset, test_opset=args.test_opset)
    print(
        f"Wrote tests/onnx_ops/: {built} test(s), {skipped} ignored "
        f"(build failed at codegen)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
