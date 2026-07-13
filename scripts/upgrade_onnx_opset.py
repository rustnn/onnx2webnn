#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upgrade ONNX model opset version using onnx.version_converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import version_converter


def main() -> int:
    parser = argparse.ArgumentParser(description="Upgrade ONNX opset version.")
    parser.add_argument("onnx_path", type=Path, help="Input .onnx file")
    parser.add_argument("target_opset", type=int, help="Target ai.onnx opset version")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <stem>_opset<N>.onnx next to input)",
    )
    args = parser.parse_args()

    input_path = args.onnx_path
    if not input_path.is_file():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_opset{args.target_opset}.onnx"
    )

    model = onnx.load(str(input_path))
    before = [(i.domain or "ai.onnx", i.version) for i in model.opset_import]
    print(f"Before: {before}")

    converted = version_converter.convert_version(model, args.target_opset)
    after = [(i.domain or "ai.onnx", i.version) for i in converted.opset_import]
    print(f"After:  {after}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(output_path))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
