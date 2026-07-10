#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate per-opset ONNX operator inventories for webnn-graph coverage tracking.

Writes docs/onnx-opsets/opset-{N}.csv for each requested opset version. Each file lists
every non-deprecated ai.onnx operator available when a model declares opset N, using the
installed onnx package schema registry (onnx.defs).

Regenerate after upgrading the onnx pip package so lists match the spec version you ship.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from onnx import defs

from webnn_onnx_ops import is_webnn_supported_op

PROJECT_ROOT = Path(__file__).resolve().parent
WEBNN_GRAPH_ROOT = PROJECT_ROOT.parent / "webnn-graph"
DEFAULT_OUT_DIR = WEBNN_GRAPH_ROOT / "docs" / "onnx-opsets"
ONNX_DOMAIN = ""  # empty string == ai.onnx in onnx.defs


def _all_op_names() -> list[str]:
    names: set[str] = set()
    for schema in defs.get_all_schemas_with_history():
        domain = schema.domain or ""
        if domain in ("", "ai.onnx"):
            names.add(schema.name)
    return sorted(names)


def _ops_at_opset(version: int) -> dict[str, int]:
    """Return op -> active schema since_version at this opset (non-deprecated only)."""
    result: dict[str, int] = {}
    for name in _all_op_names():
        try:
            if not defs.has(name, version, ONNX_DOMAIN):
                continue
            schema = defs.get_schema(name, version, ONNX_DOMAIN)
        except defs.SchemaError:
            continue
        if schema.deprecated:
            continue
        result[name] = schema.since_version
    return result


def _write_opset_csv(
    path: Path,
    version: int,
    ops: dict[str, int],
    *,
    new_in_opset: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "op_type",
                "schema_since_version",
                "new_in_opset",
                "webnn_exporter_supported",
            ],
        )
        writer.writeheader()
        for op_type in sorted(ops):
            writer.writerow(
                {
                    "op_type": op_type,
                    "schema_since_version": ops[op_type],
                    "new_in_opset": "yes" if op_type in new_in_opset else "no",
                    "webnn_exporter_supported": (
                        "yes" if is_webnn_supported_op("ai.onnx", op_type) else "no"
                    ),
                }
            )


def generate(
    versions: list[int],
    out_dir: Path,
) -> dict[int, tuple[int, int, int]]:
    """Generate CSV files. Returns version -> (total, new_in_opset, webnn_supported)."""
    prev_ops: set[str] = set()
    if versions:
        first = min(versions)
        if first > 1:
            prev_ops = set(_ops_at_opset(first - 1).keys())

    stats: dict[int, tuple[int, int, int]] = {}
    for version in sorted(versions):
        ops = _ops_at_opset(version)
        cur = set(ops.keys())
        new_in_opset = cur - prev_ops
        prev_ops = cur

        path = out_dir / f"opset-{version}.csv"
        _write_opset_csv(path, version, ops, new_in_opset=new_in_opset)

        supported = sum(
            1 for op in ops if is_webnn_supported_op("ai.onnx", op)
        )
        stats[version] = (len(ops), len(new_in_opset), supported)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min",
        type=int,
        default=11,
        help="Minimum opset version (default: 11)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=21,
        help="Maximum opset version (default: 21)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR.relative_to(WEBNN_GRAPH_ROOT)})",
    )
    args = parser.parse_args()

    if args.min > args.max:
        print("error: --min must be <= --max", file=sys.stderr)
        return 1

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("error: install onnx (see transformers.js/requirements.txt)", file=sys.stderr)
        return 1

    versions = list(range(args.min, args.max + 1))
    stats = generate(versions, args.output_dir)

    print(f"onnx package {onnx.__version__} (registry opset {defs.onnx_opset_version()})")
    print(f"Wrote {len(versions)} file(s) to {args.output_dir}")
    for version, (total, new_count, supported) in stats.items():
        pct = (100.0 * supported / total) if total else 0.0
        print(
            f"  opset-{version}.csv: {total} ops "
            f"({new_count} new vs opset {version - 1}), "
            f"exporter supports {supported} ({pct:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
