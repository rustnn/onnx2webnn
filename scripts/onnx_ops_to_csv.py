#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List ONNX operators in a model and write them to a CSV file."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import GraphProto, ModelProto, NodeProto

from webnn_onnx_ops import (
    WEBNN_MAX_OPSET,
    WEBNN_MIN_OPSET,
    is_webnn_supported_op,
    is_webnn_supported_opset,
)


def _domain(node: NodeProto) -> str:
    return node.domain if node.domain else "ai.onnx"


def _opset_by_domain(model: ModelProto) -> dict[str, int]:
    """Map operator domain to ONNX opset version from model imports."""
    opsets: dict[str, int] = {}
    for imp in model.opset_import:
        domain = imp.domain if imp.domain else "ai.onnx"
        opsets[domain] = imp.version
    return opsets


def _opset_version(opsets: dict[str, int], domain: str) -> str:
    version = opsets.get(domain)
    return str(version) if version is not None else ""


def _webnn_supported_label(domain: str, op_type: str) -> str:
    return "yes" if is_webnn_supported_op(domain, op_type) else "no"


def _iter_graphs(graph: GraphProto):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH and attr.g is not None:
                yield from _iter_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    yield from _iter_graphs(subgraph)


def _collect_nodes(model: onnx.ModelProto) -> list[NodeProto]:
    if model.graph is None:
        raise ValueError("ONNX model has no graph")
    nodes: list[NodeProto] = []
    for graph in _iter_graphs(model.graph):
        nodes.extend(graph.node)
    return nodes


def _summary_rows(
    nodes: list[NodeProto], opsets: dict[str, int], *, include_webnn: bool
) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for node in nodes:
        counts[(_domain(node), node.op_type)] += 1
    rows: list[dict[str, str | int]] = []
    for (domain, op_type), count in sorted(
        counts.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row: dict[str, str | int] = {
            "domain": domain,
            "opset_version": _opset_version(opsets, domain),
            "op_type": op_type,
            "count": count,
        }
        if include_webnn:
            row["webnn_supported"] = _webnn_supported_label(domain, op_type)
        rows.append(row)
    return rows


def _detail_rows(
    nodes: list[NodeProto], opsets: dict[str, int], *, include_webnn: bool
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, node in enumerate(nodes):
        domain = _domain(node)
        row: dict[str, str | int] = {
            "index": index,
            "domain": domain,
            "opset_version": _opset_version(opsets, domain),
            "op_type": node.op_type,
            "name": node.name or "",
            "num_inputs": len(node.input),
            "num_outputs": len(node.output),
        }
        if include_webnn:
            row["webnn_supported"] = _webnn_supported_label(domain, node.op_type)
        rows.append(row)
    return rows


def _default_output_path(input_path: Path, per_node: bool) -> Path:
    suffix = "_ops_per_node" if per_node else "_ops"
    return input_path.with_name(f"{input_path.stem}{suffix}.csv")


def _print_opset_status(opsets: dict[str, int]) -> bool:
    ok = True
    for domain, version in sorted(opsets.items()):
        supported = is_webnn_supported_opset(domain, version)
        status = "ok" if supported else "unsupported"
        print(f"  opset {domain}={version}: {status} (webnn-graph accepts {WEBNN_MIN_OPSET}-{WEBNN_MAX_OPSET} for ai.onnx)")
        if not supported:
            ok = False
    return ok


def _print_unsupported_ops(rows: list[dict[str, str | int]], *, per_node: bool) -> bool:
    if per_node:
        unsupported = sorted(
            {
                (str(row["domain"]), str(row["op_type"]))
                for row in rows
                if row.get("webnn_supported") == "no"
            }
        )
    else:
        unsupported = sorted(
            (str(row["domain"]), str(row["op_type"]))
            for row in rows
            if row.get("webnn_supported") == "no"
        )
    if not unsupported:
        print("  all operators in model are supported by webnn-graph")
        return True
    print("  unsupported operators for webnn-graph:")
    for domain, op_type in unsupported:
        if per_node:
            count = sum(
                1
                for row in rows
                if row["domain"] == domain
                and row["op_type"] == op_type
                and row.get("webnn_supported") == "no"
            )
        else:
            count = next(
                int(row["count"])
                for row in rows
                if row["domain"] == domain and row["op_type"] == op_type
            )
        print(f"    {domain}::{op_type} ({count} node(s))")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ONNX operator usage from a model to CSV."
    )
    parser.add_argument(
        "onnx_path",
        type=Path,
        help="Path to the input .onnx file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <model_stem>_ops.csv next to the input)",
    )
    parser.add_argument(
        "--per-node",
        action="store_true",
        help="Emit one row per node instead of aggregated op_type counts",
    )
    parser.add_argument(
        "--check-webnn",
        action="store_true",
        help="Add webnn_supported column and exit 1 if opset or ops are unsupported",
    )
    args = parser.parse_args()

    input_path = args.onnx_path
    if not input_path.is_file():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or _default_output_path(input_path, args.per_node)
    include_webnn = args.check_webnn

    model = onnx.load(str(input_path))
    opsets = _opset_by_domain(model)
    nodes = _collect_nodes(model)
    if args.per_node:
        fieldnames = [
            "index",
            "domain",
            "opset_version",
            "op_type",
            "name",
            "num_inputs",
            "num_outputs",
        ]
        if include_webnn:
            fieldnames.append("webnn_supported")
        rows = _detail_rows(nodes, opsets, include_webnn=include_webnn)
    else:
        fieldnames = ["domain", "opset_version", "op_type", "count"]
        if include_webnn:
            fieldnames.append("webnn_supported")
        rows = _summary_rows(nodes, opsets, include_webnn=include_webnn)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_ops = len({(r["domain"], r["op_type"]) for r in rows}) if args.per_node else len(rows)
    print(f"Wrote {len(rows)} row(s) ({unique_ops} unique op types) to {output_path}")

    if args.check_webnn:
        print("webnn-graph compatibility:")
        opset_ok = _print_opset_status(opsets)
        ops_ok = _print_unsupported_ops(rows, per_node=args.per_node)
        if not opset_ok or not ops_ok:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
