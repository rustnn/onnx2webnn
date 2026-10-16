# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator support derived from the Rust catalog (regenerate with generate_operator_catalog.py)."""

from __future__ import annotations

import json
from pathlib import Path

_CATALOG = json.loads((Path(__file__).with_name("operator_capabilities.json")).read_text())
WEBNN_SUPPORTED_ONNX_OPS: frozenset[str] = frozenset(
    entry["name"] for entry in _CATALOG
    if entry["domain"] == "ai.onnx" and entry["status"] != "Unsupported"
)

WEBNN_MIN_OPSET: int = 9  # Minimum opset for generated ORT-reference fixtures.
WEBNN_MAX_OPSET: int = 26


def is_webnn_supported_op(domain: str, op_type: str) -> bool:
    return domain == "ai.onnx" and op_type in WEBNN_SUPPORTED_ONNX_OPS
