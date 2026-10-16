#!/usr/bin/env python3
"""Regenerate or check JSON/Python and Markdown views of the Rust operator catalog."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/operator-conversion-status.md"
DATA = ROOT / "scripts/operator_capabilities.json"


def generated() -> tuple[str, str]:
    binary = os.environ.get("O2W_CATALOG_BIN")
    command = [binary, "operator-catalog"] if binary else [
        "cargo", "run", "--quiet", "--locked", "--", "operator-catalog",
    ]
    output = subprocess.check_output(command, cwd=ROOT)
    catalog = json.loads(output)
    if len([entry for entry in catalog if entry["domain"] == "ai.onnx"]) != 198:
        raise ValueError("catalog must contain 198 standard opset-26 operators")
    data = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    matrix = "".join(
        f'| `{entry["name"]}` | {entry["status"]} | {entry["scope"]} | '
        f'{entry["perf_impact"]} | {"Yes" if entry["perf_limited"] else ""} |\n'
        for entry in catalog
    )
    return data, matrix


def update_doc(original: str, matrix: str) -> str:
    first, after = original.split("## Complete opset-26 status matrix\n", 1)
    _, tail = after.split("### MoE interpretation", 1)
    rows = matrix.splitlines(keepends=True)
    standard = rows[:198]
    direct = sum("| Direct WebNN |" in row for row in standard)
    unsupported = sum("| Unsupported |" in row for row in standard)
    executable = len(standard) - unsupported
    partial = executable - direct
    before_coverage, _ = first.split("## Coverage\n", 1)
    coverage = (
        "## Coverage\n\n"
        "**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. "
        "**Regenerate:** `python3 scripts/generate_operator_catalog.py`.\n\n"
        "| Opset-26 standard ONNX coverage | Count | Share |\n"
        "|---------------------------------|------:|------:|\n"
        "| Total non-deprecated `ai.onnx` names | 198 | 100% |\n"
        f"| **Executable for at least one supported form** | **{executable}** | **{executable / 198:.1%}** |\n"
        f"| └ Direct WebNN/spec-level subset | **{direct}** | **{direct / 198:.1%}** |\n"
        f"| └ Decomposed, folded, or pattern-only | **{partial}** | **{partial / 198:.1%}** |\n"
        f"| Unsupported | {unsupported} | {unsupported / 198:.1%} |\n\n"
        "The generated Python supported-op data derives from the same Rust catalog.\n\n"
    )
    return (
        before_coverage + coverage + "## Complete opset-26 status matrix\n\n"
        "**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. "
        "**Regenerate:** `python3 scripts/generate_operator_catalog.py`.\n\n"
        "| ONNX operator | Status | Execution scope and WebNN limitation | Estimated perf impact | Performance-limited |\n"
        "|---------------|--------|--------------------------------------|-----------------------|---------------------|\n"
        + "".join(rows[:198]) + "\n## Additional contrib and fused operators\n\n"
        "**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. "
        "**Regenerate:** `python3 scripts/generate_operator_catalog.py`.\n\n"
        "| ONNX operator | Status | Execution scope and WebNN limitation | Estimated perf impact | Performance-limited |\n"
        "|---------------|--------|--------------------------------------|-----------------------|---------------------|\n"
        + "".join(rows[198:]) + "\n### MoE interpretation" + tail
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail on generated-view drift")
    args = parser.parse_args()
    data, matrix = generated()
    original = DOC.read_text()
    doc = update_doc(original, matrix)
    if args.check:
        if DATA.read_text() != data or DOC.read_text() != doc:
            raise SystemExit("operator catalog views drifted; run scripts/generate_operator_catalog.py")
    else:
        DATA.write_text(data)
        if doc != original:
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            doc = re.sub(r"(?m)^> \*\*Last edited:\*\* .*$", f"> **Last edited:** `{now}`<br>", doc)
            doc = re.sub(r"(?m)^> \*\*Checkout:\*\* .*$", f"> **Checkout:** `{branch}` at `{commit}`", doc)
            DOC.write_text(doc)


if __name__ == "__main__":
    main()
