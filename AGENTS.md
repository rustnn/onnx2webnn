# onnx2webnn

ONNX → WebNN conversion crate. Rust lowering lives in `src/onnx/`; graph validation uses
[rustnn](../rustnn) **ORT `build()`** on `MLGraphBuilder` (see `ValidatedGraph` in
`src/onnx/convert.rs`). There is no JSON IR in the committed tree.

## Build and test

```powershell
make build    # cargo build
make test     # cargo test
make check    # cargo check
make fmt      # cargo fmt
```

Or directly:

```powershell
cargo build
cargo test --lib
```

## Convert CLI

Binary: `onnx2webnn` (`src/main.rs`).

```powershell
cargo run -- convert --input model.onnx --optimize --override-dim batch_size=1
```

Dynamic symbolic dims (experimental):

```powershell
cargo run -- convert --input model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

| Flag | Source | Purpose |
|------|--------|---------|
| `--input` | required | Path to `.onnx` model |
| `--optimize` | `ConvertOptions::optimize` | Constant folding and shape propagation |
| `--override-dim NAME=VALUE` | `ConvertOptions::free_dim_overrides` | Bind symbolic dims (repeatable) |
| `--override-dims-file` | JSON file | Same overrides; object or `{ "freeDimensionOverrides": { … } }` |
| `--experimental-dynamic-inputs` | `ConvertOptions::experimental_dynamic_inputs` | Preserve unresolved symbolic input dims as dynamic metadata |
| `--debug` | global | Enable conversion debug output (`src/debug.rs`) |

Success prints `✓ ORT graph build succeeded for …` to stderr. There is no `--output` flag in
the committed CLI — validation only.

## Library API

Public exports (`src/lib.rs`):

- `convert_onnx(path, ConvertOptions) -> Result<ValidatedGraph, OnnxError>`
- `ConvertOptions`, `OnnxError`, `ValidatedGraph`

Supported ONNX opset range for `ai.onnx`: **11–18** (`MIN_SUPPORTED_OPSET` /
`MAX_SUPPORTED_OPSET` in `src/onnx/convert.rs`).

## Layout (committed)

| Path | Purpose |
|------|---------|
| `src/onnx/convert.rs` | Main lowering + ORT validation entry |
| `src/onnx/builder.rs` | `OnnxBuilder` — operand map over `MLGraphBuilder` |
| `src/onnx/builder_helpers.rs` | Shared reshape/expand/slice helpers |
| `src/onnx/shape_inference.rs` | Static shape/type propagation scaffold |
| `src/onnx/constant_folding/` | Optional ONNX constant folding (`--optimize`) |
| `src/onnx/ops/` | Per-op handlers (`OpHandler` in `ops/mod.rs`) |
| `src/protos.rs` | ONNX protobuf types |
| `scripts/` | Python tooling (test generation, model audit, opset inventory) |
| `tests/onnx_ops/` | Auto-generated per-op conversion integration tests |

## Python scripts (`scripts/`)

Install: `pip install -r requirements.txt` (from repo root; needs `onnx`, `numpy`).

| Script | Purpose |
|--------|---------|
| `webnn_onnx_ops.py` | Supported-op manifest — sync with `src/onnx/ops/*.rs` when adding handlers |
| `generate_rust_op_conversion_tests.py` | Regenerate `tests/onnx_ops/` and `tests/onnx_op_tests.rs` |
| `onnx_ops_to_csv.py` | List operators in a model; `--check-webnn` exits non-zero if unsupported ops remain |
| `upgrade_onnx_opset.py` | Upgrade a model to a target `ai.onnx` opset version |
| `generate_onnx_opsets.py` | Per-opset operator inventory CSVs (`webnn_exporter_supported` column) |

Regenerate conversion tests after changing handlers or the manifest:

```powershell
.\.venv\Scripts\python.exe scripts\generate_rust_op_conversion_tests.py --fixture-opset 26 --test-opset 18
cargo test --test onnx_op_tests
```

See [docs/operator-conversion-plan.md](docs/operator-conversion-plan.md) for the full operator rollout workflow.

## Dependencies (`Cargo.toml`)

- **rustnn** — `path = "../rustnn"`, features `["onnx-runtime"]`
- **webnn-onnx-utils** — ONNX protos, op names, data types
- **bytemuck**, **clap**, **prost**, **serde**, **serde_json**, **thiserror**, **anyhow**

## Related (outside this repo)

- [rustnn](../rustnn) — `MLGraphBuilder`, shape inference, ORT backend
- [webnn-graph](../webnn-graph) — sibling project; lowering in `src/onnx/` was extracted from it
