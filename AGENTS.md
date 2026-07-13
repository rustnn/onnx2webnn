# onnx2webnn

ONNX → WebNN conversion crate. Rust lowering lives in `src/onnx/`; graph validation uses
[rustnn](../rustnn) **ORT `build()`** on `MLGraphBuilder` (see `ValidatedGraph` in
`src/onnx/convert.rs`). There is no JSON IR in the committed tree.

## WebNN specification

**Canonical operator reference:** [W3C WebNN API](https://www.w3.org/TR/webnn/) — all
`MLGraphBuilder` methods defined by the spec (§ 7.3 Operators, § 8.9 `MLGraphBuilder`).

When adding or fixing ONNX op handlers, check the spec for the target WebNN op name, options
dictionary, and tensor limits. The spec groups operators into: tensor manipulation, quantization,
casting, math, logical, matmul, convolution, pooling, activation, normalization, reduction, and
RNN (`gru`/`lstm`).

**Editor's draft (latest):** https://webmachinelearning.github.io/webnn/

## ONNX ↔ WebNN mapping (three layers)

"Unsupported" in this crate does **not** mean "no WebNN mapping exists". There are three
independent layers:

| Layer | What it tracks | Where | Count (approx.) |
|-------|----------------|-------|-----------------|
| **1. WebNN spec** | Ops the W3C API defines | [webnn spec](https://www.w3.org/TR/webnn/) | ~105 |
| **2. Name mapping** | ONNX op name ↔ WebNN `MLGraphBuilder` method | `webnn-onnx-utils/src/operation_names.rs` | ~90 |
| **3. Exporter implementation** | ONNX ops with a Rust lowering handler | `src/onnx/ops/*.rs` + `scripts/webnn_onnx_ops.py` | ~59 |

An ONNX op can be:

- **No WebNN target** — e.g. `If`, `Loop`, `Scan` (control flow), `StringConcat` (no string
  tensors), `Compress`, `Einsum`, `Attention`. WebNN graphs are static DAGs; these are permanently
  rejected. See `docs/operator-conversion-plan.md` Stage 3.
- **Mapped but not implemented** — e.g. `BatchNormalization`, `InstanceNormalization`, `ArgMax`,
  `GatherND`, `Resize`. Listed in `operation_names.rs` but no handler in `src/onnx/ops/` yet.
  Tests expect `UnsupportedOp` until a handler lands.
- **Implemented** — listed in `scripts/webnn_onnx_ops.py` and handled by `OpRegistry`. Tests
  expect `Success` (convert + ORT vs rustnn output match).

The test manifest (`webnn_onnx_ops.py`) mirrors layer 3 only. Regenerate tests after changing it.

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

Supported ONNX opset range for `ai.onnx`: **1–26** (`MIN_SUPPORTED_OPSET` /
`MAX_SUPPORTED_OPSET` in `src/onnx/convert.rs`). The converter accepts any model
in that range.

Generated integration tests use **model opset ≥ 9** (`MIN_ORT_REFERENCE_OPSET` in
`scripts/onnx_fixture_builders.py`) so ONNX Runtime can execute the reference path.
Operators with a single schema revision at or before opset 9 are tested with a
model at opset 9 (ONNX still resolves them to their sole schema, e.g. `Ceil` v1).
Operators with multiple pre-9 schema revisions skip sub-opset-9 structure bands
until legacy attribute normalization lands; newer bands use the true latest schema
revision in range (e.g. `MaxRoiPool` at model opset 22, `Pad` at 9/17/26).

Operator dispatch and the unsupported-op pre-scan key on **`op_type` only** (standard
`ai.onnx` domain). Custom-domain ops are not supported yet; adding them requires
domain-aware handler registration and matching pre-scan logic in `convert.rs`.

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
.\.venv\Scripts\python.exe scripts\generate_rust_op_conversion_tests.py --min-opset 1 --max-opset 26
cargo test --test onnx_op_tests
```

The generator emits **one test per distinct ONNX schema structure** per operator (not a full
opset×op matrix). It compares schema fingerprints (inputs, outputs, attribute names) across ONNX
history and tests the highest buildable opset in each band — e.g. `Pad` at opsets 10, 17, and 26
(attribute `pads` vs input `pads` vs optional `axes` input). Ops with unchanged structure get a
single test at the newest opset. Unbuildable ops get an `#[ignore]` stub.

See [docs/operator-conversion-plan.md](docs/operator-conversion-plan.md) for the full operator rollout workflow.

## Dependencies (`Cargo.toml`)

- **rustnn** — `path = "../rustnn"`, features `["onnx-runtime"]`
- **webnn-onnx-utils** — ONNX protos, op names, data types
- **bytemuck**, **clap**, **prost**, **serde**, **serde_json**, **thiserror**, **anyhow**

## Related (outside this repo)

- [W3C WebNN API](https://www.w3.org/TR/webnn/) — operator spec (`MLGraphBuilder` methods)
- [rustnn](../rustnn) — `MLGraphBuilder`, shape inference, ORT backend
- [webnn-onnx-utils](../webnn-onnx-utils) — ONNX↔WebNN name mapping, protos, dtypes
- [webnn-graph](../webnn-graph) — sibling project; lowering in `src/onnx/` was extracted from it
