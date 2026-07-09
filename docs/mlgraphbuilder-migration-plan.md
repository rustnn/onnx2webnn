# MLGraphBuilder migration plan

ONNX → WebNN conversion via **`MLGraphBuilder`** only. No `GraphJson` intermediate representation, no `from_graph_json` / `shape_bridge` validation pass.

**This milestone:** validate converted graphs by compiling them with rustnn’s **ORT (CPU) backend** (`builder.build()`). **Disk export** (`.webnn`, weights, manifest) is deferred.

## Goal

Lower ONNX through `MLGraphBuilder` so rustnn provides:

- Per-op shape inference during the ONNX walk
- Dtype / graph consistency checks at each builder call
- Full-graph validation when `build(outputs)` runs `OnnxConverter` and loads an ORT session

**Success criterion:** `build()` returns `Ok(MLGraph)` for supported models (e.g. EfficientNet opset 11).

## Constraints

| Rule | Implication |
|------|-------------|
| No JSON IR in onnx2webnn | Remove `ast`, `shape_bridge`, handler `Node` emission |
| No JSON round-trip for validation | Delete `infer_webnn_shapes` / `from_graph_json` from convert path |
| Validation = ORT | `MLContext` + feature `onnx-runtime` → CPU `build()` |
| Tensor names | `sanitize_identifier(onnx_name)` → `MLOperatorOptions.label`; inputs via `input(name, …)` |
| No disk export yet | CLI may omit `--output`; success = ORT build OK |
| rustnn changes | **None** for this milestone |

## Target architecture

```text
ONNX ModelProto
  → (optional) constant folding
  → MLContext::create(accelerated: false)   # CPU ORT execution provider
  → MLGraphBuilder::new
  → OpRegistry → builder ops + MLOperand map (sanitized ONNX value names)
  → builder.build(outputs)                  # GraphInfo → OnnxConverter → ORT session
  → Ok(ValidatedGraph) or error
```

```mermaid
flowchart LR
  ONNX[ONNX ModelProto] --> Fold[Constant folding]
  Fold --> Ctx[MLContext CPU ORT]
  Ctx --> Bld[MLGraphBuilder]
  Bld --> Ops[OpRegistry handlers]
  Ops --> Build[builder.build outputs]
  Build --> OK[ValidatedGraph]
```

## Phase 1 — Builder infrastructure

**Owner:** onnx2webnn  
**Files:** `src/onnx/builder.rs`, `src/onnx/ops/mod.rs`, `Cargo.toml`, `src/lib.rs`

1. **Dependencies** (`Cargo.toml`):

   ```toml
   rustnn = { path = "../rustnn", features = ["onnx-runtime"] }
   bytemuck = "1.15"
   ```

   Add `dynamic-inputs` on rustnn when `experimental_dynamic_inputs` is required.

2. **`OnnxBuilder`**: wraps `MLGraphBuilder` + `HashMap<String, MLOperand>` (sanitized ONNX value → handle).

3. **Helpers** (`builder.rs`):

   - `webnn_id(name)` → `sanitize_identifier`
   - `resolve_operand(onnx_name)` → `MLOperand`
   - `record_output(onnx_output, operand)`
   - `descriptor_static` / `descriptor_dynamic` for `MLOperandDescriptor`

4. **`OpHandler` signature:**

   ```rust
   fn convert(
       &self,
       node: &NodeProto,
       ctx: &ConversionContext,
       b: &mut OnnxBuilder,
   ) -> Result<ConversionResult, OnnxError>;

   pub struct ConversionResult {
       pub output_mappings: HashMap<String, MLOperand>,
       pub output_types: HashMap<String, MLOperandDataType>,
   }
   ```

5. **Unit test:** one input + `add` + `build()` succeeds on CPU ORT.

**Acceptance:** `cargo test` passes; minimal graph builds under ORT.

**Estimate:** 2–3 days

---

## Phase 2 — Pilot handlers

**Files:** `elementwise.rs`, `activation.rs`, `comparison.rs`

- Map ONNX ops to `builder.add`, `relu`, `equal`, etc. (`*_with_options` where needed).
- Set `options.label = sanitize_identifier(&node.output[0])`.
- Drop `Node` / `ConstDecl` from `ConversionResult`.
- Extend op unit tests to use `OnnxBuilder` + optional `rustnn_operand_shape` checks.

**Acceptance:** Tiny ONNX models (Add, Relu) complete `build()` without error.

**Estimate:** 3–4 days

---

## Phase 3 — Structural ops

**Files:** `reshape.rs`, `matmul.rs`, `pad.rs`, `conditional.rs`, `scatter.rs`, `reduction.rs`, `conversion.rs`

- Dynamic shapes: enable rustnn `dynamic-inputs` if needed.
- Multi-output `Split`: `builder.split` → map each ONNX output to an `MLOperand`.
- Keep ONNX rank/axis helpers in `convert.rs` only (not WebNN JSON).

**Acceptance:** Fixtures under `../webnn-graph/tests/onnx/` for these op families reach `build()` Ok.

**Estimate:** ~1 week

---

## Phase 4 — Conv, pool, normalization

**Files:** `conv.rs`, `pool.rs`, `normalization.rs`

- 1D conv/pool via builder chains (`reshape` → `conv2d` → `reshape`), same semantics as today.
- ONNX attributes → `MLConv2dOptions`, `MLPool2dOptions`, `MLBatchNormalizationOptions`, etc.
- Initializers as `constant_from_slice` (avoid `constant_from_vec`, which panics in rustnn).

**Acceptance:** Birds EfficientNet (opset 11) full model `build()` succeeds.

**Estimate:** ~1 week

---

## Phase 5 — Graph walk (`convert.rs`)

1. Graph **inputs** → `builder.input(sanitized_name, descriptor)`.
2. **Initializers** → `constant_from_slice` + operand map.
3. Topological **nodes** → `OpRegistry::convert_node` + `OnnxBuilder`.
4. Graph **outputs** → `HashMap<&str, MLOperand>` → `builder.build(&outputs)`.
5. Remove `OnnxConverter`’s `GraphJson` field and all `graph.nodes` / `graph.consts` mutation.
6. Keep constant folding on `ModelProto` before lowering (unchanged).

**Return type sketch:**

```rust
pub struct ValidatedGraph<'ctx> {
    pub context: MLContext<'ctx>,
    pub graph: MLGraph<'ctx>,
}

pub fn convert_onnx(path, options) -> Result<ValidatedGraph<'_>, OnnxError>;
```

**Acceptance:** `convert_onnx` returns `Ok` when ORT accepts the full model graph.

**Estimate:** ~1 week

---

## Phase 6 — CLI and validation UX

**`convert` subcommand (validation milestone):**

| Flag | Behavior |
|------|----------|
| `--input` | Required ONNX path |
| `--optimize` | Constant folding (unchanged) |
| `--override-dim` | Symbolic dim overrides (unchanged) |
| `--experimental-dynamic-inputs` | Unchanged |
| `--output` | Optional / ignored until export milestone |
| `--validate-shapes` | Remove; validation is always via builder + ORT |

On success: print e.g. `✓ ORT graph build succeeded`.

Map `GraphBuilderError` / `ShapeInferenceError` to messages that include ONNX node context when possible.

Optional later: `--dispatch-smoke` (zeroed tensors, one `dispatch`).

**Acceptance:**

```powershell
cargo run -- convert --input Birds-Classifier-EfficientNetB2\model_opset11.onnx --optimize
```

exits 0 when ORT build succeeds (replaces shape-bridge validation in `efficientnet.cmd`).

**Estimate:** 2 days

---

## Phase 7 — Remove legacy JSON path

**Delete:**

- `src/ast.rs`
- `src/serialize.rs`
- `src/shape_bridge.rs`

**Update:**

- `src/lib.rs` — remove `infer_webnn_shapes`, `GraphJson` exports
- `Cargo.toml` — drop `webnn-graph` if unused
- `src/main.rs` — no JSON / `.webnn` write in validation milestone

**Keep:** `scripts/webnn_onnx_ops.py`, `onnx_ops_to_csv.py`, `generate_onnx_op_tests.py` (op support manifest unchanged).

**Acceptance:** `rg GraphJson src/` in onnx2webnn is empty; `cargo test` green.

**Estimate:** 1–2 days

---

## Phase 8 — Regression harness

1. Integration tests: glob `../webnn-graph/tests/onnx/*_opset*.onnx`, run `convert_onnx`, expect `build()` Ok for supported ops.
2. Update `AGENTS.md` and `README.md`: validation = ORT build; export deferred.
3. Update `efficientnet.cmd`: convert step = ORT validation only (no `.webnn` artifact until export phase).

**Estimate:** 2–3 days

---

## Deferred (later milestone)

| Item | Notes |
|------|--------|
| Disk export (`.webnn`, `.weights`, manifest) | Export backend or rustnn hook; may use `to_graph_json` inside rustnn only at write time |
| DSL golden file diffs | After export exists |
| `dispatch` parity vs ONNX Runtime | Optional QA |
| JSON / `GraphJson` IR | Not planned |

---

## Timeline

| Phase | Focus | Duration |
|-------|--------|----------|
| 1 | `OnnxBuilder` shell | 2–3 d |
| 2 | Pilot ops | 3–4 d |
| 3 | Structural ops | 7–10 d |
| 4 | Conv / pool / norm | ~5 d |
| 5 | `convert.rs` | ~5 d |
| 6 | CLI | ~2 d |
| 7 | Remove JSON | 1–2 d |
| 8 | Integration tests | 2–3 d |

**Total:** ~5–6 weeks (Phases 2–4 can run in parallel per handler file).

---

## Definition of done

- [ ] All ops in `scripts/webnn_onnx_ops.py` lower through `MLGraphBuilder`
- [ ] No `GraphJson`, `from_graph_json`, or `shape_bridge` in onnx2webnn
- [ ] EfficientNet opset 11: `convert` exits 0 (ORT `build()` OK)
- [ ] Supported op-test ONNX fixtures: `build()` OK
- [ ] `rustnn` dependency includes `onnx-runtime`
- [ ] **Not required:** `.webnn` files, weights manifest, DSL byte goldens

---

## First sprint tasks

1. Add `src/onnx/builder.rs` and change `OpHandler` trait (Phase 1).
2. Migrate `elementwise.rs` + one integration test: ONNX Add → `build()` (Phase 2).
3. Wire CLI to report ORT build success/failure (Phase 6 can land early for `efficientnet.cmd` once coverage allows).

---

## Naming reference

```text
id = sanitize_identifier(onnx_output_name)
options.label = id.clone()
let out = builder.add_with_options(a, b, options)?;
b.record_output(onnx_output_name, out);  // map raw + sanitized keys as today
```

Sanitization rules: `webnn-onnx-utils::identifiers::sanitize_for_webnn`; leading digits → `_` prefix (`onnx2webnn::sanitize_identifier`).

---

## Related docs

- [AGENTS.md](../AGENTS.md) — dev workflow (update after Phase 8)
- [README.md](../README.md) — user-facing convert CLI (update after Phase 6)
- `../rustnn/AGENTS.md` — `MLGraphBuilder`, ORT backend, `OnnxConverter` at build time
