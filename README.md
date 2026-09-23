# onnx2webnn

> **Last edited:** `2026-09-22T21:07:28Z`<br>
> **Checkout:** `fkrall/cache-backed-validation` at `a582b89`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

ONNX → WebNN lowering crate extracted from [webnn-graph](../webnn-graph). ONNX operators lower
directly to [rustnn](../rustnn) `MLGraphBuilder`; full-graph validation runs via ORT CPU
`build()` (`onnx-runtime` feature). There is no intermediate JSON IR. Without `--output`,
success means `builder.build()` returns `Ok(MLGraph)`; `--output` additionally writes a
reloadable `.webnn` graph and sibling Safetensors archive.

Decomposed attention is not promised to be bit-identical to a source backend fused-attention
kernel. This matters for quantized models that immediately feed attention into
`DynamicQuantizeLinear`: small legal floating-point differences can cross Uint8 bucket boundaries
and compound through later layers. FastVLM prefill has this limitation; its fixed sequence-1 decode
specialization passes numerical validation.

Supported ONNX opset range: **1–26** (see `MIN_SUPPORTED_OPSET` / `MAX_SUPPORTED_OPSET` in
`src/onnx/convert.rs`).

## Build

### Prerequisites

- A stable Rust toolchain.
- The Protocol Buffers compiler (`protoc`). On Debian/Ubuntu install
  `protobuf-compiler`; on macOS install `protobuf` with Homebrew. Cargo uses it
  while building the ONNX and CoreML protobuf bindings.
- A sibling `rustnn` checkout at `../rustnn`. This is the path dependency used
  by `Cargo.toml` while the stacked RustNN changes are under review.
- ONNX Runtime 1.27 or newer when running conversions or tests with the ORT
  backend. RustNN can download the pinned version into its own ignored
  `target/onnxruntime` directory:

```powershell
make -C ../rustnn onnxruntime-download
```

Set `ORT_DYLIB_PATH` to the downloaded `libonnxruntime` shared library before
running the CLI or tests. The exact filename is platform-specific; examples
are `libonnxruntime.so.1.29.0` on Linux, `libonnxruntime.1.29.0.dylib` on macOS,
and `onnxruntime.dll` on Windows. The CI workflow in `.github/workflows/ci.yml`
contains the complete cross-platform provisioning sequence.

```powershell
cargo build
# or
cargo build --release
```

`make build`, `make test`, `make fmt`, and `make check` are defined in the repo `Makefile`.

## Convert

```powershell
cargo run -- convert --input model.onnx --optimize --override-dim batch_size=1
```

Dynamic ONNX inputs (unresolved symbolic dims kept as WebNN dynamic metadata):

```powershell
cargo run -- convert --input model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

If `model.dims.json` sits beside the ONNX file and no overrides were passed on the CLI, dimension
bindings are loaded from that sidecar (`freeDimensionOverrides` or a flat JSON object).

With `--output`, artifacts are overwritten below the operating system's cache directory in
`onnx2webnn/onnx` and `onnx2webnn/webnn`; add `--validate` to reload the saved pair immediately and
compare deterministic execution against native ORT. Set `O2W_CACHE_DIR` to relocate both caches,
or `O2W_ONNX_CACHE` and `O2W_WEBNN_CACHE` to override either cache independently. RustNN stores
logical Int4/Uint4 constants as their original packed nibble bytes in a
versioned U8 Safetensors extension while the `.webnn` declaration retains the logical dtype and
shape. This is a RustNN archive convention, not a native Safetensors 4-bit dtype.

`MatMulNBits` is lowered to `dequantizeLinear` followed by ordinary floating-point `matmul` because
WebNN has no fused low-bit matmul operation. The dequantized weights and activations therefore use
the scale dtype (`float16` or `float32`; the validated q4 models use `float32`). Native ORT may use
a fused packed-weight kernel instead. In particular, `MatMulNBits accuracy_level=4` permits ORT to
quantize activations to Int8 internally, which the WebNN lowering cannot represent; such models are
recorded as validation-blocked rather than numerically supported.

`--validate-cached` reuses existing cache artifacts and is exclusive with `--output` and
`--validate`.

Merged decoders (optimum's `decoder_model_merged*.onnx`) branch at runtime on `use_cache_branch`.
WebNN has no runtime `If`, so pin the input and convert each branch separately:

```powershell
cargo run -- convert --input decoder_model_merged.onnx --optimize `
  --pin-input use_cache_branch=false `
  --override-dim batch_size=1 --override-dim decoder_sequence_length=4 `
  --override-dim past_decoder_sequence_length=0 ...
```

Pinned inputs become constants, the chosen `If` branch is inlined, and inputs the branch never
reads (e.g. the KV cache in the prefill branch) and zero-size dummy outputs are dropped.

Supporting both cache modes therefore produces two independent artifact pairs: one `.webnn` file
and one Safetensors file for prefill, and another pair for cached decoding. Shared decoder weights
are not deduplicated between the pairs. Loading both compiled graphs concurrently may duplicate
those weights in RAM and VRAM; applications can instead keep only the active graph resident and
manage graph selection and KV-cache handoff themselves.

| Flag | Purpose |
|------|---------|
| `--input` | Input `.onnx` path (required) |
| `--optimize` | Constant folding and shape propagation |
| `--override-dim NAME=VALUE` | Bind a symbolic dim (repeatable); unnamed zero dims are addressed as `<input>_dim<axis>` |
| `--override-dims-file` | JSON overrides (`freeDimensionOverrides` or flat object) |
| `--pin-input NAME=VALUE` | Freeze a graph input to `true`/`false`/an integer (repeatable) |
| `--allow-missing-external-data` | Zero-fill external tensors whose data file is absent (weight-stripped skeleton models) |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata |
| `--debug` | Verbose conversion logging (global) |
| `--output` | Overwrite cache-backed `.onnx`, `.webnn`, and Safetensors artifacts |
| `--validate` | After `--output`, reload and numerically compare against native ORT |
| `--validate-cached` | Validate existing cache artifacts without reconverting |

On success the CLI prints `✓ ORT graph build succeeded for …`.

## Model sweep

`tests/models/manifest.json` lists the transformers.js exports the converter is expected to handle,
with their dimension overrides and pinned inputs; `tests/model_skeletons.rs` converts each entry and
builds it in ORT. This broad manifest is generated by `scripts/generate_manifest.py` and intentionally
tracks the publisher repositories through their default `main` revisions. It is diagnostic coverage,
not an immutable CI gate.

No weights are downloaded by the skeleton test: it reads each export from the Hugging Face Hub with
HTTP range requests, keeps the graph and small constants, and points every large initializer at a file
that does not exist, which the converter zero-fills. A 1.4 GB export becomes a ~0.2 MB skeleton for
~10 MB of traffic. Skeletons are kept in `target/model-skeletons` (or `O2W_SKELETON_CACHE`), about
40 MB for the whole manifest, and CI caches that directory keyed on the manifest and scanner source.

### Numerical-validation CI

Every Linux pull-request job numerically validates the real-weight models in the hand-maintained
`tests/models/ci-validation.json`; the experimental, non-blocking macOS/CoreML job runs the same
set. Windows does not run numerical model validation. Each curated entry must be small enough for a
hosted runner, pass on both ORT and CoreML, and declare an immutable 40-character Hugging Face
commit `revision` plus the lowercase SHA-256 of its primary ONNX file. Downloads, external-data
sidecars, skeletons, and cache identities all use that revision. The primary
model digest is checked after download and whenever a cached model is reused.

Full-model validation downloads through the official Hugging Face Rust client and uses the standard
`HF_HUB_CACHE`/`HF_HOME` cache by default, allowing Python and Rust tools to reuse the same blobs.
`O2W_ONNX_CACHE`, followed by `O2W_CACHE_DIR/onnx`, takes precedence when set and is interpreted as
the Hub cache root. onnx2webnn stores only its completion records in the private
`.onnx2webnn-validation` directory there; model files retain the standard `blobs` and `snapshots`
layout. Xet-backed downloads use chunked transfer, while ordinary HTTP downloads are retried but
are not guaranteed to resume from a partial byte offset.

To add required CI coverage, resolve the model repository to a commit, download the exact ONNX file
from that revision, compute its SHA-256 (for example with `sha256sum`), and add the file, revision,
digest, and fixed dimension/input settings to `tests/models/ci-validation.json`. Do not add these
pins to `tests/models/manifest.json`: regenerating the broad manifest does not read or write the
curated file. Verify the candidate on Linux and macOS with:

```bash
target/release/onnx2webnn validate-models \
  --manifest tests/models/ci-validation.json \
  --selection all --weights real --jobs 1
```

The manually dispatched `Full model validation` workflow runs every entry from the generated main
manifest with real publisher weights. It requires a dedicated runner labeled
`self-hosted`, `linux`, `x64`, and `onnx2webnn-validation`, a writable
`/var/cache/onnx2webnn`, network access, a filesystem of at least 100 GiB, and approximately 32 GiB
of RAM. Downloaded source models persist in that directory; exported WebNN artifacts remain
temporary. Setup, build, and runner-contract failures fail the workflow, while model failures are
reported as diagnostic warnings and retained in a full log artifact for 14 days.

`O2W_MODELS` selects the source: `hub` (the default when `CI` is set), `dir=<path>` for full local
downloads laid out as `<org>--<repo>/onnx/<file>.onnx`, or `strip=<path>` to run local files through
the skeleton scanner. Unset outside CI, the sweep is skipped. `O2W_MODEL_FETCH_JOBS` (default 8) sets
how many skeletons are fetched at once, `O2W_MODEL_TEST_JOBS` (default 4) how many convert at once,
and `O2W_MODEL_TEST_SKIP_HEAVY` skips the entries that need more than 10 GB of RAM.

```powershell
$env:O2W_MODELS = "dir=..\transformers_js_experiments\models"; cargo test --release --test model_skeletons
```

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

## Layout

| Path | Purpose |
|------|---------|
| `src/onnx/convert.rs` | ONNX load, optional folding, lowering, ORT `build()` |
| `src/onnx/builder.rs` | `OnnxBuilder` — operand map and `MLGraphBuilder` bridge |
| `src/onnx/builder_helpers.rs` | Shared lowering helpers |
| `src/onnx/shape_inference.rs` | Static shape/type propagation |
| `src/onnx/constant_folding.rs` | Constant folding driver (with `--optimize`) |
| `src/onnx/constant_folding/evaluators/` | Per-op fold evaluators |
| `src/onnx/ops/` | ONNX op handlers (activation, conv, pool, reshape, …) |
| `src/protos.rs` | ONNX protobuf types |
| `src/debug.rs` | Debug logging toggle |

## Dependencies

- **rustnn** (`../rustnn`, `onnx-runtime`) — `MLGraphBuilder`, shape inference, ORT `build()` validation
- **webnn-onnx-utils** — ONNX protos, op names, data types

## Related

- [webnn-graph](../webnn-graph) — DSL parser, validator, JS/HTML emit (source of the extracted lowering code)
