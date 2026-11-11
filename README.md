> **Last edited:** `2026-11-08T18:11:41Z`<br>
> **Checkout:** `backend-reporting` at `b519633`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

# onnx2webnn

onnx2webnn lowers ONNX opsets **1–26** through [RustNN](https://github.com/rustnn/rustnn)'s WebNN builder.
`convert` exports reloadable `.webnn`/Safetensors artifacts. Add `--backend onnx` to build with
CPU ONNX Runtime, or select another compiled-in RustNN backend.
`onnx2webnn convert --help` lists all known and compiled-in backends directly from RustNN,
without initializing runtimes. Backend-selection errors include both lists.
`validate` loads existing graphs and numerically compares execution against native CPU ORT.
`validate-manifest` converts and validates models selected from a manifest. See [operator coverage](docs/operator-conversion-status.md)
and [model results](docs/model-validation-status-triage.md).

## Build

### Prerequisites

- A stable Rust toolchain.
- `protoc` (Debian/Ubuntu: `protobuf-compiler`; macOS: Homebrew `protobuf`).
- Cargo downloads the tested RustNN revision recorded in `Cargo.lock`; `Cargo.toml` tracks `main`.
- ONNX Runtime 1.29+ for ORT graph builds and validation. Export-only conversion does not load it.
  An optional sibling RustNN checkout can download the runtime:

```powershell
make -C ../rustnn onnxruntime-download
```

Set `ORT_DYLIB_PATH` to the downloaded shared library before ORT use; see
[CI setup](.github/workflows/ci.yml) for platform paths.
A compiled backend still needs its native runtime and compatible devices. Context initialization
errors show indented causal chains, preserving RustNN's runtime-loading and device-discovery
failure details. These are error causes, not program stack traces.
One CLI preflight checks compiled backends, requested dynamic-input support, and runtime
initialization before processing models, writing caches, or downloading manifest models.
Validation also checks native ONNX Runtime. Export-only conversion and the operator catalog
do not initialize runtimes. Graph-dependent checks remain at graph load/build time.

```powershell
cargo build --locked
# or
cargo build --release --locked
```

Enable optional backends with `--features coreml` or `--features trtx`, or forward a RustNN
feature directly, e.g. `cargo build --features rustnn/burn-cpu-runtime`. Platform and native
runtime requirements still apply.

The `Makefile` also provides `build`, `test`, `fmt`, and `check` targets.

Run `cargo update -p rustnn` to advance RustNN, test compatibility, and commit the updated
`Cargo.lock`. The weekly
[compatibility workflow](.github/workflows/rustnn-compatibility.yml) also checks compilation
against RustNN `main` on Linux and macOS; it is diagnostic and can be run manually.
For local RustNN development, override the Git dependency with
`cargo --config 'patch."https://github.com/rustnn/rustnn".rustnn.path="../rustnn"' check`.

## Convert

```powershell
cargo run -- convert --model model.onnx --override-dim batch_size=1
```

Dynamic ONNX inputs (unresolved symbolic dims kept as WebNN dynamic metadata):

```powershell
cargo run -- convert --model model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

Building or validating dynamic graphs requires `cargo build --release --features dynamic-inputs`.
Export-only conversion does not require this feature. Requests combining `--backend` and
`--experimental-dynamic-inputs` are rejected before processing models when the feature is absent;
cached validation checks the graph before reference execution. Backend-specific restrictions still apply.
Validation currently allocates tensors at maximum bounded sizes, so smaller concrete feeds can fail
with a size mismatch.

Without CLI overrides, `model.dims.json` can supply dimension bindings. WebNN exports default
to the OS cache's `onnx2webnn/webnn` directory; set `O2W_WEBNN_CACHE` to override it.
Source ONNX files are read in place.

WebNN lacks fused `MatMulNBits`; conversion dequantizes packed q4/q8 weights before floating-point
matmul. ORT `accuracy_level=4` can instead use Int8 activations, so numerical parity is not assured.

Merged decoders (optimum's `decoder_model_merged*.onnx`) branch at runtime on `use_cache_branch`.
WebNN has no runtime `If`, so pin the input and convert each branch separately:

```powershell
cargo run -- convert --model decoder_model_merged.onnx `
  --pin-input use_cache_branch=false `
  --override-dim batch_size=1 --override-dim decoder_sequence_length=4 `
  --override-dim past_decoder_sequence_length=0 ...
```

Use distinct `--output` locations for pinned branches or other specializations of the same
source model: the default WebNN cache is keyed by source path and overwritten on conversion.
Each exported branch has its own `.webnn`/Safetensors pair. Shared weights are not deduplicated;
loading both graphs may duplicate RAM/VRAM use. Proper growing-KV-cache execution still needs
dynamic shapes or further specialization.

| Flag | Purpose |
|------|---------|
| `--model PATH` | On `convert`: one ONNX model path (required) |
| `--onnx PATH` | On `validate`: reference one ONNX model path (required) |
| `-i`, `--input PATH` | On `validate` only: one complete SafeTensors execution input file; omit for deterministic feeds |
| `-o`, `--output PATH` | Write to a `.webnn` file for one model, or an extensionless directory containing `<model-stem>.webnn`. Omit to use the cache. |
| `--constant-folding` | Enable optional constant folding on `convert` and `validate-manifest` (disabled by default); required simplifications always run |
| `--override-dim NAME=VALUE` | On `convert`, specialize the exported graph; on `validate`, size execution feeds. Repeatable; unnamed zero dims use `<input>_dim<axis>` |
| `--pin-input NAME=VALUE` | On `convert`, freeze an input to a boolean/integer; on `validate`, supply the same pin for the reference run (repeatable) |
| `--allow-missing-external-data` | Zero-fill absent external tensors for structural conversion |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata during conversion |
| `--backend NAME` | Optional graph-build backend; required for every numerical validation. Names: `onnx`, `coreml`, `trtx`, `litert`, `cann`, `burn-cpu`; the runtime must be compiled in and available |
| `--webnn PATH` | On `validate`: one existing WebNN graph; omit for cache lookup |

Set logging with `RUST_LOG` (e.g. `info`, `debug`, or `onnx2webnn=trace`).

The shared `--backend` option must follow the mode name, for example
`onnx2webnn convert --backend onnx --model model.onnx`. Options before the mode are rejected.

`convert` accepts one `--model`; `validate` accepts one `--onnx`, an optional `--webnn`, and
an optional `--input`. Repeating these path options is rejected. Use separate invocations for
multiple models; the JSON override option is removed. `--override-dim` and `--pin-input` remain
repeatable for configuring dimensions and pins on the selected model. Conversion always exports;
`--backend` additionally builds the graph.
Optional constant folding is disabled by default; enable it with `--constant-folding`.
When enabled, folding preserves Squeeze axes inputs, normalizes Unsqueeze axes against the output
rank, and gathers complete axis-0 slices with the indices tensor's rank. Regression tests compare
folded and unfolded exports against native ORT, including downstream broadcasting.

On `validate`, provide a complete `--input` SafeTensors archive or omit it for deterministic
feeds. Input names, types, and shapes must match the feedable ONNX inputs. Supplied tensor
shapes can resolve symbolic dimensions. External ONNX data stays beside the source model.
Validation requires `--backend` before any model processing or output writes.

```bash
cargo run -- convert --model model.onnx --output model.webnn
cargo run -- validate --onnx model.onnx --webnn model.webnn --input inputs.safetensors --backend onnx
```

### Validate existing artifacts

`validate` never converts and errors if the graph is missing. Pass `--webnn` to select an
existing WebNN file; omit it to use the same WebNN cache lookup as `convert`. The reference
ONNX file is read directly from `--onnx`, so an explicit model/graph pair needs no ONNX cache copy. Relative and absolute paths are accepted. Reuse the
conversion's pins and choose concrete execution dimensions through overrides or `--input`
tensor shapes. These options never modify existing graphs; fixed graph dimensions must match
the supplied feeds. Omit `--input` to use deterministic feeds.

```bash
cargo run -- validate --onnx first.onnx --webnn exported/first.webnn \
  --input first.safetensors --backend onnx
```

`validate` uses `--onnx` and `--webnn` instead of `--model` and `--graph`.
`validate-cached` is renamed `validate`; `convert --validate` is removed. Conversion no longer
accepts `--input`. Use `convert` followed by `validate` for the two steps. `validate` accepts no
conversion options such as `--output`, `--constant-folding`, or `--experimental-dynamic-inputs`.
`validate-manifest` continues to convert and validate manifest models.

Operator statistics are code-backed; “fully supported” is a lowering classification, not measured
hardware parity. Performance-limited paths remain functional. Cargo features make CoreML/TensorRT
available, while `--backend` selects one; native comparison remains CPU ORT.
Deterministic fallback feeds are reproducible. Scale-aware quantization feeds use fixed model
scale/zero-point values; ambiguous quantization patterns warn and use the generic deterministic
feed, while malformed quantization parameters still fail validation.

## Model sweep

[`manifest.json`](tests/models/manifest.json) is a generated Transformers.js selection, not a
pinned CI gate. Both sweeps exclude the invalid Chronos encoder/decoder source graphs; see
[model exclusions](docs/transformersjs_excluded_models.md). The skeleton sweep fetches graph
structure and small constants with HTTP ranges, zero-fills missing large weights, and builds each
graph in ORT or CI-selected CoreML. It checks construction, not numerical correctness.

### Numerical-validation CI

PR CI numerically validates the curated
[`ci-validation.json`](tests/models/ci-validation.json): Linux/ORT blocks; macOS/CoreML is
experimental; Windows does not run this check. Each entry pins a Hugging Face commit and ONNX
SHA-256, verified on download and reuse. To add one, test both backends and record its file,
revision, digest, dimensions, and pinned inputs in the curated file—not the generated manifest.
Downloads honor `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `HF_HOME`, falling back to
the platform cache's `huggingface/hub` directory. Use `--refresh-model-cache` to force fresh downloads.

```bash
target/release/onnx2webnn validate-manifest --backend onnx \
  --manifest tests/models/ci-validation.json \
  --selection all --weights real --jobs 1
```

The manual full-manifest real-weight workflow needs a dedicated Linux runner (≥100 GiB disk,
~32 GiB RAM). Model failures are diagnostic; setup failures are fatal. Logs persist for 14 days.

For the skeleton test, `O2W_MODELS=hub|dir=<path>|strip=<path>` selects source models; unset
outside CI skips it. `O2W_MODEL_FETCH_JOBS`, `O2W_MODEL_TEST_JOBS`, and
`O2W_MODEL_TEST_SKIP_HEAVY` control concurrency and large cases.

```powershell
$env:O2W_MODELS = "dir=..\transformers_js_experiments\models"; cargo test --release --test model_skeletons
```

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

Operator handlers live in `src/onnx/ops/`; graph conversion is in `src/onnx/convert.rs`.
See [webnn-graph](../webnn-graph) for the `.webnn` format.
