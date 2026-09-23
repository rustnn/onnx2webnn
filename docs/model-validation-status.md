# Full-model numerical validation status

> **Last edited:** `2026-09-23T11:35:58Z`<br>
> **Checkout:** `fkrall/cache-backed-validation` at `ac6d5ca`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

This document tracks the **latest recorded** full-model numerical-validation sweep and a concise
history of changes that materially altered coverage. Only the latest sweep keeps a complete
per-case ledger; older ledgers and duplicated historical failure tables are intentionally omitted.

The skeleton sweep establishes broad graph-construction coverage. The manifest contains only model
selection and execution configuration; numerical status and blocker triage live in this document and
the linked failure analysis. Numerical validation additionally
exports the converted graph, reloads its `.webnn` and Safetensors artifacts, executes both the
original ONNX model and the reloaded graph on CPU ONNX Runtime, and compares matching outputs.

Proper dynamic cache-shape support is not currently available. Validation cases use fixed
dimension overrides and execute as independent snapshots, so they do not validate a complete
prefill-to-decode loop or growing KV-cache reuse. A decode artifact that
accepts a fixed past length `N` normally produces a cache of length `N + 1`, which cannot be fed
back into that same fixed-shape artifact. Repeated decoding therefore still requires proper dynamic
cache shapes, separately specialized artifacts, or a fixed-capacity cache with an explicit position.

## Broad manifest selection

[`tests/models/manifest.json`](../tests/models/manifest.json) is not a global top-50 list. The
[`scripts/generate_manifest.py` generator](../scripts/generate_manifest.py) queries Hugging Face for
repositories tagged `transformers.js`, groups them by pipeline task, and by default selects the
single most-liked repository in each task (`--top 1`). Tokenizer-only repositories, repositories
without usable ONNX exports, and repositories or components listed in the
[Transformers.js excluded-model list](transformersjs_excluded_models.md) are omitted.

For each selected repository, the generator includes the ONNX components that Transformers.js
would load for that model class, such as both encoder and merged decoder graphs for a sequence-to-
sequence model. It follows the model's `transformers.js_config` dtype when present; otherwise the
default generator mode mirrors WASM and prefers q8 exports. Missing preferred variants fall back to
an available export. Models with a `use_cache_branch` input receive separate prefill and decode
cases, while other multipart models contribute one case per required graph. Consequently the
current manifest contains 28 repositories, 44 unique ONNX files, and 51 configured cases.

The generator assigns bounded dimensions using onnx2webnn's naming policy and carries forward
existing per-file dimension overrides, pinned inputs, and other manual fields. It inspects remote
ONNX structure using HTTP range requests rather than downloading full weights. Regeneration tracks
publisher `main` revisions and is therefore diagnostic coverage, not an immutable model set. The
current population was last generated around early September 2026 and entered Git history on
13 September 2026.

## Validation automation

Pull requests use `tests/models/ci-validation.json` as the required numerical-validation contract.
Linux/ORT is blocking; macOS/CoreML runs the identical set as an experimental, non-blocking job;
Windows does not run numerical validation. Membership is defined by the curated file itself and the
runner invokes it with `--selection all`. Every entry must be suitable for hosted runners and include
a full immutable Hugging Face commit revision and a lowercase SHA-256 for its primary ONNX file.
The downloader verifies that digest both after download and on cache reuse. Full-model downloads
use the standard Hugging Face cache by default so Python and Rust clients can share immutable blobs;
`O2W_ONNX_CACHE` and `O2W_CACHE_DIR/onnx` remain explicit overrides. Completion records stay
onnx2webnn-specific and do not replace the manifest digest check.

To add a required model, first confirm it passes on ORT and CoreML, then add its exact file, commit
revision, downloaded-file digest, fixed dimension overrides, and any pinned inputs to
`tests/models/ci-validation.json`. This file is hand-maintained. The
[`scripts/generate_manifest.py`](../scripts/generate_manifest.py) generator only regenerates
`tests/models/manifest.json` and must not be used to update or carry CI pins.

The `Full model validation` workflow is manually dispatched for real publisher weights.
It runs the entire generated manifest sequentially on a runner labeled `self-hosted`, `linux`,
`x64`, and `onnx2webnn-validation`. The runner must provide network access, a writable persistent
`/var/cache/onnx2webnn`, at least a 100 GiB filesystem, and approximately 32 GiB RAM.
Downloaded model sources persist; WebNN exports are temporary. Provisioning, storage checks, and
build failures are fatal. Per-model validation failures remain diagnostic: the workflow emits a
warning and uploads its complete log for 14 days. There is no scheduled or extended-tier sweep.

### Runtime dtypes and comparison tolerances

WebNN graphs are not globally limited to floating point: RustNN records integer tensors and integer
operations where their operator contracts allow them. Matrix multiplication is different. The
portable WebNN `matmul` contract accepts matching `float16` or `float32` operands, and WebNN has no
fused equivalent of ORT's `com.microsoft.MatMulNBits`.

For q4 models, the original ONNX path gives native ORT packed Uint4 weights and a fused
`MatMulNBits` node. The reloaded path restores the same packed bytes, widens them for RustNN's
temporary ORT graph, applies `dequantizeLinear`, and runs ordinary matmul in the scale dtype. The
models in this manifest use Float32 scales and activations, so their reconstructed matmuls are
Float32. Different fused/decomposed accumulation orders are mathematically equivalent but not
bit-identical.

The validator therefore uses these output tolerances:

| Output/path | Comparison |
|-------------|------------|
| Float32 without `MatMulNBits` | `1e-5 + 1e-4 * abs(reference)` |
| Float32 from a model containing q4 `MatMulNBits` | `1e-3 + 1e-3 * abs(reference)` |
| Float32 from a model containing q8 `MatMulNBits` | `2e-3 + 2e-3 * abs(reference)` |
| Float16 | `1e-3 + 1e-2 * abs(reference)` |
| Integer and boolean | Exact |

The q4 Float32 envelope is based on a complete SmolLM2 output scan whose worst normalized
difference was `9.488e-4`. The q8 envelope is selected from each source graph's
`MatMulNBits.bits` attribute; it covers the measured Qwen output delta while remaining distinct
from q4. It does not make ORT's `accuracy_level=4` equivalent to WebNN:
level 4 permits internal Int8 activation quantization, while the WebNN lowering keeps Float32
activations. Voxtral uses level 4 on all 211 `MatMulNBits` nodes and remains blocked for this reason.

## Latest recorded sweep

- Real sweep: 2026-09-22 on `fkrall/cache-backed-validation` at `a582b89` plus the uncommitted Hugging Face cache migration worktree
- RustNN: `38022044`
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (51 cases, 44 unique ONNX files)
- Execution: one validation worker; ORT may use multiple CPU threads inside a case
- Skeleton verification: 51/51 passed; the test took 81.1s (133.2s including the release build)
- Real verification: 47/51 passed in one complete cold-cache process; wall time was 10m 43.6s and peak RSS was 30.3 GiB. No case was skipped or inferred without execution.

All 51 current manifest cases were executed. A newer checkout is not considered the tested baseline
until the real sweep is rerun and this section is replaced.

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

### Summary

| Weight mode | Pass | Fail | Download skipped | Result |
|-------------|-----:|-----:|-----------------:|--------|
| Real | 47 | 4 | 0 | Complete (51/51) |

Two cases exercise unsupported cross-backend execution modes: FastVLM precision-sensitive GQA/DQL
prefill and Voxtral level-4 quantized matmul. The two Chronos publisher artifacts are rejected by
native ORT before conversion comparison.

The real-weight failures, exact affected cases, current diagnosis, and suggested ownership are
tracked in [Real-weight validation failures](real-weight-validation-failures.md).

### Current failure families

| Code | Cases | Stage | Current cause / next action |
|------|------:|-------|-----------------------------|
| Q1 | 1 | Comparison | Unsupported execution semantics: Voxtral sets `MatMulNBits accuracy_level=4`, allowing native ORT to quantize activations to Int8; WebNN lowers to Float32 dequantize-plus-matmul. |
| U1 | 1 | Comparison | Unsupported strict equivalence: FastVLM prefill repeatedly quantizes attention results whose legal floating-point rounding differs between fused ORT kernels and the portable WebNN decomposition; decode passes. |
| O1 | 2 | Native ORT load | Upstream-blocked: both Chronos paths resolve to the same publisher artifact, which feeds a float ConstantOfShape result to Gather indices. |

Real totals: no export, reload, cached-interface, or native-ORT input failures; 2 native-ORT
model-load failures, 2 unsupported execution-mode mismatches, and 47 passes.

### Current case ledger

`PASS` means export, reload, both executions, and numerical comparison completed. Any other value
is the current first-blocker code from the table above.

`WebNN If split` identifies merged decoders whose runtime `use_cache_branch` condition must be
pinned because WebNN has no `If` operation. Each prefill/decode entry produces its own independent
`.webnn`/Safetensors pair. Loading both graphs concurrently may duplicate their shared weights in
RAM and VRAM. This is distinct from ordinary repeated cases that differ only in fixed dimensions.

| # | Manifest case | Real | WebNN If split |
|---:|---------------|------|----------------|
| 0 | `briaai--RMBG-1.4 :: model_quantized.onnx` | PASS | — |
| 1 | `openai--privacy-filter :: model_quantized.onnx` | PASS | — |
| 2 | `nomic-ai--nomic-embed-text-v1.5 :: model_quantized.onnx` | PASS | — |
| 3 | `mixedbread-ai--mxbai-embed-large-v1 :: model_quantized.onnx` | PASS | — |
| 4 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) | PASS | — |
| 5 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) | PASS | — |
| 6 | `distil-whisper--distil-large-v2 :: encoder_model_quantized.onnx` | PASS | — |
| 7 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=0`) | PASS | Prefill pair (1/2) |
| 8 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=1`) | PASS | Decode pair (2/2) |
| 9 | `jinaai--jina-reranker-v2-base-multilingual :: model_quantized.onnx` | PASS | — |
| 10 | `onnx-community--FastVLM-0.5B-ONNX :: embed_tokens_quantized.onnx` | PASS | — |
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | U1 | — |
| 12 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=1`, `past=64`) | PASS | — |
| 13 | `onnx-community--FastVLM-0.5B-ONNX :: vision_encoder_quantized.onnx` | PASS | — |
| 14 | `Marqo--marqo-fashionSigLIP :: text_model_quantized.onnx` | PASS | — |
| 15 | `Marqo--marqo-fashionSigLIP :: vision_model_quantized.onnx` | PASS | — |
| 16 | `AdamCodd--vit-base-nsfw-detector :: model_quantized.onnx` | PASS | — |
| 17 | `Xenova--nllb-200-distilled-600M :: encoder_model_quantized.onnx` | PASS | — |
| 18 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` | PASS | — |
| 19 | `onnx-community--Janus-Pro-1B-ONNX :: lm_head.onnx` | PASS | — |
| 20 | `onnx-community--Janus-Pro-1B-ONNX :: gen_head.onnx` | PASS | — |
| 21 | `onnx-community--Janus-Pro-1B-ONNX :: gen_img_embeds.onnx` | PASS | — |
| 22 | `onnx-community--Janus-Pro-1B-ONNX :: image_decode.onnx` | PASS | — |
| 23 | `Xenova--musicgen-small :: text_encoder_quantized.onnx` | PASS | — |
| 24 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=0`) | PASS | Prefill pair (1/2) |
| 25 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=1`) | PASS | Decode pair (2/2) |
| 26 | `Mozilla--distilvit :: encoder_model_quantized.onnx` | PASS | — |
| 27 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: embed_tokens_fp16.onnx` | PASS | — |
| 28 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` | Q1 | — |
| 29 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: audio_encoder_quantized.onnx` | PASS | — |
| 30 | `Xenova--LaMini-Flan-T5-783M :: encoder_model_quantized.onnx` | PASS | — |
| 31 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=0`) | PASS | Prefill pair (1/2) |
| 32 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=1`) | PASS | Decode pair (2/2) |
| 33 | `Xenova--detr-resnet-50 :: model_quantized.onnx` | PASS | — |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | PASS | — |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=0`) | PASS | Prefill pair (1/2) |
| 36 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=1`) | PASS | Decode pair (2/2) |
| 37 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model.onnx` | PASS | — |
| 38 | `Xenova--distilbart-cnn-6-6 :: encoder_model_quantized.onnx` | PASS | — |
| 39 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=0`) | PASS | Prefill pair (1/2) |
| 40 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=1`) | PASS | Decode pair (2/2) |
| 41 | `prithivMLmods--Common-Voice-Gender-Detection-ONNX :: model_quantized.onnx` | PASS | — |
| 42 | `Xenova--bert-base-multilingual-cased :: model_quantized.onnx` | PASS | — |
| 43 | `Xenova--distilbert-base-cased-distilled-squad :: model_quantized.onnx` | PASS | — |
| 44 | `onnx-community--vitpose-base-simple :: model_quantized.onnx` | PASS | — |
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` | O1 | — |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` | O1 | — |
| 47 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: embed_tokens_quantized.onnx` | PASS | — |
| 48 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_quantized.onnx` | PASS | — |
| 49 | `onnx-community--timesformer-base-finetuned-k400 :: model_quantized.onnx` | PASS | — |
| 50 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model_quantized.onnx` | PASS | — |

### Current timing and storage

| Run | Cases | Wall time |
|-----|------:|---------------:|
| Real, cold standard HF cache | 51 | 10m 43.6s |

The official Rust client populated 44 completion records, one per unique ONNX file. A subsequent
offline `match=privacy-filter` run verified warm reuse of both the primary ONNX and its external-data
sidecar.

After the sweep, the standard Hugging Face cache occupied 18 GB and temporary WebNN artifacts
occupied 32 GB; 122 GiB remained free. No selected case was skipped for download or storage.

### Current repair order

1. Track WebNN/backend capabilities that could make U1 or Q1 executable with comparable fused precision semantics.
2. Monitor the upstream Chronos repository for corrected reference exports.

## Coverage change history

This history records only changes that explain coverage movement or establish that an apparent
movement was not a functional improvement. Detailed obsolete ledgers are available through Git.

| Date / tested revisions | Change | Comparable coverage effect |
|-------------------------|--------|----------------------------|
| 2026-09-17 — validation-tier cleanup after `25ea94c` | Removed manifest validation tiers and the hand-added FP32 RoFormer smoke case, restoring the generator-owned upstream population. Status and blocker ownership now live only in these validation documents; the CLI selects `all` or `match=<text>`. | Population changed **52 → 51** by removing one passing duplicate-model variant. Comparable coverage is unchanged: skeleton **51/51** and real validation **47/51** with the same four blockers. |
| 2026-09-17 — onnx2webnn `25ea94c`, RustNN `28fb3bbe`, post-rebase complete rerun | Rebased both feature stacks onto `rustnn/onnx2webnn:main` and `rustnn/rustnn:main`, then ran all 52 skeleton cases and all 52 real-weight cases from warm caches, including heavy entries. | No regression: skeleton remained **52/52**; real validation remained **48/52** with the identical FastVLM U1, Voxtral Q1, and two Chronos O1 blockers. |
| 2026-09-17 — onnx2webnn `d4d350b`, RustNN `724d076b`, targeted FastVLM probes | Exposed layer boundaries and compared ORT CPU flash/non-flash GQA against the WebNN decomposition. Layer 0 and external rotary matched; the first failing boundary followed layer-1 GQA dynamic quantization and output projection. Native ORT first-logit output changed from `0.669124` to `1.066261` when only its GQA kernel changed, while WebNN remained `1.211192`. | Reclassified FastVLM prefill from generic N1 to unsupported U1. Coverage remains **48/52**; no tolerance was weakened. |
| 2026-09-17 — onnx2webnn `6f6ad8a` plus current Cast/Slice/comparison worktree, RustNN `65e76e67` plus Slice-backend worktree | Normalized ONNX numeric-to-Bool Cast through comparison, preserved positive Slice strides with WebNN extent semantics, and selected q4/q8 comparison envelopes from `MatMulNBits.bits`. | DETR, Donut encoder, and Qwen passed; real coverage rose **45 → 48** on the same 52 cases. FastVLM prefill, Voxtral level 4, and the two invalid Chronos artifacts remain blocked. |
| 2026-09-16 — onnx2webnn `24fdd50` plus current validator/manifest worktree, RustNN `7f07a5e1` plus packed-4-bit archive worktree | Measured complete q4 output error distributions and applied a `1e-3` absolute/relative Float32 envelope only to source graphs containing `MatMulNBits`. Classified Voxtral's `accuracy_level=4` Int8-activation execution mode as unsupported. | SmolLM2 prefill/decode and Janus passed; Voxtral remained Q1. Real passes rose **42 → 45** on the same 52 cases. |
| 2026-09-16 — onnx2webnn `24fdd50` plus current manifest/tests worktree, RustNN `7f07a5e1` plus packed-4-bit archive worktree | Stored packed Int4/Uint4 constants as versioned U8 Safetensors payloads while preserving logical dtype and shape in `.webnn`, then restored and executed them on reload. Marked the identical invalid Chronos publisher artifacts as upstream-blocked. | E1 was eliminated from the real sweep: all four q4 cases reached comparison and exposed N1. The aggregate remained **42/52** because those cases do not yet match numerically. |
| 2026-09-15 — onnx2webnn `ec5ba275` plus current validator worktree, RustNN `7f07a5e1` | Reconciled native ONNX interfaces with branch-specialized cached WebNN interfaces, dispatching only retained inputs and accepting omitted outputs only when native ORT proves they are empty. | Real passes rose **32 → 42** on the same 52 cases. All ten V1 cases passed numerically and no new blocker family appeared. |
| 2026-09-14 — onnx2webnn `aaa33e2` plus current validator worktree, RustNN `7f07a5e1` | Made standard `token_type_ids` zero, `attention_mask` one, and preserved zero-element input buffers without treating scalars as empty. | Real passes rose **28 → 32** on the same 52 cases. I1-I3 were eliminated: four cases passed, five exposed V1, and two exposed N1. |
| 2026-09-12 — onnx2webnn `4926c3e`, RustNN `7f07a5e1` | RustNN `2e22b3db` unified MLGraphBuilder recording and GraphJSON loader inference, replacing the loader-only string-based inference loop. Typed inference added the previously missing/rejected Resample2d, RoundEven/Clamp, Conv2d, LogicalNot, and LogicalAnd reload paths. | On the same 52-case manifest, real passes rose **6 → 28**. Former reload families R1–R5 were eliminated; newly reachable cases exposed N1 and V1 instead. |
| 2026-09-12 — RustNN `7f07a5e1` | Made `[]` unambiguously scalar and required completed GraphInfo descriptors to have known shapes. | Hardened the shared inference path but did not itself add the dispatch that cleared R1–R5; no separate coverage gain is attributed without a bisect. |
| 2026-09-09 — onnx2webnn `e5a5f88c`, RustNN `2783a191` | Rebased onto the newer upstream model manifest and converter work. The manifest changed from 63 to 52 cases: 21 retained, 31 added, 42 removed. | Retained cases preserved their outcomes. Real coverage changed to 6 passes because the population changed, not because comparable coverage improved. One real Voxtral download was skipped. |
| 2026-09-08/09 — onnx2webnn `f1ca548`, RustNN `b64cf495` | First complete real-weight full-model baseline with cache-backed export, reload, deterministic execution, and comparison. | 7/63 real cases passed. Every case that reached comparison matched; failures established the original reload, serialization, and input blocker families. |

### RustNN inference-refactor attribution

RustNN `2e22b3db` removed the separate ten-pass GraphJSON shape-inference implementation. Both
normal graph construction and GraphJSON loading now pass typed operations through
`GraphRecorder::record_operation` and `infer_operation_descriptors`. This directly removed:

- R1: missing Resample2d output-shape inference.
- R2: inconsistent RoundEven spelling and missing Clamp inference.
- R3: missing Conv2d output-shape inference.
- R4: inconsistent LogicalNot spelling.
- R5: inconsistent LogicalAnd spelling.

The full manifest was tested at `7f07a5e1`, not bisected at `2e22b3db`; attribution is based on
the code changes between the recorded revisions. A focused reload benchmark found the changed
graph reconstruction/inference stage about 6% faster on seven cached graphs, while end-to-end
weight-heavy loading was effectively unchanged.

## Maintenance rule

When a new complete sweep is recorded:

1. Replace the latest metadata, summary, failure counts, timing, and the **single** case ledger.
2. Add one concise history row only when a code/revision change explains coverage movement.
3. Do not append another historical case ledger or duplicate obsolete failure-family tables.
4. Distinguish manifest/population changes from improvements on comparable cases.
