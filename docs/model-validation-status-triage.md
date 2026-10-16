# Full-model numerical validation status and triage

> **Last edited:** `2026-10-16T23:16:29Z`<br>
> **Checkout:** `validation-enhancements` at `ae2e0cc`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

The skeleton sweep checks graph construction. Real-weight validation additionally exports and
reloads `.webnn`/Safetensors, then compares reloaded-graph execution against native ONNX on CPU
ORT. This page records the latest tested result and reviewed blockers. Fixed input shapes do **not**
test a growing KV cache or a full prefill-to-decode loop.

## Broad manifest selection

[`scripts/generate_manifest.py`](../scripts/generate_manifest.py) selects the most-liked
`transformers.js` repository per Hugging Face task and the ONNX components Transformers.js would
load. It excludes [known blockers](transformersjs_excluded_models.md), reuses dimension overrides,
and adds separate prefill/decode cases where needed. The current
[`manifest.json`](../tests/models/manifest.json) has **28 repositories, 44 files, 51 cases**.
Last generated around early September 2026, it follows publisher `main` revisions; it is not an
immutable top-50 benchmark.

## Validation automation

PR CI runs the hand-maintained [`ci-validation.json`](../tests/models/ci-validation.json):
Linux/ORT is blocking; macOS/CoreML is experimental; Windows does not run numerical validation.
Entries need a pinned Hugging Face commit, primary-ONNX SHA-256, dimensions, and pins. The digest
is checked on download and cache reuse. The generated broad manifest does not supply these CI pins.

The manually dispatched `Full model validation` workflow runs the broad manifest on a dedicated
Linux runner (≥100 GiB disk, ~32 GiB RAM). Model failures are diagnostic; setup failures are
fatal. It retains the log for 14 days.

### Runtime dtypes and comparison tolerances

WebNN `matmul` uses matching Float16 or Float32 inputs; it has no fused `MatMulNBits`. For q4,
the reloaded path dequantizes packed weights before matmul, so rounding can differ from native
ORT's fused kernel. The validator uses these tolerances:

| Output/path | Comparison |
|-------------|------------|
| Float32 without `MatMulNBits` | `1e-5 + 1e-4 * abs(reference)` |
| Float32 from a model containing q4 `MatMulNBits` | `1e-3 + 1e-3 * abs(reference)` |
| Float32 from a model containing q8 `MatMulNBits` | `2e-3 + 2e-3 * abs(reference)` |
| Float16 | `1e-3 + 1e-2 * abs(reference)` |
| Integer and boolean | Exact |

q4/q8 envelopes are selected from `MatMulNBits.bits`; q4 was measured on SmolLM2. They do not
cover ORT `accuracy_level=4` Int8-activation execution, which still blocks Voxtral.

## Latest recorded sweep

- Real sweep: 2026-09-22 on `fkrall/cache-backed-validation` at `a582b89` plus the uncommitted Hugging Face cache migration worktree
- RustNN: `38022044`
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (51 cases, 44 unique ONNX files)
- Skeleton verification: 51/51 passed; the test took 81.1s (133.2s including the release build)
- Real verification: **47/49 eligible passed**; 51 attempted, including two invalid source ONNX
  cases excluded from the conversion metric; complete cold-cache run, one worker, no skips;
  10m 43.6s, peak RSS 30.3 GiB

These results describe the listed checkout, not every newer revision.

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

### Summary

| Weight mode | Eligible pass | Eligible fail | Invalid source ONNX | Download skipped | Result |
|-------------|--------------:|--------------:|--------------------:|-----------------:|--------|
| Real | 47 | 2 | 2 | 0 | Complete (51 attempted; 49 eligible) |

The two O1 cases remain in the manifest and run ledger but cannot measure conversion while native
ORT rejects their source model. See [failure evidence](real-weight-validation-failures.md) for both
conversion disagreements and the upstream blockers.

### Current failure families

| Code | Cases | Stage | Current cause / next action |
|------|------:|-------|-----------------------------|
| Q1 | 1 | Comparison | Unsupported execution semantics: Voxtral sets `MatMulNBits accuracy_level=4`, allowing native ORT to quantize activations to Int8; WebNN lowers to Float32 dequantize-plus-matmul. |
| U1 | 1 | Comparison | Unsupported strict equivalence: FastVLM prefill repeatedly quantizes attention results whose legal floating-point rounding differs between fused ORT kernels and the portable WebNN decomposition; decode passes. |
| O1 | 2 | Native ORT load | Upstream-blocked: both Chronos paths resolve to the same publisher artifact, which feeds a float ConstantOfShape result to Gather indices. |

### Current case ledger

`PASS` covers export, reload, both executions, and comparison. Other codes name the first blocker.
`WebNN If split` means separate prefill/decode artifact pairs; loading both may duplicate weights
in RAM/VRAM.

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

The Hugging Face cache held 18 GB, WebNN artifacts 32 GB, and 122 GiB remained free. An offline
privacy-filter rerun confirmed ONNX and sidecar reuse.

## Coverage change history

Only coverage-changing milestones are listed; Git retains older ledgers. Earlier historical
fractions use attempted-case denominators; the current comparable row uses eligible cases.

| Date / tested revisions | Change | Comparable coverage effect |
|-------------------------|--------|----------------------------|
| 2026-09-17 — after `25ea94c` | Removed one passing FP32 RoFormer case from the generated population. | **48/50 → 47/49 eligible**; same four attempted blockers, no comparable regression. |
| 2026-09-17 — `6f6ad8a`, RustNN `65e76e67` | Fixed Bool Cast, Slice strides, and q4/q8 tolerances. | **45 → 48/52**; DETR, Donut encoder, and Qwen passed. |
| 2026-09-16 — `24fdd50`, RustNN `7f07a5e1` | Added packed-4-bit reload and measured q4 tolerance. | q4 export blocker cleared; SmolLM2 and Janus passed, **42 → 45/52**. |
| 2026-09-15 — `ec5ba275`, RustNN `7f07a5e1` | Reconciled branch-specialized WebNN inputs/outputs with native ONNX. | **32 → 42/52**. |
| 2026-09-14 — `aaa33e2`, RustNN `7f07a5e1` | Fixed semantic token/mask feeds and zero-element buffers. | **28 → 32/52**. |
| 2026-09-12 — `4926c3e`, RustNN `7f07a5e1` | Unified builder/loader shape inference in RustNN. | **6 → 28/52**; prior reload blockers R1–R5 cleared. |
| 2026-09-09 — `e5a5f88c`, RustNN `2783a191` | Changed manifest population from 63 to 52 cases. | 6 passes on the new population; not a comparable regression from 7/63. |
| 2026-09-08/09 — `f1ca548`, RustNN `b64cf495` | First full real-weight baseline. | **7/63** passed. |

## Maintenance rule

After a complete sweep, replace the current result and single ledger. Add history only for
comparable coverage changes; distinguish those from manifest population changes.
