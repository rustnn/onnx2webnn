# Real-weight validation failures

> **Last edited:** `2026-09-17T16:52:20Z`<br>
> **Checkout:** `fkrall/cache-backed-validation` at `25ea94c`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

This document is the current triage record for failures from full-model validation with publisher
weights. Generated-weight failures are intentionally out of scope. The summary and complete case
ledger remain in [Full-model numerical validation status](model-validation-status.md).

## Recorded run

- Date: 2026-09-17
- Tested executable: onnx2webnn `25ea94c`
- RustNN: `28fb3bbe`
- Manifest: `tests/models/manifest.json` (51 current cases; verified within a completed 52-case superset)
- Runtime: CPU ONNX Runtime 1.29.0
- Cache state: complete; no model downloads or skips
- Disk guard: minimum observed free space 78.7 GiB; stop threshold 15 GiB was not approached
- Result for the current manifest: 47 passed and 4 failed
- Recorded 52-case superset wall time: 7m 2.3s, one validation worker; peak RSS 26.4 GiB

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

The current manifest is the tested 52-case run minus the passing FP32 Tiny RoFormer case; every
retained case was executed in that run.

The recorded result for each case is its first blocker. A case that fails before comparison may
contain further converter, serialization, reload, execution, or numerical issues that are not yet
observable.

## Failure groups

| Code | Cases | Furthest stage | Classification | Current ownership |
|------|------:|----------------|----------------|-------------------|
| U1 | 1 | Output comparison | Unsupported strict fused/decomposed attention equivalence across dynamic quantization boundaries | FastVLM prefill remains blocked; its decode specialization passes. |
| Q1 | 1 | Output comparison | Unsupported `MatMulNBits accuracy_level=4` execution semantics | Requires a WebNN/backend mechanism for Int8-quantized activations with packed q4 weights; do not relax tolerance to hide it. |
| O1 | 2 | Native ORT model load | The publisher ONNX is rejected before conversion can be compared | Blocked upstream pending a corrected publisher artifact. |

The real sweep has no deterministic-input or cached-interface failures. U1 and Q1 both reach
comparison, but the source and reconstructed graphs do not promise the same numerical execution
mode. O1 fails in the reference model itself.

## U1: precision-sensitive fused attention followed by dynamic quantization

| # | Case | Comparison diagnostics |
|--:|------|------------------------|
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | `9,702,133 / 9,705,344` logits exceed tolerance; maximum absolute error `10.0572`, mean absolute error `0.760946`, RMSE `1.05325`. |

Boundary probes rule out a shape, rotary, cache-layout, or serialization defect. Layer 0 query
projection, external `RotaryEmbedding`, decomposed `GroupQueryAttention`, normalization, and MLP
all pass the normal Float32 envelope. Layer 1 GQA output also remains within that envelope, but
the immediately following `DynamicQuantizeLinear` makes the small difference discrete: the output
projection is the first failing observed boundary (`825 / 57,344` values, maximum absolute error
`0.001003`). Repeating this pattern through 24 layers produces the final logit disagreement above.

ORT 1.29 CPU GQA has multiple valid kernels. With its default tiled flash-attention kernel the
first native logit is `0.669124`; setting `ORT_GQA_DISABLE_FLASH_ATTENTION=1` selects ORT non-flash
fallback and changes that same native logit to `1.066261`. The decomposed WebNN result is stable at
`1.211192`. The source backend therefore materially disagrees with itself after repeated
quantization when only its legal attention kernel changes.

WebNN has no fused GQA primitive, and a decomposition into `matmul`, mask, `softmax`, and `matmul`
cannot guarantee bit-identical rounding to either backend-specific fused kernel. Weakening the
final tolerance would hide different Uint8 quantization decisions rather than accommodate ordinary
output accumulation error. This fixed prefill specialization is therefore recorded as unsupported
for strict cross-backend numerical validation; the sequence-1 decode specialization passes.

## Q1: unsupported `MatMulNBits accuracy_level=4`

Voxtral's decoder sets `accuracy_level=4` on all 211 `MatMulNBits` nodes. Native ORT may quantize
Float32 activations to Int8 internally in that mode. WebNN has no fused low-bit matmul operator, so
onnx2webnn emits packed Uint4 weights followed by `dequantizeLinear` and ordinary Float32 matmul.
The two paths therefore do not promise the same numerical algorithm.

This is not a packed-weight serialization problem. All 211 source weight payloads matched their
saved q4 tensors byte-for-byte by SHA-256; archive marker, U8 storage shape, nibble ordering, and
generated `0x88` zero points were also correct. Direct output analysis found mean absolute error
`0.0401`, maximum error `0.367`, and correlation `0.999502`. Changing only the source attributes
to `accuracy_level=0` reduced those figures to `9.66e-6`, `9.32e-5`, and `0.99999999997`.

This document records case 28 as Q1. Both `all` and `match` attempt it normally, so a future
implementation becomes visible. It is not counted as numerically supported.

## O1: source ONNX rejected by native ORT

Both Chronos files fail during native ORT model loading. A `Gather` indices input is the float
output of `ConstantOfShape`; Gather indices must be integer.

| # | Case |
|--:|------|
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` |

The two manifest paths resolve to the same invalid publisher artifact. This document records them
as O1; `all` and `match` still attempt them normally, so a future publisher correction will be
visible. Until the source model runs in native ORT, these cases cannot
provide a numerical oracle and are not evidence for or against onnx2webnn correctness.

## Repair order

1. Track U1 until WebNN or a backend-specific fusion can provide comparable attention precision
   across the dynamic-quantization boundaries.
2. Track Q1 until WebNN or a backend-specific fusion can express Int8 activations with packed q4 weights.
3. Monitor the upstream Chronos repository for corrected reference exports.

## Maintenance rule

This file describes only the latest complete real-weight run. Replace counts, cases, diagnoses,
and exact errors when a new run changes first blockers; do not append obsolete full failure tables.
Record revision-attributed coverage changes in the main status file's history section.
