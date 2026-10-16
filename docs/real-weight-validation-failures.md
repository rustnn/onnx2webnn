# Real-weight validation failures

> **Last edited:** `2026-10-16T23:16:29Z`<br>
> **Checkout:** `validation-enhancements` at `ae2e0cc`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

Reviewed first blockers for real-weight validation. See the
[51-case ledger](model-validation-status-triage.md); these diagnoses are not inferred automatically.

## Recorded run

- Date: 2026-09-17
- Tested executable: onnx2webnn `25ea94c`
- RustNN: `28fb3bbe`
- Manifest: 51 current cases within a completed 52-case sweep
- Runtime: CPU ONNX Runtime 1.29.0
- Result for the current manifest: 47/49 eligible passed; two invalid source ONNX cases were
  attempted but excluded from the conversion metric. The 52-case run took 7m 2.3s, peak RSS
  26.4 GiB.

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

The 51-case manifest omits one passing FP32 Tiny RoFormer case from that run. A first blocker may
hide later failures.

## Failure groups

| Code | Cases | Furthest stage | Classification | Current ownership |
|------|------:|----------------|----------------|-------------------|
| U1 | 1 | Comparison | FastVLM attention rounding crosses dynamic-quantization boundaries | Needs comparable fused/decomposed precision; decode passes. |
| Q1 | 1 | Comparison | Voxtral uses `MatMulNBits accuracy_level=4` | Needs Int8-activation/q4 execution parity. |
| O1 | 2 | Native ORT load | Publisher Chronos graph is invalid | Await corrected source export. |

## U1: precision-sensitive fused attention followed by dynamic quantization

| # | Case | Comparison diagnostics |
|--:|------|------------------------|
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | `9,702,133 / 9,705,344` logits exceed tolerance; maximum absolute error `10.0572`, mean absolute error `0.760946`, RMSE `1.05325`. |

The first divergent boundary is the projection after layer-1 `GroupQueryAttention` and
`DynamicQuantizeLinear` (825/57,344 values; maximum error `0.001003`). ORT's first logit changes
from `0.669124` to `1.066261` when switching its GQA kernel; WebNN yields `1.211192`.
WebNN's decomposed attention cannot guarantee the fused kernel's rounding, and dynamic
quantization amplifies the difference. A looser final tolerance would hide different Uint8
decisions. The sequence-1 decode case passes.

## Q1: unsupported `MatMulNBits accuracy_level=4`

Voxtral has 211 `MatMulNBits` nodes with `accuracy_level=4`: ORT may use Int8 activations, while
WebNN dequantizes packed q4 weights into Float32 matmul. All 211 saved weight payloads matched the
source byte-for-byte. Output mean/max errors were `0.0401`/`0.367`; changing only the source to
`accuracy_level=0` reduced them to `9.66e-6`/`9.32e-5`. Case 28 is not a numerical pass.

## O1: source ONNX rejected by native ORT

Both Chronos files fail during native ORT model loading. A `Gather` indices input is the float
output of `ConstantOfShape`; Gather indices must be integer.

| # | Case |
|--:|------|
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` |

Both paths resolve to the same invalid publisher artifact. They remain in `all` and `match`, but
cannot validate onnx2webnn until native ORT accepts the source.
