# Transformers.js Excluded Models

> **Last edited:** `2026-10-16T18:05:10Z`<br>
> **Checkout:** `validation-enhancements` at `afb644f`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

[`scripts/generate_manifest.py`](../scripts/generate_manifest.py) reads this exclusion list.
Headings identify repositories; backticked `onnx/<component>*.onnx` patterns exclude only matching
components. Without a pattern, the whole repository is excluded. Reasons are informational, not
parsed. See [operator status](operator-conversion-status.md) for general support.

The 23 September 2026 skeleton audit found no fully supported excluded component. Some entries
with unresolved dimensions were not fully retested.

## [onnx-community/dpt-dinov2-small-kitti](https://huggingface.co/onnx-community/dpt-dinov2-small-kitti)

- `onnx/model*.onnx`: cubic `Resize`; WebNN supports nearest/linear.

## [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)

- `onnx/model*.onnx`: `STFT`, `Atan`, `NonZero`; q8 also uses unsupported ORT fusions
  `DynamicQuantizeLSTM`, `SkipLayerNormalization`, `FastGelu`, and `FusedMatMul`.

## [onnx-community/grounding-dino-tiny-ONNX](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX)

- `onnx/model*.onnx`: data-dependent `NonZero`; unsupported `GridSample` and `EyeLike`.

## [onnx-community/Janus-Pro-1B-ONNX](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX)

- `onnx/prepare_inputs_embeds*.onnx`: data-dependent `NonZero`; other exports convert.

## [huggingworld/Qwen2.5-VL-3B-Instruct-ONNX](https://huggingface.co/huggingworld/Qwen2.5-VL-3B-Instruct-ONNX)

- `onnx/vision_encoder*.onnx`: `TopK`, `NonZero`, `MultiHeadAttention`; embed_tokens/decoder convert.

## [Xenova/musicgen-small](https://huggingface.co/Xenova/musicgen-small)

- `onnx/build_delay_pattern_mask*.onnx`: data-dependent `NonZero`.
- `onnx/encodec_decode*.onnx`: dynamic pads; q8 also uses `DynamicQuantizeLSTM`.

## [onnx-community/sam3-tracker-ONNX](https://huggingface.co/onnx-community/sam3-tracker-ONNX)

- `onnx/vision_encoder*.onnx`: windowed-attention shape mismatch (9216 vs. 9 rows).
- `onnx/prompt_encoder_mask_decoder*.onnx`: nonconstant `Range` limit.

## [Xenova/RTMO-l](https://huggingface.co/Xenova/RTMO-l)

- `onnx/model*.onnx`: data-dependent `TopK`/`NonMaxSuppression` detection output.

## [Xenova/movenet-singlepose-thunder](https://huggingface.co/Xenova/movenet-singlepose-thunder)

- `onnx/model*.onnx`: keypoint broadcast mismatch (17 vs. 2 dimensions).

## [Xenova/bart-large-mnli](https://huggingface.co/Xenova/bart-large-mnli)

- `onnx/model*.onnx`: data-dependent `NonZero` in EOS-token pooling.

## [D4ve-R/wavlm-base-plus-sv](https://huggingface.co/D4ve-R/wavlm-base-plus-sv)

- `onnx/model*.onnx`: TDNN `Gather` axis 4 exceeds rank 3; probe needs ≥400 audio samples.

## [BricksDisplay/silero-vad-6.2](https://huggingface.co/BricksDisplay/silero-vad-6.2)

- `onnx/model*.onnx`: pinning `sr` leaves shape-gated `If` branches unresolved.

## [Xenova/4x_APISR_GRL_GAN_generator-onnx](https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx)

- `onnx/model*.onnx`: window-partition reshape has wrong rank (4 vs. 6).

## [AdamCodd/distilroberta-nsfw-prompt-stable-diffusion](https://huggingface.co/AdamCodd/distilroberta-nsfw-prompt-stable-diffusion)

- Gated repository (HTTP 401 without a token); not evaluated

## [Mozilla/distilvit](https://huggingface.co/Mozilla/distilvit)

- `onnx/decoder_model_merged*.onnx`: cache-branch reshape target is not constant-folded;
  encoder converts.

## [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)

- `onnx/model*.onnx`: q8 export uses unsupported `DynamicQuantizeLSTM`.

## [Xenova/nllb-200-distilled-600M](https://huggingface.co/Xenova/nllb-200-distilled-600M)

- `onnx/decoder_model_merged*.onnx` — prefill fails: bias reshape uses `decoder_sequence_length`
  instead of vocab size 256206. Decode builds; encoder converts.
