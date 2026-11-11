# UltraQuant explainer video: README hand-off

Video: `video/ultraquant_explainer.mp4` (1080p, 30 fps, burned-in captions; subtitles also in `video/ultraquant_explainer.srt`). Poster: `video/poster.png`. Preview: `video/preview.gif`.
Release link: https://github.com/Bobliuuu/ultraquant-more/blob/main/video/ultraquant_explainer.mp4
Rebuild: `video/build.sh` (needs ffmpeg, python3 with numpy/matplotlib/pillow, macOS `say`). Scenes are code in `video/src/scenes.py`; narration in `video/script.md`; every chart reads `results/*.json`.

## README hero text (embeddable)

```markdown
<p align="center"><a href="https://github.com/Bobliuuu/ultraquant-more/blob/main/video/ultraquant_explainer.mp4"><img src="video/poster.png" width="720" alt="UltraQuant explainer video"></a></p>

**UltraQuant: 4-bit KV caching for context-heavy agents.** FP4 (E2M1) values with UE8M0 group scales
(32 elements per scale) cut KV cache to 68 bytes per head-token (fp16: 256, fp8: 128). Our contribution is
a fused multi-token (K-query) paged-attention decode kernel: each KV page is read once and reused for all K
speculative / MTP-verify tokens, instead of K independent passes. Measured at B=8, ctx 32768:
4.1x over K independent queries at K=8 on MI300X, 6.8x on B300 (tuned).
[Watch the 3-minute explainer](https://github.com/Bobliuuu/ultraquant-more/blob/main/video/ultraquant_explainer.mp4).
```

## How it works (5 bullets for the README)

1. Decode attention does about 1 FLOP per KV byte, far below the ~164 (MI300X) / ~274 (B300) FLOP/byte ridge computed from `results/*_peak.json`, so it is memory-bound: fewer KV bytes means faster tokens.
2. KV is stored as E2M1 FP4 with one UE8M0 (power-of-two, 1 byte) scale per 32 elements: 64 B values + 4 B scales = 68 B per head-token at D=128, about 3.8x smaller than fp16.
3. Speculative decoding / MTP verify scores K candidate tokens per step. A naive kernel runs K independent queries, so each re-reads every KV page (K times the traffic).
4. The fused kernel stacks the K tokens (times the grouped query heads) along the M dimension of one `tl.dot`, so each FP4 page is loaded and dequantised once and serves all K tokens.
5. Tuning (bit-trick E2M1 to bf16 dequant, warps/stages, KV split) gives K=8 fused 0.54 ms vs 2.22 ms independent on MI300X (4.1x) and 0.42 vs 2.85 ms on B300 (6.8x); K=1 shows no gain, as expected. MI300X (gfx942) dequantises FP4 in software; native scaled MFMA is on CDNA4 (gfx950).

## Scene to results-file map

| Scene | Uses |
|---|---|
| Memory-bound | `results/mi300x_peak.json`, `results/b300_peak.json` |
| FP4 + UE8M0 packing | arithmetic only (D=128: 64 B + 4 B scales) |
| K-query trick | schematic (K=4 shown) |
| Results | `results/mi300x_b8_c32k_tuned.json`, `results/b300_b8_c32k_tuned.json`, `results/*_tuning_notes.md` |
| Caveats | same JSONs (achieved GB/s, max error) and peaks |
| One op, three vendors (auto-included only when `results/cuda_*.json` or `results/sycl_*.json` exist) | those files |

Claims policy: the video states only numbers present in results files. Cross-vendor / novelty wording is deliberately absent from the video until `cuda/PRIOR_ART.md` and `sycl/PRIOR_ART.md` back it.
