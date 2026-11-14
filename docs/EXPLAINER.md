# AllQuant explainer video: README hand-off

Video: `video/ultraquant_explainer.mp4` (1080p, 24 fps, no voice-over, burned-in captions and background music; subtitles also in `video/ultraquant_explainer.srt`). Poster: `video/poster.png`. Preview: `video/preview.gif`.
Rebuild: `python3 tools/ports_data.py && python3 ~/video-readme-maker/vrm.py video README.spec.yml --silent` ([video-readme-maker](https://github.com/Bobliuuu/video-readme-maker); needs ffmpeg, node and Chrome). Scenes and caption text are in `README.spec.yml` (`video.scenes`); every number is read from `results/`. The older `video/build.sh` pipeline is kept but does not produce this video.

## README hero text (embeddable)

```markdown
<p align="center"><a href="https://github.com/Bobliuuu/all-quant/blob/main/video/ultraquant_explainer.mp4"><img src="video/poster.png" width="720" alt="UltraQuant explainer video"></a></p>

**UltraQuant: 4-bit KV caching for context-heavy agents.** FP4 (E2M1) values with UE8M0 group scales
(32 elements per scale): 68 bytes per head-token (fp16: 256, fp8: 128). Our contribution is a fused multi-token
(K-query) paged-attention decode kernel: each KV page is read once and reused for all K speculative / MTP-verify
tokens, instead of K independent passes. The same op runs on AMD (Triton, MI300X), NVIDIA (CUDA: B300, H100, A100,
L40S, RTX PRO 6000) and Intel (SYCL/XMX, Arc B390). At K=8, B=8, ctx 32768, fusing beats K independent passes by
2.5x to 6.8x depending on backend (4.1x MI300X Triton, 3.3x B300 CUDA, 5.6x H100 CUDA, 3.1x Arc SYCL/XMX).
[Watch the 6.8-minute explainer](https://github.com/Bobliuuu/all-quant/blob/main/video/ultraquant_explainer.mp4).
```

## How it works (5 bullets for the README)

1. Decode attention does about 1 FLOP per KV byte, far below the ~164 (MI300X) / ~274 (B300) FLOP/byte ridge from `results/*_peak.json`: it is memory-bound, so fewer KV bytes means faster tokens.
2. KV is stored as E2M1 FP4 with one UE8M0 (power-of-two, 1 byte) scale per 32 elements: 64 B values + 4 B scales = 68 B per head-token at D=128, 3.76x smaller than fp16.
3. Speculative decoding / MTP verify scores K candidate tokens per step; a naive kernel runs K independent queries and re-reads every KV page K times.
4. The fused kernel stacks the K tokens (times the grouped query heads) along the M dimension of one matmul, so each FP4 page is loaded and dequantised once and serves all K tokens.
5. Ports: Triton (AMD, NVIDIA), CUDA (5 NVIDIA GPUs, 0 failures on the reference vectors) and SYCL XMX/ESIMD (Intel Arc B390, 36/36 cases). Positioning: NVIDIA's FlashInfer trtllm-gen NVFP4 kernels already fuse K-query verify and are faster on B300; ours is the UltraQuant-format version across three vendors (no public CUDA/SYCL port of the format found; implementation contribution, not a new FP4 algorithm). MI300X (gfx942) dequantises FP4 in software; native scaled MFMA is on CDNA4 (gfx950).

## Scene to results-file map

| Scene | Uses |
|---|---|
| One op, three vendors | `mi300x_b8_c32k_tuned.json`, `b300_b8_c32k_tuned.json`, `cuda2_*_b8_c32k.json`, `sycl_arcb390_xmx_sweep.jsonl`, `sycl2_summary.json` (ESIMD callout; correctness line: `cuda/README.md`, `sycl/README.md`) |
| Memory-bound | `mi300x_peak.json`, `b300_peak.json` |
| FP4 + UE8M0 packing | arithmetic only (D=128: 64 B + 4 B scales) |
| K-query trick | schematic (K=4 shown) |
| Triton kernels | `mi300x_b8_c32k_tuned.json`, `b300_b8_c32k_tuned.json`, `*_tuning_notes.md` |
| Honest positioning (B300) | `priorart_B300.json`, `cuda2_b300_b8_c32k.json`, `b300_b8_c32k_tuned.json`, `cuda/PRIOR_ART.md` |
| Quality | `quality_summary.md` (Qwen3-8B), `nvfp4_summary.md` |
| Whole decode step | `e2e_summary.md` (MI300X), `cuda2_e2e_mistral7b_h100_b8_c32k.json` (H100); projected from GPU kernel time |
| Real-world proxy | `realworld_summary.md` (stock vLLM H100, bf16 vs fp8 KV; not FP4) |
| Caveats | the JSONs above and `mi300x_peak.json` |

h2h and real_kv summaries exist but are not used in the video.

Claims policy: only numbers present in results files; "no public port found", never "does not exist"; no "first FP4 KV" claim.
