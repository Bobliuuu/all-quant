# Comparison-method ports to AMD (Triton / ROCm, MI300X): porting log

Goal: speed AND accuracy of the comparison methods measured with real fused paged-decode kernels in the same harness as UltraQuant
(`uqmt/attn.py`) and the TurboQuant port (`amd-experiments`), instead of quantise-dequantise-only simulation.
Shape everywhere: B8, ctx 32768, Hq64/Hkv8/D128, page 64, K = 1/2/4/8 query tokens. Harness: `common.py` / `run_port.py` (cold-cache timing),
own-cache correctness `mp_run.py`, quality `mp_quality.py` (wikitext-2 ppl delta + attention cosine on real KV, Mistral-7B-v0.1 and Qwen3-8B, through
`bench/quality/qeval.py`). Shared helpers (`mp_common.py`): split-KV launch, combine kernel, nsplit tuner (8/16/32, picked per K).
Every port keeps one KV pass for all K*G query rows ("fused"); `--fused=False` style baselines (K independent calls) are in `ms_per_call_indep` where recorded.

LOC = kernel + port module (no comments counted separately). Hours = agent wall-clock estimate incl. debugging, excluding time queued on the shared GPU lock.

| method | source | kernel design | LOC | hours | deviations from the published scheme |
|---|---|---|---|---|---|
| FP8 e4m3 (`fp8/`) | FP8 formats https://arxiv.org/abs/2209.05433 ; vLLM fp8 KV https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html | 1 B/elem, per-tensor fp32 scale for K and V; k_scale folded into softmax scale, v_scale in the combine kernel; fp8->bf16 in registers, bf16 MFMA | 128 | 0.7 | e4m3fnuz (gfx942 native; max 240) instead of e4m3fn; scale = amax of the whole written tensor (vLLM: first batch); q not quantised, no FP8 MFMA |
| NVFP4 (`nvfp4/`) | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | existing `uqmt/nvfp4.py` kernel (E4M3 scale/16 decoded by integer bit placement, per-tensor fp32 scale folded) behind the shared harness | 62 (+278 reused) | 0.3 | none in the format; per-tensor scale = amax/(6*448) of the written tensor; q bf16 (no FP4 q) |
| MXFP4 (`mxfp4/`) | OCP MX v1.0 https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf , https://arxiv.org/abs/2310.10537 | the UltraQuant UE8M0/32 kernel, unmodified (same bytes) with the OCP quantiser: exponent = floor(log2 amax) - 2, RNE, saturating | 71 | 0.4 | none; no rotation. Speed is identical to UltraQuant BY CONSTRUCTION, only the quantiser differs |
| KIVI-4 (`kivi4/`) | https://arxiv.org/abs/2402.02750 , https://github.com/jy-yuan/KIVI | paged 4-bit K (per-channel, 32-token groups, fp16 scale/min) + V (per-token, 32-channel groups) with in-kernel dequant, plus fp16(bf16) residual window of 128..159 tokens read in the same pass by the last split program | 245 | 1.5 | channel-pair packing for K (KIVI packs along tokens: same bytes, same math); bf16 residual; static cache (flush rule reproduced through n_q, no streaming append); scale/min broadcast-loaded per page |
| QServe KV4 (`qserve_int4/`) | https://arxiv.org/abs/2405.04532 , https://github.com/mit-han-lab/omniserve | per-token-head asym int4 (fp16 s,z) with s,z folded through both dots (no per-element scale work); SmoothAttention lambda (alpha 0.5, RoPE-paired) multiplied onto q in-kernel | 174 | 1.0 | lambda from the written K (post-RoPE) not offline calibration of pre-RoPE activations; QServe's CUDA fp16 PRMT kernel replaced by Triton bf16; QServe has no multi-query design -> "straightforward K-query" (tile of K*G rows); KV only (no W4A8) |

Plumbing (harness adapters, tuner, quality hook, sync script): ~0.8 h.

Quality notes
* The quality rows use each port's own quantiser (`quality_methods()` in every `port.py`); for NVFP4/MXFP4/KIVI/QServe this is the same math the kernel reads
  (own-cache check `rel_err_vs_own_dequant`, ~2e-3 for FP8/FP4 formats, ~5e-3 for KIVI/QServe where bf16 rounding of p*scale and residual enters).
* FP8 / QServe lambda scales are calibrated online from the first K/V a layer sees (class-level state), a stand-in for offline calibration.
* ppl protocol = `bench/quality/qeval.py` ppl_run (8 windows x 2048, chunked prefill 1024 + token-by-token decode). fp16 baseline re-run on the same box/transformers version.
* The MI300X box is shared: all GPU work ran under `flock /root/gpu.lock`.
