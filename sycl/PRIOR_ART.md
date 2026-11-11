# Prior art: low-bit KV-cache attention on Intel GPUs (checked 2026-11-11)

Method: web search plus primary-source reads of `main`/`master` of each repo (raw files, release notes). Nothing was run on hardware. "Not found" means a targeted search/read did not find it; it is not proof of absence. GitHub API rate limit was hit late, so Triton-XPU, Gaudi and SGLang quantized-KV coverage is thinner (flagged below).

## Headline: two things that make a naive "does not exist" claim FALSE

1. **llama.cpp SYCL has a native 4-bit KV decode kernel on Xe2 XMX (DPAS).** Merged 2026-10-29 ("[SYCL] add DPAS flash attention kernel for quantized KV cache (#31125)", commit f7c01c52). It takes q4_0 / q8_0 K and V, F32 queries with `Q->ne[1]` in 1..31 (so several query tokens share one KV read; GQA group up to 32), head sizes 64/96/128/256/512, on BMG/LNL/PTL/WCL. It is an integer 4-bit format with an f16 scale per 32 values (not FP4/MX), and it is a contiguous ggml KV cache (not paged).
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/fattn-dpas.cpp
2. **intel/sycl-tla has MXFP4 (e2m1 + UE8M0-style block scale) flash-attention PREFILL for Crescent Island ("cri", Xe3p).** Q and K are `mx_float4_t<float_e2m1_t>`, V is bf16. Its own source comments say "BlockScale does not support CachedKV/PagedKV", decode sets `BlockScale = false`, and the MXFP4 benchmark is `CachedKV=false, PagedKV=false`. So: prefill only, not a KV cache, not paged, not decode.
   https://github.com/intel/sycl-tla/blob/main/examples/06_bmg_flash_attention/06_xe_fmha_fwd_mxfp.cpp
   https://github.com/intel/sycl-tla/blob/main/benchmarks/flash_attention/benchmarks_prefill_mxfp4.cpp
   (first CRI code upstreamed in "Upstream CRI code (#833)", 2026-07-10)

Also: **upstream vLLM already contains the FP4 E2M1 + UE8M0 per-32-group 4-bit KV *format*** ("UltraQuant", `ultraquant_4bit`, PR #57057, 2026-10-06), paged, with a Triton fallback for decode and small-chunk continuation. It is ROCm-oriented (FlyDSL D=256 scaled-MFMA kernel on CDNA4). I found no XPU routing for it (see vLLM row). Any claim must be scoped to Intel GPUs and must not claim the format is novel.
   https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/ultraquant/format.py
   https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/ultraquant_attn.py

## Per-project findings

### Intel Extension for PyTorch (IPEX)
- Repo is **archived**; README: development discontinued after 2.8, project maintained only until end of March 2026, features upstreamed to PyTorch. https://github.com/intel/intel-extension-for-pytorch (README)
- Historic: vLLM-XPU FP8 KV cache was to use IPEX kernels (Intel blog, vLLM V1 era). https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Optimize-LLM-serving-with-vLLM-on-Intel-GPUs/post/1678793
- 4-bit/FP4 KV: not found. Status: none (archived).

### oneDNN SDPA (incl. Graph API)
- Float SDPA: f32/bf16/f16; GPU optimized path for f16/bf16 on XMX parts. https://github.com/uxlfoundation/oneDNN/blob/main/doc/graph_api/fusion_patterns/sdpa.md
- Quantized SDPA: u8/s8 (INT8) or f8_e4m3/f8_e5m2 (FP8) with scales/zero-points, static or dynamic. https://github.com/uxlfoundation/oneDNN/blob/main/doc/graph_api/fusion_patterns/sdpa_quantized.md
  (The doc has a per-device support table; I did not verify which rows are GPU-enabled.)
- 4-bit / FP4 / MX / paged KV in the documented pattern: not found. Status: shipping for f16/bf16; INT8/FP8 documented, GPU coverage unverified.
- llama.cpp SYCL uses oneDNN SDPA only for native F16 KV (needs >=32 query tokens); quantized KV is dequantized to F16 first (`fattn-onednn.cpp`).

### llama.cpp SYCL backend
- Flash attention is implemented in SYCL (not CPU fallback): `fattn.cpp` selects among VEC, DPAS, oneDNN, TILE, MKL kernels. https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-sycl
- VEC kernels are instantiated for all combinations of F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 K and V (`fattn.cpp`). q4_0/q8_0 K/V decode on Xe2 XMX has the dedicated DPAS kernel above (arch-gated: BMG g21/g31, LNL, PTL, WCL).
- Caveat: an earlier issue reported corrupted output / segfault for FA and q8_0 KV on an Xe2 iGPU via SYCL (https://github.com/ggml-org/llama.cpp/issues/19276, date/status not rechecked, likely predates the DPAS work).
- FP4/MXFP4 KV: not found (`ggml_type` list has MXFP4/NVFP4 as weight types; the SYCL FA kernels only list the legacy q-types). Paged: no (ggml contiguous KV; llama.cpp has no paged KV). Status: shipping (merged Oct 2026).
- Multi-token: DPAS kernel handles Q batches of 1..31 tokens with one pass over K/V; this is generic batching, not a speculative-verify-specific kernel, but it covers the K-query-per-KV-read shape for 4-bit INT KV.

### vLLM XPU backend
- `vllm/platforms/xpu.py` (main): `supported_quantization` includes fp8, mxfp4, mxfp8, awq, gptq (weights/activations). Attention selection routes `turboquant_*` KV dtypes to the TurboQuant backend (Triton-based, 3-4 bit MSE centroid keys / uniform values, not FP4/MX). https://github.com/vllm-project/vllm/blob/main/vllm/platforms/xpu.py
- FP8 KV on XPU: documented in Intel llm-scaler (`--kv-cache-dtype fp8_e4m3`, static scales) https://github.com/intel/llm-scaler/blob/main/vllm/README.md (section 3.5). Earlier vLLM versions gated XPU fp8 KV to the Triton backend (v0.10-0.11 docs via search; not re-verified on main).
- `ultraquant_4bit` is accepted by `config/cache.py` and `platforms/interface.py`, and has a Triton unified fallback, but `xpu.py` has no `ultraquant` reference, so I found no XPU attention routing for it. Whether it runs on XPU via the generic Triton path: **not verified**. Intel XPU batch-invariance mode rejects quantized KV (xpu.py).
- Status: FP8 KV shipping/experimental; TurboQuant on XPU experimental (unverified on hardware); FP4/MX KV on XPU: not found.

### OpenVINO / OpenVINO GenAI (GPU plugin)
- INT8 KV compression default on all GPU platforms since 2025.1; default switched from by-token to by-channel (keys) in 2025.3; 2026.2.1 notes say INT4 KV-cache compression is enabled for GPUs. Release notes: https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-1.html , https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-3.html , 2026.2.1 via https://docs.openvino.ai/nightly/about-openvino/release-notes-openvino.html (I did not open the 2026.2.1 page directly; relayed from search summary).
- OVMS doc: KV precision default u8, options u4/f16/f32, u4 example on GPU. https://docs.openvino.ai/2026/model-server/ovms_demo_long_context.html
- Property: `ov::hint::kv_cache_precision` (`KV_CACHE_PRECISION`). Formats are integer (u8/u4/i8) with per-token / per-channel quant modes; group scales detail not verified. FP4/MX KV: not found. PagedAttention is OpenVINO's LLM execution model (PagedAttentionExtension op exists); whether the INT4 path is fused for multi-token verify: not verified.
- Status: shipping (INT8 certain; INT4 per release notes).

### Intel Gaudi (HPU)
- FP8 KV cache via vLLM-Gaudi plugin: `--quantization inc --kv-cache-dtype fp8_inc` with calibrated per-tensor scales; not fully supported with torch.compile mode. https://docs.vllm.ai/projects/gaudi/en/latest/configuration/quantization/inc.html
- Not an Intel GPU (Gaudi accelerator). 4-bit KV / FP4 KV: not found. Status: shipping FP8.

### SYCL-TLA / cutlass-sycl (intel/sycl-tla, releases v0.5..v0.9.3, latest 2026-09-23)
- Flash attention for BMG (BF16 perf work in v0.8) and Xe3p/CRI; README lists Crescent Island as Xe3p target (`-DDPCPP_SYCL_TARGET=intel_gpu_cri`). https://github.com/intel/sycl-tla
- FP8: CRI decode benchmark with FP16 Q and E4M3 K/V (`benchmarks_decode_fp8kv_fp16mma.cpp`): `CachedKV=false, PagedKV=false`.
- MXFP4/MXFP8: prefill only, see headline item 2. Block-scaled e2m1 GEMM examples 50/51 (`xe35`). No FP4 KV-cache decode kernel found.
- Legacy decode kernels (`xe_flash_attn_decode*`) exist for BF16 with cachedKV/paged variants; their dtypes are not FP4.
- Status: shipping (BF16/FP8), experimental/pre-silicon (CRI MXFP4 prefill).

### SGLang XPU
- `xpu_backend.py` uses `sgl_kernel` `flash_attn_with_kvcache` (paged, page sizes 1/64/128) and has speculative target-verify metadata; but for quantized KV it only guards (DCP with quantized KV unsupported; checks fp16/bf16). https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/xpu_backend.py
- 4-bit/FP4 KV: not found. Whether FP8 KV works on XPU: not verified. Status: fp16/bf16 paged + spec verify shipping; low-bit KV none found.

### Intel llm-scaler (vLLM) and ipex-llm
- llm-scaler: weights `sym_int4`, `fp8`, `mxfp4` (gpt-oss only); KV `fp8_e4m3` documented. No 4-bit KV found. https://github.com/intel/llm-scaler/blob/main/vllm/README.md
- ipex-llm (archived-era): `--kv-cache-dtype fp8`, GQA models only (per search summary of its vLLM-Serving example). https://github.com/intel/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving
- Status: FP8 KV shipping; FP4 KV none found.

### Triton XPU (intel-xpu-backend-for-triton) and FlashInfer
- Triton-XPU: could not complete code search (rate limit). README mentions no fp4/e2m1/scaled_dot content in my grep. FP4 KV attention kernel there: not found (not conclusive). https://github.com/intel/intel-xpu-backend-for-triton
- FlashInfer: README GPU table is NVIDIA only (SM 7.5..12.1); FP4 listed is NVFP4/MXFP4 GEMM for Blackwell. No XPU/SYCL backend found. https://github.com/flashinfer-ai/flashinfer

### Hardware: does Xe/Xe2/Xe3 XMX natively do FP4/MXFP4?
- Xe2 (Battlemage B580): XMX types reported as INT2/INT4/INT8/FP16/BF16/TF32; FP8/FP4 not listed. https://chipsandcheese.com/p/intels-battlemage-architecture , https://techpowerup.com/review/intel-arc-b580-battlemage-unboxing-preview/2.html . (Secondary sources; absence of FP8/FP4 in a list, not an Intel statement. Intel's own optimization guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html). So on Xe2 FP4/MX must be dequantized in software (as llama.cpp's DPAS kernel does for q4_0).
- Xe3p / Crescent Island: Hot Chips 2026 coverage says supported formats range from FP4 / MXFP4 (microscaling) up to FP64; 32 Xe3p cores, 256 XMX engines, up to 480 GB LPDDR5X; customer sampling targeted 2H 2026. https://www.tomshardware.com/pc-components/gpus/hot-chips-2026-intel-dives-deep-on-crescent-island-ai-accelerator-larger-caches-and-deeper-xmx-engines-target-maximum-ai-flops-per-watt , https://hothardware.com/news/intel-crescent-island-hot-chips-2026 . Intel slide PDF (https://cdrdv2-public.intel.com/926893/Crescent%20Island%20GPU%20Hot%20Chips%202026.pdf) could not be text-extracted here. A secondary table lists only FP4 and FP8 without MX; "native XMX execution of MXFP4" is plausible (sycl-tla has block-scaled e2m1 GEMM for xe35) but I did not find an Intel statement of rates. Mark as: supported per Intel/press, throughput unverified. Crescent Island is not generally available.

## Prior art summary

Intel GPU backend? = does the project run on Intel GPUs (Xe/Arc/Max/CRI). "Fused K-query" = a kernel that handles multiple query tokens against one KV read in the low-bit path.

| Project | Intel GPU backend? | KV lowest bit-width | FP4/MX KV? | Paged? | K-query verify fused? | Link |
|---|---|---|---|---|---|---|
| IPEX | Yes (archived) | fp8 (historic); no 4-bit found | No | Yes (fp16) | n/a | https://github.com/intel/intel-extension-for-pytorch |
| oneDNN SDPA | Yes | 8 (u8/s8, f8 documented) | No | No (not found) | Generic multi-query, no low-bit<8 | https://github.com/uxlfoundation/oneDNN/blob/main/doc/graph_api/fusion_patterns/sdpa_quantized.md |
| llama.cpp SYCL | Yes | 4 (q4_0, INT, f16 scale/32) | No (not found) | No | Yes for q4_0/q8_0 (Q<=31, Xe2 DPAS) | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/fattn-dpas.cpp |
| vLLM XPU | Yes | 8 (fp8); TurboQuant 3-4 bit experimental | No (FP4 KV not routed on XPU; unverified) | Yes | Not found for low-bit | https://github.com/vllm-project/vllm/blob/main/vllm/platforms/xpu.py |
| vLLM UltraQuant (ROCm) | No XPU routing found (Triton fallback unverified) | 4 (FP4 E2M1 + UE8M0 /32) | Yes, but not on Intel as far as found | Yes | Triton small-chunk continuation (not Intel) | https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/ultraquant_attn.py |
| OpenVINO / GenAI | Yes | 4 (u4/int4, per OV 2026.2.1 notes) | No (not found) | Yes (OV paged attention) | Not verified | https://docs.openvino.ai/2026/model-server/ovms_demo_long_context.html |
| Intel Gaudi | No (HPU) | 8 (fp8_inc) | No | Yes | n/a | https://docs.vllm.ai/projects/gaudi/en/latest/configuration/quantization/inc.html |
| SYCL-TLA / cutlass-sycl | Yes (BMG, CRI) | 4 for Q,K in CRI prefill (MXFP4); 8 for KV decode | Yes, prefill only, V bf16 | No (explicitly unsupported with block scale) | No (decode has BlockScale=false) | https://github.com/intel/sycl-tla/blob/main/examples/06_bmg_flash_attention/06_xe_fmha_fwd_mxfp.cpp |
| SGLang XPU | Yes | 16 (only fp16/bf16 verified) | No | Yes | fp16 spec verify only | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/xpu_backend.py |
| llm-scaler / ipex-llm | Yes | 8 (fp8_e4m3 / fp8) | No (MXFP4 weights only) | Yes (vLLM) | n/a | https://github.com/intel/llm-scaler/blob/main/vllm/README.md |
| Triton-XPU | Yes | not found (UNVERIFIED) | not found (UNVERIFIED) | n/a | n/a | https://github.com/intel/intel-xpu-backend-for-triton |
| FlashInfer | No (NVIDIA only) | 4 (NVFP4/FP8 on Blackwell) | Yes, not Intel | Yes | Yes, not Intel | https://github.com/flashinfer-ai/flashinfer |

## Exact claim that is safe to make

"As of 2026-11-11, in the public sources we checked (llama.cpp SYCL, oneDNN, OpenVINO, vLLM XPU, SGLang XPU, Intel llm-scaler/IPEX-LLM, SYCL-TLA), we found no open implementation of a paged KV-cache decode-attention kernel for Intel GPUs that stores K and V as FP4 E2M1 with UE8M0 block scales and verifies K speculative query tokens against a single read of the quantized KV pages. Closest existing work: llama.cpp's SYCL DPAS decode kernel (merged 2026-10-29) reads q4_0/q8_0 integer-4-bit KV for up to 31 query tokens on Xe2, but is not FP4/MX and not paged; Intel's SYCL-TLA has MXFP4 (e2m1 + block scale) flash-attention prefill for Crescent Island, with V in bf16 and explicitly no cached/paged KV; vLLM main has an FP4-E2M1/UE8M0 per-32 KV format (UltraQuant) for ROCm that we found no Intel-GPU routing for; OpenVINO GPU supports INT4 KV compression (integer, not FP4/MX)." Avoid "first FP4 KV cache", "first 4-bit KV on Intel GPUs", or "first multi-token 4-bit decode on Intel": all are false or not provable. Unverified items that could weaken even the scoped claim: whether vLLM's UltraQuant Triton fallback actually runs on XPU, Triton-XPU/SGLang private or unreleased branches, OpenVINO's INT4 GPU kernel internals, and any Intel-internal CRI work beyond the public SYCL-TLA tree.
