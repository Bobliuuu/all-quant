# UltraQuant: detailed results and tables

The full tables behind the [README](../README.md), moved here so the README stays scannable. Headline numbers, charts and the main tables in the README are generated from `results/` by `python3 ~/video-readme-maker/vrm.py all README.spec.yml` (see `tools/readme_data.py`); the tables below are the earlier hand-written detail and each names its source file. Contents: three-vendor tables and correctness, where we stand against other kernels, accuracy of the quantisers, real-model whole-step results, serving-level evaluation, porting effort.

## Three vendors, one op

Single layer, B=8, ctx=32768, Hq64/Hkv8/D128, page 64, best tuned config per K, random data. Sources: `results/mi300x_b8_c32k_tuned.json`, `results/cuda_summary_b8_c32k.json`, `results/b300_b8_c32k_tuned.json`.

| GPU / backend | K=1 ms | K=2 | K=4 | K=8 | FP4 GB/s at K=1 (% of measured HBM copy) |
|---|---|---|---|---|---|
| **MI300X**, Triton (HBM copy 3.9 TB/s) | 0.361 | 0.373 | 0.453 | 0.542 | 789 (20%) |
| **B300**, CUDA (6.57 TB/s) | 0.139 | 0.142 | 0.255 | 0.307 | 2055 (31%) |
| B300, Triton | 0.308 | 0.310 | 0.425 | 0.494 | 925 |
| **H100**, CUDA (3.03 TB/s) | 0.163 | 0.164 | 0.215 | 0.215 | 1747 (58%) |
| **A100 80GB SXM**, CUDA (1.75 TB/s) | 0.256 | 0.266 | 0.465 | 0.713 | 1113 (64%) |
| **L40S**, CUDA | 0.391 | 0.395 | 0.431 | 0.494 | 729 |
| **RTX PRO 6000**, CUDA | 0.208 | 0.210 | 0.288 | 0.381 | 1375 (94%) |

Fused versus the same kernel with KV sharing off (one launch, KV read K times; CUDA, same GPU), K=2 / 4 / 8: H100 2.0x / 2.9x / 5.6x, B300 2.05x / 2.1x / 3.3x, A100 2.0x / 2.1x / 2.6x, RTX PRO 6000 1.45x / 1.9x / 2.7x, L40S 1.1x / 1.55x / 2.5x; MI300X Triton 1.83x / 2.59x / 4.10x; B300 Triton 1.79x / 2.82x / 6.77x. K=1 gains nothing from fusion, as expected. The H100 K>=4 rows use the sm_90a wgmma variant, the B300 K=8 row the tcgen05 variant (1.14x over the `mma.sync` kernel; the rest of the B300 rows are unchanged `mma.sync` kernels). CUDA numbers are `results/cuda2_summary.json`; `copy_` underreports GDDR read bandwidth, so L40S percentages are not meaningful.

**Intel Arc B390 (SYCL; ESIMD/DPAS path, XMX `joint_matrix` path)**, Hq32/Hkv8/D128, B8, 16k context, ms per layer call (`results/sycl2_summary.json`, baselines `results/sycl2_baselines_*.json`):

| K | UltraQuant ESIMD | UltraQuant XMX (earlier) | torch-XPU f16 SDPA | IPEX paged f16 | llama.cpp q4_0 KV |
|---|---|---|---|---|---|
| 1 | **1.72** | 2.62 | 4.90 | 4.99 | 2.03 |
| 4 | **1.77** | 3.49 | 5.87 | 10.03 | 2.29 |
| 8 | **3.47** | 7.24 | 5.81 | 10.06 | 3.88 |

The ESIMD path is 1.4x to 2.2x faster than the earlier XMX path. Against the baselines on this part: ahead of torch-XPU and IPEX f16 attention while reading 3.8x fewer bytes, roughly level with llama.cpp's integer q4_0 KV at K=1 (2.03 vs 1.72 ms), and ahead of it at K>=4 (llama.cpp is non-paged integer 4-bit, not the UltraQuant format). The measured random-data read ceiling of this GPU is about 105-113 GB/s (uniform data reads 2.5-4x faster because of compression and is not a valid peak): ESIMD reaches ~75% of it at K=1 and K=4 and ~38% at K=8; the earlier XMX path ~50% at K=1. FP4 is decoded in software; XMX has no fp4.

**Correctness:** CUDA passes the 15 CPU reference vectors in every launchable config on all five NVIDIA GPUs (1260-1856 launches each, `results/cuda_*_vectors_check.log`); error vs the fp32 reference at the benchmark shape is 1.1e-4 to 1.5e-4. SYCL passes 36/36 vectors per path on Arc B390 (UE8M0; NVFP4 10/10), and also on CPU (OpenCL), A100 and H100. The CUDA NVFP4 path passes its own vectors (2016 launches on H100) and the cuda2 harness has 0 failures on all five GPUs (`results/cuda2_*_vectors*.log`). The Triton kernel passes 50 pytest cases on MI300X (ragged lengths, head_dim 64/128, GQA 1-16, K up to 16, pages 16-128, negative-zero codes, scale extremes, CUDA/HIP-graph replay); `tests/` run on CPU without a GPU, GPU tests skip cleanly.

## Where we stand

| Comparison | Result |
|---|---|
| MI300X, real Mistral-7B shapes: ours vs best fp16 attention (cold cache) | 1.0-1.75x faster per call (parity at B=32 / 4k / K=8); vs torch SDPA 3.3-18x at B<=8 (SDPA is a weak baseline there) |
| MI300X, K=8, B8 / 32k: ours vs K calls of amd-experiments' TQ3 kernel | 0.48 ms vs 8.5 ms (their kernel is K=1 only) |
| B300, K=1/2/4/8: ours CUDA | 0.139 / 0.142 / 0.255 / 0.307 ms |
| B300: **FlashInfer trtllm-gen NVFP4** (fp8 q, NVFP4 E4M3/16 scales, tcgen05) | **0.066 / 0.077 / 0.097 / 0.094 ms, 1.8x to 3.3x faster than our port (2.1x at K=1, 1.8x at K=2, 3.3x at K=8)**, and it already does fused K-query verify |
| B300: BitDecoding-FP4 (single-query; K independent decodes) | 0.068 / 0.131 / 0.261 / 0.521 ms |
| B300: FlashInfer trtllm-gen fp8 KV, fp8 q (2x the bytes) | 0.080 / 0.081 / 0.083 / 0.087 ms |
| vLLM upstream UltraQuant backend (PR #57057) | exists since 2026-10-06; ROCm-listed, single-query decode paths in the source we read; run by us at serving level on MI300X with Qwen3-8B (see "Serving-level evaluation"), no kernel-level timing comparison with ours |
| Arc B390: ours (ESIMD) vs torch-XPU f16 SDPA / IPEX / llama.cpp q4_0 | B8/16k K=1/4/8: 1.72 / 1.77 / 3.47 ms vs 4.90 / 5.87 / 5.81 / 4.99-10.1 / 2.03 / 2.29 / 3.88 (llama.cpp competitive at K=1, ours ahead at K>=4) |

So: **we do not beat vendor-tuned NVFP4 kernels on B300**, and fused FP4 K-query verify exists there already (FlashInfer). What this repo offers is the exact UltraQuant format (UE8M0 per 32, bf16 queries, any page size, ragged lengths and causal verify masks), K-query sharing, and one validated implementation across AMD, NVIDIA and Intel. The CUDA port is 1.1x to 2.8x faster than our Triton kernel on every NVIDIA GPU measured (H100 2.1-2.3x, B300 1.6-2.2x).

## Accuracy: how the quantisers compare

Quantise-on-write / dequantise-on-read through a real HF cache path on Mistral-7B-v0.1 and Qwen3-8B (wikitext-2 perplexity; full table with C4, needle, multi-key, KV-retrieval and spec-acceptance columns in [results/quality_summary.md](../results/quality_summary.md)). `official` = repo code, `official-port` = the amd-experiments kernels run unchanged, `simulated` = our quantise-dequantise reimplementation of the published scheme. Single run per cell, small samples.

**Speed and accuracy together, real fused AMD kernels for every method on the same MI300X shapes** (B8, ctx 32k, cold cache; FP8, NVFP4, MXFP4, KIVI-4, QServe-int4, Planar/Iso/Rotor-3, SpectralQuant, TurboQuant-3, UltraQuant): [one comparison table and a speed-vs-accuracy chart in the blog](../blog/fused-fp4-kv-verify-three-vendors.md#part-2-how-ultraquant-relates-to-other-schemes-and-whether-it-is-the-best), raw data [results/ports_summary.md](../results/ports_summary.md), porting notes [bench/ports/PORTING_LOG.md](../bench/ports/PORTING_LOG.md). UltraQuant is not the most accurate there (QServe-int4 at equal bytes, NVFP4 for +4 B, FP8 and KIVI-4); FP8 is faster at K=1 (0.29 vs 0.36 ms). It is faster and more accurate than TurboQuant-3 and SpectralQuant and far ahead of Planar/Iso/Rotor at 3 bits. The table below is the earlier quantise-dequantise accuracy sweep; some of its numbers differ slightly from the ported-kernel runs (for example NVFP4 +0.052 here against +0.056 there on Mistral).

| method | kind | B/vec | Mistral dppl | Qwen3-8B dppl | Qwen3 attn cos |
|---|---|---|---|---|---|
| fp16 | reference | 256 | 4.873 (ppl) | 8.522 (ppl) | 1.000 |
| fp8 e4m3 | simulated | 128 | +0.008 | +0.007 | 0.996 |
| KIVI-4 | simulated | 80 | +0.007 | +0.025 | 0.998 |
| KVQuant-4 | simulated | 85 | +0.041 | -0.007 | 0.982 |
| **NVFP4 (E4M3/16)** | simulated, kernel-checked | 72 | +0.052 | +0.103 | 0.980 |
| **UltraQuant FP4/UE8M0 (this repo)** | official | 68 | +0.139 | +0.562 | 0.932 |
| UltraQuant + paper scale rule (c=0.156) | simulated | 68 | +0.310 (no rotation) | +0.929 (no rotation) | 0.894 |
| UltraQuant-style + Hadamard on K | simulated | 68 | +0.035 | +0.521 | 0.964 |
| MXFP4 | simulated | 68 | +0.190 | +0.481 | 0.923 |
| TurboQuant 4-bit | official-port | 68 | +0.026 | +0.581 | 0.979 |
| TurboQuant 3-bit | official-port | 52 | +0.142 | +1.995 | 0.933 |
| Planar / Iso / Rotor 3-bit | official-port | 52 | +6.1 / +2.5 / n/a | +16.7 / +13.9 / +13.4 | 0.70-0.72 |

Reading: FP8 and KIVI/KVQuant at 4-5 bits are the accuracy leaders; plain 68-byte FP4 costs 0.14 ppl on Mistral and 0.56 on Qwen3-8B, and **NVFP4 (+4 bytes per vector) cuts that penalty about 9x on Qwen3-8B (through the kernel: +0.06 vs +0.55 for MXFP4; 5x in the table's fake-quant column) and 2.7x on Mistral**, at ~1-2% more kernel time. The paper's constant c = 0.156 did not transfer to these two models; our scale rule equals the paper's `round(log2(c*amax))` at c = 0.236 ([docs/SCALE_RULE.md](../docs/SCALE_RULE.md)). Speculative acceptance drops with 4-bit KV (Qwen3-8B accept@8 0.265 FP4 vs 0.985 fp16 in the quality harness; 0.46 NVFP4), which offsets part of the per-step gain, so quality and acceptance must be validated per model.

**Recommendation:** for new deployments where 72 vs 68 bytes does not matter, use the **NVFP4 scale path** (`uqmt.nvfp4`: E4M3 scale per 16 elements plus a per-layer fp32 tensor scale, integer-only scale decode, identical on AMD and NVIDIA). Keep UE8M0/32 when bit-compatibility with upstream UltraQuant caches is required.

## Real-model whole-step results (MI300X, Mistral-7B-v0.1)

Hand-rolled decode loop over the HF modules so only the attention backend changes; real prefill KV; GPU kernel time summed from the profiler ("graph-mode" projection, because the eager loop is launch-bound). Full tables: [results/e2e_summary.md](../results/e2e_summary.md), kernel head-to-head [results/h2h_summary.md](../results/h2h_summary.md), real-KV accuracy [results/real_kv_summary.md](../results/real_kv_summary.md), overview [results/OVERNIGHT_REPORT.md](../results/OVERNIGHT_REPORT.md).

Whole-step speedup of fused FP4 vs the best fp16 attention (higher is better):

| B, ctx | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 1, 8k | 0.97x | 1.10x | 1.10x | 1.13x |
| 1, 32k | 0.97x | 1.11x | 1.14x | 1.15x |
| 8, 8k | 1.08x | 1.21x | 1.24x | 1.29x |
| 8, 16k | 1.07x | 1.24x | 1.33x | 1.41x |
| 8, 32k | 1.13x | 1.29x | 1.46x | **1.60x** |

**Amdahl framing:** attention is 4-17% of the step at batch 1 and 27-57% at batch 8 (8k to 32k context), so the whole-step gain is bounded; even free attention would give only 2.4x at B=8 / 32k. At batch 1 expect no gain (0.96-1.15x). The value is (1) **3.76x KV capacity** (B=8 / 32k: 34.4 GB of fp16 KV becomes 9.1 GB), which is what lets a memory-bound server hold longer contexts or bigger batches, and (2) 1.3-1.6x on **large-batch, long-context speculative verify**. On a real 3-turn agent-like run at B=1 the lower draft acceptance under FP4 KV cancelled the per-step gain (projected 440 vs 460 tok/s). Attention-output cosine on real Mistral K/V is 0.975 for FP4/UE8M0, equal to TurboQuant-3 (0.975), with perplexity +3.0% vs +2.7% on 4k windows.

**Same step on H100 with the CUDA port** (Mistral-7B, B=8, ctx 32768, eager HF-module loop, CUDA-graph replay time; fp16 baseline = torch SDPA, the faster of the two fp16 formulations there; `results/cuda2_e2e_mistral7b_h100_b8_c32k.json`): K=1 14.4 ms vs 22.0 ms (**1.52x**), K=8 16.2 ms vs 34.8 ms (**2.15x**); K=8 with the same kernel and KV sharing off is 50.4 ms, so K-query sharing is worth 3.1x at the step level. (B300 ratios from the same harness are not quoted: the fp16 SDPA of that torch build is unusually slow and flatters us.)

### Serving-level evaluation: status

[bench/realworld/](../bench/realworld/) holds an agent / RAG / needle / speculative-decoding / greedy-drift harness for vLLM.

**MI300X, vLLM 0.33.0, Qwen3-8B, bf16 vs fp8 vs FP4 KV.** These runs use **vLLM's own native `ultraquant_4bit` backend** (its FlyDSL/Triton kernel), **not our uqmt kernel**, so this validates the KV-capacity story, not a speed claim for our kernel. Qwen3-8B only (Mistral-7B cannot use the backend because of its sliding window). The FP4 runs needed `--attention-backend TRITON_ATTN`, so attention backend and KV format are confounded in the speed numbers. Single runs on a GPU shared with other jobs, no CIs.

| metric | bf16 KV | fp8 KV | FP4 (`ultraquant_4bit`) KV |
|---|---|---|---|
| KV capacity, tokens (gmu 0.9) | 1.09M | 2.18M (2.00x) | 3.16M (2.89x) |
| RAG HotpotQA-in-16k F1, n=100 | 0.655 | 0.647 | 0.618 |
| Greedy drift vs bf16, median first divergent token (24 prompts x 512) | reference | 24 | 10 |
| Agent, 32 x 24k-token sessions, KV oversubscribed: out tok/s | 47.9 | 38.1 | 59.4 |
| Agent TTFT p50 (s) | 69 | 74 | 30 |
| Plain decode (n-gram spec K=0, conc 8) tok/s | 1052 | 1091 | 767 |

Capacity (2.89x, below 4x because vLLM keeps four layers at bf16) and the oversubscribed-agent gain (bf16 KV holds 185k tokens, FP4 544k) are real; plain decode is slower with FP4; the quality cost is real (F1 about 0.04 lower; greedy output diverges from bf16 after about 10 tokens, earlier than fp8). Speculative K=4/8 runs exist but are noisy (bf16 K=4 below bf16 K=0) and are not used here. Full table and caveats: [results/realworld_mi300x_summary.md](../results/realworld_mi300x_summary.md), raw `results/realworld_mi300x_*.json`.

Earlier proxy on Modal H100 (stock vLLM, Mistral-7B and Qwen3-8B, fp8 vs bf16 KV): [results/realworld_summary.md](../results/realworld_summary.md).

## Porting effort

Porting the op to a new vendor took hours, versus about two weeks for TurboQuant-on-AMD: the CUDA port took about 2.5 hours of agent time (~1.5 kLOC) and SYCL a similar overnight session, against 15 commits over two weeks, a HIP library, a 1,834-line Triton file and custom HSACO/MFMA loaders for amd-experiments ([cuda/PORTING_LOG.md](../cuda/PORTING_LOG.md), [sycl/PORTING_LOG.md](../sycl/PORTING_LOG.md)). Not like-for-like: TurboQuant also needs rotations, Lloyd-Max codebooks and bit-plane packing, and the UltraQuant ports are one op without engine integration. Both end up ALU/latency-bound for the same reason: decoding sub-byte KV costs more issue slots than its bytes cost in memory time at decode sizes.
