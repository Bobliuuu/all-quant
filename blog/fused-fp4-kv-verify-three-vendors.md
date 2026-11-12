# One FP4 KV-cache verify kernel, three GPU vendors

*UltraQuant (`uqmt`): what we built, what we measured, and what is actually new.*

Speculative decoding asks the target model to check K draft tokens in one step. In a naive serving stack that check reads the whole KV cache K times, once per drafted token. At long context the KV cache is the dominant memory traffic, so verification cost grows almost linearly with K, which cancels much of the point of speculating.

This repo implements the fix for a specific 4-bit cache format: **one pass over the KV pages serves all K speculative queries and the whole GQA group**. We ran the same operation on AMD, NVIDIA and Intel GPUs, with the same cache layout and the same reference vectors.

## The format

UltraQuant stores KV as FP4 (E2M1) values with one UE8M0 power-of-two scale per 32 channels. At head_dim 128 that is 68 bytes per head per token for K or V, which is **3.76x smaller than fp16 and 1.88x smaller than fp8**. That is the capacity win: the same GPU holds 3.76x more KV tokens.

## The operation

```
 plain decode of K speculative tokens          fused K-query (this repo)
 q0 -> [read all KV pages] -> o0               q0 \
 q1 -> [read all KV pages] -> o1                q1  |  one tile: M = K x GQA group
 ...                                            ... |-> [read each KV page ONCE] -> o0..oK-1
 qK-1 -> [read all KV pages] -> oK-1            qK-1/    (FP4 -> bf16 in registers, per-query
 HBM traffic: K x KV                                      causal mask, split-KV + LSE reduce)
                                                HBM traffic: 1 x KV
```

Query j of K sees `seq_len - K + j + 1` positions. Each tile stacks the K queries and the GQA group into the M dimension of one matrix multiply, FP4 is expanded to bf16 in registers, and split-KV partial results are merged with a log-sum-exp reduction. Ragged lengths, empty sequences, head_dim 64/128, GQA ratios 1 to 16 and pages 16 to 128 are all covered by tests.

## What we built

| Vendor | Implementation | Hardware measured |
|---|---|---|
| AMD | Triton kernel (`uqmt/attn.py`) | MI300X |
| NVIDIA | Native CUDA with `mma.sync`, plus sm90 and sm100 (tcgen05) variants | A100, H100, L40S, B300, RTX PRO 6000 |
| Intel | SYCL with XMX `joint_matrix`, plus an ESIMD/DPAS path | Arc B390 |

All three read the same paged cache and are checked against the same CPU reference vectors. The layout is byte-compatible (a pure permute) with the upstream vLLM/SGLang UltraQuant cache. We also added an NVFP4 scale path (E4M3 scale per 16 elements plus a per-layer fp32 tensor scale) that runs identically on AMD and NVIDIA.

## What we measured

One attention layer, batch 8, context 32768, Hq64/Hkv8/D128, page 64, synthetic KV:

| | MI300X (Triton) | B300 (CUDA) | H100 (CUDA) |
|---|---|---|---|
| K=1 decode | 0.36 ms | 0.139 ms | 0.163 ms |
| K=8 verify, fused | 0.54 ms | 0.307 ms | 0.215 ms |
| K=8 fused vs K independent launches | 4.1x | 3.3x | 5.6x |

- **Fusion works as designed.** Fused K=8 costs about 1.5x a K=1 call on MI300X, not 8x. K=1 gains nothing, as expected.
- **Real shapes.** On Mistral-7B shapes on MI300X the fused kernel is 1.0x to 1.75x faster per call than the best fp16 attention while reading 3.76x fewer bytes.
- **Whole decode step (MI300X, Mistral-7B, projected from GPU kernel time).** 1.3x to 1.6x at batch 8 with 16k to 32k context and K=8, about 1.0x at batch 1. Attention is only 4-17% of the step at batch 1 and 27-57% at batch 8, so this is bounded by Amdahl's law.
- **Intel.** On Arc B390 (B8, 16k) the ESIMD/DPAS path takes 1.72 / 1.77 / 3.47 ms at K=1/4/8 versus 4.90 / 5.87 / 5.81 ms for torch-XPU f16 SDPA, while reading 3.8x fewer bytes; it is 1.4-2.2x faster than our earlier XMX path. llama.cpp's integer q4_0 KV is competitive at K=1 (2.03 ms) and slower at K>=4. FP4 is decoded in software, since XMX has no fp4; the kernel reaches about 75% of the GPU's measured random-data read ceiling at K=1.
- **Correctness.** CUDA passes all 15 reference vectors in every launchable configuration on all five NVIDIA GPUs. SYCL passes 36/36 vectors per path on Arc B390, and on CPU, A100 and H100. Triton passes 50 pytest cases on MI300X.
- **Accuracy.** Plain FP4/UE8M0 costs +0.14 perplexity on Mistral-7B and +0.56 on Qwen3-8B (wikitext-2). The NVFP4 scale path cuts the Qwen3-8B penalty to about +0.06 through the kernel, for 4 more bytes per vector. FP8 and KIVI-4 remain the accuracy leaders. Acceptance of speculative drafts drops under 4-bit KV, so quality needs to be validated per model.

Full tables and raw data are in the [README](../README.md) and `results/`.

## What we contribute, and what is not new

We are careful here, because most of the pieces already exist.

**Not new, and we do not claim it:**
- Sharing KV reads across several queries. FlashInfer multi-query decode, tree attention and other verify kernels already do it.
- A new FP4 algorithm. The format is UltraQuant's, from its paper and the AMD work around it.
- Fused FP4 multi-token verify on NVIDIA in general. FlashInfer's trtllm-gen kernels do it for NVFP4 on Blackwell, and **on B300 they are faster than our port** (K=8: 0.094 ms vs 0.307 ms). We do not beat vendor-tuned NVFP4 kernels.

**What we contribute, as far as we can find:**
1. **The UltraQuant format with fused K-query verify, ported to CUDA and SYCL.** The UltraQuant format and a vLLM backend for it already exist upstream (vLLM PR #57057, merged 2026-10-06): AMD gfx950 FlyDSL decode plus a Triton fallback, listed for ROCm only in the source we read, with single-query decode paths. We did not run it. In the public sources we checked we found no CUDA or SYCL implementation of the format with fused K-query verify, and no FP4 KV decode on Intel/SYCL. The prior-art surveys with their search trail are in [cuda/PRIOR_ART.md](../cuda/PRIOR_ART.md) and [sycl/PRIOR_ART.md](../sycl/PRIOR_ART.md). The wording is "no public port found", not "does not exist"; the search was not exhaustive.
2. **One validated op across three vendors.** Same layout, same vectors, same semantics (bf16 queries, any page size, ragged lengths, per-query causal masks), and a layout that drops into the upstream cache by permutation. Cross-vendor portability of a quantised-KV kernel is what the vendor-specific kernels do not offer.
3. **A measured comparison of the effort.** Porting this op to a new vendor took hours of agent time (CUDA about 2.5 hours, roughly 1.5 kLOC), against about two weeks of human work for TurboQuant on AMD. That comparison is not like-for-like: TurboQuant also needs rotations, codebooks and bit-plane packing, and our ports are one op without engine integration. Both end up ALU/latency-bound for the same reason: decoding sub-byte KV costs more issue slots than its bytes cost in memory time at decode sizes.
4. **A like-for-like quality comparison of KV quantisers** (UltraQuant, NVFP4, MXFP4, TurboQuant, PlanarQuant/IsoQuant/RotorQuant, KIVI, KVQuant, QServe-style int4, FP8) on two models through the same cache path.

## What this does not show yet

- No end-to-end serving throughput for FP4 KV. The step numbers above are kernel-level measurements and a projection.
- MI300X (gfx942) has no native scaled FP4 MFMA, so FP4 is dequantised in software. The native path needs gfx950 hardware, which we have not measured.
- At batch 1 there is no gain. The reliable benefit is **3.76x more KV tokens per GB**, plus cheaper verification at large batch and long context.
- Single runs on small samples for the quality tables.

## Try it

```bash
pip install git+https://github.com/Bobliuuu/ultraquant-more
```

```python
import torch, uqmt
cache = uqmt.build_paged_cache(k, v, page_size=64, seq_lens=[4096] * 8)
out = cache.decode(q)   # q: [B, K, Hq, D], K = 1 + speculative tokens
```

CUDA and SYCL builds: [cuda/README.md](../cuda/README.md), [sycl/README.md](../sycl/README.md). Explainer video: [video/ultraquant_explainer.mp4](../video/ultraquant_explainer.mp4).
