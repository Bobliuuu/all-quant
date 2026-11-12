# UltraQuant (uqmt): fused K-query FP4 KV-cache verify attention, on AMD, NVIDIA and Intel

<p align="center"><a href="video/ultraquant_explainer.mp4"><img src="video/poster.png" width="720" alt="UltraQuant explainer video"></a></p>

This repo implements one operation: **decode / speculative-verify attention over a 4-bit paged KV cache in the UltraQuant format** (FP4 e2m1 values, one UE8M0 power-of-two scale per 32 channels: 68 bytes per head per token for K or V at head_dim 128, **3.76x smaller than fp16, 1.88x smaller than fp8**), where **all K speculative queries and the GQA group share a single pass over the KV pages**.

The same op is ported to **AMD (Triton, MI300X)**, **NVIDIA (native CUDA, `mma.sync`; A100, H100, L40S, B300, RTX PRO 6000)** and **Intel (SYCL / XMX, Arc B390)**. All three consume the same cache layout, are checked against the same CPU reference vectors, and the layout is byte-compatible (a pure permute) with the upstream vLLM/SGLang UltraQuant cache ([cuda/UPSTREAM_COMPAT.md](cuda/UPSTREAM_COMPAT.md)). We found no public CUDA or SYCL port of the UltraQuant format ([cuda/PRIOR_ART.md](cuda/PRIOR_ART.md), [sycl/PRIOR_ART.md](sycl/PRIOR_ART.md)); K-query KV sharing itself is not new, and on B300 vendor NVFP4 kernels are faster (see "Where we stand").

**Headline numbers** (measured; one attention layer, batch 8, context 32768, Hq64/Hkv8/D128, page 64, synthetic KV):

| | MI300X (Triton) | B300 (CUDA) | H100 (CUDA) |
|---|---|---|---|
| K=1 decode | 0.36 ms | 0.140 ms | 0.161 ms |
| K=8 verify, fused | 0.54 ms | 0.350 ms | 0.431 ms |
| K=8 fused vs K independent launches | 4.1x | 2.9x (CUDA), 6.8x (Triton) | 2.7x |

On real Mistral-7B on MI300X the fused kernel is **1.0-1.75x faster than the best fp16 attention per call** (typically 1.3-1.6x) while reading 3.76x fewer bytes, and a whole decode step is **1.3-1.6x faster at batch 8 / 16k-32k context with K=8**, but **about 1.0x at batch 1**. The robust win is memory: **3.76x more KV tokens per GB**.

```
 plain decode of K speculative tokens          UltraQuant fused K-query
 q0 -> [read all KV pages] -> o0               q0 \
 q1 -> [read all KV pages] -> o1                q1  |  one tile: M = K x GQA group
 ...                                            ... |-> [read each KV page ONCE] -> o0..oK-1
 qK-1 -> [read all KV pages] -> oK-1            qK-1/    (FP4 -> bf16 in registers, per-query
 HBM traffic: K x KV                                      causal mask, split-KV + LSE reduce)
                                                HBM traffic: 1 x KV
```

## Install and use (Triton reference, `uqmt`)

```bash
pip install git+https://github.com/Bobliuuu/ultraquant-more   # or: pip install -e .
```

Importing needs only PyTorch (quantise / dequantise / reference / cache builder run on CPU); the kernel needs a GPU and Triton (ROCm 7 + Triton 3.9 and B300 tested). CUDA and SYCL builds: [cuda/README.md](cuda/README.md), [sycl/README.md](sycl/README.md).

```python
import torch, uqmt

k = torch.randn(8, 8, 4096, 128, device="cuda"); v = torch.randn_like(k)      # dense [B, Hkv, T, D]
cache = uqmt.build_paged_cache(k, v, page_size=64, seq_lens=[4096] * 8)      # paged FP4 + page table

q = torch.randn(8, 4, 64, 128, device="cuda", dtype=torch.bfloat16)          # [B, K, Hq, D], K = 1 + spec tokens
out = cache.decode(q)            # seq_lens includes the K new tokens; nsplit chosen automatically

ws = uqmt.UQWorkspace.create(8, 4, 64, 8, 128, nsplit=uqmt.auto_nsplit(8, 8, 64))   # CUDA-graph friendly
out = uqmt.uq_decode(q, cache.kc, cache.ks, cache.vc, cache.vs,
                     cache.block_table, cache.seq_lens, workspace=ws)
ref = uqmt.reference(q, cache.kc, cache.ks, cache.vc, cache.vs, cache.block_table, cache.seq_lens)
```

Public API: `uq_decode`, `quantize_fp4`, `dequantize_fp4`, `reference`, `build_paged_cache` / `PagedFP4Cache`, `UQWorkspace`, `auto_nsplit`, `uqmt.vllm_adapter` (varlen verify wrapper, cache-append hook), plus `uqmt.quant` (paper scale rule, Hadamard, NVFP4 quantiser variants), `uqmt.tree` (tree-masked verify for EAGLE/Medusa-style trees) and `uqmt.nvfp4` (NVFP4 scale path, below).

Layout (uint8): `kc, vc [pages, Hkv, PAGE, D/2]` (low nibble = even element), `ks, vs [pages, Hkv, PAGE, D/32]` (UE8M0, value 2^(s-127)). Query token j of K attends to `seq_len - K + j + 1` positions; rows with nothing visible return zeros.

**Supported:** head_dim 64/128; any GQA ratio (tested 1,2,4,7,8,16); K 1-16; page 16/32/64/128; ragged, empty and shorter-than-K sequences; int32/int64 block tables; bf16 queries. `nsplit` is auto-selected (about 4 programs per compute unit, rounded down to a power of two, clamped to the page count); override with `nsplit=`.

## Three vendors, one op

Single layer, B=8, ctx=32768, Hq64/Hkv8/D128, page 64, best tuned config per K, random data. Sources: `results/mi300x_b8_c32k_tuned.json`, `results/cuda_summary_b8_c32k.json`, `results/b300_b8_c32k_tuned.json`.

| GPU / backend | K=1 ms | K=2 | K=4 | K=8 | FP4 GB/s at K=1 (% of measured HBM copy) |
|---|---|---|---|---|---|
| **MI300X**, Triton (HBM copy 3.9 TB/s) | 0.361 | 0.373 | 0.453 | 0.542 | 789 (20%) |
| **B300**, CUDA (6.57 TB/s) | 0.140 | 0.142 | 0.253 | 0.350 | 2045 (31%) |
| B300, Triton | 0.308 | 0.310 | 0.425 | 0.494 | 925 |
| **H100**, CUDA (3.03 TB/s) | 0.161 | 0.163 | 0.291 | 0.431 | 1767 (58%) |
| **A100 80GB SXM**, CUDA (1.75 TB/s) | 0.259 | 0.268 | 0.458 | 0.709 | 1102 (63%) |
| **L40S**, CUDA | 0.390 | n/a here | n/a here | 0.523 | 731 |
| **RTX PRO 6000**, CUDA | 0.206 | 0.208 | 0.277 | 0.362 | 1388 (95%) |

Fused versus the same kernel run K times (CUDA, same GPU): 1.9-2.1x at K=2, 2.1x at K=4, 2.4-2.9x at K=8 (RTX PRO 6000 1.35x / 1.8x / 2.7x); MI300X Triton 1.83x / 2.59x / 4.10x; B300 Triton 1.79x / 2.82x / 6.77x. K=1 gains nothing from fusion, as expected. (L40S K=2/4 rows are in `results/cuda_summary_b8_c32k.json`; `copy_` underreports GDDR read bandwidth, so percentages there are not meaningful.)

**Intel Arc B390 (SYCL, XMX `joint_matrix`)**, Hq32/Hkv8/D128, ms per layer call (`results/sycl_arcb390_xmx_sweep.log`): B8 / 16k ctx K=1 2.48, K=4 3.19, K=8 6.30; B2 / 16k K=1 0.67, K=4 0.87. That is 1.4-2.4x faster than torch-XPU dense f16 SDPA at K=1-4 (5.01 ms at B8/16k/K=1) while reading 3.8x fewer bytes, at 5-13% of the part's read bandwidth; at K=8 it is not faster (0.7-1.0x of SDPA, which batches the 8 queries densely). FP4 is decoded in software; XMX has no fp4.

**Correctness:** CUDA passes the 15 CPU reference vectors in every launchable config on all five NVIDIA GPUs (1260-1856 launches each, `results/cuda_*_vectors_check.log`); error vs the fp32 reference at the benchmark shape is 1.1e-4 to 1.5e-4. SYCL passes 36/36 vectors per path on Arc B390, and also on CPU (OpenCL), A100 and H100. The Triton kernel passes 50 pytest cases on MI300X (ragged lengths, head_dim 64/128, GQA 1-16, K up to 16, pages 16-128, negative-zero codes, scale extremes, CUDA/HIP-graph replay); `tests/` run on CPU without a GPU, GPU tests skip cleanly.

## Where we stand

| Comparison | Result |
|---|---|
| MI300X, real Mistral-7B shapes: ours vs best fp16 attention (cold cache) | 1.0-1.75x faster per call (parity at B=32 / 4k / K=8); vs torch SDPA 3.3-18x at B<=8 (SDPA is a weak baseline there) |
| MI300X, K=8, B8 / 32k: ours vs K calls of amd-experiments' TQ3 kernel | 0.48 ms vs 8.5 ms (their kernel is K=1 only) |
| B300, K=1/2/4/8: ours CUDA | 0.140 / 0.142 / 0.253 / 0.350 ms |
| B300: **FlashInfer trtllm-gen NVFP4** (fp8 q, NVFP4 E4M3/16 scales, tcgen05) | **0.066 / 0.077 / 0.097 / 0.094 ms, 2.1x (K=1) to 3.7x (K=8) faster than our port**, and it already does fused K-query verify |
| B300: BitDecoding-FP4 (single-query; K independent decodes) | 0.068 / 0.131 / 0.261 / 0.521 ms |
| B300: FlashInfer trtllm-gen fp8 KV, fp8 q (2x the bytes) | 0.080 / 0.081 / 0.083 / 0.087 ms |
| Arc B390: ours vs torch-XPU f16 SDPA | 1.4-2.4x faster at K=1-4 (0.7-1.0x at K=8), 3.8x fewer bytes |

So: **we do not beat vendor-tuned NVFP4 kernels on B300**, and fused FP4 K-query verify exists there already (FlashInfer). What this repo offers is the exact UltraQuant format (UE8M0 per 32, bf16 queries, any page size, ragged lengths and causal verify masks), K-query sharing, and one validated implementation across AMD, NVIDIA and Intel. The CUDA port is 1.05-2.8x faster than our Triton kernel on every NVIDIA GPU measured.

## Accuracy: how the quantisers compare

Quantise-on-write / dequantise-on-read through a real HF cache path on Mistral-7B-v0.1 and Qwen3-8B (wikitext-2 perplexity; full table with C4, needle, multi-key, KV-retrieval and spec-acceptance columns in [results/quality_summary.md](results/quality_summary.md)). `official` = repo code, `official-port` = the amd-experiments kernels run unchanged, `simulated` = our quantise-dequantise reimplementation of the published scheme. Single run per cell, small samples.

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

Reading: FP8 and KIVI/KVQuant at 4-5 bits are the accuracy leaders; plain 68-byte FP4 costs 0.14 ppl on Mistral and 0.56 on Qwen3-8B, and **NVFP4 (+4 bytes per vector) cuts that penalty about 9x on Qwen3-8B (through the kernel: +0.06 vs +0.55 for MXFP4; 5x in the table's fake-quant column) and 2.7x on Mistral**, at ~1-2% more kernel time. The paper's constant c = 0.156 did not transfer to these two models; our scale rule equals the paper's `round(log2(c*amax))` at c = 0.236 ([docs/SCALE_RULE.md](docs/SCALE_RULE.md)). Speculative acceptance drops with 4-bit KV (Qwen3-8B accept@8 0.265 FP4 vs 0.985 fp16 in the quality harness; 0.46 NVFP4), which offsets part of the per-step gain, so quality and acceptance must be validated per model.

**Recommendation:** for new deployments where 72 vs 68 bytes does not matter, use the **NVFP4 scale path** (`uqmt.nvfp4`: E4M3 scale per 16 elements plus a per-layer fp32 tensor scale, integer-only scale decode, identical on AMD and NVIDIA). Keep UE8M0/32 when bit-compatibility with upstream UltraQuant caches is required.

## Real-model whole-step results (MI300X, Mistral-7B-v0.1)

Hand-rolled decode loop over the HF modules so only the attention backend changes; real prefill KV; GPU kernel time summed from the profiler ("graph-mode" projection, because the eager loop is launch-bound). Full tables: [results/e2e_summary.md](results/e2e_summary.md), kernel head-to-head [results/h2h_summary.md](results/h2h_summary.md), real-KV accuracy [results/real_kv_summary.md](results/real_kv_summary.md), overview [results/OVERNIGHT_REPORT.md](results/OVERNIGHT_REPORT.md).

Whole-step speedup of fused FP4 vs the best fp16 attention (higher is better):

| B, ctx | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 1, 8k | 0.97x | 1.10x | 1.10x | 1.13x |
| 1, 32k | 0.97x | 1.11x | 1.14x | 1.15x |
| 8, 8k | 1.08x | 1.21x | 1.24x | 1.29x |
| 8, 16k | 1.07x | 1.24x | 1.33x | 1.41x |
| 8, 32k | 1.13x | 1.29x | 1.46x | **1.60x** |

**Amdahl framing:** attention is 4-17% of the step at batch 1 and 27-57% at batch 8 (8k to 32k context), so the whole-step gain is bounded; even free attention would give only 2.4x at B=8 / 32k. At batch 1 expect no gain (0.96-1.15x). The value is (1) **3.76x KV capacity** (B=8 / 32k: 34.4 GB of fp16 KV becomes 9.1 GB), which is what lets a memory-bound server hold longer contexts or bigger batches, and (2) 1.3-1.6x on **large-batch, long-context speculative verify**. On a real 3-turn agent-like run at B=1 the lower draft acceptance under FP4 KV cancelled the per-step gain (projected 440 vs 460 tok/s). Attention-output cosine on real Mistral K/V is 0.975 for FP4/UE8M0, equal to TurboQuant-3 (0.975), with perplexity +3.0% vs +2.7% on 4k windows.

Real-world agent/RAG/needle harness: [bench/realworld/](bench/realworld/); H100 stock-vLLM bf16-vs-fp8 KV proxy results are in [results/realworld_summary.md](results/realworld_summary.md). No UltraQuant FP4 serving-level number is reported here.

## Porting effort

Porting the op to a new vendor took hours, versus about two weeks for TurboQuant-on-AMD: the CUDA port took about 2.5 hours of agent time (~1.5 kLOC) and SYCL a similar overnight session, against 15 commits over two weeks, a HIP library, a 1,834-line Triton file and custom HSACO/MFMA loaders for amd-experiments ([cuda/PORTING_LOG.md](cuda/PORTING_LOG.md), [sycl/PORTING_LOG.md](sycl/PORTING_LOG.md)). Not like-for-like: TurboQuant also needs rotations, Lloyd-Max codebooks and bit-plane packing, and the UltraQuant ports are one op without engine integration. Both end up ALU/latency-bound for the same reason: decoding sub-byte KV costs more issue slots than its bytes cost in memory time at decode sizes.

## Integration



## Related work and further reading

[docs/RELATED_WORK.md](docs/RELATED_WORK.md) (UltraQuant paper and AMD blog, KIVI, KVQuant, QServe, TurboQuant, NVFP4/MXFP4, FlashInfer, with verification status), [docs/methods.json](docs/methods.json), [docs/USE_CASES.md](docs/USE_CASES.md), [docs/GAPS.md](docs/GAPS.md), [docs/EXPLAINER.md](docs/EXPLAINER.md). The AMD blog names a fused K-query UltraQuant kernel as an open opportunity; this repo supplies one.

## Video

[3.7-minute explainer](video/ultraquant_explainer.mp4) (subtitles in [video/ultraquant_explainer.srt](video/ultraquant_explainer.srt)).

## Reproduce

```bash
pip install -e . pytest && pytest tests -q                  # CPU tests always; GPU tests need a GPU
cd uqmt && python bench.py --batch 8 --ctx 32768 --ks 1 2 4 8 --nsplit 16 --check   # Triton, fused vs K independent
UQ_WARPS=8 UQ_STAGES=3 python bench.py --batch 8 --ctx 32768 --ks 8                  # B300 K=8 config
python peak.py peak.json                                    # device ceilings
cmake -S cuda -B build && cmake --build build -j && ./build/test_vectors cuda/vectors 1e-2   # CUDA vs reference vectors
```

Tuning notes: `results/mi300x_tuning_notes.md`, `results/b300_tuning_notes.md`, `results/cuda_tuning_notes.md`, `results/sycl_tuning_notes.md`. Best Triton configs: MI300X `UQ_WARPS=2 UQ_STAGES=2 nsplit=16`; B300 K=1 `4/3/64`, K=8 `8/3/16`.

## Limitations

* **Software FP4 dequant on gfx942 and Intel Arc.** MI300X has no FP4 or scaled-FP4 MFMA and Arc's XMX has no fp4; e2m1 is decoded in registers. The kernel reaches only **4-26% of MI300X HBM peak** (20% at K=1 B8/32k), so most of the 3.76x byte saving is not converted into time. Native scaled MFMA needs gfx950 (MI350/MI355X); not measured here.
* **No native block-scaled FP4 MMA path on NVIDIA** (tcgen05 on B300; `mma.sync` block-scale on sm_120 studied but not fused). On B300 K=8 the CUDA kernel is at the legacy `mma.sync` bf16 ceiling. Vendor NVFP4 kernels are faster there.
* **Mostly synthetic and microbenchmark data.** Real-model numbers cover Mistral-7B-v0.1 (kernel/whole-step, MI300X) and Mistral-7B / Qwen3-8B (accuracy); perplexity and cosine come from single runs on small samples (8 to 16 windows); task accuracy (SWE-bench-style, RULER) is not measured. Many methods in the accuracy table are our own simulations of the published schemes.
* The whole-step results are a decode loop of our own over HF modules, not a served vLLM run; wall-clock steps are launch-bound, so tokens/s are projections from GPU kernel time. Draft acceptance was measured only at B=1 with a prompt-lookup drafter.
* SemKV and StarKV were not benchmarked (no public code found).
* K=1 gains nothing from fusion. The fp8-MFMA score path (`UQ_FP8K=1`) is experimental, head_dim 128 only, and slower on MI300X.

## What we tried

An fp8-MFMA K-score path and a scale-broadcast variant; neither beat the default bf16 path on MI300X.
