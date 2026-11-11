# Use cases and back-of-envelope models

**Legend.** [measured] = in this repo's results/ (kernel microbenchmarks only: MI300X B=8, ctx 32768, Hq64/Hkv8/D128, page 64; results/mi300x_b8_c32k_tuned.json). [paper] = reported by AMD in arXiv 2606.20474 or the ROCm blogs (see RELATED_WORK.md; not reproduced here). [est] = arithmetic or assumption in this file. **Nothing in this repo has been measured end to end (no model, no serving stack, no task accuracy).**

## 1. KV bytes per token [est, exact arithmetic]
Bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bits/8. FP4+UE8M0/32 = 4.25 bits/elt (17 B per 32 values); FP8 = 8 bits (per-tensor scale ignored); fp16 = 16.

| Model (config from memory, check the HF config) | elts/token | fp16 | fp8 | FP4 (4.25 b) | FP4 vs fp8 |
|---|---|---|---|---|---|
| Qwen3-8B: 36 L, 8 KV heads, d128 | 73,728 | 144 KiB | 72 KiB | 38.2 KiB | 1.88x |
| Llama-3.1-70B: 80 L, 8 KV heads, d128 | 163,840 | 320 KiB | 160 KiB | 85.0 KiB | 1.88x |
| "DeepSeek-style GQA" (hypothetical 61 L, 8 KV heads, d128) | 124,928 | 244 KiB | 122 KiB | 64.8 KiB | 1.88x |

Real DeepSeek V2/V3 use MLA (compressed latent, ~576 values/token/layer from memory [unverified]); the FP4 kernel here does not cover MLA, and UltraQuant's SGLang backend rejects MLA too. Hybrid models (Qwen3.8: 23 of 92 layers with KV [paper]) shrink all rows proportionally and shrink the benefit too.

## 2. Concurrent sessions per device [est]
Assumptions: 90% of HBM usable, weights fixed, no activation/fragmentation overhead (optimistic), one session = one full context of KV, no prefix sharing. Weights: Qwen3-8B fp16 16 GB (4.5 GB 4-bit on the 16 GB Arc row); Llama-3.1-70B fp8 weights 70 GB in every KV column. Sessions (floor):

| Device | Model | ctx | fp16 KV | fp8 KV | FP4 KV |
|---|---|---|---|---|---|
| MI300X 192 GB | Qwen3-8B | 32K | 32 | 64 | 122 |
| MI300X 192 GB | Qwen3-8B | 128K | 8 | 16 | 30 |
| MI300X 192 GB | Llama-70B | 32K | 9 | 19 | 36 |
| MI300X 192 GB | Llama-70B | 128K | 2 | 4 | 9 |
| B300 288 GB | Qwen3-8B | 32K | 50 | 100 | 189 |
| B300 288 GB | Qwen3-8B | 128K | 12 | 25 | 47 |
| B300 288 GB | Llama-70B | 32K | 17 | 35 | 66 |
| B300 288 GB | Llama-70B | 128K | 4 | 8 | 16 |
| 16 GB Arc-class | Qwen3-8B (4-bit w) | 32K | 2 | 4 | 7 |
| 16 GB Arc-class | Qwen3-8B (4-bit w) | 128K | 0 | 1 | 1 |

Reading: FP4 is ~1.88x FP8 and ~3.8x fp16 in capacity at equal weights. Paper reports 3.2-3.4x BF16 logical capacity [paper] (consistent with 3.76x minus the BF16 boundary layers). Concurrency only turns into throughput when decode is memory-bound and the workload would otherwise evict; paper: no gain over FP8 at short context, ~1-3% loss [paper]. The Arc row is only an arithmetic illustration: we have no Arc numbers, and our kernels are Triton ROCm/CUDA-tested only (B300 and MI300X).

## 3. Multi-token (MTP / spec-decode) verify speedup from our measured kernel

Measured (B8 ctx32K, K = verified query tokens per sequence), attention kernel only:

| K | fused ms | K independent queries ms | fused speedup over independent | fused ms per query slot relative to K=1 fused |
|---|---|---|---|---|
| 1 | 0.361 | 0.353 | 0.98x | 1.00 |
| 2 | 0.373 | 0.684 | 1.83x | 0.97 |
| 4 | 0.453 | 1.174 | 2.59x | 0.80 |
| 8 | 0.542 | 2.220 | 4.10x | 0.75 |

Derived [est, from the same numbers]: attention time per *candidate slot* vs single-token decode (0.361 ms): fused is 1.9x / 3.2x / 5.3x cheaper at K=2/4/8, independent only 1.06x / 1.23x / 1.30x. So "independent" verification barely amortises the KV read, which is the situation AMD describes for its UltraQuant+MTP path ("treats these as independent queries") [paper/blog]. AMD's MTP verify scores 3 positions and commits ~2.3 tokens [blog]. Interpolating our K=2 and K=4 rows for K=3 (not measured): fused ~0.41 ms, independent ~0.93 ms per verify step; per *committed* token (divide by 2.3): fused ~0.18 ms (2.0x better than K=1 decode), independent ~0.40 ms (0.9x, i.e. slightly worse than plain decode). Upper-bound reading of why MTP regresses at high concurrency for independent verify [est, not tested]. Caveats: attention is only part of a decode step (MoE/GEMMs, comms dominate in big models, so end-to-end gain is a fraction of this); the K=8 chain acceptance is far lower than 8; non-chain tree masks are not supported yet (GAPS.md #1); batch 8 only; K>1 rows have GQA group 8 padded to 16 (M tile) so small K have free MMA rows.

## 4. Use cases ranked by who benefits and why

1. **Long-context multi-turn agents (coding agents, tool loops) on KV-capacity-bound serving.** Long shared prefixes, short outputs, many turns, prefix cache hit 91-95% in AMD's trace [blog]. The win is residency, not decode FLOPs: FP4 keeps prefixes resident when FP8 evicts (3.47x P50 TTFT in late rounds, 1.63x output throughput vs FP8 [paper]; Claude-Code-trace goodput 0.174 vs 0.140 req/s/GPU on MiniMax, 0.135 vs 0.067 on Qwen3-235B [paper]). Who benefits: operators of agent backends running near HBM limit. Our code contributes only the decode kernel, so this case is a [paper] claim for us.
2. **Coding agents specifically**: same as above, plus accuracy risk: SWE-bench Lite 62/62/59 (BF16/FP8/UltraQuant) on MiniMax and 28/21/24 on Qwen3-235B [paper]; the sample is 100 tasks, treat as noisy.
3. **Speculative decoding / MTP serving with big-batch verify.** The unclaimed niche: fused K-query FP4 verify. [measured] 1.8-4.1x kernel speedup at K=2-8 over independent queries; end-to-end unknown. Pays when batch x ctx is large enough that KV reads dominate; AMD saw MTP help up to C=16 and regress at C=32 [blog].
4. **Memory-capacity-bound serving of large models** (70B-class on 192 GB MI300X): FP4 turns 4 sessions at 128K into 9 (table above) [est]. MI300X lacks FP4 hardware so decode is dequant-bound [measured: ~30% of time is dequant ALU; K=1 reaches ~790 GB/s of 5.3 TB/s peak, notes say HBM copy peak measured 3.9 TB/s] and may not beat FP8 on speed; the benefit is capacity.
5. **RAG with large retrieved contexts**: long prefill per request, shorter reuse than agents; benefit mainly capacity; accuracy risk on retrieval (official RULER on MiniMax 74.03 -> 67.08, NIAH/LongBench-E on Llama-3.1-8B at parity [paper]).
6. **Prefix-cache offload (LMCache, DRAM/disk tiers)**: 4-bit means ~1.7x less data copied and 3.2x more tokens resident in HBM (TurboQuant 4-bit, not UltraQuant [blog]); the LMCache/UltraQuant combination is listed by AMD as future work, and SGLang's UltraQuant backend rejects LMCache/hierarchical cache [PR].
7. **Edge / Intel Arc / cost per token**: arithmetic only [est]: a 16 GB card has little room after weights (7 sessions at 32K in the table); no Intel kernel exists here and no data. Cost per token not modelled: needs a measured end-to-end tokens/s, which we lack; do not quote a $/token figure.

## 5. What would convert estimates to measurements
Quantise real K/V from a GQA model (GAPS #2), run attention-output error and perplexity/NIAH with simulated FP4; add the tree-mask kernel and bench K in {2,3,4,8} on B=1..64 and ctx 8K..128K on MI300X and B300; only then run an out-of-tree vLLM backend (GAPS #3) with ngram/MTP drafting for an end-to-end number.
