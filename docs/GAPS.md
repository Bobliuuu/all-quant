# Implementation gaps: this repo vs the UltraQuant paper and related techniques

Value = expected benefit to the project's claim ("fused K-query FP4 decode, where AMD says the opportunity is"). Effort in engineer-days (estimate). Evidence tags: **[measured]** = in results/ in this repo; **[paper]** = from arXiv 2606.20474 / AMD blogs (see RELATED_WORK.md); **[est]** = estimate.

## Ranking

| # | Gap | Value | Effort | Needs GPU type |
|---|---|---|---|---|
| 1 | Tree-structured spec-verify masks | High | 2-3 d | MI300X/B300 |
| 2 | Accuracy harness + scale rule (c=0.156) + K rotation | High | 3-5 d | any (CPU for MSE) + 1 GPU for tasks |
| 3 | vLLM backend integration (verify path) | High | 5-10 d | MI300X |
| 4 | Quantise-on-write / fused append kernel | Medium-high | 2-4 d | MI300X |
| 5 | Native scaled-MFMA path on gfx950 | High on MI355X, zero on MI300X | 5-10 d + hardware | MI355X |
| 6 | Per-layer mixed precision (boundary BF16) | Medium | 1-2 d | any |
| 7 | NVFP4 (E4M3, 16-group) variant | Medium | 2-3 d | B300 |
| 8 | SGLang backend | Medium | 5+ d | MI300X |
| 9 | Long-context accuracy benchmarks (RULER/NIAH/LongBench) | High but dependent on 2 | 3-5 d | 1 GPU |

## Gap details

### 1. Tree-structured spec-verify masks (value High, effort ~2-3 d)
Today `seq_len - K + j + 1` gives every query j a causal prefix; that covers chain MTP (the AMD blog's K=3, ~2.3 accepted) but not tree drafts (Medusa/EAGLE-3 trees, SpecInfer), where candidate j sees only its ancestors among the K tree tokens. Sketch: keep the shared-prefix loop (KV pages are read once for all K x G rows, so the cache part needs no mask: all queries see the full committed prefix of length `seq_len - K`). Add one extra argument `tree_mask` uint8/bit-packed `[B, K, K]` (or int32 bitmask per query row) and a tail step that handles only the last block containing the K spec tokens: load the bit for (j, key position - (seq_len-K)) and `tl.where` to -inf. This mirrors LongSpec's hybrid approach (prefix path unmasked, small masked kernel for spec tokens) and FlashInfer's custom-mask path (FlashInfer PR #2105 adds mask + q_seq_len to XQA decode). Tests: compare with torch reference for chain (must equal current causal) and for random trees; keep `fused=False` baseline reading the same mask. Note the spec tokens' own K/V in the cache are quantised in our layout, so the mask must also be applied to FP4-dequantised values (fine, same path). SGLang's UltraQuant backend rejects spec decoding today [paper/PR], so this is the unclaimed ground.

### 2. Accuracy: scale rule, rotation, harness (value High, effort ~3-5 d)
We measure kernel error vs a dequantised reference (~1e-4) but nothing about task accuracy, which the paper shows is sensitive to scale choice (c=0.156: +4.4 pp over FP8; c=1.0: -4.3 pp) [paper]. Sketch: (a) add `scale_mode` to `quantize_fp4`: current `ceil(log2(amax/6))` vs paper `round(log2(c*amax))` with `c` configurable and a clamp to 6; (b) add `hadamard(x, d)` helper (fast WHT, d power of 2) applied to K and Q outside the kernel (invariance makes it exact in fp32; q.k scores unchanged), leave V unrotated; (c) a CPU/GPU script that captures real K/V from a small HF model (Qwen3-8B or smaller) and reports MSE, attention-output relative error, and perplexity delta for {scale rule} x {rotation} x {first/last layers bf16}. This is cheap and answers whether our format reproduces the paper's claim before any serving work. Caveat: the paper's c was fit on MiniMax/Qwen keys and reported to transfer by head dim; verify on our model.

### 3. vLLM backend integration (value High, effort ~5-10 d)
The AMD blog says vLLM expands MTP verification into B*K single-token rows and lacks a fused K-query UltraQuant kernel; upstream UltraQuant is not yet in vLLM mainline [blog]. Sketch: implement an attention backend that (a) registers `kv_cache_dtype="uq_fp4"` with page layout matching ours (`[pages, Hkv, PAGE, D/2]` + scales, or adapt our kernel to vLLM's block layout), (b) in `forward`, detects `max_query_len > 1` with uniform K per sequence (spec-decode verify) and routes to `fused=True` with q reshaped to `[B, K, Hq, D]`; non-uniform (chunked prefill) goes to a prefill path (gap 4), (c) computes `seq_lens` including K spec tokens (our convention). Start as an out-of-tree backend plugin and test on a small GQA model with ngram or MTP drafting so no draft model is needed. Risk: vLLM's block table dtype/layout and the 1-token-per-row flattening; budget time for that.

### 4. Quantise-on-write kernel (value Medium-high, effort ~2-4 d)
`quantize_fp4` is torch-side: 6 elementwise passes with an `argmin` over an 8-entry grid and a temporary `[...,8]` tensor, unfit for the decode hot loop (K/V appended every step) and wasteful for prefill. Sketch: Triton kernel one program per (token, head, group-of-32): load 32 values, absmax via `tl.max`, compute exponent with `tl.log2`/`floor`, scale, round to e2m1 by threshold compare chain (thresholds 0.25,0.75,1.25,1.75,2.5,3.5,5.0, ties-to-even as in hardware) or the bit-trick inverse of our dequant, pack nibbles, scatter-write to the page via the slot mapping, optionally fusing the Hadamard (gap 2) in registers (SGLang's store kernel does fused rotation+quantise+scatter [PR]). Verify bit-exactness against `quantize_fp4`. Also removes the Python-level float32 materialisation, which is the memory blow-up for long prefills.

### 5. Native scaled MFMA on gfx950 (value High on MI355X, effort 5-10 d, hardware needed)
MI300X has no FP8xFP4 scaled MFMA, so `UQ_FP8K` emulation is slower than the bf16 path [measured: K=8 0.66 vs 0.54 ms]. On gfx950 `v_mfma_scale_f32_16x16x128_f8f6f4` removes BF16 materialisation and folds UE8M0 into the instruction [paper]. Triton exposes `tl.dot_scaled` (OCP-MX scaled dot) for supported targets; first try `tl.dot_scaled` with FP8 Q (e4m3), FP4 K (e2m1), UE8M0 scales `[M, K/32]`; our `[PAGE, D/32]` scale layout may need transposing/swizzling to what the lowering expects. Gate with an MI355X smoke test; do not touch default path. Cannot be benchmarked here without MI355X (not available to this repo as of this writing); until then claims about it must stay estimates. Also needs the PV side: P in fp8/bf16 against FP4 V with scales per 32 along the *sequence* axis of V in the contraction, which is a different grouping from today's per-head_dim groups (the paper's V uses "FP4/scale path"; details unverified).

### 6. Per-layer mixed precision (value Medium, effort ~1-2 d)
Paper keeps first and last two attention layers BF16 (n=2) and TurboQuant blog skips boundary layers for full-attention models; hybrid models (e.g., 23/92 layers with KV) skip nothing extra [paper]. Sketch: a `PerLayerKVConfig` (dict layer -> {bf16, fp4}) consumed by the cache allocator; dispatch kernel vs bf16 reference per layer. Without a model integration this is a config class plus tests; its real value shows with gap 3. LMCache needs a layout-aware connector for the mixed pool [blog].

### 7. NVFP4 vs MXFP4 variant (value Medium, effort ~2-3 d)
NVFP4: E2M1 codes, 16-element groups with FP8 E4M3 scales plus an FP32 per-tensor scale; MXFP4 (ours): 32-element groups, UE8M0 [NVIDIA/OCP specs; not re-fetched, unverified here]. Sketch: parameterise `BLOCK` and the scale codec (E4M3 scale needs a real fp8 dequant of the scale before multiply instead of `exp2`); smaller groups reduce MSE (0.5 b/elt more scale cost: 4.5 vs 4.25 bits) and may suit head_dim 64. B300 supports native FP4 so this is the natural CUDA-side study; kernel change is mostly in the scale path and tile layout. Test kernel error and MSE against MXFP4 on captured K/V (gap 2 harness).

### 8. SGLang backend (value Medium, effort 5+ d)
SGLang PR #41949 exists but rejects spec decoding, LMCache, hierarchical cache [PR]. Our fused kernel could be offered as the spec-decode path there; depends on gap 3 and 1 being done first.

### 9. Long-context accuracy benchmarks (value High, effort 3-5 d)
Paper: RULER on MiniMax-M2.5 fell 74.03 -> 67.08, NIAH/LongBench-E on Llama-3.1-8B were at parity, GPQA/GSM8K within ~2 pp, SWE-bench Lite -3 to -4 tasks [paper]. We have none. Sketch: once gap 2 gives us a quantise->dequant path, run NIAH and RULER subset on a small model with simulated KV quant in HF (no serving), at 4k-32k, for BF16 vs FP8 vs our FP4 variants; report with seeds. Keep it simulation-only and label it so.

## Smaller items
- `seq_lens` includes K spec tokens: needs documenting for the vLLM adapter (they pass committed length in some backends).
- Q is bf16; FP8-Q path exists but is off by default and slower on MI300X [measured].
- No MLA support (paper/PRs also exclude MLA).
- Only head_dim multiples of 32; 64/128/256 tested? (check tests/ before claiming; not audited here).
