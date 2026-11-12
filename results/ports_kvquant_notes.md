# KVQuant-4 (dense nuq4) fused paged-decode port on MI300X: notes

Sources: paper https://arxiv.org/abs/2401.18079 (Hooper et al., NeurIPS 2024); code https://github.com/SqueezeAILab/KVQuant. Code: `bench/ports/kvquant/` (`kernel.py` Triton split-KV kernel, `core.py` quantiser math, `port.py` harness adapter, `check_kvq.py`, `quality_kvq.py`, `finalize.py`). Raw: `results/ports_kvquant.json`, `results/kvquant_quality_{mistral7b,qwen3_8b}.json/.log`, `results/ports_kvquant_*.log`.
Shapes/protocol as `ports_spectralquant.json`: B8, ctx 32768, Hq64/Hkv8/D128, page 64 (paged block table), cold cache, K = 1/2/4/8 verify block.

## What is implemented
- K: **pre-RoPE**, per-channel static (zero-point, scale) per (kv head, channel) fitted offline (full min/max), then a per-layer 16-level **non-uniform codebook** (1-D k-means on the normalised values). The decode kernel dequantises (LUT gather x scale + zp) and **applies RoPE in-kernel** (cos/sin table, rotate-half). Packing: byte j = channel j (low nibble) | channel j+64 (high nibble), so one byte is exactly one RoPE pair.
- V: per-token online (fp16 scale, zp over 128 channels), per-layer nuq4 codebook.
- Bytes/vec: K 64, V 68 (4 B per-token sc+zp), average **66** (3.9x vs fp16). Static per-channel tables and codebooks are not counted (per layer/head, token-independent).
- Harness convention: `write()` receives post-RoPE K (the harness's fp16 truth); the port un-rotates in fp32, quantises, kernel re-rotates.

## Outliers / sink: NOT in the kernel
Dense-and-sparse outliers are **not included** in the ported path (nor the first-token-fp16 attention-sink trick, nor Fisher-weighted k-means: there are no gradients here, codebooks use uniform weights). Because the range is full min/max, no values are clipped, so the dense path is self-consistent. The paper's 1%-outlier setting is therefore not reproduced; real KVQuant-4 (1%) numbers are better than the dense-only numbers below. The quality script has an `--outlier PCT` / `--sink N` diagnostic (fake-quant, not run for lack of GPU time on the shared box).

## Results (see JSON)
Correctness (own dequantised cache, fp32 reference incl. RoPE): rel err 0.0087 / 0.0053 at K=1/2 (bf16 P/V rounding level); K=4/8: see JSON (`rel_err_vs_own_dequant`) if present.
Speed (cold, ms): K=1 0.822 (337 GB/s), K=2 0.780 (355 GB/s); K=4/8: see JSON if present (Triton compile of the larger BLOCK_M variants took >15 min each on the box; a first attempt was killed by me at the 15 min mark).
Quality (WikiText-2, qeval ppl protocol; attention cosine on real KV): Mistral-7B fp16 4.872 -> 5.093 (**+0.220**), attn cos 0.811; Qwen3-8B 8.522 -> 9.585 (**+1.064**), attn cos 0.864. Per-layer k_rel 0.067, v_rel 0.10. The low attention cosine (as for SpectralQuant) is likely driven by the attention-sink token (unverified here; the SpectralQuant diagnostic showed this for that method); the paper's fp16 sink token and outliers are likely why this dense-only result is worse than published KVQuant-4 numbers (roughly +0.02 ppl on LLaMA-7B with 1% outliers + sink).
Synthetic-KV attention cosine (bench) 0.949/0.958 at K=1/2 (synthetic has 4 outlier channels at 4x; calibration on 4096 tokens clamps the rest of the 32k).

## Deviations
1. No sparse outliers, no sink token, no Fisher-weighted codebooks (see above). 2. Codebook is pooled over heads per layer (paper: per layer too); calibration 16x2048 WikiText-2 train tokens (not the paper's exact set). 3. Write path (quantise-on-write) is torch (bucketize), not Triton; no write timing. 4. LUT dequant is a global-memory gather (L1-resident, 16 entries); needs `UQ_STAGES=1` (LDS limit) and compiles slowly on this Triton/ROCm. 5. cos/sin read from an fp32 table (no in-kernel sincos). 6. K-query verify handled natively (BLOCK_M = K x 8 rows) as other ports. 7. nsplit tuned from {16,32} per K.

## Effort
~330 LOC (kernel 85, core 80, port 70, check 25, quality 125 incl. attention-error copy). About 2.5 h wall, mostly queueing on the shared GPU lock and Triton compile time.
