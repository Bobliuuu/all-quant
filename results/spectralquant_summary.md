# SpectralQuant on AMD MI300X: a faithful Triton port vs UltraQuant and our TurboQuant port

All numbers: MI300X, ROCm 7.1, Triton 3.9, B8, ctx 32768, Hq64/Hkv8/D128, K = 1/2/4/8 query tokens (causal verify block), **cold** timing (256 MB Infinity Cache flushed before every timed call, mean of 10 calls), same box and harness (`bench/ports/run_port.py`) for all methods. GB/s = bytes of KV actually stored / time (one pass).
Raw: `results/ports_spectralquant.json`, `results/spectralquant_bench*.log`, `results/spectralquant_tune.json`, `results/spectralquant_quality_*.json`. Code: `bench/spectralquant/` (`sq_triton.py` kernels, `test_sq.py`, `quality_sq.py`, `tune_sq.py`), adapter `bench/ports/spectralquant/port.py`.

## 1. What SpectralQuant is (and what is ambiguous)

There is no arXiv paper. It is a 2026 open-source project by Dynamis-Labs / Sentra, "SpectralQuant: KV cache compression via spectral rotation" (blog title "3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure", April 2026). Sources:
- repo: https://github.com/Dynamis-Labs/spectralquant (paper PDF in-repo `paper_output/spectralquant.pdf`, per the blog; arXiv link listed as "to come")
- blog: https://www.sentra.app/blog/3-percent-is-all-you-need
- PyPI package `spectralquant` 0.3.1 (MIT; homepage https://github.com/Dynamis-Labs/spectralquant_package) - **this is the code I read** (wheel source, pure PyTorch, no CUDA kernels: `engine.py`, `kernels/compress_keys.py`, `_water_fill.py`, `presets.py`, `calibrate.py`).
- Not to be confused with SpectrumQuant ("Resonating with RoPE", ACL 2026, DCT-based key quantiser, https://aclanthology.org/2026.acl-long.1732/) and VeloxQuant-MLX's third-party spectral module.
I could not fetch the repo's paper PDF or main-branch README (the pkb mirror page says the repo was not accessible in April 2026), so the scheme below is what the **shipped 0.3.1 package code** does, not independently validated against the PDF. The headline claims (5.95x, "identical perplexity 9.51", 4.5x faster attention, B200) come from the blog and are the authors' own.

Scheme (preset `standard`, package default engine): per (layer, kv head) eigendecomposition of the uncentred second-moment matrix of post-RoPE K and, separately, V from a short WikiText-2 calibration; rotate into the eigenbasis; `d_eff` = smallest prefix with 90% of the variance; signal dims get water-filled bits (greedy, gain lam_i/4^b_i, budget d_eff x 3 for K, x 4 for V, 1..5 / 1..6 bits) with Lloyd-Max codebooks for N(0, sigma_i); tail dims get 2 bits (unit Lloyd-Max codebook scaled by the per-dim sigma, floor 1/sqrt(D)); selective QJL residual sketch on the d_eff dims exists but is **off by default** (the package docstring says it hurts). No per-token scale: codebooks are global per head.

## 2. The port (what was built)

- `sq_triton.py`: calibration (host, torch), **quantise-on-write kernel** (rotation as fp32 `tl.dot`, binary-search nearest Lloyd-Max level with per-dim thresholds, bit-plane packing), **fused split-KV decode kernel** (bit-plane unpack + per-dim LUT dequant of K and V in the rotated basis, online softmax, causal K-query mask, GQA rows = G x K) + reduce kernel; q is rotated by Vrot_k once per call and the output by Vrot_v^T (both included in the timing).
- Storage: dims sorted by eigenvalue so bits/dim is non-increasing; plane0 = 2 low bits of every dim (32 B), planes 3..6 hold bit j-1 of the first N_j dims (N_j/8 B each). B8/ctx32k synthetic-LLM run: 38 B (K) + 59 B (V). On the real models the plane geometry is unified over layers: Mistral-7B 46 B (K) + 64 B (V) = 55 B/vec average (ideal, unpadded 48.0), Qwen3-8B 54.5 B (46.2 ideal) = **4.7x vs fp16 packed (5.3x Mistral / 5.5x Qwen3 ideal, unpadded)**, vs the claimed 5.95x (the claim ignores plane padding and per-head tables; mean d_eff at 90%: Mistral K 68 / V 94, Qwen3 K 44 / V 92 of 128).
- Correctness (`test_sq.py`): Triton quantise+pack+dequant equals a torch nearest-centroid reference except ties (3e-5 of elements differ by >1e-2); fused decode equals attention over the dequantised cache, cos 0.999998 (rel 1.9e-3, fp16 P/V rounding) at K = 1/2/4/8.

### Deviations (honest list)
1. Written from the shipped package, not the (unfetched) PDF; no GPU kernel exists upstream, so "Triton port" = our own kernels for the package's PyTorch semantics.
2. Exact Lloyd-Max codebooks (fp64 truncated-Gaussian means) instead of the package's scipy loop with sigma rounded to 2 decimals.
3. QJL not implemented (off by default upstream; r_norms/signs not stored). Bytes exclude them, as the package's real storage would.
4. Variable bits per dim stored as bit-planes (fixed layout per model), so there is padding overhead; K/V dequant is a global-memory LUT gather per element (no register-resident codebook).
5. Calibration: 16 x 2048 WikiText-2 *train* tokens per model (post-RoPE K/V, via a recording cache), evaluation on WikiText-2 test, same ppl protocol as `bench/quality/qeval.py` (chunked prefill 256, teacher-forced token decode, 8 windows x 2048). Attention cosine: same per-layer metric as the quality table (T=2048, 128 queries).
6. Speed bench uses synthetic anisotropic KV (`make_kv("llm")`, power-law spectrum, random per-head rotation, 4 outlier key channels), calibrated on 4096 tokens; real-KV quality is measured separately on the models.
7. No paging (contiguous [B,Hkv,S,*] planes) and fp16 PV/QK dots. The package's own fused attention is plain PyTorch matmul; there is no upstream speed number to reproduce on AMD.

Effort: ~440 lines (kernels+calibration 261, quality 109, test/tune/adapter 70), about 4 h wall including a long wait for the shared GPU lock, one shared-memory failure (num_stages), one launch-parameter sweep (1.7x speedup from nsplit/BN tuning: 2.01 -> 1.43 ms at K=1).

## 3. Speed (B8, ctx32768, Hq64, cold cache, ms per call; GB/s of stored KV)

| method | B/vec (K or V avg) | K=1 | K=2 | K=4 | K=8 | GB/s @K=1 / K=8 |
|---|---|---|---|---|---|---|
| fp16 SDPA (reference) | 256 | 1.475 | 1.478 | 1.479 | 1.476 | 728 / 728 |
| **UltraQuant fp4 fused (uqmt)** | 68 | **0.383** | **0.390** | **0.492** | **0.577** | 744 / 494 |
| TurboQuant tq3 (amd-experiments fused; K=1 kernel, K calls) | 52 | 1.153 | 2.226 | 4.371 | 8.662 | 189 / 25 |
| PlanarQuant-3 port (block-rot fork, shared kernel) | 52 | 0.78 | 0.77 | 0.97 | 1.58 | see `ports_planar3.json` |
| IsoQuant-3 port | 52 | 0.79 | 0.76 | 0.96 | 1.58 | |
| RotorQuant-3 port | 52 | 0.77 | 0.77 | 0.97 | 1.58 | |
| **SpectralQuant port (tuned)** | 48.5 | 1.429 | 1.498 | 1.456 | 1.733 | 142 / 117 |
| SpectralQuant port (untuned nsplit16/BN64) | 48.5 | 2.012 | 2.059 | 2.318 | 2.186 | 101 / 93 |

SpectralQuant write path (quantise 1/2/4/8 new tokens x 64 heads, incl. rotation): 0.10 ms per decode step.

**Where UltraQuant wins/loses.** UltraQuant is **3.7x / 3.1x / 3.0x / 3.0x faster** than the SpectralQuant port at K = 1/2/4/8, and the SpectralQuant port is barely faster than fp16 SDPA (1.03x at K=1, 0.85x at K=8) despite reading 5x fewer bytes: it is dequant-bound (about 6 dependent byte loads and one LUT gather per element), not bandwidth-bound (117-142 GB/s vs 728 GB/s fp16). SpectralQuant is the *only* method here that stores fewer bytes (48.5 vs 68 B/vec, 1.4x) and it loses on speed by a wide margin. This is a statement about this naive Triton port; a register-resident codebook / `tl.gather` or a HIP kernel could narrow it, which I did not attempt. UltraQuant does not lose to SpectralQuant on any axis measured except stored bytes and (at equal bytes) nothing else.

## 4. Accuracy

| method | B/vec | ppl delta Mistral-7B (fp16 4.872) | ppl delta Qwen3-8B (fp16 8.522) | real-KV attn cos (Mistral / Qwen3) | synthetic-LLM attn cos @B8 ctx32k (K=1) |
|---|---|---|---|---|---|
| UltraQuant fp4 (uq_fp4, official) | 68 | +0.139 | +0.562 | 0.963 / 0.932 | 0.965 |
| TurboQuant turbo3 (amd-experiments) | 52 | +0.142 | +1.995 | 0.968 / 0.933 | 0.937 |
| TurboQuant turbo4 | 68 | +0.026 | +0.581 | 0.992 / 0.979 | n/a |
| **SpectralQuant port (standard preset)** | 55 padded / 48 ideal | **+5.664** (10.54) | **+7.018** (15.54) | **0.787 / 0.790** | 0.931 |
| SpectralQuant + first token kept fp16 (diagnostic, not in the published scheme) | 55 | +4.234 (9.11) | +0.450 (8.97) | 0.887 / 0.955 | n/a |

(UltraQuant / TurboQuant rows are copied from `results/quality_summary.md`, measured earlier with the same protocol on CUDA; they were not re-measured on this box. The speed rows were.)

SpectralQuant at its default `standard` preset (~2.9 bits per element) is **far** off the claimed "identical perplexity": +5.7 ppl on Mistral-7B and +7.0 on Qwen3-8B with the shipped quantiser. Per-layer key reconstruction error is a steady 16-17% (rel-L2), as expected for ~3 bits with Gaussian codebooks, but attention is much more sensitive because (a) global per-head codebooks have no per-token scale, so heavy-tailed tokens are clipped, and (b) the attention-sink token (position 0) is an extreme outlier: keeping *only that one token* in fp16 cuts the Qwen3 ppl delta from +7.0 to +0.45, but Mistral stays at +4.2. The blog's "9.51 ppl identical to fp16" was on Qwen2.5-14B with a different setup; I could not reproduce that class of result, and I cannot rule out that the authors' PDF recipe differs from the 0.3.1 package defaults (e.g. other presets: `high` claims +11% ppl at 6.55x on their own sweep, not run here).
This is a **single calibration set and a single preset**; a different preset (more bits) would trade bytes for accuracy and was not swept.

## 5. Takeaways
- At matched models and harness, UltraQuant (4.25 bits, no calibration, no rotation) beats the SpectralQuant port on speed (3.0-3.7x) and on accuracy (ppl delta +0.14/+0.56 vs +5.7/+7.0, attention cosine 0.96/0.93 vs 0.79); SpectralQuant stores 1.4x fewer bytes (ideal) but is not on the accuracy-per-byte frontier here either (TurboQuant turbo3 at 52 B gets +0.14/+2.0).
- The K-query story: UltraQuant fused time grows 1.5x from K=1 to K=8, the SpectralQuant port 1.2x (both far below TurboQuant's linear K calls, 7.5x).
- Other methods ported under this scope (Planar/Iso/Rotor: `results/ports_blockrot_notes.md`; KVQuant: see `results/ports_kvquant*.json` if present) use the same harness and cold timing, so the speed columns are directly comparable.
