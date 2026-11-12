# PlanarQuant / IsoQuant / RotorQuant: fused AMD Triton decode ports (MI300X)

Shapes: B8, ctx 32768, Hq64/Hkv8/D128, K = 1/2/4/8 query tokens (causal verify block), cold timing (256 MB Infinity Cache flushed before each call), synthetic
"llm" KV (power-law spectrum, random per-head rotation, 4 outlier key channels; `bench/ports/common.py`). Raw data: `results/ports_{planar3,iso3,rotor3,planar4}.json`
(`baselines` block in planar3 = UltraQuant fused, TQ3 amd-experiments, fp16 SDPA from the same process/data). Code: `bench/ports/blockrot_common.py`.

## Speed (ms per call, cold; GB/s = stored-bytes single pass)

| method (B/vec) | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| UltraQuant fp4 fused (68) | 0.382 (747) | 0.386 (739) | 0.477 (598) | 0.585 (487) |
| TQ3 amd-experiments fused, K independent calls (52) | 1.168 | 2.233 | 4.377 | 8.668 |
| fp16 SDPA (256) | 1.475 | 1.476 | 1.480 | 1.477 |
| planar3 fused (52) | 0.777 (281) | 0.769 (284) | 0.965 (226) | 1.575 (138) |
| iso3 fused (52) | 0.790 | 0.763 | 0.964 | 1.584 |
| rotor3 fused (52) | 0.766 | 0.769 | 0.966 | 1.583 |
| planar4 fused (68) | 1.095 | 1.155 | 1.323 | 2.195 |

All three 3-bit methods time identically: after rotating q (and un-rotating the output) outside the kernel the decode kernel is the same codebook-dequant + online softmax; the
rotation type only changes a 128x128 matrix. UltraQuant is 2.0x (K=1), 2.0x (K=2), 2.0x (K=4), 2.7x (K=8) faster than these 3-bit kernels, and 3.1x/5.8x/9.2x/14.8x faster than TQ3 at K=1/2/4/8 (TQ3 has no K-query mode, so K independent calls). The 3-bit kernels beat TQ3 at every K (1.5x at K=1, 5.5x at K=8).
These kernels are ALU-bound on bit-plane unpack + 8-way select (only 138-284 GB/s of 5.3 TB/s), not bandwidth bound; fp16 SDPA is slower only at K<=4 thanks to the 5x byte saving. At K=8 they are on par with fp16 SDPA (1.58 vs 1.48 ms).
Quantise-on-write is a Triton kernel (dense 128x128 rotation by tl.dot, midpoint compares, bit-plane pack); verified bit-compatible with the torch fake-quant (`bench/ports/test_blockrot.py`; decode vs torch attention on the dequantised cache cos 0.999998).

## Accuracy (real models, from results/quality_summary.md; same quantiser maths)

| method | Mistral-7B wiki2 dppl | Qwen3-8B wiki2 dppl | attn cos Mistral / Qwen3 | synthetic attn cos (K=1) |
|---|---|---|---|---|
| UltraQuant fp4 (official) | 0.139 | 0.562 | 0.963 / 0.932 | 0.965 |
| TQ3 (official-port) | 0.142 | 1.995 | 0.968 / 0.933 | 0.937 |
| planar3 | 6.101 | 16.703 | 0.699 / 0.716 | 0.896 |
| iso3 | 2.537 | 13.870 | 0.714 / 0.721 | 0.916 |
| rotor3 | 3.089 | 13.395 | 0.710 / 0.722 | 0.905 |
| planar4 | 2.155 | 17.188 | 0.746 / 0.756 | 0.947 |

## Why the ppl is bad: not a port bug, a property of block-local rotations (`results/ports_blockrot_diag_realkv.json`)
Checks done: (1) our dense-matrix port reproduces the amd-experiments quantiser (planar/iso cos 1.0000 vs original compress/decompress; kernel dequant == fake-quant to bf16 rounding);
(2) the rotations are orthogonal (err 1e-7) and the norm is stored and re-applied correctly; (3) 3-bit round-trip cos on Gaussian-like data is 0.975-0.978, normal for 3 bits.
On REAL KV (2048 wikitext tokens, per-layer mean quantise-dequantise cosine, 3-bit): Mistral K planar/iso/rotor 0.949/0.956/0.942, TQ3 0.983; Qwen3-8B K 0.919/0.938/0.918 (worst layer 0.66 / 0.82 / 0.67), TQ3 0.983.
Qwen3 keys put on average 26% of the energy in one channel (Mistral 6.6%). Block-local 2D/3D/4D rotations cannot spread a dominant channel across the vector, the fixed unit-vector codebook (N(0,1/128) scale) then clips it. A labelled DIAGNOSTIC variant
(Walsh-Hadamard pre-mix before the block rotation, not the published scheme, not ported to Triton) restores 0.980-0.984 on both models, i.e. TurboQuant's full-width rotation is what matters. 4-bit planar being no better than 3-bit on Qwen3 (17.2 vs 16.7 dppl) fits the clipping explanation.
We do not claim the upstream RotorQuant repo behaves this way with its own pipeline; this is our MI300X port of its quantisers, evaluated through the same harness as everything else.

## Deviations
1. Storage packed as bit-planes (b x 16 B + fp32 norm = 52/68 B). The amd-experiments code keeps one int8 per index (132 B in memory) and only nominally counts 52 B.
2. Block rotation applied as a dense 128x128 matmul outside the kernel (q in, output out) instead of per-block FMAs.
3. Rotor: head_dim 128 gives 43 groups of 3 = 129 dims; the original quantises the pad-derived 129th rotated coordinate; the port stores 128 and drops it (fake-quant cos vs original 0.993 on a heavy-tailed test, ~0.5% of energy). The Triton path's sandwich (grade-1 part) is used; the original's torch fallback uses different formulas.
4. Single seed-42 rotation shared across heads/layers (as the original). No QJL. Attention-sink / edge-layer handling: none (as the original).
5. ppl/needle numbers are taken from quality_summary.md (simulated cache with the identical quantiser), not rerun through the Triton kernels.

## Effort
~230 lines (blockrot_common.py 169, test 27, diagnostic 35) for all three methods + 4-bit; ~3 h wall, of which about half was waiting on the shared GPU lock. One kernel rewrite (byte-gather + LUT gather 1.20 ms -> expand-bits-in-registers + select chain 0.78 ms at K=1).
Kernel is untuned beyond that (BN=32 because BN=64 exceeded the 64 KB LDS with the 8-row-padded layout; nsplit 16 fixed).
