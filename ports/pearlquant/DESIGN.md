# pq-repro design: a PearlQuant-style E8 KV-cache codec

This is our reproduction, built from public information only. It is **not** Pearl
Research Labs' code, and the name "PearlQuant" refers only to their method. Pearl has
released no code (the `pearl-research-labs` GitHub org has no repositories), so every
item marked *ours* below is a choice we made to fill a gap.

Sources: Pearl's blog post (https://pearlresearch.ai/research/blog/pearlquant, read
2026-11-13), NestQuant (Savkin, Porat, Ordentlich, Polyanskiy, arXiv:2502.09720), Conway &
Sloane (SPLAG) for E8, and HyperQuant (MIT, github.com/moonmath-ai/HyperQuant) as a comparison.

## 1. What Pearl states and what we chose

| Item | Pearl's blog | pq-repro |
|---|---|---|
| Quantizer type | "high-dimensional lattice quantization, rounding groups of values together"; "NestQuant [3] inspired"; vector quantization | same |
| Group size / lattice | "groups of eight values"; "the E8 lattice" | E8, groups of 8 |
| Preprocessing | "After a random rotation, KV values behave like a Gaussian cloud" | *ours:* randomized Hadamard transform (fixed random signs, then normalized Walsh-Hadamard) over each head vector (D = 128). |
| Codebook | "we only assign indices to probable lattice points under the Gaussian i.i.d. distribution"; "For groups of d values at R bits per value, the codebook contains 2^dR vectors" | *ours (main):* the 2^b E8 points of smallest norm (for an iid Gaussian, lattice-point probability falls with norm), "ball" codebook. *Ours (ablation):* NestQuant's Voronoi code E8/qE8. |
| Index assignment | not disclosed | *ours:* leader decomposition (RE8-style, as in AMR-WB+/EVS lattice VQ): idx = offset[leader] + perm_rank * nsign + sign_bits. Decodes with a small table (L leaders, 32 B each) and integer arithmetic, no codebook table. |
| Scale | not disclosed | *ours:* one fp16 scale per vector (128 values), +0.125 bpv. The encoder tries 8 gain multipliers (±25%) around a calibrated gain alpha(b) and keeps the one with the lowest error; that search is free because the stored scale absorbs it. |
| Overload (point outside the codebook) | not disclosed | *ours:* shrink the 8-vector radially in 3% steps until its nearest E8 point is a codeword (approximate projection onto the ball). |
| Rates | PQ-3.2 / 2.3 / 2.1 as model-level averages over GLM-5.3's NoPE and RoPE parts; Fig. 5 shows 2.75 and 3.75 bits on a Gaussian | *ours:* fixed rate b bits per group of 8; rate = b/8 + 16/128 bpv. Model configs average K and V: 3.19 (K 25 / V 24 bits), 2.31 (18/17), 2.125 (16/16). |
| K/V or NoPE/RoPE bit split | not disclosed | *ours:* K gets the extra bit when the split is uneven (ablation `pq-3.2-vk` swaps it). |
| Entropy coding | not mentioned; their fractional rates are consistent with fixed rate plus side info, but we cannot tell | *ours:* none (fixed rate, so every vector is randomly accessible in the kernel). |
| Outliers / sinks / centering | not disclosed | *ours, ablation:* K mean-centering (`-kc`): subtract a per-(layer, head) mean mu of the first 256 prefill keys before encoding. It shifts every logit of a query by the same q·mu, so the softmax is exactly unchanged and the kernel never sees mu. `-kvc` also centers V (one add of mu_v to the output). |
| Kernel | Blackwell, CuTe DSL; reads the compressed cache directly, decodes codewords on chip into Tensor Cores, no HBM round trip | *ours:* plain CUDA/HIP C++ and SYCL split-KV decode kernels, CUDA-core FMA (no tensor cores), codes decoded into shared memory per 32-token tile. Query rotated once in the kernel, output un-rotated in the reduce kernel, so K and V are never inverse-rotated. |
| Model | GLM-5.3 (DSA / MLA cache: NoPE in FP8 + RoPE in BF16, 9.1 bpv baseline) on 8xB200 | Qwen3-8B (GQA, 8 KV heads, head_dim 128), one GPU. Not comparable in absolute numbers. |

The 9.1 bpv baseline is consistent with a DeepSeek-V3.2-style FP8 MLA cache:
512 NoPE values in FP8 + 4 fp32 scales + 64 RoPE values in BF16 = 656 bytes per token
= 9.11 bits per value over 576 values. That arithmetic is our inference, not a Pearl statement.

## 2. Rate accounting

Per head vector (D = 128):

| format | bits | bpv |
|---|---|---|
| pq ball b bits/group | 16·b + 16 (fp16 scale) | b/8 + 0.125 |
| pq Voronoi q | 16·ceil(8·log2 q) + 16 | ceil(8 log2 q)/8 + 0.125 |
| FP8 e4m3 + fp32 scale per vector (our reference) | 1024 + 32 | 8.25 |
| NVFP4 (e2m1, e4m3 scale / 16, fp32 per tensor) | | 4.5 |
| int4, fp16 absmax scale / 32 | | 4.5 |
| TurboQuant-MSE b bits (our re-implementation: rotation + Lloyd-Max + fp16 norm) | | b + 0.125 |

In the kernels a pq vector is exactly 2·b bytes of codes (16 indices of b bits pack into
b 16-bit words) plus 2 bytes of scale, so the bytes read per token equal the accounted rate.
Leader tables are per layer-independent constants (b=25: 460 leaders, 14.7 KB).

## 3. Why the ball codebook (and not Voronoi) is the main variant

On iid Gaussian data (results/rd_gaussian.json) the ball codebook wastes fewer bits
per value than the Voronoi code at every matched rate, it allows any integer number of bits per
group (fractional bpv in 1/8 steps, which is how 3.2/2.3/2.1 averages come out), and it matches
Pearl's "probable lattice points" sentence literally. Decode cost is similar: Voronoi needs a
base-q unpack, a matrix product and a full E8 nearest-point (two D8 roundings); the ball needs a
binary search over L leaders plus a multiset-permutation unrank (at most 8 x ndistinct steps).

## 4. Known differences that matter for the kernel

GLM-5.3 uses MLA: one latent cache entry per token is shared by all query heads, so decoding a
codeword is amortized over every head (and over speculative tokens). With GQA (Qwen3-8B: 4 query
heads per KV head) the per-codeword ALU cost is amortized over only 4 dot products, so our kernel
is ALU-bound on decode rather than bandwidth-bound. We report an MQA-like shape (one KV head,
16 to 32 query heads) to show how the cost behaves when it is amortized the way MLA amortizes it.
