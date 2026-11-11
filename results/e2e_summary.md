# End-to-end decode on Mistral-7B-v0.1 (AMD Instinct MI300X VF)

Hand-rolled layer loop over the HF modules (embed, RMSNorm, q/k/v proj, RoPE, attention, o_proj, SwiGLU MLP, lm_head) so only the attention backend changes; weights fp16, eager PyTorch (no CUDA graphs: HIP graph capture of this loop core-dumped, not investigated).
Context KV = real Mistral prefill of python source (sliding window off), replicated across the batch (timing is data-independent). New-token K/V are quantised and appended by a Triton kernel inside the step for the FP4 backends (included in the step time; bit-exact with `quantize_fp4`).
Backends: `fp16` = torch SDPA over contiguous fp16 KV (GQA via [B,Hkv,G*K,D] reshape, block-causal mask for K>1); `fp16_bmm` = bmm+softmax+bmm on the same layout (much faster than SDPA at small batch); tables use `fp16 best` = the faster of the two per config; `uq_fused` = fused K-query FP4 kernel (pinned 1ec6fbe, nsplit 32 at B=1 else 16); `uq_kindep` = same kernel, K independent single-query problems.
`gpu` = sum of GPU kernel durations per step from torch.profiler (what a graph-captured/persistent decode loop approaches); `wall` = eager step time (launch-bound: ~20-40 ms regardless of work for small batches). Tokens/s below use `gpu` (graph-mode projection) and are marked as projections.

## B=1, ctx=8192: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 9.8 (bmm) | 0.4 | 4.9 | 4.5 | 10.1 | 1.1 | - | 0.97x | 1.00x | 16.9 | 21.1 |
| 2 | 11.4 (bmm) | 0.4 | 4.9 | 6.0 | 10.4 | 1.1 | 11.7 | 1.10x | 1.13x | 24.1 | 21.1 |
| 4 | 11.6 (bmm) | 0.4 | 5.0 | 6.2 | 10.5 | 1.1 | 12.1 | 1.10x | 1.15x | 23.0 | 21.7 |
| 8 | 12.3 (bmm) | 1.1 | 5.1 | 6.2 | 10.9 | 1.4 | 13.3 | 1.13x | 1.22x | 22.8 | 21.4 |

## B=8, ctx=8192: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 13.4 (bmm) | 3.6 | 5.0 | 4.8 | 12.5 | 3.0 | - | 1.08x | 1.00x | 19.0 | 22.1 |
| 2 | 15.4 (bmm) | 4.0 | 5.1 | 6.3 | 12.8 | 3.1 | 16.7 | 1.21x | 1.31x | 23.9 | 22.7 |
| 4 | 16.8 (bmm) | 4.8 | 5.7 | 6.3 | 13.5 | 3.2 | 21.0 | 1.24x | 1.56x | 22.4 | 22.9 |
| 8 | 20.3 (bmm) | 7.6 | 6.5 | 6.2 | 15.8 | 4.1 | 30.3 | 1.29x | 1.92x | 25.4 | 21.9 |

## B=1, ctx=16384: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 10.4 (bmm) | 1.0 | 4.9 | 4.5 | 10.8 | 1.8 | - | 0.96x | 1.00x | 17.1 | 23.0 |
| 2 | 12.0 (bmm) | 1.0 | 4.9 | 6.1 | 11.0 | 1.8 | 12.9 | 1.09x | 1.17x | 22.9 | 21.4 |
| 4 | 12.4 (bmm) | 1.2 | 5.0 | 6.2 | 11.1 | 1.8 | 13.2 | 1.12x | 1.19x | 21.9 | 22.1 |
| 8 | 13.3 (bmm) | 2.1 | 5.0 | 6.2 | 11.7 | 2.3 | 16.2 | 1.14x | 1.38x | 26.4 | 23.0 |

## B=8, ctx=16384: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 17.0 (bmm) | 7.0 | 5.0 | 4.9 | 15.8 | 5.7 | - | 1.07x | 1.00x | 18.3 | 22.1 |
| 2 | 19.9 (bmm) | 8.4 | 5.1 | 6.3 | 16.0 | 5.9 | 22.0 | 1.24x | 1.38x | 22.8 | 22.3 |
| 4 | 22.3 (bmm) | 10.2 | 5.6 | 6.4 | 16.8 | 6.0 | 30.0 | 1.33x | 1.79x | 23.2 | 22.4 |
| 8 | 27.6 (bmm) | 14.8 | 6.5 | 6.3 | 19.5 | 7.6 | 46.2 | 1.41x | 2.37x | 26.5 | 21.5 |

## B=1, ctx=32768: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 11.7 (bmm) | 2.0 | 4.9 | 4.8 | 12.1 | 3.1 | - | 0.97x | 1.00x | 18.1 | 21.3 |
| 2 | 13.7 (bmm) | 2.3 | 5.0 | 6.4 | 12.3 | 3.1 | 15.1 | 1.11x | 1.23x | 22.8 | 22.0 |
| 4 | 14.3 (bmm) | 2.7 | 5.1 | 6.6 | 12.5 | 3.1 | 16.0 | 1.14x | 1.28x | 22.2 | 21.3 |
| 8 | 15.7 (bmm) | 4.1 | 5.1 | 6.5 | 13.6 | 3.9 | 21.4 | 1.15x | 1.58x | 22.8 | 21.3 |

## B=8, ctx=32768: step latency by K (ms)

| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 24.0 (bmm) | 13.8 | 5.1 | 5.1 | 21.2 | 10.8 | - | 1.13x | 1.00x | 22.9 | 21.7 |
| 2 | 27.9 (bmm) | 16.1 | 5.1 | 6.7 | 21.6 | 11.1 | 31.5 | 1.29x | 1.46x | 26.4 | 24.6 |
| 4 | 32.4 (bmm) | 20.1 | 5.6 | 6.7 | 22.2 | 11.3 | 46.1 | 1.46x | 2.07x | 30.9 | 21.8 |
| 8 | 41.8 (bmm) | 28.6 | 6.6 | 6.6 | 26.1 | 14.1 | 80.0 | 1.60x | 3.07x | 40.5 | 25.2 |

## Projected throughput (tokens/s, aggregate over batch) at assumed per-token acceptance a

Tokens per verify step with K-1 drafts = sum_{i<K} a^i (iid acceptance). Each column takes its best K in {1,2,4,8}. Baseline = best fp16 backend (per config min of torch-SDPA and bmm attention) plain decode (K=1). Uses the graph-mode `gpu` step times above.

| B | ctx | a | fp16 plain | fp16 spec (best K) | UQ plain K=1 | UQ fused spec (best K) | UQ K-indep spec (best K) | UQ fused vs fp16 plain | UQ fused vs fp16 spec | UQ fused vs UQ K-indep |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 8192 | 0.5 | 102 | 162 (K=4) | 99 | 183 (K=8) | 155 (K=4) | 1.79x | 1.13x | 1.18x |
| 1 | 8192 | 0.7 | 102 | 255 (K=8) | 99 | 288 (K=8) | 236 (K=8) | 2.83x | 1.13x | 1.22x |
| 1 | 8192 | 0.9 | 102 | 463 (K=8) | 99 | 522 (K=8) | 428 (K=8) | 5.13x | 1.13x | 1.22x |
| 8 | 8192 | 0.5 | 596 | 895 (K=4) | 642 | 1111 (K=4) | 719 (K=2) | 1.86x | 1.24x | 1.55x |
| 8 | 8192 | 0.7 | 596 | 1239 (K=8) | 642 | 1594 (K=8) | 964 (K=4) | 2.67x | 1.29x | 1.65x |
| 8 | 8192 | 0.9 | 596 | 2246 (K=8) | 642 | 2890 (K=8) | 1505 (K=8) | 4.85x | 1.29x | 1.92x |
| 1 | 16384 | 0.5 | 96 | 151 (K=4) | 92 | 170 (K=8) | 142 (K=4) | 1.77x | 1.12x | 1.19x |
| 1 | 16384 | 0.7 | 96 | 236 (K=8) | 92 | 268 (K=8) | 193 (K=8) | 2.80x | 1.14x | 1.38x |
| 1 | 16384 | 0.9 | 96 | 427 (K=8) | 92 | 485 (K=8) | 351 (K=8) | 5.07x | 1.14x | 1.38x |
| 8 | 16384 | 0.5 | 472 | 674 (K=4) | 505 | 895 (K=4) | 546 (K=2) | 1.90x | 1.33x | 1.64x |
| 8 | 16384 | 0.7 | 472 | 911 (K=4) | 505 | 1287 (K=8) | 675 (K=4) | 2.73x | 1.41x | 1.91x |
| 8 | 16384 | 0.9 | 472 | 1650 (K=8) | 505 | 2334 (K=8) | 986 (K=8) | 4.95x | 1.41x | 2.37x |
| 1 | 32768 | 0.5 | 85 | 131 (K=4) | 83 | 150 (K=4) | 117 (K=4) | 1.76x | 1.14x | 1.28x |
| 1 | 32768 | 0.7 | 85 | 201 (K=8) | 83 | 231 (K=8) | 158 (K=4) | 2.71x | 1.15x | 1.46x |
| 1 | 32768 | 0.9 | 85 | 364 (K=8) | 83 | 419 (K=8) | 266 (K=8) | 4.92x | 1.15x | 1.58x |
| 8 | 32768 | 0.5 | 333 | 463 (K=4) | 377 | 675 (K=4) | 381 (K=2) | 2.02x | 1.46x | 1.77x |
| 8 | 32768 | 0.7 | 333 | 625 (K=4) | 377 | 963 (K=8) | 440 (K=4) | 2.89x | 1.54x | 2.19x |
| 8 | 32768 | 0.9 | 333 | 1091 (K=8) | 377 | 1746 (K=8) | 597 (K=4) | 5.24x | 1.60x | 2.92x |

## Where the step time goes, fp16 K=1 (GPU kernel time share) - Amdahl view

| B | ctx | step ms | attention | GEMM | other | max whole-step speedup if attention were free | UQ fused K=1 whole-step speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 8192 | 9.8 | 4% | 50% | 46% | 1.04x | 0.97x |
| 8 | 8192 | 13.4 | 26% | 37% | 36% | 1.36x | 1.08x |
| 1 | 16384 | 10.4 | 10% | 47% | 43% | 1.11x | 0.96x |
| 8 | 16384 | 17.0 | 41% | 30% | 29% | 1.71x | 1.07x |
| 1 | 32768 | 11.7 | 17% | 42% | 41% | 1.21x | 0.97x |
| 8 | 32768 | 24.0 | 58% | 21% | 21% | 2.36x | 1.13x |

amd-experiments' Amdahl finding (GEMM ~40-44%, attention ~30% of decode step) was measured with their stack on a different batch/context; compare the shares above at the matching B/ctx.

## KV memory (bytes at ctx, all 32 layers, whole batch)

| B | ctx | fp16 KV GB | FP4 KV GB | ratio |
|--:|--:|--:|--:|--:|
| 1 | 8192 | 1.07 | 0.29 | 3.76x |
| 8 | 8192 | 8.59 | 2.28 | 3.76x |
| 1 | 16384 | 2.15 | 0.57 | 3.76x |
| 8 | 16384 | 17.18 | 4.56 | 3.76x |
| 1 | 32768 | 4.29 | 1.14 | 3.76x |
| 8 | 32768 | 34.36 | 9.13 | 3.76x |

## Real multi-turn agent-like decode, B=1, ctx=16384, prompt-lookup drafter (n=3), Kmax=8, greedy

Context = real python source (transformers modeling files). Each of 3 turns: a 'copy this code' tool-output (~150-210 tokens) is ingested in 16-token chunks through the same step path, then 96 tokens are generated. Acceptance is real (verified against each backend's own greedy argmax). Wall tok/s is eager (launch-bound, ~20 ms/step floor) so it understates kernel differences; the `projected` column re-prices the measured tokens/step with the graph-mode GPU step time of the sweep at B=1 (K=8 row for K>1 runs, K=1 row otherwise).

| backend | Kmax | tokens/step | drafts accepted/step | gen tok/s (eager wall) | projected tok/s (GPU step time) | ingest tok/s (eager, 16-token chunks) | KV GB |
|--|--:|--:|--:|--:|--:|--:|--:|
| fp16 | 1 | 1.00 | 0.00 | 31 | 31 | 357 | 2.26 |
| fp16_bmm | 1 | 1.00 | 0.00 | 56 | 96 | 689 | 2.26 |
| fp16_bmm | 8 | 1.62 | 0.62 | 74 | 121 | 704 | 2.26 |
| uq_fused | 1 | 1.00 | 0.00 | 46 | 92 | 729 | 0.60 |
| uq_fused | 8 | 1.84 | 0.84 | 84 | 156 | 697 | 0.60 |
| uq_kindep | 8 | 1.84 | 0.84 | 70 | 113 | 594 | 0.60 |

Teacher-forced top-1 agreement with the fp16 (SDPA, K=1) greedy continuation (same 3x96 tokens fed to every backend; fraction of positions whose argmax equals the fp16 token):

| backend | block size K | top-1 agreement |
|--|--:|--:|
| fp16_bmm | 1 | 99.7% |
| uq_fused | 1 | 72.9% |
| uq_fused | 8 | 72.9% |
| uq_kindep | 8 | 72.9% |
| fp16_bmm | 8 | 99.7% |

## Real multi-turn agent-like decode, B=1, ctx=8192, prompt-lookup drafter (n=3), Kmax=8, greedy

Context = real python source (transformers modeling files). Each of 3 turns: a 'copy this code' tool-output (~150-210 tokens) is ingested in 16-token chunks through the same step path, then 96 tokens are generated. Acceptance is real (verified against each backend's own greedy argmax). Wall tok/s is eager (launch-bound, ~20 ms/step floor) so it understates kernel differences; the `projected` column re-prices the measured tokens/step with the graph-mode GPU step time of the sweep at B=1 (K=8 row for K>1 runs, K=1 row otherwise).

| backend | Kmax | tokens/step | drafts accepted/step | gen tok/s (eager wall) | projected tok/s (GPU step time) | ingest tok/s (eager, 16-token chunks) | KV GB |
|--|--:|--:|--:|--:|--:|--:|--:|
| fp16 | 1 | 1.00 | 0.00 | 48 | 48 | 558 | 1.19 |
| fp16_bmm | 1 | 1.00 | 0.00 | 57 | 102 | 671 | 1.19 |
| fp16_bmm | 8 | 5.66 | 4.66 | 247 | 460 | 709 | 1.19 |
| uq_fused | 1 | 1.00 | 0.00 | 47 | 99 | 734 | 0.32 |
| uq_fused | 8 | 4.80 | 3.80 | 227 | 440 | 738 | 0.32 |
| uq_kindep | 8 | 5.84 | 4.84 | 225 | 439 | 578 | 0.32 |

Teacher-forced top-1 agreement with the fp16 (SDPA, K=1) greedy continuation (same 3x96 tokens fed to every backend; fraction of positions whose argmax equals the fp16 token):

| backend | block size K | top-1 agreement |
|--|--:|--:|
| fp16_bmm | 1 | 100.0% |
| uq_fused | 1 | 99.0% |
| uq_fused | 8 | 99.0% |
| uq_kindep | 8 | 99.3% |
| fp16_bmm | 8 | 100.0% |
