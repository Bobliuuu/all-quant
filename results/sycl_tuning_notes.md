> CORRECTION (2026-11-12): every "% of read peak" figure in this file (1%, 10%) used a 430-455 GB/s peak that was measured on uniform data and is invalid. The random-data read ceiling is ~105-113 GB/s, so the XMX kernel was at ~50% of peak at K=1. See `results/sycl2_profile.md`.

# SYCL tuning notes (Arc B390, Xe3 iGPU, 96 XVEs, 128 KB SLM/core)

Peak (copy kernel, `bench --peakonly`): ~535 GB/s copy (r+w), ~430-455 GB/s read-only. "% of peak" below is FP4 KV
bytes read once (unique bytes) / read-only peak. Raw data: `results/sycl_arcb390_*_sweep.jsonl`.

## What moved the needle (B8, ctx16384, K4, Hq32/Hkv8/D128, ms)
| step | ms | note |
|---|---|---|
| portable v1 (scalar FMA from SLM, wg128) | 26.6 | compute bound: ~1% of read peak |
| XMX joint_matrix 8x16x16 bf16, float-math dequant, wg128 | 7.5 | 3.5x |
| + bit-fold dequant (exponent add, no float math, 32-bit VNNI stores) | 4.2 | |
| + 256-entry byte->bf16-pair LUT in SLM with mask | 3.5 | |
| + wg256 (tn32), nsplit 16 | 3.19 | best: 45 GB/s FP4 KV, 10% of read peak |

## Dead ends (kept so nobody repeats them)
- 3-barrier schedule with double-buffered V and K+V dequantised together: 1.5-2x slower. SLM per work-group grew
  (3 tiles), occupancy fell; Xe cores run 64 hw threads, SLM 128 KB shared.
- 16-byte loads (one scale per 32 elems per thread) for the dequant: 25% slower than 4-byte loads (register pressure).
- tn=64 tiles: slower than tn=32 (SLM footprint). K=1 (nsplit 8): wg256/tn32 2.5 ms; wg128/tn16 2.8; wg512/tn32 4.8; wg256/tn16 4.6; wg64/tn32 ~5.4 (sharp optimum, probably SLM-size/occupancy interplay; not fully understood).
- sub-group size 32 for the portable path: no gain over 16 on this part.

## Bottleneck analysis (ablation, B8 ctx16384, nsplit 8, wg256)
Removing dequant (UQ_NODEQ ablation, since removed) cut 2.5 -> 1.4 ms; removing the MMAs cut 2.5 -> 2.1 ms; removing both left 1.36 ms
for barriers + softmax + O rescale. So per-tile fixed cost (5 work-group barriers, row softmax with sub-group reductions)
and FP4 unpack are co-dominant; the kernel is NOT memory bound (10% of read peak) and XMX itself is cheap.
Next steps if continuing: sub-group-private pipelines (no WG barriers), softmax from joint_matrix accumulators in registers
(drop the S SLM round trip), persistent kernel with page prefetch.

## nsplit
Best nsplit grows with ctx / shrinks with B*Hkv: B2: 32, B8: 8-16 (grid = B*Hkv*nsplit work-groups vs 12 Xe cores x 4 wg256).
The auto heuristic in bench.cpp (4*EUs/(B*Hkv), pow2) lands within ~10% of best.

## Fused vs K-independent (same kernel, K=1 over B*K pseudo-sequences, KV re-read K times)
Time ratio K-indep/fused, XMX path: K=2 1.8x, K=4 2.7x, K=8 3.0x. Cost of verifying K tokens vs 1 token (fused):
K=2 1.0x, K=4 1.3x, K=8 2.5x of the single-token time (K=8 -> M=32 rows -> 4 row tiles: QK/softmax/PV scale with M).
