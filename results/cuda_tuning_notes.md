# CUDA port: tuning notes (B8, ctx 32768, Hq 64, Hkv 8, D 128, PAGE 64)

All numbers are Modal runs (CUDA 12.9.1, torch cu129); raw logs in `results/cuda_*`.  GPU labels: "H100" = H100 80GB HBM3 (`gpu="H100!"`;
plain `"H100"` sometimes returns an H200 on Modal - early v1 logs show an H200).

## Iteration history (K=1 / K=8 fused ms on B300 unless noted)
| iter | change | B300 K=1 | B300 K=8 | note |
|---|---|---|---|---|
| v1 | CTA-wide cp.async ring, smem bf16 dequant (bit trick), ldmatrix + mma, warps split rows/keys | 0.218 | 0.350 | correct on first run; Triton tuned 0.308 / 0.494 |
| ablate v1 | remove dequant ALU / remove MMA / both (`UQ_ABL`) | 0.162 / 0.203 / 0.143 | 0.289 / 0.309 / 0.255 | even with no dequant and no MMA the kernel is 0.143 ms: structure (2 CTA barriers per page, small work per barrier) is the limit; L2-resident (same page for all) = same time as HBM-resident => not DRAM bound |
| W1 | design W: per-warp rings, no CTA barriers, K dequant straight into B fragments (permuted reduction order), V via warp-private smem tile + ldmatrix.trans | 0.194 | 0.56 | K=1,2 improve, MB=2 (K>=4) spills 1 KB of registers and loses to v1 |
| W2 | page-split (warps take different pages, KPW=64) + prefetched page ids | 0.226 | 0.72 | worse: bigger per-warp state, fewer warps |
| prof | `clock64` phase profile (warp 0, K=1): wait 1.0k, K-dequant+S 4.6-5.1k, softmax 0.7k, PV 5.5k cycles per page per warp | | | latency/issue bound, not memory bound |
| W3 | V dequantised straight into PV B fragments (prmt-paired keys, bf16x2 pair scale), no smem tile/ldmatrix/syncwarp | 0.175 | - | correct; ~no speed change => V smem round trip was not the limiter |
| load-only | skip all compute after the cp.async wait (`UQ_ABL=4`) | 0.055 (79% of peak) | 0.106 | the loader/pipeline can deliver 79% (B300) / 87% (H100) of HBM peak; compute path is the gap |
| balance | nsplit grid widened to any integer; B*Hkv*nsplit CTAs should land on ~k x SM count | **0.1395** | 0.350 | nsplit=9 (576 CTAs on 148 SMs) beats 8 / 16: grid imbalance was ~20% |

## Findings
* **K=1/2 (BLOCK_M=16)**: design W wins (no barriers, no register spills, 2-4 warps).  Best configs: 2 warps/CTA, 2 stages, nsplit 9 (B300),
  4 warps, 2-3 stages, nsplit 4 (H100), 4 warps nsplit 10 (A100).
* **K>=4**: variant 1 wins (cooperative dequant is shared by the 2-4 row-warps; W with MB=2 spills: ptxas reports 700 B stack / ~1 KB spill
  stores for BM=32/64 MB=2, 0 for BM=16 MB=1).  Best: 4 warps, 2 stages, nsplit 16-24 (B300).
* **B300 K=8 is at the `mma.sync` ceiling**: 4*B*Hq*K*ctx*D = 68.7 GFLOP per step; the native-FP4 study measured legacy bf16 `mma.sync`
  at ~188 TFLOP/s on B300 (M=64) => 365 us minimum; we measure 350 us.  Only tcgen05 (UMMA) can go beyond.  On H100 the same arithmetic
  gives a much higher ceiling (K=8 is at 0.43 ms, i.e. well below it): H100 K=4/8 are the weakest rows (1.7x / 1.05x vs Triton) and the
  clearest next target (software dequant ALU + 16 row-warps' worth of mma issue).
* Register budget vs occupancy: design W with MB=1 keeps acc (64) + Q fragments (32) + S (32) + dequant temporaries in registers; the profile
  (12k cycles per page per warp, ~5-6 cycles/instruction) says the SM runs out of independent warps long before it runs out of bandwidth.
  An independent-accumulator split of the S chain (two mma chains) gave ~10% on that phase only.
* Not tried / ideas with real upside: (1) fp8 (e4m3) K path with Q split into hi+lo e4m3 (K dequant ~0.9 instr/element instead of ~2.5, needs sm_89+/sm_90+);
  (2) tcgen05 path for B300 (QK block-scaled native 1.5x measured at M=64, ~4x in the mainloop for K>=8 per `native_fp4/NOTES.md`);
  (3) TMA/cp.async.bulk loads and warp specialisation (producer warp) on H100/B300; (4) CUDA-graph friendly workspace API;
  (5) hardware `cvt.rn.f16x2.e2m1x2` on sm_100+ (fp16 operands).

## Native FP4 block-scaled MMA on B300 (answer to "does it beat software dequant?")
Yes for the QK half, but only ~1.5x as measured (CUTLASS tcgen05 mxf8f6f4, e4m3 Q x e2m1 K, ue8m0 scales, M padded 64->128, scores written to HBM) and
only at M>=64 (K>=8 with G=8); at M=32 the bf16 `mma.sync` path ties.  The mainloop alone reaches ~4x; a fused warp-specialised kernel (S kept in TMEM,
paged TMA gathers) was not built, so the whole-kernel win is projected at 1.2-1.6x, not measured.  `mma.sync` with block scales does not exist on sm_103a
(ptxas rejects it); on sm_120a it does and was validated.  V cannot use block scales.  Details and logs: `cuda/native_fp4/NOTES.md`.

## Tuning protocol
`bench_cuda.py --tune`: grid over variant {1,2}, mb {1,2}, warps {1..16}, stages {2,3,4}, nsplit {4..48 incl. non-powers of two}; 10 timed launches per
candidate, then 30 iterations at the winner (CUDA events, back-to-back launches, no CUDA graphs).  Triton is tuned over warps {2,4} x stages {2,3} x nsplit {16,32,64}.
