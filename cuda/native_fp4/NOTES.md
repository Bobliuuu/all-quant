# Native FP4 block-scaled MMA for the QK^T half of UltraQuant decode

Status 2026-11-11. Everything here was run on Modal (CUDA 12.9.1, torch 2.13+cu129, CUTLASS commit b2f249a).
Logs: `results/cuda_native_*.log`, summary `results/cuda_native_summary.json`. Driver: `cuda/modal/modal_native.py`.

## What works
| piece | file | status |
|---|---|---|
| sm_120a native `mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X m16n8k32 e4m3 x e2m1, ue8m0` QK^T, D=128, M=32/64 | `qk_mmasync.cu` (`qk_native`) | validated vs torch fp32 (max abs err 3.8e-6 of |S|max 52, exact on 99.999% of entries) on RTX PRO 6000 |
| bf16 `mma.sync m16n8k16` + software FP4->bf16 dequant baseline, same data/layout/k-permutation | `qk_mmasync.cu` (`qk_bf16`) | validated identically on RTX PRO 6000 and B300 |
| sm_103a (B300) tcgen05 `kind::mxf8f6f4` block-scaled GEMM, A=e4m3 (Q rows), B=e2m1 (K cache), ue8m0 SF, via CUTLASS 4.x CollectiveBuilder | `qk_cutlass_sm100.cu` (from CUTLASS ex. 72c, tile/cluster/verify configurable) | runs and passes CUTLASS's own host reference at 128x2048x128 (random data; NOT the real cache SF layout, NOT cross-checked against torch) |
| data/reference generator (uqmt `quantize_fp4`, Q -> e4m3 + ue8m0/32) | `gen_data.py` | used by both |

The cache layout of `uqmt/attn.py` is directly usable: K bytes (`[key][D/2]`, low nibble = even element) are exactly the packed e2m1 B operand
CUTLASS expects (TMA unpacks 4b->8b containers); `ks` ue8m0 per 32 along D is exactly the k-block-32 SFB (it only needs re-tiling into the
128x4 interleaved SF atom layout for tcgen05, because SF live in TMEM).

## PTX forms (verified by ptxas / CUTLASS headers, not just memory)
* sm_100a / sm_103a (B200/B300): block-scaled MMA exists ONLY as `tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4|mxf4|mxf4nvf4}.block_scale` (A from SMEM or TMEM, B SMEM, SF in TMEM, accumulator in TMEM, M=128 (1-CTA)/256 (2-CTA) for block-scaled).
  The warp-level `mma.sync...block_scale` is **rejected by ptxas on sm_103a** ("Instruction 'mma with block scale' not supported on .target 'sm_103a'"; evidence: `results/cuda_native_b300_qk_bf16.log`, the `-DUQ_NATIVE` compile step). Legacy `mma.sync` (bf16/fp8/fp16, no block scale) still works on sm_100/103, which is what the bf16 baseline uses. sm_103a adds the "ultra" FP4 tcgen05 variants (CUTLASS ex. 89/90, mxf4/nvf4 with A and B both fp4) - not usable for e4m3 x e2m1.
* sm_120a (RTX PRO 6000 / GeForce Blackwell): no tcgen05/TMEM. Block-scaled FP4 is `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.{e4m3,e2m1,...}.{e2m1,...}.f32.ue8m0` and `kind::mxf4[nvf4].block_scale.scale_vec::{2X,4X}.m16n8k64` (both operands fp4, two e2m1 per byte; used by CUTLASS ex. 79). Plain `-arch=sm_120a` in nvcc 12.9 emitted non-`a` PTX for me; use `-gencode arch=compute_120a,code=sm_120a`.
* Contract details found (from `cute/arch/mma_sm120.hpp`, `mma_traits_sm120.hpp`, confirmed empirically):
  * e2m1 inside the 8-bit container of mxf8f6f4 sits in bits [5:2] (`0b00ABCD00`, i.e. nibble << 2); sign = nibble bit 3. (mxf4 m16n8k64 uses packed nibbles instead.)
  * A frag (m16 x k32, 4 regs): a0 = row g, k 4*tig..+3; a1 = row g+8; a2 = row g, k 16+4*tig..; a3 = row g+8 (g = lane/4, tig = lane%4). B frag (k32 x n8, 2 regs): b0 = k 4*tig..+3, n = g; b1 = k 16+4*tig... C as sm_80.
  * Scale operands: `{sf, byte-id, thread-id}` per matrix; with byte-id = thread-id = 0, SFA for row g comes from the lane with lane%4==0 and SFA for row g+8 from lane%4==1 (SFALayout stride 8 on lane bit 0), SFB for column n=g from lane%4==0. I simply pass the right byte in every lane (A: row g + 8*(lane&1); B: key g). Validated.
  * Reduction axis permutation trick: k is a free order as long as A and B agree and each mma's 32 slots lie in one scale block. Thread `tig` owns elements 8*tig..8*tig+7 of each 32-block, so one 4-byte load per (key, k-block) feeds b0,b1; Q fragments are loaded with the same permutation.
  * nibble -> container repack costs ~4 ALU ops per register (`prmt` + 2 shifts + `lop3`); `ldmatrix.b8x16.b4x16_p64` (+ `<<2`) is the CUTLASS way and could remove it (not tried).

## Numbers
All QK^T only, head_dim 128, scores NOT stored (per-thread running max/sum so nothing is dead-code-eliminated), K+scales read from global (524288 keys = 34 MB, L2 resident on both chips), grid = SMs x {2,4,8} CTAs of 128 threads, best of the three shown.

RTX PRO 6000 (sm_120a, 188 SMs): native vs bf16-software-dequant
| M (query rows) | bf16 keys/us (TFLOP/s) | native keys/us (TFLOP/s) | speedup |
|---|---|---|---|
| 64 | 17140 (281) | 27978 (458) | 1.38x / 1.74x / 1.85x for grid x2 / x4 / x8 |
| 32 | 22958 (188) | 40511 (332) | 2.01x / 1.76x / 2.03x |

B300 (sm_103a, 148 SMs), bf16 `mma.sync` + software dequant (the only warp-level path there):
| M | keys/us | TFLOP/s |
|---|---|---|
| 64 | 10947 / 9219 / 11443 (grid x2/x4/x8) | 179 / 151 / 188 |
| 32 | 13724 / 18162 / 18174 | 112 / 149 / 149 |

B300 native tcgen05 mxf8f6f4 (CUTLASS-generated, M padded to 128, S written to global as bf16, K=128):
| tile | N=16k keys | N=262k | N=1M keys | keys/us at 1M | TFLOP/s (padded M=128) |
|---|---|---|---|---|---|
| 128x128, 1 CTA | 3177 | 11643 | 12708 | 12.7k | 416 |
| 128x256, 1 CTA | 2578 | 13424 | **17132** | 17.1k | 561 |
| 256x128, 2 CTA | 2406 | 6326 | 6877 | 6.9k | 225 |
Mainloop-only ceiling (large K so epilogue amortised): 128x128 K=4096 N=16k: 1586 TFLOP/s; 128x256 K=1024 N=262k: 1219 TFLOP/s. At K=128 one key costs 2*128*128 = 32.8 kFLOP, so 1586 TFLOP/s corresponds to ~48k keys/us for M=128 padded rows.

## Interpretation (B300)
* Measured, conservative: tcgen05 native (with M padded 64->128 and a full bf16 S store to HBM, i.e. handicapped) = 17.1k keys/us vs 11.4k for the software-dequant mma.sync path at M=64 => **1.5x**. At M=32 the bf16 path (18.2k) is on par with the padded tcgen05 GEMM: no gain unless >=64 real rows (K x G_PAD).
* Projected: the tcgen05 mainloop alone sustains ~48k keys/us (4.2x over bf16 sw-dequant at M=64) but only if S stays in TMEM, is read with `tcgen05.ld` straight into the softmax, and the kernel is warp-specialised (TMA producer / MMA warp / softmax warps). That fused kernel was NOT built here (no hand-written tcgen05 PTX; only CUTLASS GEMM). The mainloop also needs the K tile landed in SMEM by TMA (no register path), so the paged layout needs a page-table-driven TMA (gather of 64 B rows) - feasible per page (PAGE keys contiguous per head) but untested.
* Roofline context: K+V FP4 traffic is 136 B/key, i.e. ~59k keys/us at 8 TB/s. The sw-dequant QK alone (11k keys/us at M=64) is therefore ~5x ALU/tensor-bound, so at M>=64 the op is compute-bound on B300 and native QK has real headroom; at M=32 or less it is already near the bf16 mma.sync ceiling.

## PV half (P x V)
V is e2m1 with a ue8m0 scale per 32 elements along D, i.e. per (key, D-group). In O[m,d] = sum_key P[m,key] V[key,d], the reduction axis is `key` while the scale axis is `d` (the N axis), and the scale also changes per key. Block-scaled MMA only applies a scale along the reduction (k-block of 32) axis, with an independent scale per M row / N column, so it cannot apply V's scale. (Transposing roles, O^T = V^T P^T, makes D the M axis and key the reduction axis: still per-(key, D-group) scales => no.) Options:
1. Hybrid (recommended): QK native block-scaled; softmax in fp32; P -> bf16 (or fp8); V dequantised in registers/SMEM to bf16 (sm_120: mma.sync bf16; B300: either mma.sync bf16 with sw dequant, or dequant to SMEM + tcgen05 bf16/fp8 kind::f16/f8f6f4 with P from TMEM).
2. Unvalidated idea: fold V's scale into P per D-group: for group g, A_g[m,key] = P[m,key]*s_g[key] (fp8 e4m3), B = V e2m1 columns of group g with unit SF; 4 GEMMs with N=32 (same total FLOPs, 4 fp8 copies of P). Risk: fp8 P*scale rounding (3-bit mantissa) -> needs an accuracy study; extra SMEM traffic for 4 P copies.
No PV kernel was built or measured; the whole-kernel projection below assumes PV ~ the same cost as the bf16 QK path (same MMA count, dequant on V).

## Whole-kernel projection (B300, M=64; estimates, not measured)
Per-key time = QK + PV (softmax and merge ignored). Baseline both halves bf16 sw-dequant: 1/11.4k + 1/11.4k => 5.7k keys/us.
* Hybrid with the measured (handicapped) CUTLASS QK 17.1k: 1/17.1k + 1/11.4k => 6.8k keys/us = **1.2x**.
* Hybrid with an ideal fused tcgen05 QK at the ~48k mainloop rate: 1/48k + 1/11.4k => 9.2k keys/us = **1.6x** (Amdahl cap ~2x if PV stays bf16 sw-dequant).
* To go beyond ~1.6x PV must also leave mma.sync + ALU dequant (option 2 above or tcgen05 bf16/fp8 PV with V dequantised once per tile in SMEM).
On RTX PRO 6000 (measured QK 1.4-1.85x at M=64) the same Amdahl arithmetic gives ~1.2-1.4x whole-kernel.

## Caveats / not done
* No fused softmax+PV kernel, no paged/TMA gather, no hand-written tcgen05 PTX, no Q-row padding strategy beyond M=128 in CUTLASS (M=64 is not offered for block-scaled 1-CTA tiles).
* B300 tcgen05 correctness was validated against CUTLASS's reference on random data only; the real cache SF re-tiling into the interleaved SF layout is not implemented.
* CUTLASS runs of the 256x256 2-CTA and 128x64 variants were not completed (run stopped to save time); logs hold the three finished variants.
* e4m3 quantisation of Q (per-row/per-32 ue8m0 scale) adds error vs bf16 Q that the torch reference includes on both paths (reference uses the dequantised fp8 Q), so the comparison is exact-arithmetic-equal but real-Q fidelity was not evaluated.
* Timings include L2-resident K (34 MB); HBM-resident behaviour was not measured.
