# CUDA port effort log (UltraQuant K-query FP4 decode)

Executed by one agent plus four short-lived research subagents (prior art + baselines, native FP4 MMA study, upstream bit-compat,
none of them wrote kernel code).  Wall clock for the whole port from reading the Triton reference to the five-GPU result tables:
about **2.5 hours** (single session, 2026-11-11, mostly waiting on Modal image builds and compile/bench jobs).
There is no local nvcc/GPU on the dev machine: every compile and every test ran on Modal, so one iteration = edit + `modal run`
(3-8 minutes, dominated by image/JIT build and the tuning sweep).

## Effort
| item | amount |
|---|---|
| kernel iterations measured on hardware | 9 (v1, 2 ablations, W1, W2, profile, W3, load-only, nsplit balance) |
| Modal runs | ~45 (check x7, bench x14, standalone tune/ablate/prof x16, ptxas, plus subagent runs) |
| GPUs | B300, H100 (+H200 by accident), A100, L40S, RTX PRO 6000 |
| LOC (new) | `uq_decode.cu` 560, `uq_decode_w.cuh` 310, host/binding 120, harness+bench ~290, vector generator 120, Modal ~75 (~1.5 kLOC) |
| bugs found on hardware | 0 correctness bugs in the kernels (first run was correct on H100/B300); 4 harness/tooling bugs (below) |

## Perf per iteration (B300, fused ms; Triton tuned 0.308 / 0.310 / 0.425 / 0.494 for K=1/2/4/8)
| iter | K=1 | K=2 | K=4 | K=8 | what |
|---|---|---|---|---|---|
| v1 | 0.218 | 0.219 | 0.253 | 0.350 | smem dequant, CTA barriers |
| W1 | 0.194 | 0.200 | 0.288 | 0.560 | warp-independent, K in registers |
| W3 | 0.175 | 0.178 | (v1) 0.253 | (v1) 0.350 | V straight into B fragments |
| balance | **0.1395** | **0.1419** | 0.2527 | 0.3504 | nsplit grid incl. non-powers of two |
H100: 0.249 (v1, H200 host) -> 0.1614 K=1; A100 0.2588; see `README.md` for all GPUs.

## Pitfalls hit
1. `pip install torch` pulled a CUDA-13 wheel while the image had nvcc 12.9 (JIT extension mismatch) and the install step stalled for 30+ minutes; fixed with `--index-url https://download.pytorch.org/whl/cu129`.
2. `modal run --extra "--flag"`: Modal parses the value as an option; must be `--extra="--flag"` (two runs silently did nothing).
3. Modal `gpu="H100"` may return an H200; use `"H100!"`.
4. Error criterion: the requested 1e-3 max abs error is unreachable on short sequences because bf16 output + bf16 P rounding alone give ~1e-2 on O(1) outputs; at the benchmark shape it is 1.1e-4..1.5e-4.  Both kernels and Triton share it.
5. Macro hygiene (`CK(e)` shadowing) and a sticky CUDA error: a failed `cudaFuncSetAttribute` for a too-large config poisoned the next successful launch's `cudaGetLastError()` (A100/L40S/RTX tuning crashed); fixed by checking the opt-in smem limit and clearing errors.
6. macOS `sed -i` needs `''`; a failed sed in an `&&` chain silently skipped the Modal launches.
7. ptxas spills (1 KB) in the MB=2 variant of the warp-independent design were invisible until `-Xptxas -v`; always check.
8. Performance model mistakes: first guess "ALU-bound dequant" was wrong (removing dequant and MMA entirely still gave 0.143 ms); "latency to the block table" was wrong (prefetching page ids changed nothing); the load-only and `clock64` ablations were what located the problem (compute-path latency / occupancy).

## Comparison with the earlier TurboQuant-to-AMD port (Bobliuuu/amd-experiments)
Facts from `git log`, `kernels/hip/*`, `tq_triton.py`, `tq_hsaco_loader.py`, `current.md`, `docs/`:
* **Scope/time**: amd-experiments is 15 commits over 2026-04-13..04-26 (about two weeks) producing a HIP library (`turboquant_mi300x.hip.cpp`, 713 LOC + header 283 + tests 718 + bench 283),
  a 1834-line Triton fused-attention file, a 594-line HSACO loader and a 499-line MFMA loader, a pure-PyTorch wrapper, a vLLM integration and a long benchmark suite.  This port: ~2.5 hours,
  ~1.5 kLOC, one op, one layout, five NVIDIA GPUs, with a vector-based correctness harness.  Not like-for-like: TurboQuant needed a rotation + Lloyd-Max codebooks + bitplane packing and QJL
  residuals (4 kernels); UltraQuant is a fixed FP4/UE8M0 format with a published layout and an existing Triton reference to match bit for bit.
* **What was hard there**: (1) toolchain/ABI: a system `hipcc` build produced COV6 code objects that the HIP runtime inside PyTorch rejected (`hipErrorNoBinaryForGpu 209`), so the HIP kernels could not be called
  from the PyTorch process; the workaround was a COV5 HSACO loaded via `hipModuleLoad` from PyTorch's own libamdhip64, and ultimately a pure-PyTorch (rocBLAS/MFMA) wrapper for the hot path;
  (2) the standalone HIP kernels were validated (16/16 tests) but fast paths never replaced the PyTorch path: decompress 198 GB/s standalone vs 58 GB/s through the wrapper; the Triton kernel was "written, not end-to-end
  tested"; GQA kernels were committed "untested"; (3) end-to-end: the attention win did not show up in vLLM decode throughput (+8-10% from a mixed GEMM + attention profile; AWQ 35% slower; CUDA graphs leg
  aborted), i.e. the integration/Amdahl work dominated; (4) a shell/benchmark-harness breakage ("primary shell process in a broken state") cost a phase.
* **What was hard here**: (1) nothing about the toolchain: CUDA/PyTorch extension builds were routine; the cost was iteration latency (no local GPU) and tuning, not ABI; (2) performance, not correctness: the first kernel was right and 1.4x faster than
  Triton, but 4-10x off the HBM roofline and ~3x off vendor NVFP4 kernels; finding why took ablations/profiling (below roofline because of instruction latency at register-limited occupancy, not memory);
  (3) hardware matrix: sm_80/89/90/103/120 required the same source to survive smem limits (99 KB on L40S/RTX), register spills and tuning per arch; (4) the native FP4 tensor-core path (the NVIDIA analogue of CDNA4 scaled MFMA) is
  gated by ISA: `mma.sync` block-scale exists only on sm_120a, B300 needs tcgen05/TMEM/CUTLASS-style machinery, which we measured but did not turn into a fused kernel.
* **Common to both**: the high-level kernel (fused dequant + attention) is easy to get correct and 1.3-2x better than a generic baseline; the last 2-3x to the roofline is instruction-level work; benchmarks need explicit
  caveats (what peak, which clocks, which config) to avoid overstating.  Differences in outcome: AMD stayed a standalone/prototype (library + Triton), this one has a vector-validated CUDA kernel, build files and a five-GPU table, but
  neither was integrated into a serving stack by the porting agent.

## Addendum 2026-11-11 (cuda2-agent): H100 wgmma path, B300 tcgen05 variant, NVFP4, workspace API

Everything below ran on Modal (CUDA 12.9.1, torch cu129; H100 = `gpu="H100!"`, i.e. a real H100 80GB HBM3, not an H200).  Raw logs/JSON:
`results/cuda2_*`; summary in the previous schema: `results/cuda2_summary.json`.  Dev harnesses: `bench/cuda2/` (`modal_cuda2.py`, `bench_sm90.cu`,
`modal_cuda2_b300.py`, `bench_sm100.cu`, `test_vectors_nv.cu`, `test_graph.py`, `quality_cuda2.py`, `modal_cuda2_model.py`).

### Goal 1: H100 variant 3 (`uq_decode_sm90.cuh`, sm_90a)
Design: 1 producer warpgroup (one lane) issues `cp.async.bulk` (1D TMA, no tensor map) copies of the four contiguous per-(page,head) FP4 slices into a raw
ring (mbarrier `complete_tx`); 2 dequant warpgroups expand FP4/UE8M0 to bf16 into 128B-swizzled K/V tiles (fence.proxy.async, mbarrier handshake);
2 MMA warpgroups (alternate pages, own online-softmax state, merged through smem at the end) run `wgmma.m64n64k16` for S = Q K^T (A = Q held in registers,
B = K tile K-major) and `wgmma.m64n128k16` for O += P V (A = P in registers, B = V tile MN-major via the transpose bit, LBO 8192 / SBO 1024); `setmaxnreg`
rebalances registers (DQ 56, producer 24, MMA the rest).  M = 64 rows (K*pow2(G) <= 64, i.e. K <= 8 for G = 8), D = 128, PAGE = 64.
Correct on the first bring-up (vectors 420/420).  Then a perf hunt; what each step bought (H100, K=8, B8 ctx32768, ms; K=1 is the same because the kernel is K-independent):

| step | change | K=8 ms | finding |
|---|---|---|---|
| v0 | first wgmma kernel (ALU dequant, Q in smem, producer = dequant thread 0, `named_bar` per page) | 0.375 | K-independent => not MMA bound; 2500 cycles/page |
| ablate | no dequant 0.18 / no MMA+softmax 0.26 / neither 0.117 (80% of HBM peak) | | dequant side dominates, but also additive => poor overlap |
| v1 | cheaper pair order (`deq_word`), Q stored permuted, per-chunk scale factors | 0.333 | |
| v2 | 2-4 dequant WGs, `ex2.approx`, fma-based softmax, skip masks on interior pages, skip softmax for all-padding warps, `setmaxnreg` | 0.252 | K=1/4/8 identical => structural |
| v3 | 32-replica byte->bf16x2 LUT in smem (`prmt`+`imad`+`lds`+`hfma2` per 2 elements instead of ~8 int ops) | 0.242 | int-ALU was not the (only) limiter |
| v4 | Q fragments in registers (`wgmma` RS for S) | 0.238 | smem bandwidth not the limiter either |
| prof | `clock64` split per role (UQ90_PROF): DQ 805 cyc/page of work + 591 cyc waiting at the per-page barrier; MMA 900 cyc softmax per page | | the per-page `named_bar` serialises the two DQ warpgroups |
| v5 | dedicated producer warpgroup + per-stage `rfree` mbarrier (raw stage released right after the LDS, no CTA barrier), lazy O rescale (FA4-style, only when the max grows by > 2^8) | **0.2115** | 2.0x vs the previous H100 K=8 (0.431) |

Tried and dropped: two independent S accumulation chains (S wait 902 -> 418 cycles but +32 registers spill the MMA warpgroups under the 96-register launch
cap of a 640-thread CTA; net slower); deeper raw rings (4 -> 8/12 stages: no gain); ND = 3 dequant WGs (no gain).  Pitfall: the register file is split per SM
sub-partition (16K regs, ceil(warps/4) warps each), so a 17-warp CTA gets 96 regs/thread, not 120; a `setmaxnreg.inc` that asks for more than the launch
allocation minus what the `dec` released hangs (the first producer-warp attempt deadlocked on exactly this; mbarrier waits therefore spin with a bounded
`__trap()` instead of hanging).  `ptxas` C7510 "wgmma serialized due to function call" appeared when a `printf` was left in the wait loop.
Final H100 B8 ctx32768 (tuned, same job as Triton): K=1 0.1633, K=2 0.1635 (variant 2, unchanged, 58% of the 3.0 TB/s copy peak), **K=4 0.2145 (was 0.2906, 1.35x), K=8 0.2152 (was 0.4310, 2.00x)**.
Variant 3 needs ~0.21 ms at every K (it is K-independent, 44% of peak), so it only pays for K >= 4 (auto-selected on sm_90 for >= 32 query rows); K=1/2 stay on variant 2.
Where it stops: per page per MMA warpgroup S+wait 900, softmax 1100, PV 520 cycles and ~1500 cycles per page per CTA in total; the remaining gap to the 0.117 ms
load-only floor is instruction issue + the dependent S -> softmax -> PV chain at 1 CTA/SM, not memory.

### Goal 2: B300 variant 4 (`uq_decode_sm100.cuh`, tcgen05; written by a forked sub-agent, see its commits 586b380 / 21d0376)
Raw-PTX tcgen05 (kind::f16, bf16) variant of the same architecture: M = 128 rows, Q and P in TMEM (TS MMA), S and P double-buffered in TMEM, 4 softmax warps
(one thread per TMEM lane, no shuffles), bulk-copy producer, 256 dequant threads, conditional O rescale in TMEM.  Correct (vectors 60/60, plus a fp32 GPU
reference cross-check at ctx 8192).  Performance on B300 B8 ctx32768: 0.350/0.354/0.351/0.362 ms at K=1/2/4/8 in its own bench (K-independent again; skeleton alone,
no dequant and no softmax math, is already 0.20 ms) => only K=8 beats the legacy kernel (0.3074 ms tuned in the final run vs 0.3504 before, 1.14x); K=1/2/4 stay on
variant 2 / 1.  The native block-scaled FP4 QK step (kind::mxf8f6f4) was NOT built.  FlashInfer trtllm-gen NVFP4 (0.066 / 0.094 ms at K=1/8) is still 2.1x / 3.3x faster:
the gap is the per-page serial chain (and dequant) of a software-dequant design, not a missing instruction.  Variant 4 is hooked as `variant=4` (explicit request, fmt 0 only).

### Goal 3: NVFP4 (fmt=1)
`UQParams.fmt/k_ts/v_ts`; scales uint8 [pages,Hkv,PAGE,D/16] (raw e4m3fn), `k_ts` folded into sm_scale at launch, `v_ts` applied in the reduce kernel.  The E4M3 scale
is decoded exactly (`fp16(b<<7) * 256`, same trick as `uqmt/nvfp4.py`, then exact to bf16; fp4 x e4m3 has <= 5 significant bits so the bf16 product is exact).
Implemented in variant 1 (all GPUs, `GS=16` template) and variant 3 (H100: 256-entry E4M3 -> bf16x2 table in smem, two scales per 32-element chunk).  Variant 2 has no NVFP4
form (fmt=1 requests fall back to variant 1).  Vectors: `tools/cuda2_gen_vectors_nvfp4.py` (snapshot of `quantize_nvfp4`, fp32 reference) -> `cuda/vectors_nvfp4/`,
harness `bench/cuda2/test_vectors_nv.cu`: **H100 2016 launches (variants 1 and 3) 0 failures**, worst error beyond bf16 rounding 1.7e-3 (`results/cuda2_h100_vectors_nvfp4.log`).

### Goal 4: workspace / CUDA graphs
`uq_workspace_bytes`, `uq_decode_launch_ws` (C), `uq_decode_cuda(..., workspace=, out=)` and `workspace_floats` (Python): scratch and output are caller owned, no allocation and no
sync in the launch path (the one-time `cudaFuncSetAttribute` happens on the first call: warm up before capture).  `bench/cuda2/test_graph.py`: capture + replay is bit-equal to
eager for UE8M0 and NVFP4 at K = 1, 4, 8 with changed `seq_lens`/q (the capture allocates nothing beyond the empty-capture baseline).  D = 256 not done.

### Goal 5: real model
See the tables in the "cuda2" section of `README.md` (perplexity Triton vs CUDA through Qwen3-8B on H100/B300; whole decode step vs fp16 SDPA on Mistral-7B, B8 ctx32768).

### Goal 5 results and a default-config bug
Perplexity (Qwen3-8B, H100) and whole-step times (Mistral-7B, B8 ctx32768, K=1/8, H100 and B300) are tabulated in `README.md` ("Real-model checks"); raw data `results/cuda2_quality_qwen3_8b_*.json`, `results/cuda2_e2e_mistral7b_*.json`.
Summary: ppl within ~0.2% of the Triton kernels (same size as the Triton K=1 vs K=8 spread); H100 step 14.4 / 16.2 ms vs fp16 SDPA 22.0 / 34.8 ms (1.5x / 2.15x), B300 11.1 / 14.8 vs 32.8 / 40.7 ms.
The first B300 K=8 whole-step run (103 ms) exposed that `variant=0` with >= 32 rows dispatched design W with MB=2 (700 B of stack, ~2.7 ms per call); the default now uses variant 1 for BM >= 32 (`dispatch_dp`).
