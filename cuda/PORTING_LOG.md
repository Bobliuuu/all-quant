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
