# Porting log: UltraQuant fused K-query FP4 decode -> SYCL (Intel Arc)

Effort record (2026-11-11 overnight). Agent-driven; wall clock approximate from the session.

## Timeline and iterations
| # | what | LOC (cumulative kernel hdr) | result |
|---|---|---|---|
| 0 | inventory both Intel boxes; discover DPC++ install route | - | box 1: Arc B390 Xe3 driver 32.0.101.9033, no oneAPI; box 2: Arc 140V (Lunar Lake) GPU in Windows error 39 (driver 32.0.101.5564 from 2024), unusable without admin driver reinstall (restart-device denied by policy) |
| 1 | DPC++ v7.2.0 `sycl_windows.tar.gz` (372 MB) extracted to user dir; MSVC (VS 18 Community) for linking; `sycl-ls` sees Level Zero + OpenCL GPU | - | ~25 min including discovery; no admin needed |
| 2 | vector generator (CPU torch, 8 cases), portable kernel (`uq_decode.hpp`), harness | ~270 | first run on GPU: 5/7 pass, 2 "fail" only because my tolerance ignored bf16 P rounding; fp32 P -> all pass. Time to first correct: ~1.5 h session time |
| 3 | benchmark harness + copy-kernel peak | +160 | portable: 1-3% of read peak, ALU bound |
| 4 | XMX joint_matrix path | +190 | first compile+run correct on first attempt; 3.5x faster |
| 5 | bit-fold dequant, LUT dequant | | 4.2 -> 3.5 ms (B8 ctx16k K4) |
| 6 | tuning sweeps; 3-barrier and 16B-load variants (both slower, reverted) | | wg256/tn32 best: 3.19 ms |

Code size: `uq_decode.hpp` ~255 lines (portable kernel + reduce + dispatch), `uq_xmx.hpp` ~200, harness `test_vectors.cpp` ~90, `bench.cpp` ~170, generator ~120.

## Pitfalls hit (in order)
- `scp` is unavailable on the Windows boxes: ship sources with `tar czf - | ssh ... tar xzf -` (dev/sync.sh).
- Non-interactive ssh to the jump host needs `BatchMode=yes` or it falls to a missing askpass program; cmd.exe `%`-expansion and `=` in
  `build.bat -DX=1` (cmd splits on `=`: use value-less macros).
- `sycl::property::in_order` moved: `sycl::property::queue::in_order`. `nd_item::barrier` is deprecated: `sycl::group_barrier`.
- `auto a = f<T>(), b = f<U>()` mixed deductions fail; trivial but cost a build round-trip each time (builds are ~40 s on the box).
- Float denormals: the Triton bit-trick (bf16 bits * 2^126) depends on subnormal 0.5; GPUs may flush, so the exact table / integer bit-fold is used.
- bf16 output + bf16 P rounding => max abs error looks bad against 1e-3 on short sequences (cancellation), though it is just rounding.
- Performance surprises: SLM footprint and work-group size interact sharply (wg256 2x faster than neighbours); a "cleaner" 3-barrier
  double-buffered schedule was 2x slower due to SLM-limited occupancy; wider 16B loads were slower (registers).
- Second Intel box GPU driver in error 39 (old 2024 driver): cannot be repaired from a non-admin ssh session.
- sycl-ls on DPC++ release shows the GPU through "Unified Runtime over Level-Zero V2" adapter, no CPU device (no OpenCL CPU runtime installed).

## Toolchain installation pain on Windows/Arc
Low: one 372 MB tarball + existing MSVC. Pain points were only the remote-shell ergonomics (cmd vs PowerShell, no scp, `taskkill` denied,
shared GPU lock) and the second box's broken driver. Compare ROCm: see below.

## Comparison with the earlier TurboQuant-on-AMD port (Bobliuuu/amd-experiments)
Facts from `sycl/AMD_PORT_NOTES.md` (read-only study of the repo; AMD numbers are the repo's own reports, not re-run).
| | AMD TurboQuant (MI300X) | UltraQuant SYCL (Arc B390) |
|---|---|---|
| calendar | 13 active days (Apr 13-26), 15 bulk commits, per-iteration history not recoverable | one overnight session, commits per milestone |
| kernels | HIP C++ ~1.0k LOC (+1.3k tests/bench), Triton 1834 LOC, only Triton ran end-to-end | one SYCL source ~450 LOC kernels (portable + XMX), one harness |
| correctness | cosine thresholds (>=0.92..0.99), statistical | bit-exact vectors, max abs err vs fp32 reference, per-case, with edge cases |
| perf vs peak | fused attention ~2.8% of 5.3 TB/s (146 GB/s), crossover with FP16 at ~16K ctx | ~10-13% of read peak (45-58 GB/s of FP4 KV) |
| toolchain pain | COV5/COV6 mismatch (error 209), arch-string features, wheel repins, vLLM stack, profiler counters blocked | 372 MB tarball, no admin; remote shell friction; one box with a broken driver |
| portability | separate HIP (wave64, MFMA, ballot) and Triton paths | one SYCL source; portable path + Intel-only XMX path behind a macro |
| end to end | vLLM parity to +10% (Amdahl: attention 30% of step) | not integrated (kernel level only) |
Both efforts end up ALU/latency-bound, far from bandwidth peak, for the same reason: dequantising sub-byte KV costs more issue slots than the bytes cost memory time on this size of decode problem.

## Addendum 2026-11-11 (sycl2 pass): ESIMD/DPAS path, NVFP4, corrected bandwidth ceiling
(earlier text above is kept unchanged; where it quotes "% of read peak" see item 1.)

1. **The read peak was wrong.** The 430-455 GB/s figure came from a read kernel over a uniformly filled buffer; this GPU serves uniform data 2.5-4x faster than random data
   (compression). With random bytes the same kernel, plain SYCL uint4 loads and ESIMD block loads over the paged layout all give 105-113 GB/s (LPDDR5X); torch SDPA and llama.cpp reach 90-112 GB/s.
   `bench` now prints both peaks and uses the random one. B8/ctx16k FP4 KV (134 MB) therefore has a memory floor of ~1.2 ms. The old XMX kernel was at ~50% of the true peak at K=1.
2. **Sub-group-private joint_matrix pipeline (`uq_xmx2.hpp`, `--path xmx2`)**: no work-group barriers, per-sub-group KV stream and online softmax, O in registers. Correct (36/36) but 2.5x SLOWER (6.4 ms vs 2.5 at K=1):
   per-lane SLM LUT gathers + scalar SLM stores dominate. Negative result, kept.
3. **ESIMD path (`uq_esimd.hpp`, `--path esimd`)**: one hardware thread per (b, kv-head, split); 256 B block loads; FP4->bf16 through a 256-entry byte LUT held in 16 GRFs (register-indirect `iselect`);
   UE8M0 scale folded as a 16-bit exponent add on the looked-up words (zero codes decode to +-2^-100, scale byte clamped to [105,250], no mask needed); S^T = K Q^T on DPAS with tokens as M rows and queries as N lanes
   (so the online softmax is lane-parallel, no cross-lane reductions); V decoded straight into the VNNI B operand (token pairs per word); P goes through a 512 B SLM transpose; M up to 16 per thread (RC 4/8, 1-2 row tiles,
   GRF 128 for RC4, 256 otherwise), M>16 splits into query-group threads. No barrier in the hot loop. Debugging notes: a 32-bit add on packed bf16 pairs leaks a carry into the high half (+0.6% bias): all scale adds are 16-bit lane adds.
   Software prefetch gave nothing. 2D transposed block loads and gathers run at 121 GB/s vs 264+ for contiguous blocks, hence the S^T formulation (K tile used in its natural row layout).
   Passes the 36/36 vectors including rotated K/Q (`rot_h128_b2k4`) and the upstream-rule case (`paper_rule_b2k4`). Dispatch `--path best` = ESIMD when supported (UE8M0, D128, PAGE%16==0), else XMX.
4. **NVFP4**: `Args.nv`, scales `uint8 [pages,Hkv,PAGE,D/16]` (raw e4m3fn), exact integer e4m3 decode, k_ts folded into sm_scale, v_ts applied in the reduce kernel; portable and XMX paths (bf16 product exact). 10/10 vectors from
   `bench/sycl2/gen_vectors_nvfp4.py` (frozen copy of the uqmt quantiser; includes rotated, k_ts/v_ts large and small, GQA 1, short rows). ESIMD falls back to XMX for NVFP4.
5. **Results** (`results/sycl2_summary.json`, sweeps `results/sycl2_bench_{xmx,esimd}.jsonl`): ESIMD vs the barrier XMX kernel, best nsplit each: 1.4-1.6x at K=1/2, 2.0x at K=4, 2.1-2.2x at K=8; B8 ctx16k K=1 2.62 -> 1.72 ms (76% of the random-data peak),
   K=4 3.49 -> 1.77 ms (74%), K=8 7.24 -> 3.47 ms (38%). The "3x over the old XMX numbers" target was not reached: the memory floor bounds K=1 to ~2.2x and the K=8 path is ALU-bound.
6. Baselines on the same Arc (random data): torch XPU SDPA routes to oneDNN `micro_sdpa` (90-112 GB/s of dense f16 KV); IPEX 2.8 (needs torch 2.8) paged/dense 82-111 GB/s; llama.cpp `test-backend-ops` built from source (35 min, MKL import lib workaround),
   q4_0 KV at B8/ctx16k n_q=1 2.0 ms, q4_0 n_q=8 cliff 3.9 ms. Files: `results/sycl2_baselines_*`. Full profile/ablation: `results/sycl2_profile.md`.
7. csc-arc-2 (Lunar Lake Arc 140V) still has its GPU in Windows error 39 (CM_PROB_DRIVER_FAILED_LOAD, 2024 driver); `sycl-ls` sees no device; not touched further (no bypass attempted).
