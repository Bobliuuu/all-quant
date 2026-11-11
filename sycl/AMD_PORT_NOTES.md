# AMD (MI300X) TurboQuant port: factual notes for SYCL/Arc comparison

Source: `~/amd-experiments-ro` (clone of Bobliuuu/amd-experiments), read-only. Paths below are relative to that repo. Draft, not committed.

## 1. Timeline (from `git log`)
- 15 commits. First 2026-04-13 (`31743be` "inital commit"); last code commit 2026-04-26 (`ef7444c`); `27a7c17` (2026-11-07) is only a .gitignore chore. Active span 13 days (Apr 13 to Apr 26); 4 commit-days after day 1 (Apr 15, 19, 25, 26).
- Plan (`plan.md`, 776 lines) budgeted 25 days / 6 phases: env, baselines, HIP port (days 6-10), fused Triton attention (11-15), vLLM integration (16-20), report (21-25). Actual history does not follow it (see caveat).
- Milestones in order:
  - Apr 13 (7 commits: `31743be`, `044f2c9`, `c8c7341`, `45213ae`, `9871379`, `480ae12`, `a542376`): repo, HIP kernels, Triton kernels, benchmarks, report and figures, all pushed in a day.
  - Apr 15 `ea03c21` "add everything" (177 files, +219k/-170k lines) and `18e95eb`: vLLM V1 backend, scripts, results.
  - Apr 19 `634389c` "fix all benchmarks", `0ce79d6`: Phase D whole-step baseline and rocprof closure (`docs/decode_whole_step_amdahl_outcome.md`, dated 2026-04-19).
  - Apr 25 `8a04ca0`: `--swa` flag.
  - Apr 26 `93e2b50` "untested gqa kernels, theoretically they make sense to me" (+396 lines in `kernels/tq_triton.py`), `ef7444c`: GQA runbook.
- Caveat: commits are bulk dumps (squashed or AI-generated working trees). No per-iteration history exists, so time-to-first-correct and per-kernel iteration counts cannot be recovered from git. The GQA kernel is explicitly UNTESTED at HEAD (`docs/gqa_kernel_test_runbook.md`: "Step 1 correctness, must pass before anything else"). No GQA results are checked in except `results/bench_tq_gqa_decode_sweep.json`.

## 2. Approaches tried
1. **HIP C++ standalone** (`kernels/hip/`): ported from the `domvox/turboquant-hip` reference (RDNA3/gfx1100, mirrored in `kernels/ref/`) to gfx942. Quantize, dequantize, fused-dot, QJL kernels, and a C API plus tests.
   - Build: `hipcc --offload-arch=gfx942:sramecc+:xnack- -mwavefrontsize64 -DCDNA3 -DAMD_MFMA_AVAILABLE` (`kernels/hip/build_mi300x.sh`).
   - MI300X-specific changes (`report/mi300x_vs_turboquant_hip.md`):
     - Wave64 `__ballot` bit-packing replaces LDS `atomicOr`.
     - Register-resident indices instead of an `s_indices` LDS array.
     - Wavefront-leader 64-bit plane writes.
     - Fixed Wave64 reduction offsets (32 down to 1).
     - New QJL key-residual kernel (`tqm_qjl_kernel`).
2. **HIP MFMA rotation kernel** (`kernels/hip/tq_mfma_rotate.hip.cpp`, `mfma_f32_16x16x16f16`) loaded as a COV5 HSACO through `kernels/tq_mfma_loader.py` and `tq_hsaco_loader.py`. Result: 0.78-0.90x of `torch.matmul` (rocBLAS) at every size (`results/bench_mfma_rotate.json`; e.g. n=65536: 84.2 us vs 65.8 us). Correct (cos 1.0, max err 1.7e-3) but not faster, so the Python path stays on rocBLAS (`kernels/turboquant_mi300x.py`, pure PyTorch).
3. **Triton** (`kernels/tq_triton.py`): the main production path. Kernels:
   - v2 bit-plane fused attention `_tq3_attention_kernel`
   - v3 compact-bitplane `_tq3v3_attention_kernel`
   - nibble variant `_tq3_nibble_attention_kernel`
   - Split-K `_tq3_splitk_partial_kernel` + `_reduce_kernel`
   - GQA `_tq3_gqa_attention_kernel` + `_gqa_splitk_partial_kernel` (untested)
   - Block-method (Planar/Iso/Rotor) quant kernels in `kernels/block_quant_rocm.py` (886 lines).
4. **Pure PyTorch fallback** (`kernels/turboquant_mi300x.py`, 481 lines): used for every end-to-end benchmark.
5. **vLLM V1 integration** (`tq_backends/`): a `CUSTOM` attention backend bridge plus `scripts/install_turboquant_vllm_backend.sh`, a `tq3` `CacheDType` patch, and `scripts/patch_vllm_rocm_sliding_window_custom_paged.py`.
6. Not tried: CK/aiter custom kernels, hipBLASLt epilogue fusion (documented as "outside repo").

## 3. Lines of code (`wc -l`)
| File | LOC |
|---|---|
| `kernels/hip/turboquant_mi300x.hip.cpp` | 713 |
| `kernels/hip/turboquant_mi300x.h` | 283 |
| `kernels/hip/turboquant_mi300x_test.cpp` | 718 |
| `kernels/hip/tq_hip_benchmark_mi300x.cpp` | 283 |
| `kernels/hip/tq_mfma_rotate.hip.cpp` | 215 |
| `kernels/hip/build_mi300x.sh` | 138 |
| `kernels/tq_triton.py` | 1834 |
| `kernels/block_quant_rocm.py` | 886 |
| `kernels/turboquant_mi300x.py` | 481 |
| `kernels/tq_hsaco_loader.py` + `tq_mfma_loader.py` | 594 + 499 |
| `tq_backends/vllm_v1_{turboquant,isoquant}_bridge.py` | 209 + 198 |

- HIP kernel .cpp+.h: ~996 lines. All HIP C++ with tests and bench: 2350.
- Triton attention kernel file: 1834 lines, of which the GQA addition is ~396 (`93e2b50`).
- Triton `@triton.jit` bodies, by line range in `kernels/tq_triton.py`: v2 `_tq3_attention_kernel` ~206 (116-321), v3 ~178 (323-499), nibble ~127, split-K partial ~144, split-K reduce ~80, GQA ~110 plus split-K GQA ~134.
- `benchmarks/*.py`: about 19k lines in total (about 40 scripts). Docs and reports are far larger than the kernel code.

## 4. Correctness methodology and tolerances
- HIP (`kernels/hip/turboquant_mi300x_test.cpp`, 9+ tests, exit 0 on pass; unit-test thresholds are looser than the quoted averages):
  - T2: TQ3 round-trip min cosine over 256 samples > 0.92 (average about 0.983 expected).
  - T4: TQ4 min cosine > 0.95 (average about 0.996 expected).
  - MSE within +/-15% of the analytic expectation (tol 0.15).
  - Codebook equals the domvox values; QJL sign and residual sanity; Wave64 assumption check.
- Reconstruction: random N(0,1), 512 vectors, head_dim=128. Cosine thresholds >= 0.97 (3-bit) and >= 0.99 (4-bit); 6/6 pass (`report/final_report_v2.md` section 2.2).
- Triton attention checked against FP16 SDPA and a Python decompress+SDPA path by cosine (`benchmarks/validate_triton_e2e.py`).
- GQA validation (`benchmarks/validate_tq_gqa_fused_decode.py`, 12 cases): cos(gqa, decompress) >= 0.92 and cos(gqa, expand) >= 0.99. Not recorded as run at HEAD.
- Model level: PPL (WikiText-2, Mistral-7B: TQ4 7.77 vs FP16 7.82, per `docs/swa_audit_and_gqa_plan.md`) and NIAH up to 65k.
- Methodology is cosine and statistical, not bit-exact. A later audit (`report/adversarial_data_provenance_audit.md`, `docs/swa_audit_and_gqa_plan.md`) found two broken PPL benches and that synthetic benches ignore the sliding window.

## 5. Performance achieved
- Fused TQ3 Triton Split-K decode attention, B=1, H=32, D=128 (`results/bench_triton_attention.json`, `_splitk.json`; MI300X VF):
  | seq_k | FP16 SDPA (ms) | fused (ms) | FP16/fused | eff BW (GB/s) |
  |---|---|---|---|---|
  | 1024 | 0.045 | 0.163 | 0.28x | 21 |
  | 4096 | 0.185 | 0.411 | 0.45x | 33 |
  | 16384 | 0.721 | 0.443 | 1.63x | 123 |
  | 32768 | 1.58 | 0.831 | 1.91x | 131 |
  | 65536 | 3.16 | 1.67 | 1.89x | 130 |
  | 131072 | 6.31 | 2.99 | 2.11x | 146 |
  - The split-K file shows a lower 1.86x at 131072.
  - Crossover with FP16 at about 16K tokens.
  - Fused is about 20x faster than Python decompress+SDPA at 64K.
- "Eff BW" is about 146 GB/s of 5300 GB/s (about 2.8%). Stated in the report: "kernels are not memory-BW limited... launch overhead and instruction latency". The FP16 baseline reached 0.69 TB/s (13% of peak) using `torch.bmm`, not flash attention (`report/final_report_v2.md` line 474).
- GQA fused vs expand+decompress (`results/bench_tq_gqa_decode_sweep.json`): 6.3-7.5x faster (fused 5.6-77.8 ms for 1K-16K). That "speedup" is measured against a decompress path, not FP16.
- HIP rocprofv3 (`report/paper.md` Table 4): TQ3 compress 3122 us, 128 VGPR (about 25% occupancy); decompress 171 us, 76 VGPR, about 200 GB/s (3.8% of peak); fused dot 39 us, 12 VGPR.
- Block-kernel GB/s: Iso 21.8 compress / 38.3 decompress; Turbo 2.9 / 4.4 (WHT rotation bound).
- vLLM Mistral-7B, kv-heavy (1024 in / 256 out / 32 prompts, `docs/decode_whole_step_amdahl_outcome.md`): FP16 ~1763 tok/s; TQ decompress ~1933; TQ fused ~1914 (+8-10%, within noise; some other runs show parity).
  - Rocprof: ~43% hipBLASLt GEMM, ~30% paged attention, so Amdahl-limited.

## 6. Pitfalls, dead ends, bugs
- **Code-object version (COV) mismatch**: system `hipcc` on ROCm 7.2 emits COV6; the PyTorch-linked HIP runtime rejects it with `hipErrorNoBinaryForGpu (209)` (`kernels/hip/README.md`, `kernels/tq_hsaco_loader.py`). Fixes: `-mcode-object-version=5 --genco` HSACO plus `hipModuleLoad`, or avoid ctypes entirely. Final decision: standalone HIP only validates in the container; Python uses Triton or PyTorch.
- **Arch string must include features**: `gfx942:sramecc+:xnack-`; mismatch also gives 209 (`report/mi300x_vs_turboquant_hip.md` section 6).
- **Wave64**: reference port assumed `warpSize`-generic or atomic LDS packing; rewrote to 64-bit ballot (two masks per 128 dims) and fixed shuffle offsets.
- **MFMA hand kernel lost to rocBLAS** (section 2 above). The rotation (WHT/dense 128x128) dominates TQ compress, which is 6-9x slower than Planar/Iso.
- **Triton v3 (compact bit-plane) was 10% slower than v2** despite 8x fewer VMEM instructions (3D reshape VALU cost, `report/paper.md` lines 147-156). Real bottleneck was grid parallelism: B=1 with 8 KV heads gives 8 wavefronts on 1216 SIMDs (0.66%). Fix: Split-K. BLOCK_KV sweep: 1024-4096 equal, 8192 regressed (1.50 ms vs 0.83).
- **GQA**: Triton kernel needed `repeat_interleave` expansion first (`expand_tq_compressed_for_gqa`); native GQA kernel written but untested.
- **Profiler**: counters (FETCH_SIZE, VALU) blocked on VF (virtualized) MI300X (`ROCPROFILER_STATUS_ERROR_PROFILE_EXCEEDS_HW_LIMIT`); only the kernel timeline was available.
- **Stack**:
  - vLLM wheel repins torch; `uv pip install vllm` fails on URL deps (`docs/rocm72_uv_torch_vllm_venv.md`).
  - In-repo `vllm/` stub shadowed PyPI vLLM.
  - V1 multiprocessing hid backend registration (needs `VLLM_ENABLE_V1_MULTIPROCESSING=0`).
  - vLLM `gpu_memory_utilization` 0.85 failed mid-run, 0.15 worked (allocator snapshot).
  - CUDA-graph leg aborts (exit=-6).
  - ROCm custom paged attention spuriously disabled when sliding_window == max_len-1 (patched).
- Negative results: SwiGLU Triton fusion no_go; AWQ ~35% slower.
- Environment dependence: all numbers from one shared VF MI300X in a Primus docker (`rocm/primus:v26.2`, ROCm 7.2, torch 2.10, vLLM 0.19). `docker_run_amd_mi300x.sh` and the frozen venv are required for reproduction.

## 7. Reused vs rewritten
- Reused: algorithm and codebooks from TurboQuant paper and `domvox/turboquant-hip` (HIP structure, 128-dim bit-plane layout, 52 B/vector TQ3 block, 4.92x ratio); `rotorquant` Triton design pattern; stock vLLM V1, torch SDPA/rocBLAS as baselines.
- Rewritten: HIP kernels re-done for Wave64/MFMA; QJL kernel added; fused dequant+attention (Triton, original); the vLLM bridge; the Split-K and GQA kernels.
- Triton, being portable, is the only part that ran end-to-end. HIP stayed standalone.

## 8. What worked
- Triton Split-K fused TQ3 attention: correct (cos match to 4 decimals) and 1.6-2.1x faster than FP16 SDPA for isolated attention at >= 16K context.
- 4.92x KV capacity and PPL parity.
- vLLM end-to-end wiring (tok/s parity to +10%).
- HIP standalone test suite passes in the container.

## Questions to answer for the SYCL comparison
1. LOC per component (kernel, host/launch, tests, build): AMD ~996 HIP kernel LOC (.cpp+.h) / 1834 Triton / 2350 all HIP C++.
2. Calendar time and number of agent iterations to first correct output; AMD cannot be extracted from git (bulk commits). Define for SYCL via the commit log.
3. Correctness tolerance: cosine vs FP16 reference (AMD >= 0.92 to 0.99) vs UltraQuant's own metric (max abs err or logit match). Same input shapes (B=1, H=32, D=128)?
4. Time-to-X%-of-peak-bandwidth: AMD best was about 2.8% of 5.3 TB/s (146 GB/s) for fused attention; FP16 baseline 13%. Report ours as % of Arc B-series peak BW and vs an FP16/oneDNN SDPA baseline.
5. Latency vs context length and crossover point against FP16 (AMD about 16K); tokens/s on a real model.
6. Parallelism lesson: AMD B=1 decode starved by grid size (8 wavefronts on 1216 SIMDs); did SYCL need split-K or sub-group tuning, and what was the work-group count vs Xe core count?
7. Toolchain pain: compiler/ABI/runtime mismatches (AMD COV5/COV6 error 209, arch-string features, wheel repins) vs oneAPI/Level Zero/IGC issues; counts of workarounds and hours lost.
8. Portability: single source for all targets? AMD needed separate HIP (Wave64, MFMA, ballot) and Triton paths. Report SYCL sub-group size (16/32), XMX/joint_matrix use and fallbacks.
9. Profiling observability: AMD counters blocked on VF, timeline only; what Arc offered (VTune/unitrace), and whether occupancy/BW attribution was possible.
10. Integration depth and end-to-end effect: AMD vLLM tok/s was parity to +10% (Amdahl: 43% GEMM, 30% attention); report whole-step share of attention and the end-to-end tok/s delta for SYCL, plus reproducibility (pinned stack, number of commands to rerun).
