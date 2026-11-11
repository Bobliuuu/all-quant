# UltraQuant native CUDA kernel

Fused multi-token (K-query) FP4 / UE8M0 paged decode attention in CUDA C++ (`mma.sync` bf16 tensor cores,
software FP4 dequant, `cp.async` pipelines, split-KV + reduce).  Same op, same cache layouts and same
semantics as the Triton reference `uqmt/attn.py`; each KV page is read **once** for all K queries x GQA heads.

```
kc/vc  uint8 [pages, Hkv, PAGE, D/2]    low nibble = even element (e2m1)
ks/vs  uint8 [pages, Hkv, PAGE, D/32]   UE8M0, value 2^(s-127)
q      bf16  [B, K, Hq, D]              block_table int32 [B, max_pages]
seq_lens int32 [B] including the K spec tokens; query j sees seq_len-K+j+1 keys
```

Supported: D in {64, 128}, PAGE in {16, 32, 64, 128} (D=128, PAGE=64 is the tuned shape), K*pow2(G) <= 128
(K in 1..16 for G=8), ragged `seq_lens`, shuffled page tables, seq_len < K (zero rows), sm_80 and newer
(tested: A100, H100, L40S, B300 sm_103, RTX PRO 6000 sm_120).  Output is bf16 [B,K,Hq,D].  Added 2026-11-11 (cuda2, see the
section at the end): an sm_90a wgmma variant (H100), a tcgen05 variant (B300, explicit request), NVFP4 (E4M3 scale per 16 +
per-tensor fp32, `fmt=1`) and a CUDA-graph friendly workspace API.  Not supported yet: D=256, the native block-scaled FP4 MMA
path (see below).

## Files
| file | what |
|---|---|
| `uq_decode.h`, `uq_decode.cu`, `uq_decode_w.cuh` | kernels + host dispatch (`uq_decode_launch`) |
| `uq_ext.cpp`, `uq_cuda.py`, `setup.py` | PyTorch binding: JIT (`torch.utils.cpp_extension.load`) or `pip install ./cuda` |
| `CMakeLists.txt`, `test_vectors.cu` | CMake build + standalone harness checking `vectors/` (no torch needed) |
| `vectors/`, `../tools/gen_vectors.py` | CPU reference vectors (15 cases, 4.8 MB) and their generator (snapshot of quantize/reference) |
| `bench_cuda.py`, `bench_standalone.cu` | torch benchmark/tuner vs Triton snapshot (`ref/`), and a torch-free tuner with ablation/profiling flags |
| `modal/modal_cuda.py` | Modal runner (`--mode check|bench_cuda|sa|prof|ablate|ptxas`) |
| `PRIOR_ART.md`, `UPSTREAM_COMPAT.md`, `native_fp4/NOTES.md` | prior art, upstream vLLM/SGLang bit-compat, native FP4 MMA study |
| `PORTING_LOG.md`, `../results/cuda_tuning_notes.md` | effort log, tuning notes |

## Build / run
```bash
# standalone harness (CMake; tested on CMake 3.22 + CUDA 12.9, ctest passes on H100; add -DCMAKE_CUDA_ARCHITECTURES=100 for B300)
cmake -S cuda -B build && cmake --build build -j && ./build/test_vectors cuda/vectors 1e-2
# or directly:  nvcc -O3 -std=c++17 -arch=native -o test_vectors test_vectors.cu uq_decode.cu && ./test_vectors vectors 1e-2
# PyTorch
pip install --no-build-isolation ./cuda   # needs torch + nvcc + a host g++ (set CC=gcc CXX=g++ if your python reports clang++); or just: import uq_cuda  (JIT-builds uq_cuda_ext on first use)
```
```python
from uq_cuda import uq_decode_cuda          # same argument order as uqmt.attn.uq_decode
out = uq_decode_cuda(q, kc, ks, vc, vs, block_table, seq_lens, fused=True)      # nsplit auto
out = uq_decode_cuda(..., variant=2, mb=1, nwarps=2, stages=2, nsplit=9)        # explicit tuned config
```
`fused=False` runs K independent single-query problems (KV re-read K times) for the baseline.

## Design (two kernels, picked by the tuner)
* **variant 1 (smem-dequant, `uq_split_kernel`)**: CTA-wide `cp.async` ring; every thread dequantises its share
  of the page to bf16 in smem once, then `ldmatrix` + `mma.m16n8k16`; warps split the BLOCK_M query rows
  (each row-warp reuses the dequantised page) and, when there are few rows, the keys of a page.  Best for
  K >= 4 (BLOCK_M 32/64/128).
* **variant 2 (design W, `uq_split_w`)**: warp-independent, no CTA barriers in the main loop.  Each warp has its own
  `cp.async` ring and processes distinct pages (flash-decoding at warp granularity).  K is dequantised straight
  into the `mma` B-fragment registers with a *permuted reduction order* (thread t of a quad owns the 16 bytes =
  one UE8M0 group of its key row; Q is gathered with the same permutation), V is dequantised straight into the
  PV B fragments (keys paired across registers via `prmt`, scale applied as a bf16x2 pair; output columns are
  permuted and un-permuted in the epilogue).  Best for K = 1, 2 (BLOCK_M = 16).
* **dequant**: `bf16 bits = sign<<15 | code3<<6` (value x 2^-126, exact incl. the 0.5 subnormal) times a power
  of two built from the UE8M0 byte (two multiplies only when scale > 2^1).  Hardware `cvt.rn.f16x2.e2m1x2` exists only on
  sm_100+ and was not tried (it would also force fp16 operands; untested).
* **split-KV**: grid (B, Hkv, nsplit), nsplit any integer 1..128, split boundaries per sequence; partial
  (o, lse) in fp32 scratch; one warp per output row reduces them.  Empty splits write only `lse = -inf`.

## Correctness
* `test_vectors` sweeps every launchable config (both variants, mb, warps, stages, nsplit in {1,3,8,16}) over 15
  reference cases (K in 1,2,3,4,8; GQA 4/8; D 64/128; PAGE 16..128; ragged; shuffled pages; seq_len < K):
  **0 failures on A100, H100, L40S, B300, RTX PRO 6000** (1260-1856 launches each; logs `results/cuda_*_vectors_check.log`).
* Error vs the fp32 CPU reference: **max abs 1.1e-4 .. 1.5e-4 at the benchmark shape (B8 ctx32768, K 1..8)**, all five GPUs
  (`cuda_err` in `results/cuda_*_b8_c32k.json`), i.e. inside the 1e-3 target.  On the small test vectors the worst abs error is
  9.7e-3: the outputs there are O(1) (few keys) and the bf16 rounding of P and of the output alone gives ~2^-8 relative error, so the
  harness tolerance is `1e-2 + 2^-6 |ref|`; both kernels and Triton have the same behaviour (the error is identical across all
  configs, i.e. it is arithmetic, not indexing).
* Upstream vLLM/SGLang cache bytes are layout-compatible (pure permute); see `UPSTREAM_COMPAT.md` for the
  quantiser difference (their scale rule is `0.156*amax` round-half-up, not `ceil(log2(amax/6))`), and for
  the decode comparison (1 bf16 ulp).

## Results: B8, ctx 32768, Hq 64, Hkv 8, D 128, PAGE 64 (285 MB of FP4 KV per step)
Best tuned config per K (grid over variant, warps, stages, nsplit, mb); same-job tuned Triton snapshot for comparison.
`% peak` = FP4 KV bytes / time vs the GPU's `torch copy_` bandwidth (read+write; on GDDR GPUs `copy_` under-reports the read-only peak, hence >100% on L40S).

| GPU (HBM copy peak) | K | CUDA ms | GB/s | % peak | Triton ms | CUDA vs Triton | CUDA fused vs CUDA K-independent |
|---|---|---|---|---|---|---|---|
| B300 (6570 GB/s) | 1 | 0.1395 | 2045 | 31.1% | 0.3082 | 2.21x | 1.1x |
| | 2 | 0.1419 | 2010 | 30.6% | 0.3103 | 2.19x | 2.1x |
| | 4 | 0.2527 | 1129 | 17.2% | 0.4251 | 1.68x | 2.1x |
| | 8 | 0.3504 | 814 | 12.4% | 0.4944 | 1.41x | 2.9x |
| H100 80GB (3033) | 1 | 0.1614 | 1767 | 58.2% | 0.3714 | 2.30x | 1.0x |
| | 2 | 0.1632 | 1748 | 57.6% | 0.3725 | 2.28x | 2.0x |
| | 4 | 0.2906 | 981 | 32.4% | 0.4969 | 1.71x | 2.1x |
| | 8 | 0.4310 | 662 | 21.8% | 0.4534 | 1.05x | 2.7x |
| A100 80GB SXM (1749) | 1 | 0.2588 | 1102 | 63.0% | 0.7300 | 2.82x | 1.1x |
| | 2 | 0.2676 | 1066 | 60.9% | 0.7377 | 2.76x | 1.9x |
| | 4 | 0.4576 | 623 | 35.6% | 0.9824 | 2.15x | 2.1x |
| | 8 | 0.7085 | 403 | 23.0% | 1.4045 | 1.98x | 2.6x |
| L40S (643) | 1 | 0.3902 | 731 | 114% | 0.4360 | 1.12x | 1.0x |
| | 8 | 0.5230 | 545 | 85% | 0.6249 | 1.19x | 2.4x |
| RTX PRO 6000 (1462) | 1 | 0.2055 | 1388 | 95% | 0.2689 | 1.31x | 1.0x |
| | 2 | 0.2082 | 1370 | 94% | 0.2711 | 1.30x | 1.35x |
| | 4 | 0.2767 | 1031 | 70% | 0.3829 | 1.38x | 1.8x |
| | 8 | 0.3615 | 789 | 54% | 0.4597 | 1.27x | 2.7x |

(full per-K rows incl. L40S K=2,4: `results/cuda_summary_b8_c32k.json`.)  The "K-independent" column is our own kernel run K times
on B*K pseudo-sequences, i.e. what fusing the K queries buys on the same hardware.

### Where this stands against other NVIDIA kernels (B300, same shapes; from `PRIOR_ART.md`, run in a separate image)
| implementation | K=1 ms | K=2 | K=4 | K=8 | K queries share one KV read | KV format |
|---|---|---|---|---|---|---|
| **this CUDA port** | 0.140 | 0.142 | 0.253 | 0.350 | yes | UltraQuant FP4 + UE8M0/32 (bf16 q) |
| Triton reference (same job) | 0.308 | 0.310 | 0.425 | 0.494 | yes | same |
| BitDecoding-FP4 (K independent decodes) | 0.068 | 0.131 | 0.261 | 0.521 | no | NVFP4 (E4M3/16) |
| FlashInfer trtllm-gen NVFP4, fp8 q | 0.066 | 0.077 | 0.097 | 0.094 | yes | NVFP4 (E4M3/16 + fp32), fp8 q |
| FlashInfer trtllm-gen fp8 KV, fp8 q | 0.080 | 0.081 | 0.083 | 0.087 | yes | fp8 e4m3 (2x the bytes) |

**Honest reading**: on B300 the vendor-tuned NVFP4 kernels are 2.1x (K=1) to 3.7x (K=8) faster than this port; they use the
tcgen05 tensor cores / TMA, a different scale format and fp8 queries.  What this port offers is the exact UltraQuant
format (UE8M0 per 32, bf16 queries, any page size/ragged/verify masks), K-query sharing, and one code base across
A100/H100/L40S/B300/RTX PRO 6000 (and the sibling SYCL/HIP/Triton ports); it is 1.05x-2.8x faster than the Triton
kernel everywhere.  We do not claim to beat FlashInfer/BitDecoding NVFP4 on B300.

## What limits it (details in `../results/cuda_tuning_notes.md`)
* Load-only ablation (all compute removed) reaches 79% (B300) / 87% (H100) of HBM peak, so the pipeline/loader is fine;
  the rest is instruction issue/latency of software FP4 dequant + `mma.sync` at the occupancy these register budgets allow.
* On B300 legacy `mma.sync` bf16 tops out near 190 TFLOP/s: K=8 needs ~69 GFLOP/step = 365 us at that rate, and the kernel
  runs at 350 us, i.e. **K=8 on B300 is at the `mma.sync` ceiling**; going further needs tcgen05.
* Native FP4 block-scaled MMA (`native_fp4/NOTES.md`): on B300 only `tcgen05.mma` has block-scaled forms (`mma.sync ...
  block_scale` is rejected by ptxas on sm_103a); measured with CUTLASS ~1.5x over the software-dequant QK at M=64, projected
  1.2-1.6x for the whole kernel, not built fused.  On RTX PRO 6000 (sm_120a) `mma.sync.kind::mxf8f6f4.block_scale` QK^T is
  validated (1.4-2x faster than software dequant for QK).  PV cannot use block scales (V's scale is along the output
  axis, not the reduction axis).

## cuda2 update (2026-11-11): H100 wgmma path, B300 tcgen05 variant, NVFP4, graph-friendly API
Full account (design, ablations, what failed): `PORTING_LOG.md` addendum.  Data: `../results/cuda2_summary.json` (same schema as `cuda_summary_b8_c32k.json`, plus `prev_cuda_fused_ms`).

New kernels / features
* **variant 3 (H100, sm_90a)**: warp-specialised wgmma kernel (`uq_decode_sm90.cuh`): bulk-copy (1D TMA) producer warpgroup, 2 dequant warpgroups (LUT dequant to 128B-swizzled bf16 tiles),
  2 MMA warpgroups (`wgmma.m64n64k16` with Q in registers, `m64n128k16` with P in registers and V MN-major, lazy softmax rescale, `setmaxnreg`).  D = 128, PAGE = 64, K*pow2(G) <= 64.  Auto-selected on sm_90
  (build with `-gencode arch=compute_90a,code=sm_90a -DUQ_ENABLE_SM90`; `uq_cuda.py` / `setup.py` do this on H100) for >= 32 query rows; `variant=3, nwarps=#MMA WGs, stages, mb=#dequant WGs` to force.
* **variant 4 (B200/B300, sm_100a/103a)**: raw-PTX tcgen05 bf16 UMMA kernel with TMEM accumulators (`uq_decode_sm100.cuh`, built with `-DUQ_ENABLE_SM100`, request with `variant=4`).  Correct; only faster than the
  mma.sync kernels at K = 8 (1.14x); see the log.
* **NVFP4** (`fmt=1`, `k_ts`, `v_ts`): E4M3 scale per 16, per-tensor fp32; layout of `uqmt/nvfp4.py`; variants 1 (all GPUs) and 3 (H100).  Vectors: `vectors_nvfp4/` (`tools/cuda2_gen_vectors_nvfp4.py`).
* **Graph-friendly API**: `uq_decode_cuda(..., workspace=ws, out=out)`, `workspace_floats(...)`, C: `uq_workspace_bytes`, `uq_decode_launch_ws`; captured/replayed bit-exact (`bench/cuda2/test_graph.py`).

B8, ctx 32768, Hq 64 / Hkv 8, D 128, PAGE 64 (285 MB of FP4 KV per step), tuned, same-job Triton snapshot (UE8M0/32, bf16 q):

| GPU | K | previous CUDA ms | **cuda2 ms** | speedup | cuda2 % of copy peak | Triton ms | cuda2 vs Triton | best variant |
|---|---|---|---|---|---|---|---|---|
| H100 | 1 | 0.1614 | 0.1633 | 1.0x | 58% | 0.3739 | 2.29x | 2 |
| | 2 | 0.1632 | 0.1635 | 1.0x | 58% | 0.3777 | 2.31x | 2 |
| | 4 | 0.2906 | **0.2145** | 1.35x | 44% | 0.4938 | 2.30x | 3 (wgmma) |
| | 8 | 0.4310 | **0.2152** | **2.00x** | 44% | 0.4562 | 2.12x | 3 (wgmma) |
| B300 | 1 | 0.1395 | 0.1388 | 1.0x | 31% | 0.3076 | 2.2x | 2 |
| | 2 | 0.1419 | 0.1416 | 1.0x | 31% | 0.3096 | 2.2x | 2 |
| | 4 | 0.2527 | 0.2549 | 1.0x | 17% | 0.4250 | 1.67x | 1 |
| | 8 | 0.3504 | **0.3074** | 1.14x | 14% | 0.4945 | 1.61x | 4 (tcgen05) |
| A100 / L40S / RTX PRO 6000 | 1-8 | | unchanged within noise (0.95-1.06x) | | | | | 1 / 2 |

NVFP4 (E4M3/16 + per-tensor, same shape; Triton = `uqmt/nvfp4.py`): H100 K=1/2/4/8 = 0.275 / 0.275 / **0.214** / **0.216** ms (1.4x / 1.4x / 2.3x / 2.4x over Triton NVFP4), B300 0.235 / 0.236 / 0.280 / 0.373 ms
(1.3-1.5x), A100 0.458 / 0.470 / 0.519 / 0.816 ms (1.3-1.55x); max abs error vs the fp32 reference 1.1e-4 .. 1.6e-4 (`results/cuda2_*_nvfp4_b8_c32k.json`).
Correctness: vector harness 0 failures on A100 (1756 launches), H100 (2096, incl. variant 3), L40S (1260), B300 (1856), RTX PRO 6000 (1260) and 2016 NVFP4 launches on H100 (`results/cuda2_*_vectors*.log`).  The H100
variant 3 rounds P up to 2^8 in bf16 (lazy rescale) so its worst error beyond bf16 rounding on the tiny test vectors is 2.8e-3 (was 2.1e-3), the benchmark-shape error stays 1.4e-4.

Honest status vs the vendor kernels: B300 K=8 0.307 ms vs FlashInfer trtllm-gen NVFP4 0.094 ms (3.3x slower; was 3.7x) and K=1 0.139 vs 0.066 (2.1x, unchanged).  The tcgen05 kernel exists and is correct, but a software-dequant
design is bound by the per-page S -> softmax -> PV chain and dequant issue, not by the MMA: even the skeleton without dequant and softmax is 0.20 ms.  Native block-scaled QK (mxf8f6f4) is still not built.

Real-model checks (Qwen3-8B / Mistral-7B-v0.1, `bench/cuda2/quality_cuda2.py`, `bench/cuda2/modal_cuda2_model.py`; raw: `results/cuda2_quality_*`, `results/cuda2_e2e_*`)
* **Perplexity through the kernels** (wikitext-2, 8 windows x 2048, 1024-token chunked prefill then teacher-forced decode of the remaining tokens in steps of K query tokens, every layer's decode attention on the paged quantised cache;
  Triton uqmt kernels vs this CUDA port, H100): fp16 8.5228 | UE8M0 K=1 Triton 9.1031 vs CUDA 9.1185 (+0.17%), K=8 9.0788 vs 9.0866 (+0.09%, variant 3) | NVFP4 K=1 8.6089 vs 8.6250 (+0.19%), K=8 8.5977 vs 8.5926 (-0.06%).
  So perplexity is unchanged to within ~0.2%, which is the same size as the spread between the *Triton* K=1 and K=8 runs (0.27% / 0.12%: different GEMM shapes in the rest of the fp16 model); it is not bit-identical.
  B300 shows the same picture (UE8M0 K=1 7.0844 vs 7.0658, K=8 7.0584 vs 7.0428; NVFP4 6.7511 vs 6.7521, 6.7366 vs 6.7328) but its absolute ppl (fp16 6.61 vs 8.52 on H100) is NOT comparable to H100: the chunked-prefill SDPA path of
  the B300 torch build gives a different baseline, independent of the decode kernels; only the within-GPU backend comparison is meaningful.
* **Whole decode step, Mistral-7B, B=8, ctx 32768** (eager HF-module layer loop, CUDA-graph replay time, ms; `bench/e2e_decode.py --mode sweep` with the CUDA port behind the `bench/cuda2/shim`; KV 34.4 GB fp16 vs 9.1 GB FP4):
  | GPU | K | fp16 SDPA | fp16 bmm | UltraQuant CUDA (fused) | vs fp16 SDPA |
  |---|---|---|---|---|---|
  | H100 | 1 | 22.0 | 44.7 | **14.4** | 1.52x |
  | H100 | 8 | 34.8 | 47.0 | **16.2** | 2.15x |
  | B300 | 1 | 32.8 | 33.3 | **11.1** | 2.95x |
  | B300 | 8 | 40.7 | 26.7 | **14.8** | 2.75x |
  K=8 "kernel-independent" (the same kernel run K times) is 50.4 ms on H100 / 42.5 ms on B300, i.e. K-query sharing is worth 3.1x / 2.9x at the step level.  B300 fp16 SDPA is slow in this torch build (the bmm formulation is faster at K=8), so the B300 ratios
  flatter us.  A first B300 K=8 run took 103 ms because the *default* configuration (variant 0) picked the spilling warp-independent design for >= 32 query rows; fixed (variant 0 now uses variant 1 there; the tuned configs are unchanged).
