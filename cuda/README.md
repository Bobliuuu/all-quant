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
(tested: A100, H100, L40S, B300 sm_103, RTX PRO 6000 sm_120).  Output is bf16 [B,K,Hq,D].  Not supported
yet: D=256, a CUDA-graph workspace API (the extension allocates scratch per call unless `mo/ml` are passed),
the native block-scaled FP4 MMA path (see below).

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
