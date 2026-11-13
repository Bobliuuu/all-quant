# Porting log: pq-repro decode attention on CUDA, HIP and SYCL

One algorithm, one cache layout, one set of CPU-generated test vectors
(`tools/gen_vectors.py`, 8 cases: GQA 4/8, MQA 16, b = 14 to 30 bits per group,
T = 1 to 1000, uneven K/V rates). Every backend runs every case for the three formats
(pq, bf16, fp8) at four split counts: 96 checks, pass = relative L2 error < 1e-4 against
float64 attention over the *decoded* cache.

## Order of work and what broke

1. **Decoder on the CPU first.** `tests/test_decode_cpu.cpp` decodes all 2^16 indices at
   b = 16 and 200k random indices at b = 25 and 30 with the shared header and compares with
   the Python `unrank`. 0 mismatches for v1 (branchy), v2 (branch-free) and v3 (SoA + radix).
Timings below are B=8, T=32768, Hq=32, Hkv=8, b=25/24, us per call; each is read from the
file cited next to it (`results/kernels/history/`, `results/kernels/`).

2. **v1 kernel (branchy decode, fp32 smem tile, tables in smem), CUDA on H100 (Modal).** 96/96 on
   the first run. Slow: 6388 us vs 733 us for the same skeleton reading bf16
   (`history/cuda_h100_v1.jsonl`). Per-codeword cost, not bytes, was the problem.
3. **HIP (MI300X).** Same source through hipcc, no edits: 96/96.
4. **SYCL (Arc B390).** Built with DPC++ (`kernels/sycl/build.bat`). First runs failed
   nondeterministically: (a) the default SYCL queue is out-of-order, so the reduce kernel
   could run before the split kernel; fixed with `property::queue::in_order()`. (b) After that
   fix, the v1 serial softmax (one work-item per head loops over 32 tokens) still gave wrong
   results on the Arc iGPU for bf16/fp8 and for pq with nsplit > 1, while the same CUDA and HIP
   code passed. We did not root-cause it; the v3 rewrite (softmax parallel over (head, token))
   passes 96/96 on Arc. Treat it as an open question, not a known compiler bug.
5. **v2 branch-free decode** (prefix-count compare instead of a data-dependent loop with a
   division): first MI300X timing 6704 us (`history/hip_mi300x_v2_branchfree.jsonl`).
6. **v3 SoA tables + radix bucket lookup** instead of a binary search over 32-byte rows: 6198 us
   with the tables in LDS (`history/hip_mi300x_v3_soa_lds_tables.jsonl`). Moving the tables to
   global memory (L1/L2 resident) gave 2688 us (`history/hip_mi300x_v3_tables_lds_vs_global.jsonl`):
   22 KB of tables in LDS had limited occupancy to one 128-thread block per CU.
   `PQ_TAB_SMEM=1` still selects LDS tables (`hip_mi300x_bench_tabsmem.jsonl`).
7. **v4 compact tile**: decoded lattice coordinates stored as exact int8 (2x in [-31, 31]) plus
   one f32 scale per token; bf16/fp8 tiles stay in their storage type and are converted on read.
   LDS per block dropped from ~21 KB to ~9 KB.
8. **ND specialization**: leader tables at b <= 25 have at most 5 distinct |values|, so the
   compare loop runs 5 instead of 8 times and the values fit in 32 bits. Final MI300X timing in
   `hip_mi300x_bench.jsonl` (see the README kernel table). Same change on CUDA and SYCL; every
   backend 96/96 after it.

## Where it ends up

Final timings per device are in the README kernel table and `results/SUMMARY.md`, both generated
from `results/kernels/*bench*.jsonl` by `tools/make_tables.py`.

The pq kernel reads 5x fewer bytes than bf16 (102 vs 512 bytes per token and KV head) but is
ALU-bound on codeword decode at GQA-4: each decoded group of 8 serves only 4 query heads.
The fp8 column is our skeleton with a software e4m3 decode; it is not a tuned FP8 kernel.
On H100/B200 the gap to bf16 is larger than on MI300X; INT32 issue rate is half the FP32 rate on those parts.

Bench data are random bytes (every b-bit index is a valid codeword), random-mantissa bf16 and random fp8
values, so the Arc numbers are not inflated by the iGPU's data-dependent compression (the
ultraquant notes saw ~110 GB/s on random data vs 430+ GB/s on uniform data on this box).
Correctness on 7 device/backend pairs: CUDA H100, CUDA B200, HIP MI300X, SYCL Arc B390, SYCL Arc 140V (final source),
SYCL H100 (CUDA backend), SYCL Xeon CPU (OpenCL): 96/96 each.

## What was not done

- No tensor cores (Pearl decodes into Blackwell tensor cores via CuTe DSL). Our dot products
  are CUDA-core / VALU FMAs.
- No paged KV (contiguous per sequence and head), no fusion with the KV write path.
- Decode is per tile into shared memory; register-level fusion into an MMA fragment was not tried.
