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
2. **v1 kernel (CUDA, H100, Modal).** 96/96 on the first run. Slow: B=8, T=32k, GQA-4,
   b=25/24 took 6.4 ms vs 0.73 ms for the same skeleton reading bf16. Per-codeword cost was
   the problem, not bytes.
3. **v1 on HIP (MI300X).** Same source through hipcc, no edits: 96/96. 6.7 ms.
4. **v1 on SYCL (Arc B390).** Built with DPC++ (`kernels/sycl/build.bat`). First runs failed
   nondeterministically: (a) the default SYCL queue is out-of-order, so the reduce kernel
   could run before the split kernel. Fixed with `property::queue::in_order()`. (b) After that
   fix, the v1 serial softmax (one work-item per head loops over 32 tokens) still gave wrong
   results on the Arc iGPU for bf16/fp8 and for pq with nsplit > 1, while the identical CUDA
   and HIP code passed. We did not root-cause it; the v3 rewrite (parallel softmax over
   (head, token), same barriers pattern as CUDA) passes 96/96 on Arc. Treat this as an open
   question, not a known compiler bug.
5. **v2 branch-free decode** (prefix-count compare instead of a data-dependent loop with
   division): 6.7 -> 6.2 ms on MI300X. Small.
6. **v3 SoA tables + radix bucket lookup** instead of binary search over 32-byte rows: small
   gain while the tables lived in LDS. Moving the tables to global memory (L1/L2 resident)
   took MI300X from 6.19 ms to 2.69 ms: the 22 KB of tables in LDS limited occupancy to one
   128-thread block per CU. (`PQ_TAB_SMEM=1` still selects LDS tables:
   `results/kernels/hip_mi300x_bench_tabsmem.jsonl`.)
7. **v4 compact smem tile**: decoded lattice coordinates are stored as exact int8 (2x in
   [-31, 31]) plus one f32 scale per token, bf16/fp8 tiles stay in their storage type and are
   converted on read. LDS per block dropped from ~21 KB to ~9 KB. MI300X 2.69 -> 1.59 ms.
8. **ND specialization**: the leader tables at b <= 25 have at most 5 distinct |values|, so the
   compare loop runs 5 instead of 8 times and the values fit in 32 bits. 1.59 -> 1.41 ms.
   Same change on CUDA and SYCL; all three backends 96/96 after it.

## Where it ends up (B=8, T=32768, Hq=32, Hkv=8, D=128, b = 25/24 = 3.19 bpv)

| GPU | pq (ours) | bf16, same skeleton | fp8, same skeleton | vendor bf16 |
|---|---|---|---|---|
| MI300X (HIP) | 1.41 ms | 0.95 ms | 1.25 ms | not measured (no ROCm torch on the shared box) |
| H100 (CUDA) | 1.86 ms | 0.53 ms | 1.07 ms | torch SDPA 0.36 ms |
| B200 (CUDA, same code, sm_100a, no tcgen05) | 1.67 ms | 0.52 ms | 1.06 ms | torch SDPA 0.22 ms |
| H100 (SYCL, DPC++ CUDA backend) | 2.17 ms | 0.67 ms | 2.46 ms | |
| Arc B390 (SYCL), B=4 | 10.1 ms | 6.09 ms | 7.41 ms | not measured |

The pq kernel reads 5x fewer bytes than bf16 (102 vs 512 bytes per token and KV head) but is
ALU-bound on codeword decode at GQA-4: each decoded group of 8 serves only 4 query heads.
The fp8 column is our skeleton with a software e4m3 decode; it is not a tuned FP8 kernel.
On H100 the gap is largest because INT32 throughput is half the FP32 rate there.

Bench data are random bytes (every b-bit index is a valid codeword), random-mantissa bf16 and random fp8
values, so the Arc numbers are not inflated by the iGPU's data-dependent compression (the
ultraquant notes saw ~110 GB/s on random data vs 430+ GB/s on uniform data on this box).
Correctness on 6 device/backend pairs: CUDA H100, CUDA B200, HIP MI300X, SYCL Arc B390,
SYCL H100 (CUDA backend), SYCL Xeon CPU (OpenCL): 96/96 each.

## What was not done

- No tensor cores (Pearl decodes into Blackwell tensor cores via CuTe DSL). Our dot products
  are CUDA-core / VALU FMAs.
- No paged KV (contiguous per sequence and head), no fusion with the KV write path.
- Decode is per tile into shared memory; register-level fusion into an MMA fragment was not tried.
