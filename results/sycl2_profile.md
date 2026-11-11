# Where the time goes on Arc B390 (SYCL decode kernels), 2026-11-11

Box: Arc B390 (Xe3, 96 XVE, 12 Xe cores, 128 KB SLM/core), Level Zero, open-source DPC++ 7.2.0, no admin.
Hardware profilers: `unitrace`/`onetrace`/VTune/`zet` metric groups are not installed on the box and a user-space install is not
possible without a driver-side metrics library (checked: no `ocloc`, no `unitrace`, no `onetrace` on PATH or in the DPC++ tree;
the Level Zero metrics API needs the metrics-discovery runtime which is not shipped with the consumer driver). So the "profile" below is a
controlled ablation study (remove one stage, keep everything else, time the whole kernel) plus pure-load micro-probes.
All numbers: B=8, ctx=16384, Hq32/Hkv8, D128, PAGE64, K=1, shuffled pages, random data, ms per call (best of 3x10).

## 0. The bandwidth ceiling was mis-measured (important)
Previous notes quoted a read peak of 430-455 GB/s (copy kernel ~535 GB/s). Those kernels read a **uniformly filled** buffer.
On this part uniform-data buffers are served 2.5-4x faster than incompressible ones (memory/cache compression):

| probe | data | GB/s |
|---|---|---|
| `bench --peakonly` read kernel | uniform fill | 290 (this session; earlier 430-455) |
| `bench --peakonly` read kernel | random bytes | **105** |
| `esimd/ldprobe` paged KV pattern (SYCL uint4 and ESIMD block loads, 4 KB chunks, shuffled pages) | uniform | 500 |
| same | **random** | **113** (every access shape: SYCL wg256, ESIMD 256 B/512 B blocks, 4 KB per iteration) |
| torch XPU dense SDPA (earlier baseline) | random | 90-110 |

A real KV stream of FP4 codes is incompressible, so the honest ceiling is ~105-113 GB/s (LPDDR5X, 128-bit). For B8/ctx16k the FP4 KV
is 134 MB, i.e. a **memory floor of ~1.2 ms** per call, not the 0.3 ms implied by the old peak. All "% of peak" in the sycl2 files use the random-data peak
(`bench` now prints both). Consequences: the previous XMX kernel was at ~50% of the true peak at K=1 (not 13%), and the achievable speedup at K=1 is
bounded by ~2x; at K=8 (6.3 ms before) it is up to ~5x.

## 1. Sub-group-private joint_matrix kernel (`xmx2`, no work-group barriers): worse
K=1, nsplit 16, wg128: 6.36 ms (old barrier kernel 2.48). Ablation: skip K+V dequant 1.79 ms; skip softmax 6.04. Dequant is 4.5 of 6.4 ms:
per-lane 256-entry SLM LUT gathers (random addresses, bank conflicts) plus 128 scalar SLM stores per token. Removing barriers was not the lever; with one sub-group
per XVE-thread and 10 KB SLM per sub-group, occupancy is 1 sub-group per XVE and every SLM round trip is exposed. Kept in `uq_xmx2.hpp` as a negative result.

## 2. ESIMD path (`uq_esimd.hpp`), first correct version, ablation (nsplit 8, 4 threads/WG, 2.46 ms total)
| variant | ms | delta |
|---|---|---|
| full | 2.46 | |
| no K decode (raw words fed to DPAS) | 1.66 | K decode = 0.80 |
| no V decode | 1.75 | V decode = 0.71 |
| no P V DPAS | 1.82 | PV = 0.64 |
| no K decode, no V decode, no PV | 1.40 | |
| ... and no global loads at all | 0.46 | loads = 0.94 of the remaining 1.40 |
| ... and no S DPAS | 1.36 | |
| ... and no exp2/softmax | 1.18 | |
| ... and no P SLM transpose | 1.30 | |
| loads only (same access pattern, nothing else) | 1.29 | = the memory floor (113 GB/s) |
Findings: (a) once the decode was made cheap the kernel is bounded below by the 1.2 ms load time, so everything else has to hide under it; (b) register-indirect
LUT gathers (`iselect`) cost ~20-25 cycles each, 64+ per tile per operand: K+V decode were 1.5 ms of ALU. Replacing the per-token `iselect` for the scale offset by a pure
region read (`replicate_vs_w_hs`) and writing the decoded words straight into the DPAS operand registers (16-bit adds into strided regions) took K=1 from 2.05 to 1.72 ms;
(c) occupancy/wave quantisation matters: nsplit must make `B*Hkv*nsplit` ~ a multiple of the resident-thread capacity (768 threads with 128 GRF, 384 with 256 GRF); powers of two
are not optimal (nsplit 12 vs 16: 2.05 vs 2.93 ms before the decode optimisation); (d) software prefetch (`prefetch<uint32_t,64>` 1..6 tiles ahead) gave 0%: the load side is already at
the DRAM ceiling, the loss is ALU instruction throughput (~2000 instructions per 16-token tile before the optimisation, ~2 cycles each on an XVE).
(e) GRF 256 (needed for K>=4 because the O accumulators are 128 registers) costs half the resident threads; K=1/2 stay on 128 GRF.

## 3. Load-pattern micro-probes (random data)
| probe | result |
|---|---|
| 2D block load (`load_2d`, transposed, 16 tokens x 32 B) | 121 GB/s (half-line accesses) |
| 16-lane dword gather, VS=8 | 121 GB/s |
| contiguous 256 B `block_load` x4 per tile | 264 GB/s (uniform-data; same path as the kernel) |
| K path only (2D load + register LUT decode + DPAS, 16 tokens): decode+DPAS add < 10% over the load | decode is not the K-side limiter in isolation |

## 4. What limits the shipped ESIMD kernel now
K=1 1.72 ms (floor 1.18): ~70% of the memory floor, ALU instruction throughput on K/V decode (~0.4 ms of un-hidden ALU) and per-tile fixed work. K=8 (M=32) runs two query-group
threads that decode the same tile twice (the O accumulators for 32 rows do not fit in 256 GRF), so it is ALU-bound at 2x the K=4 time.
Not done (ideas): share the decode between the two query groups (needs O in SLM), arithmetic bf16-denormal decode ((code&7)<<6 as a denormal, scale on the accumulators) if DPAS
handles bf16 denormals exactly (untested), NVFP4 in ESIMD (the E4M3 scale is not a power of two, so it cannot be an exponent add; needs per-group accumulators for K and per-element multiplies for V).
