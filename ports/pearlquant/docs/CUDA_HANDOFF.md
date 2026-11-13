# PearlQuant-style E8 KV codec: Blackwell/NVIDIA CUDA kernel handoff

Supersedes placeholder item `20261113-pearlquant-style-e8-kv-codec-blackwell-kernel`.
From: pearlquant-repro (local repo `~/code/pearlquant-repro`, to be merged under
`ports/pearlquant/` of Bobliuuu/all-quant). Not pushed.

## What it is
Fused split-KV decode attention over an E8 "probable points" (ball) codebook KV cache:
16 indices of b bits per 128-dim head vector + fp16 scale = 2b+2 bytes/vector (b=25/24 -> 102 B/token/KV-head
vs 512 bf16, 264 fp8). Random Hadamard rotation folded into the query and the output. Decode on chip
(leader radix lookup + branch-free multiset-permutation unrank) into an int8 smem tile; CUDA-core FMAs, no tensor cores.
Source: `kernels/cuda/pq_attn.cuh`, `kernels/cuda/pq_bench.cu`, `kernels/common/pq_ball_decode.h`.

## Current metrics (all from files in results/kernels/)
- Correctness: 96/96 shared-vector checks on H100 (`cuda_h100_check.jsonl`) and B200 (`cuda_b200_check.jsonl`),
  worst rel. L2 error 1.1e-6 vs float64 over the decoded cache. Same on HIP MI300X and SYCL (Arc, H100, CPU).
- Latency, B=8, T=32768, Hq=32, Hkv=8, D=128, pq b=25/24 (us/call):
  - H100: pq 1862, our bf16 skeleton 536, our fp8 skeleton 1070, torch SDPA bf16 360 (`cuda_h100_bench.jsonl`, `cuda_h100_torch_sdpa.jsonl`)
  - B200 (sm_100a, same code): pq 1672, bf16 522, fp8 1062, torch SDPA 221 (`cuda_b200_bench.jsonl`)
  - B=1: H100 pq 283 vs SDPA 63; B200 pq 250 vs SDPA 47.
- Bound: integer ALU in codeword decode (≈ same time at 16/16 bits as at 25/24), not HBM.
  Effective KV bandwidth only ~130 GB/s on B200.
- Note: these were measured before the last register-pressure fix (dropped the vv[32] register array:
  VGPRs 143 -> 52 on MI300X); CUDA not re-measured after it, per the routing change.
- Spend on NVIDIA for the kernel part: ~3 Modal runs H100 (~$0.3 each) + 1 B200 (~$0.5); whole project ≈ $7 Modal.

## Reproduce
```
cd ~/code/pearlquant-repro
python tools/gen_vectors.py kernels/vectors          # vectors + leader tables (CPU, ~1 min)
modal run kernels/modal_kernels.py --gpu B200 --what cuda   # app pq-kernels, fn cuda_b200; image nvidia/cuda:12.8.1-devel + torch 2.8.0 cu128
modal run kernels/modal_kernels.py --gpu H100 --what cuda   # fn cuda_h100
modal run kernels/modal_kernels.py --gpu H100 --what sycl   # DPC++ nightly-2026-11-04 CUDA backend
# by hand: nvcc -O3 -std=c++17 -arch=sm_100a kernels/cuda/pq_bench.cu -o pq_bench
#          ./pq_bench check kernels/vectors ; ./pq_bench bench kernels/vectors/tables 8 32768 32 8 25 24 0 30
```
Results land in Modal volume `pq-results` under `kern_<GPU>_<what>/`.

## Next steps (expected gain)
1. Re-run H100/B200 with the current source (vv[32] removed, occupancy fix that helped MI300X). Cheap, likely 1.2-2x.
2. Decode once per token into bf16 and use tensor cores (mma.sync on H100, tcgen05/UMMA on B200) for QK and PV; GQA-4
   leaves tensor cores idle today. Expected: removes the FMA/smem-read share (~30%?), not the decode cost.
3. Cut decode cost (the real bottleneck): (a) per-leader precomputed permutation tables for the most frequent leaders
   (few KB, L1-resident), (b) warp-cooperative unrank (one leader lookup per 8 lanes), (c) 16-bit packed arithmetic.
   Target: decode < 50 int ops/group, which would make pq ≈ fp8 speed at GQA-4 (Pearl claims ≈ NVFP4 on B200).
4. MLA-style shape (one latent shared by all heads, as in Pearl's GLM-5.3): decode amortizes over 64-128 heads; build
   a D=576 (512 NoPE + 64 RoPE) variant and measure; this is where Pearl's numbers come from.
5. Paged KV + vLLM backend hook (cache write path: encode kernel; the Python encoder exists in `pqrepro/codec.py`).
