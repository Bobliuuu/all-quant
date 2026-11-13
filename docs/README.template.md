# pq-repro: a PearlQuant-style E8 KV-cache codec, reproduced from public information

Pearl Research Labs described **PearlQuant**, an E8-lattice KV-cache quantizer with a fused
Blackwell decode-attention kernel, in a blog post
(https://pearlresearch.ai/research/blog/pearlquant). They released no code and no format.
This directory is an independent reconstruction: **pq-repro**, a PearlQuant-*style* codec. It is
not Pearl's code, it runs a different model, and every gap in their description was filled by a
choice of ours, listed in [DESIGN.md](DESIGN.md). Numbers attributed to Pearl are quoted from
their blog; every other number on this page is generated from `results/` by
`tools/make_tables.py` (full tables: [results/SUMMARY.md](results/SUMMARY.md)).

What is here:

1. `pqrepro/`: the codec (random Hadamard rotation, fp16 per-vector scale, E8 "probable points"
   ball codebook with leader indexing; NestQuant-style Voronoi codebook as an ablation), plus
   scalar baselines (FP8, NVFP4, int4, a TurboQuant-MSE re-implementation).
2. `experiments/rd_gaussian.py`: Pearl's "wasted bits" rate-distortion comparison on an iid Gaussian.
3. `experiments/kl_pg19.py`, `experiments/niah.py`: KL divergence from an FP8 KV cache on PG-19
   (Pearl's metric) and a small needle-in-a-haystack check, on Qwen3-8B.
4. `kernels/`: fused split-KV decode attention that reads the compressed cache and decodes codewords
   on chip, in CUDA and HIP (one source) and SYCL, checked on shared test vectors.

## What Pearl disclosed and what we chose

| | Pearl (blog) | pq-repro (ours) |
|---|---|---|
| lattice, group | E8, groups of 8, "NestQuant inspired" | same |
| rotation | "a random rotation" | random signs + Walsh-Hadamard per 128-dim head vector |
| codebook | "only assign indices to probable lattice points under the Gaussian i.i.d. distribution" | the 2^b smallest-norm E8 points; index = leader offset + permutation rank + sign bits |
| scale | not disclosed | one fp16 per vector (+0.125 bpv), best of 8 gains |
| rates | 3.2 / 2.3 / 2.1 bpv model averages (NoPE/RoPE split not disclosed) | K/V bits per group of 8: 25/24, 18/17, 16/16 -> {{bits:pq-3.2}}, {{bits:pq-2.3}}, {{bits:pq-2.1}} bpv incl. scales |
| outliers | not disclosed | optional K mean-centering (`-kc`): exactly softmax-invariant, free in the kernel |
| model | GLM-5.3 (MLA/DSA), 8xB200 | Qwen3-8B (GQA, head_dim 128), one GPU |
| kernel | CuTe DSL, Blackwell tensor cores | portable CUDA/HIP/SYCL, CUDA-core FMAs, no tensor cores |

## 1. Rate-distortion (iid Gaussian)

![wasted bits, Pearl vs ours](results/figures/rd_vs_pearl.png)

The shape of Pearl's Fig. 5 reproduces. Wasted bits R - R*(D) (lower is better):

| | Pearl's Fig. 5 | ours |
|---|---|---|
| E8 codec, 2.75 bpv | 0.39 | {{rd:pq-ball b=22:known}} (no side info), {{rd:pq-ball b=22:norm}} (free exact norm), {{rd:pq-ball b=21:fp16}} (fp16 scale counted) |
| E8 codec, 3.75 bpv | 0.42 | {{rd:pq-ball b=30:known}} / {{rd:pq-ball b=30:norm}} / {{rd:pq-ball b=29:fp16}} |
| TurboQuant-MSE 3 / 4 bits | 0.56 / 0.60 | {{rd:turboquant-mse 3b:norm}} / {{rd:turboquant-mse 4b:norm}} (our re-implementation) |
| NVFP4, 4.5 bpv | 1.15 | {{rd:nvfp4:fp16}} |
| FP8 e4m3, 8 bpv | 2.65 | {{rd:fp8 e4m3 (no scale):known}} |

We do not know which side-information accounting Pearl used; with no side info our 2.75-bpv point
matches theirs and our 3.75-bpv point is worse. The Voronoi (NestQuant) codebook is worse than the
ball at equal rate (e.g. {{rd:pq-voronoi q=6:fp16}} vs {{rd:pq-ball b=21:fp16}} at 2.75 bpv incl. scale),
and HyperQuant's entropy-coded E8 (variable rate, not randomly accessible) beats every fixed-rate
code here ({{rdhq:2.75}} / {{rdhq:3.75}}).

![wasted bits vs rate](results/figures/rd_waste.png)

## 2. Model quality (Qwen3-8B, not GLM-5.3)

KL divergence of next-token distributions from an FP8 KV cache (e4m3, fp32 scale per 128-value
vector), PG-19 test books, 4 books x 16,384 tokens. The cache is quantized at write time (K after
RoPE), so every query sees the decoded cache, as in a real decode step. FP8 reference perplexity
{{klref_ppl}}. `-kc`: our K mean-centering; `-kvc`: also V; `-vk`: V gets the extra bit; `vor-`:
Voronoi codebook.

{{TABLE_KL}}

Read with care:
- The ordering matches Pearl's (KL grows steadily as the rate drops), and near 3.2 bpv the E8
  codec beats a scalar Lloyd-Max codec at 3.125 bpv ({{kl:pq-3.2-kc}} vs {{kl:tq-mse-3-kc}}, both with
  K-centering). In absolute terms our KL is roughly 2x Pearl's at each rate, on a different model
  (GQA, no MLA latent) with our own guesses for every undisclosed choice.
- Without K mean-centering, Qwen3-8B's keys (a large shared component, worst in layer 0) break
  every low-bit rotated codec at 16k context ({{kl:pq-2.3}} at 2.3 bpv, {{kl:tq-mse-3}} for
  TurboQuant-MSE at 3 bits). Centering is exact for attention (each query's logits shift by one
  constant) and costs nothing in the kernel. Pearl does not say whether they do anything like it.

Needle in a haystack (our own RULER-style generator, not the RULER suite; 8k tokens, 12 prompts
per task, greedy decoding, cache quantized for prefill and every generated token):

{{TABLE_NIAH}}

## 3. Kernels

One source for CUDA and HIP (`kernels/cuda/pq_attn.cuh`), one SYCL port (`kernels/sycl/pq_sycl.cpp`),
one decoder header shared by all three (`kernels/common/pq_ball_decode.h`). Split-KV flash decoding:
per 32-token tile the block copies the packed codes to shared memory, each thread decodes groups of 8
(leader lookup, branch-free multiset-permutation unrank, signs) into exact int8 lattice coordinates,
and the query (rotated once per call) is dotted against them; V is accumulated in the rotated domain
and un-rotated once in the reduce kernel. A cached vector costs exactly 2·b + 2 bytes.

**Correctness**: {{checks}} checks pass across all device/backend pairs (8 cases x {pq, bf16, fp8} x
4 split counts each, relative L2 error < 1e-4 against float64 attention over the decoded cache):

{{TABLE_CHECK}}

**Speed**, us per call, B=8, 32k context, Qwen3-8B-like GQA (Hq=32, Hkv=8, D=128), pq at 3.19 bpv:

{{TABLE_KERNEL}}

The pq cache is 5x smaller than bf16, but the kernel is **slower** than bf16 everywhere: it is
bound by codeword decode (integer ALU), not by memory. At GQA-4 each decoded group serves only 4 query
heads; Pearl's GLM-5.3 uses MLA, where one decoded latent serves every head, and their kernel feeds
tensor cores. We did not reach their "about as fast as NVFP4" result. The optimization history (mostly occupancy: leader tables moved out of
shared memory, compact int8 tile) is in [kernels/PORTING_LOG.md](kernels/PORTING_LOG.md).

## Reproduce

```
python tests/test_codec.py                                    # codec self-checks (CPU)
python experiments/rd_gaussian.py --out results/rd_gaussian.json && python tools/plot_rd.py
modal run experiments/modal_quality.py --task kl --args "--nseq 4 --ctx 16384 --configs fp8,pq-3.2-kc --out /out/x.json"
python tools/gen_vectors.py kernels/vectors                   # shared test vectors + leader tables
nvcc  -O3 -std=c++17 -arch=sm_90a kernels/cuda/pq_bench.cu -o pq_bench     # or:
hipcc -O3 -std=c++17 --offload-arch=gfx942 kernels/cuda/pq_bench.cu -o pq_bench
./pq_bench check kernels/vectors && ./pq_bench bench kernels/vectors/tables 8 32768 32 8 25 24
clang++ -fsycl -O3 -std=c++17 kernels/sycl/pq_sycl.cpp -o pq_sycl && ./pq_sycl check kernels/vectors
modal run kernels/modal_kernels.py --gpu H100 --what cuda     # also: --gpu B200, --what sycl
python tools/make_tables.py                                   # README.md + results/SUMMARY.md
```

## Limitations

- Different model and attention type than Pearl (GQA vs MLA), so absolute KL values are not
  comparable; the per-component rate split, scale format and index assignment are ours.
- KL uses 4 PG-19 books x 16k tokens; NIAH is our own small generator, not RULER.
- The kernels are correct on every backend we ran but slower than bf16; no tensor cores, no
  paged KV, no vLLM integration; serving results (TTFT, throughput) were not attempted.
- Intel numbers are from one Arc B390 iGPU (Xe3, shared system memory), not a datacenter part.
