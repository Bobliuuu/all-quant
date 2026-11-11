# UltraQuant (uqmt): fused K-query FP4 paged decode attention

A Triton kernel that verifies **K speculative tokens in one pass over a 4-bit KV cache**. The KV cache is FP4 (e2m1) with UE8M0 scales per 32 channels (68 bytes per head per token for K or V at head_dim 128, **3.76x smaller than fp16, 1.88x smaller than fp8**). All K queries and the GQA group heads form the M dimension of one `tl.dot`, so every KV page is read once instead of K times.

**Headline (measured, one attention layer, batch 8, context 32768, Hq64/Hkv8/D128, page 64):**

| | MI300X | B300 |
|---|---|---|
| K=8 verify, fused | **0.54 ms** | **0.42 ms** |
| vs. K independent single-query launches | **4.1x faster** | **6.8x faster** |
| K=1 decode | 0.36 ms (797 GB/s of FP4 bytes) | 0.29 ms (990 GB/s) |

Correctness: max abs error ~1e-4 against the dequantised reference at these sizes; 50 pytest cases (ragged lengths, head_dim 64/128, GQA 1-16, K up to 16, pages 16-128, negative-zero codes, scale extremes, CUDA-graph replay) pass on MI300X.

```
 plain decode of K speculative tokens          UltraQuant fused K-query
 (what an engine does today)

 q0 -> [read all KV pages] -> o0               q0 \
 q1 -> [read all KV pages] -> o1                q1  |  one tile: M = K x GQA group
 ...                                            ... |-> [read each KV page ONCE] -> o0..oK-1
 qK-1 -> [read all KV pages] -> oK-1            qK-1/    (FP4 -> bf16 in registers, causal
                                                         mask per query, split-KV + LSE reduce)
 HBM traffic: K x KV                            HBM traffic: 1 x KV
```

## Install

```bash
pip install git+https://github.com/Bobliuuu/ultraquant-more   # or: pip install -e .
```

Importing needs only PyTorch (quantise / dequantise / reference / cache builder run on CPU). Running the kernel needs a GPU and Triton (ROCm 7 + Triton 3.9 and CUDA B300 tested). `pytest tests` skips GPU tests cleanly when there is no GPU.

## API

```python
import torch, uqmt

# dense KV [B, Hkv, T, D] -> paged FP4 cache (page table, ragged lengths, optional page shuffle)
k = torch.randn(8, 8, 4096, 128, device="cuda"); v = torch.randn_like(k)
cache = uqmt.build_paged_cache(k, v, page_size=64, seq_lens=[4096] * 8)

# K = 1 + number of speculative tokens; seq_lens already includes the K new tokens
q = torch.randn(8, 4, 64, 128, device="cuda", dtype=torch.bfloat16)   # [B, K, Hq, D]
out = cache.decode(q)                      # [B, K, Hq, D] bf16; nsplit chosen automatically

# CUDA-graph friendly: preallocate once, no host syncs inside uq_decode
ws = uqmt.UQWorkspace.create(8, 4, 64, 8, 128, nsplit=uqmt.auto_nsplit(8, 8, 64))
out = uqmt.uq_decode(q, cache.kc, cache.ks, cache.vc, cache.vs,
                     cache.block_table, cache.seq_lens, workspace=ws)

ref = uqmt.reference(q, cache.kc, cache.ks, cache.vc, cache.vs, cache.block_table, cache.seq_lens)
```

Functions: `uq_decode`, `quantize_fp4`, `dequantize_fp4`, `reference`, `build_paged_cache` (returns `PagedFP4Cache`), `UQWorkspace`, `auto_nsplit`, and `uqmt.vllm_adapter` (varlen verify-step wrapper and a cache-append hook for engine integration).

Cache layouts (uint8): `kc, vc [pages, Hkv, PAGE, D/2]` (low nibble = even element), `ks, vs [pages, Hkv, PAGE, D/32]` (UE8M0, value = 2^(s-127)). Query token j of K attends to `seq_len - K + j + 1` positions; rows with no visible position return zeros.

**Auto `nsplit`**: the grid is (B, Hkv, nsplit); `auto_nsplit` targets about 4 programs per compute unit (halved for very wide verify tiles), rounds down to a power of two, and clamps to `[1, min(64, pages)]`. Override with `nsplit=` (any power of two; results are independent of it).

## Supported shapes

| | |
|---|---|
| head_dim | 64, 128 |
| GQA ratio Hq/Hkv | any (tested 1, 2, 4, 7, 8, 16) |
| K (query tokens per sequence) | 1 to 16 |
| page size | 16, 32, 64, 128 (power of two >= 16) |
| sequence lengths | ragged per sequence; shorter than a page, shorter than K, and empty sequences handled |
| query dtype | bf16 (KV dequantised in registers) |
| block table | int32 or int64 |

## Results: MI300X vs B300

Single layer, B=8, ctx=32768, Hq=64, Hkv=8, D=128, page 64, random data, best config per K. Source: `results/mi300x_b8_c32k_tuned.json`, `results/b300_b8_c32k_tuned.json`, `results/*_tuning_notes.md`.

| K | MI300X fused ms | MI300X speedup vs K launches | B300 fused ms | B300 speedup vs K launches |
|---|---|---|---|---|
| 1 | 0.361 | 1.0x | 0.288 | 1.0x |
| 2 | 0.373 | 1.83x | 0.357 | 1.79x |
| 4 | 0.453 | 2.59x | 0.422 | 2.82x |
| 8 | 0.542 | 4.10x | 0.421 | 6.77x |

Best configs: MI300X `UQ_WARPS=2 UQ_STAGES=2 nsplit=16`; B300 K=1 `warps=4 stages=3 nsplit=64`, K=8 `warps=8 stages=3 nsplit=16`, K=2 and K=4 on the default config (`warps=4 stages=2 nsplit=16`). Measured device ceilings (`results/*_peak.json`): MI300X HBM copy 3.9 TB/s, bf16 640 TFLOPS, fp8 1.12 PFLOPS; B300 6.4 TB/s, 1.77 PFLOPS bf16, 3.5 PFLOPS fp8.

How far from the roofline: at K=1 the MI300X kernel moves 797 GB/s of FP4 bytes, equal to 3.0 TB/s of fp16-equivalent bytes (about 77% of the copy rate) but 20% of the FP4 byte floor, because e2m1 is decoded in software on gfx942. A whole-decode-step bound from the KV bytes alone is 1.28x vs fp16 and 1.16x vs fp8 (attention ~30% of a vLLM decode step)

### Other backends

CUDA (`mma.sync`) and SYCL ports of the same algorithm live in [`cuda/`](cuda/) and [`sycl/`](sycl/), each with its own README/log and validation vectors generated from the Python reference.

## Head-to-head vs amd-experiments and end-to-end

<!-- PLACEHOLDER: filled from results/h2h_summary.md, results/real_kv_summary.md, results/e2e_summary.md when those exist. -->
_Pending: head-to-head kernel timings against the amd-experiments TurboQuant/fp16 kernels, real-model KV statistics, and end-to-end serving results are being measured and will be summarised here._

## How it compares to other KV quantisation

Survey with sources and verification status: [docs/RELATED_WORK.md](docs/RELATED_WORK.md), machine-readable [docs/methods.json](docs/methods.json), use-case arithmetic [docs/USE_CASES.md](docs/USE_CASES.md). In short: the published UltraQuant (AMD, arXiv 2606.20474) uses the same FP4/UE8M0-per-32 storage with key rotation and a CDNA4 scaled-MFMA kernel, and notes a fused K-query kernel as an open opportunity; this repo supplies that fused K-query kernel (portable Triton, in-register dequant). Alternatives (fp8 KV, KIVI/KVQuant-style int2-4, TurboQuant, NVFP4/MXFP4) trade accuracy, bytes and hardware support differently; see the table in RELATED_WORK.md.

## Video

An explainer video is in [`video/`](video/).

## Reproduce

```bash
pip install -e . pytest
pytest tests -q                                           # CPU tests always; GPU tests need a GPU
cd uqmt
python bench.py --batch 8 --ctx 32768 --ks 1 2 4 8 --nsplit 16 --check   # fused vs K independent
UQ_WARPS=8 UQ_STAGES=3 python bench.py --batch 8 --ctx 32768 --ks 8      # B300 K=8 config
python peak.py peak.json                                  # device ceilings (matmul TFLOPS, HBM copy)
```

`UQ_WARPS`, `UQ_STAGES` set the Triton launch config; `UQ_FP8K=1` enables the experimental fp8-MFMA score path (head_dim 128 only).

## Caveats

* **Software FP4 dequant on gfx942.** MI300X has no FP4 or scaled-FP4 MFMA; e2m1 is decoded in registers (bit-placement trick). Native `mfma_scale` FP4 needs gfx950 (MI350/MI355X); not measured here.
* **Synthetic data.** All timings above use random normal KV. No model-level accuracy has been measured with this quantiser by this repo; kernel error is vs the dequantised reference.
* **K=1 gains nothing from fusion**; the benefit is the K-query verify step and the 3.76x KV capacity.
* The fp8-MFMA score path (`UQ_FP8K=1`) is experimental, slower on MI300X (K=8 0.66 vs 0.54 ms) and off by default.
* Per-block scale is `2^ceil(log2(amax/6))` with no rotation of K; the paper's rotated, clipped variant is explored separately.

## What we tried

An fp8-MFMA variant of the K-score (flag above) and a scale-broadcast variant; neither beat the default bf16 path on MI300X.
