# NVFP4 scale path for the fused K-query decode (nvfp4-agent)

Code: `uqmt/nvfp4.py` (`uq_decode_nvfp4`, `build_paged_cache_nvfp4`, `reference_nvfp4`), tests `tests/test_nvfp4.py`,
bench `bench/nvfp4/`. attn.py / tree.py / quant.py untouched; `quant.quantize_nvfp4` already matches the NVIDIA NVFP4
spec (E4M3 scale per 16, ts = amax/(6*448), scale = e4m3(amax_g/6/ts)) and is used unchanged.

## Format and kernel
- Cache: kc/vc uint8 packed pairs as before; scales uint8 `[pages,Hkv,PAGE,D/16]` = raw e4m3fn bytes (sign bit 0);
  per-layer fp32 `k_ts`, `v_ts`. `k_ts` is folded into the softmax scale, `v_ts` is applied in the reduce kernel: no per-element cost.
- E4M3 decode is exact integer bit placement, no fp8 hardware type: `fp16bits = (byte & 0x7F) << 7`, value = fp16(bits) * 256
  (fp16 bias 15 vs e4m3fn bias 7; subnormals included). Verified for all 127 finite bytes (CPU test) and inside the kernel
  (GPU test injects every scale byte). It is pure integer/bitcast so it is identical on NVIDIA (e4m3fn) and on AMD
  gfx942 (e4m3fnuz hardware) with no lossy conversion; cache format is e4m3fn everywhere.
- fp4*scale has <=4 significant bits so the bf16 dequant is exact.
- Bytes per 128-d vector: NVFP4 = 64 + 8 = **72** (3.56x vs fp16) vs MXFP4 64 + 4 = **68** (3.76x); +4 B per layer-tensor for ts.

## Correctness
`tests/test_nvfp4.py`: 45 passed on A100 and on H100 (K 1..16 x D 64/128, GQA 1/2/4/8, ragged + short + fully masked
rows, shuffled pages, nsplit/baseline invariance, tensor-scale ranges 1e-3..300, every E4M3 scale byte). Kernel vs fp32 torch
reference rel. error 0.0021 (bf16 rounding only) at K=1,4,8,16 on both GPUs. AMD: pending (see below).

## Speed (B=8, ctx 32768, Hq32/Hkv8, D128; ms per call, best nsplit in {auto,8,16,32}; random valid cache)
| GPU | K | NVFP4 fused | MXFP4 fused | NVFP4 vs MXFP4 | NVFP4 K-indep | MXFP4 K-indep | fused speedup (NVFP4) | GB/s nv / mx (own bytes) |
|---|---|---|---|---|---|---|---|---|
| A100 80GB PCIe | 1 | 0.612 | 0.604 | +1.3% | 0.612 | 0.604 | 1.00x | 493 / 472 |
| A100 80GB PCIe | 2 | 0.617 | 0.611 | +1.0% | 1.165 | 1.151 | 1.89x | 489 / 467 |
| A100 80GB PCIe | 4 | 0.632 | 0.623 | +1.5% | 2.222 | 2.193 | 3.52x | 478 / 458 |
| A100 80GB PCIe | 8 | 0.908 | 0.897 | +1.2% | 4.273 | 4.209 | 4.70x | 333 / 318 |
| H100 80GB HBM3 | 1 | 0.347 | 0.344 | +0.9% | 0.342 | 0.334 | 0.99x | 871 / 830 |
| H100 80GB HBM3 | 2 | 0.355 | 0.346 | +2.5% | 0.681 | 0.666 | 1.92x | 852 / 824 |
| H100 80GB HBM3 | 4 | 0.362 | 0.356 | +1.8% | 1.284 | 1.238 | 3.55x | 835 / 802 |
| H100 80GB HBM3 | 8 | 0.439 | 0.439 | +0.1% | 2.433 | 2.364 | 5.54x | 688 / 650 |

Scale-decode + 5.9% extra bytes cost NVFP4 only about +0.1..+2.5% time over MXFP4 (fused); fused-vs-independent speedup is
unchanged (about 1.9x/3.5x/5.4x at K=2/4/8 on H100). In effective-bandwidth terms NVFP4 is not slower per byte.

## Quality through the kernel (H100, wikitext-2, 8 windows x 2048, prefill 1024 chunked + 1023 decode steps)
Every cached token is stored in the quantised paged cache; decode steps of all layers run through `uq_decode_nvfp4` /
`attn.uq_decode` (K=1); prefill chunks (256 queries > kernel max 16) read the dequantised cache via SDPA.
NVFP4 per-tensor scale is static per layer, ts = 2*amax/(6*448) from the first prefill chunk (2x headroom).
Attention error: K=8 verify queries on real post-RoPE KV (T=2048, all layers), kernel output vs exact fp32 attention.

| model | fp16 ppl | NVFP4 kernel ppl (delta) | MXFP4 kernel ppl (delta) | attn cos / relerr NVFP4 | attn cos / relerr MXFP4 | NVFP4 kernel vs own torch ref relerr |
|---|---|---|---|---|---|---|
| qwen3_8b | 8.522 | 8.583 (+0.061) | 9.072 (+0.550) | 0.9805 / 0.189 | 0.9349 / 0.334 | 0.0017 |
| mistral7b | 4.873 | 4.920 (+0.047) | 5.003 (+0.130) | 0.9888 / 0.153 | 0.9606 / 0.282 | 0.0017 |

fp16 ppl matches the quality-agent baseline (8.522 / 4.873). Deltas: Qwen3-8B NVFP4 +0.06 vs MXFP4 +0.55 (fake-quant table: +0.10 / +0.56);
Mistral +0.048 vs +0.130 (fake-quant: +0.052 / +0.139). The kernel path agrees with fake-quant (static ts is slightly better than the
per-write ts used there). NVFP4 cuts the KV-quantisation perplexity penalty ~9x on Qwen3-8B and ~2.7x on Mistral for +4 B/vector.

## AMD MI300X
`magni-amd` refused ssh connections (checked at start), so no gfx942 run. Pending: `bash bench/nvfp4/run_amd.sh`
(rsync to /root/uqmt-nvfp4/, `flock /root/gpu.lock`, <=10 min per hold, pytest + bench, results copied back to results/nvfp4_*_mi300x.*).
The e4m3 decode is integer-only, so no fnuz-specific path is needed.

## Reproduce
`modal run --detach bench/nvfp4/modal_nvfp4.py::main` (tests + A100/H100 bench), `::qmain` (H100 quality); raw JSON/logs in results/nvfp4_*.
