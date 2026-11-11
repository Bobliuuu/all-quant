# SYCL attention decode baselines (Arc B390 box)

Date 2026-11-11. All GPU runs went through the shared GPU lock wrapper. Raw data: `results/sycl_baseline_llamacpp_fa_kv.{log,json}`, `results/sycl_baseline_torchxpu_sdpa.{log,json}`.

## 1. llama.cpp SYCL FlashAttention with quantised KV

- Build: official release `llama-b12283-bin-win-sycl-x64.zip` (commit f3b407ca1, 2026-11-11, newer than PR #31125). `ggml-sycl.dll` needed `libimalloc.dll` and `mkl_intel_thread.3.dll`, which are not in the zip; copied from an existing oneMKL conda env on the box. Env defaults: GGML_SYCL_FA_DPAS=1, GGML_SYCL_FA_ONEDNN=1, XMX on.
- `test-backend-ops` is NOT in the release zip, so the kernel-only perf run (`test-backend-ops perf -o FLASH_ATTN_EXT -b SYCL0`) was not done (building from source not attempted). Fallback: `llama-bench` on Qwen3-0.6B-Q8_0 (28 layers, 16 heads, 8 KV heads, head_dim 128, i.e. GQA 2, not our 32/8 GQA 4), 1 sequence only.
- Command: `llama-bench -m Qwen3-0.6B-Q8_0.gguf -fa 1 -ctk T -ctv T -ngl 99 -d 0,4096,16384 -p 1,2,4,8 -n 64 -r 3 -o jsonl` for T in f16, q8_0, q4_0 (first pass used `-o json`, whose output is truncated when redirected to a file; the q8_0/q4_0 depth-16384 numbers come from a rerun with jsonl).
- Metric: whole-model step time. attn marginal ms = step(depth) - step(depth 0), summed over 28 layers (weights cancel). eff GB/s = KV bytes read at that depth (f16 4096 B, q8_0 2176 B, q4_0 1152 B per token per layer) / marginal ms. n_query>1 = prompt-processing of n tokens at depth (one ubatch), K-verify shape.

Decode (n_query=1, tg64):

| KV | depth | tok/s | step ms | attn marginal ms (28 layers) | KV MB read | eff GB/s |
|---|---|---|---|---|---|---|
| f16 | 0 | 129.7 | 7.71 | - | - | - |
| f16 | 4096 | 72.8 | 13.73 | 6.02 | 470 | 78 |
| f16 | 16384 | 30.4 | 32.86 | 25.15 | 1879 | 75 |
| q8_0 | 4096 | 83.6 | 11.97 | 4.02 | 250 | 62 |
| q8_0 | 16384 | 44.2 | 22.63 | 14.68 | 998 | 68 |
| q4_0 | 4096 | 91.9 | 10.88 | 3.32 | 132 | 40 |
| q4_0 | 16384 | 50.7 | 19.72 | 12.15 | 529 | 44 |

Delta vs f16 KV at depth 16384: q8_0 1.45x faster, q4_0 1.67x faster (attention part only: 1.71x, 2.07x). Quantised KV reads 1.9x/3.6x fewer bytes but speedup is below that: dequant-bound (effective GB/s drops from ~75 to ~68 to ~44). The 4-bit path does not reach bandwidth speed-of-light on this GPU.

K-verify (n_query 2/4/8, depth 16384, tokens/s of one ubatch; step ms = n/tok-s):

| KV | n=2 | n=4 | n=8 |
|---|---|---|---|
| f16 | 62.1 (32.2 ms) | 120.2 (33.3 ms) | 216.4 (37.0 ms) |
| q8_0 | 83.6 (23.9 ms) | 163.0 (24.5 ms) | 303.2 (26.4 ms) |
| q4_0 | 95.1 (21.0 ms) | 185.9 (21.5 ms) | 349.0 (22.9 ms) |

Step time is nearly flat in n_query (1..8), i.e. K-verify tokens cost close to nothing extra once KV is read. Depth-4096 and depth-0 numbers are in the json. Caveats: n_query>=2 depth-0 baselines run through non-FA-dominated paths, so the marginal-ms/GB/s for n_query>=2 is noisy (some values implausible, e.g. q8_0 n=2 d4096); trust n_query=1 and the raw step times. batch 2/8 sequences and n_head 32 not expressible; kernel-only ms not obtained.

Supports K-query verify: yes (handles 1..31 query tokens; run at 2/4/8). Supports FP4/MX: no (only f16/q8_0/q4_0 KV; no MX/FP4 path).

## 2. PyTorch XPU dense SDPA reference

- torch 2.15.0+xpu in a user-space venv (pip, nothing system-wide changed). `F.scaled_dot_product_attention(Q,K,V,enable_gqa=True)`, H=32, Hkv=8, D=128, one layer, dense f16/bf16 KV, 30 iterations after 5 warmup, synchronised wall time. Script is embedded in the log's producer; command pattern `python sdpa.py`.
- f16 (bf16 within a few percent, see json), ms per layer / eff GB/s of KV bytes:

| batch | KV len | n_q=1 | n_q=4 |
|---|---|---|---|
| 1 | 4096 | 0.173 / 97 | 0.254 / 66 |
| 1 | 16384 | 0.727 / 92 | 0.674 / 100 |
| 2 | 4096 | 0.333 / 101 | 0.324 / 104 |
| 2 | 16384 | 1.206 / 111 | 1.479 / 91 |
| 8 | 4096 | 1.305 / 103 | 1.276 / 105 |
| 8 | 16384 | 5.01 / 107 | 6.11 / 88 |

Dense SDPA reaches ~90-110 GB/s on KV, i.e. the practical memory ceiling of this GPU is roughly 100-110 GB/s. Caveat: batch-1 4096 working set (17 MB) may be partly cache-resident; times include launch overhead (one call per layer). Whether the torch path uses oneDNN SDPA internally was not verified. Supports K-query verify: yes (any n_query). FP4/MX: no.
