# MI300X serving-level KV-cache results (vLLM 0.33.0 ROCm, Qwen3-8B)

Box: 1x MI300X 192 GB, vLLM 0.33.0+rocm100 (torch 2.13 ROCm), bf16 weights, `--max-model-len 32768`, prefix caching on, greedy decoding,
harness `bench/realworld/`. Raw data: `results/realworld_mi300x_*.json` (one file per config x workload; spec K=4/8 files end in `_v2`).
`ultraquant_4bit` is vLLM's native backend (FP4 E2M1 + UE8M0 g32 scales, Hadamard keys); vLLM keeps layers 0,1,34,35 at bf16, so the
capacity gain is below 4x. All numbers are single runs (no CIs).

| metric | bf16 KV | fp8 KV | ultraquant_4bit KV |
|---|---|---|---|
| KV capacity, tokens (gmu 0.9, ~150 GiB cache) | 1,092,352 | 2,184,656 | 3,157,312 |
| capacity vs bf16 | 1.00x | 2.00x | 2.89x |
| Agent (32 distinct 24k-token sessions x 4 turns, gmu 0.25) KV tokens | 185,088 | 370,144 | 544,352 |
| Agent out tok/s | 47.9 | 38.1 | 59.4 |
| Agent TTFT p50 / p99 (s) | 68.9 / 81.0 | 74.3 / 85.0 | 30.4 / 56.9 |
| RAG HotpotQA-in-16k, n=100, F1 | 0.655 | 0.647 | 0.618 |
| RAG EM | 0.550 | 0.550 | 0.510 |
| RAG TTFT p50 (s) | 20.8 | 22.8 | 11.9 |
| n-gram spec decode, code-edit, conc 8, K=0, out tok/s | 1052 | 1091 | 767 |
| same, K=4 | 855 | 1226 | 1477 |
| same, K=8 | 1406 | 1683 | 1776 |
| Greedy drift vs bf16 (24 prompts x 512 tok): identical sequences (bf16 = reference itself) | 24 | 0 | 0 |
| median / mean first-divergence token | 512 / 512 | 24 / 30 | 10 / 11 |

Reading (measured only):
- **ultraquant_4bit runs end to end on MI300X** and serves coherent text. KV capacity is 2.89x bf16 (fp8: 2.00x).
- Quality: RAG F1 0.618 (FP4) vs 0.655 (bf16) and 0.647 (fp8), n=100: about 0.036 F1 lower for FP4, fp8 within noise.
  Greedy drift: fp8 and FP4 both diverge from bf16 within tens of tokens (median 24 / 10); FP4 diverges earlier.
- Speed: plain decode is slower with FP4 (K=0: 767 vs 1052 tok/s bf16). In the agent trace where bf16 KV (185,088 tokens) is oversubscribed,
  FP4 (544,352 tokens) gave 59.4 vs 47.9 tok/s and TTFT p50 30 s vs 69 s.

Caveats:
- Import fix: `import vllm` aborts with `LLVM ERROR: support is already registered for analysis: PointerFlowAnalysisResult` unless `torch` is
  imported first (duplicate LLVM registration; import order only). Fix used: a `.pth` file in the venv containing `import torch`.
- Mistral-7B-v0.1 cannot use ultraquant_4bit (backend rejects sliding-window models), so all three KV formats were run on Qwen3-8B only.
- vLLM 0.33.0 selector bug: the bf16 boundary layers fall back to ROCM_ATTN, whose KV layout (LHBNC/LBHNC) is incompatible with
  UltraQuant's LBNHC, so startup fails with "No KV cache layout satisfies every supported set". Workaround `--attention-backend TRITON_ATTN`
  was used for the ultraquant_4bit runs only; bf16/fp8 used default backends, so backend and KV format are confounded in speed numbers.
- The GPU was shared (via flock) with other jobs; speed runs are single and noisy (e.g. bf16 K=4 is below bf16 K=0), so do not read small
  differences. The agent run where fp8 is slower than bf16 is likewise noise-level. Spec-decode `tokens_per_step` in the JSONs is invalid
  (the metric scrape double-counts per-position counters) and is not used. The first spec K=4/8 attempt silently ran without speculation and was
  discarded; the `_v2` files are the valid ones.
