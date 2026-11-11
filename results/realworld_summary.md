# Real-world (serving-level) evaluation: status and findings

**Status: PARTIAL.** The MI300X box (`magni-amd`) stopped accepting SSH (port 22 refused, host pings) about 20 minutes into the
session and stayed down for the rest of it (several hours). Everything below that ran on a GPU ran on **Modal H100 with stock vLLM,
bf16 KV vs fp8 KV**, i.e. a CUDA proxy for "smaller KV". **No UltraQuant FP4 number has been measured at serving level yet.**

## What is established on the MI300X side (before the outage)
- vLLM **0.33.0** installs from `https://wheels.vllm.ai/rocm/` into `/root/vllm-venv` (torch 2.13.0+rocm10.0.0, HIP 7.15) and sees the MI300X.
  Models Mistral-7B-v0.1 and Qwen3-8B are in `/root/models`.
- **vLLM 0.33 already ships an UltraQuant backend** (`--kv-cache-dtype ultraquant_4bit`, `vllm/v1/attention/backends/ultraquant_attn.py`):
  FP4 E2M1 + UE8M0 group-32 scales **with Hadamard-rotated keys**, FlyDSL D=256 decode kernel + Triton unified fallback, bf16 only, no sink
  models. This is a different cache layout from `uqmt/attn.py`, so our K-query kernel cannot be dropped in without a re-layout; the serving
  test should use the native backend first (pending).
- First server launch died with `LLVM ERROR: support is already registered for analysis: PointerFlowAnalysisResult` during
  `import vllm.entrypoints.cli.main` (not at `import vllm`). Not yet diagnosed (the box went down while bisecting imports).

## H100, vLLM stock, Mistral-7B-v0.1 (and Qwen3-8B where noted), bf16 KV vs fp8 KV
Raw data `results/realworld_cuda_h100_*.json`, tables `results/realworld_cuda_h100_tables.md`, harness `bench/realworld/`.

| question | bf16 KV | fp8 KV (2x smaller) | verdict |
|---|---|---|---|
| KV capacity (80GB, 0.9 util) | 730,641 tok (Mistral), 380,528 (Qwen3-8B) | 1,461,360 / 758,016 | exactly 2x, as expected |
| Agent, 48 distinct 24k-token sessions (KV demand > bf16 capacity), tok/s | 646 | 632 | **no gain**; TTFT p99 ~31-32 s both |
| Agent, 32 sessions sharing a 24k prefix, tok/s | 1431 | 1412 | no gain (prefix cache keeps it small) |
| HotpotQA-in-16k-noise F1 (n=300, Mistral base) | 0.236 | 0.234 | no measurable quality loss |
| HotpotQA F1 (Qwen3-8B) | 0.588 | 0.581 | no measurable loss |
| Needle (Mistral base, 15-30k code haystack) | 0.225 | 0.225 | harness too hard for this base model; uninformative |
| Greedy drift, 24 prompts x 512 tok | bf16 vs bf16 rerun: 19/24 identical, median first divergence token 513 | fp8 vs bf16: 1/24 identical, **median divergence at token 30** | **fp8 KV drifts early**; bf16 run-to-run noise floor is far later |
| n-gram spec decode, code edit, conc=32, K=4, tok/s | 5088 (K=0: 2709) | 6339 (K=0: 2969) | fp8 +25% at K=4, +15% at K=8 |
| same, conc=8, K=8 | 3111 | 3605 | fp8 +16% |
| same, conc=8, K=2 / conc=32, K=2 | 2181 / 4430 | 2211 / 4842 | +1% / +9% |
| tokens/step (accepted+1), K=8 | 7.26 | 7.83-8.24 | fp8 does not hurt acceptance here (copy-heavy task) |

Reading: on H100 the KV format changes **capacity** (2x) and **verify-step cost at K>=4 with batch 32** (up to +25%), but the 48-session
agent trace was bound by prefill/queueing, not KV capacity, so capacity did not turn into throughput. Spec decoding itself is the big
lever (K=4: 1.9x at conc 32, 2.7x at conc 8) independent of KV format. Drift is the real cost: a lossy KV cache changes greedy output within
tens of tokens, so any acceptance rate / task-quality claim must be measured, not assumed.

Caveats: single runs, no CIs; GPU memory not recorded on Modal; n-gram acceptance is near-ceiling (copy task), inflating tokens/step;
tok/s for spec runs uses `usage.completion_tokens`, per-token ITL spreads chunk gaps evenly; RAG/needle prompts are not tuned for a base model.

## Pending (needs the MI300X back)
1. Diagnose the LLVM double-registration crash (try `VLLM_USE_FLYDSL=0`/uninstalling flydsl, or import-order isolation) and launch
   `--kv-cache-dtype {auto,fp8,ultraquant_4bit}` servers with `bench/realworld/run_cfg.py` under `flock /root/gpu.lock`.
2. Run the same agent/RAG/spec/drift jobs with FP4; measure KV capacity gain (expected ~3.5-4x), HIP graphs / prefix caching interaction
   (`--enforce-eager`, `--no-enable-prefix-caching`), `torch.profiler` step breakdown (rocprofv3 core-dumps here).
3. uqmt-vllm shim for our own layout (not written: format differs from the native backend).
