# Real AMD (Triton/ROCm) ports of the comparison methods, MI300X

B8, ctx 32768, Hq64/Hkv8/D128, page 64, cold-cache timing (shared harness bench/ports/common.py). HBM copy peak 3909 GB/s (results/mi300x_peak.json). GB/s counts stored KV bytes only (no fp16 residual, no q/out).

| method | B/vec(128) | ms K=1 | K=2 | K=4 | K=8 | GB/s K=1 (% of HBM copy peak) | kernel attn cos (synthetic LLM-like KV) | ppl delta wikitext2 mistral / qwen3 | attn cos real KV mistral / qwen3 | note |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 SDPA (reference) | 256 | 1.473 | 1.476 | 1.478 | 1.477 | 729 (19%) | 1.000 | - / - | - / - | torch SDPA, KV re-read not needed (K rows batched) |
| **UltraQuant FP4** (this repo) | 68 | 0.361 | 0.360 | 0.459 | 0.551 | 790 (20%) | 0.965 | 0.139 / 0.562 | 0.963 / 0.932 | real kernel; quality row = repo quantiser (official) |
| TurboQuant-3 (amd-experiments) | 52 | 1.168 | 2.234 | 4.360 | 8.651 | 187 (5%) | 0.937 | 0.142 / 1.995 | 0.968 / 0.933 | no K-query mode: K independent calls |
| fp8 (port) | 128 | 0.294 | 0.291 | 0.421 | 0.510 | 1825 (47%) | 0.998 | 0.008 / 0.032 | 0.998 / 0.997 | per-tensor e4m3fnuz scale; fused K-query |
| iso3 | 52 | 0.790 | 0.763 | 0.964 | 1.584 | 276 (7%) | 0.916 | - / - | - / - |  |
| kivi4 (port) | 80 | 0.540 | 0.533 | 0.634 | 0.784 | 621 (16%) | 0.993 | 0.010 / 0.026 | 0.996 / 0.998 | per-channel K / per-token V + 128-token bf16 residual; fused |
| mxfp4 (port) | 68 | 0.389 | 0.387 | 0.478 | 0.538 | 734 (19%) | 0.962 | 0.190 / 0.482 | 0.943 / 0.923 | OCP floor-exp UE8M0/32; same kernel as UltraQuant |
| nvfp4 (port) | 72 | 0.411 | 0.400 | 0.495 | 0.536 | 735 (19%) | 0.981 | 0.056 / 0.081 | 0.989 / 0.979 | E4M3/16 + tensor scale; fused |
| planar3 | 52 | 0.777 | 0.769 | 0.965 | 1.575 | 281 (7%) | 0.896 | - / - | - / - |  |
| planar4 | 68 | 1.095 | 1.155 | 1.323 | 2.195 | 260 (7%) | 0.947 | - / - | - / - |  |
| qserve_int4 (port) | 68 | 0.397 | 0.375 | 0.489 | 0.629 | 719 (18%) | 0.981 | 0.047 / 0.005 | 0.985 / 0.963 | per-token asym int4 + SmoothAttention; straightforward K-query |
| rotor3 | 52 | 0.766 | 0.769 | 0.966 | 1.583 | 285 (7%) | 0.905 | - / - | - / - |  |
| spectralquant | 48.5 | 1.429 | 1.498 | 1.456 | 1.733 | 142 (4%) | 0.931 | - / - | - / - |  |

## Reading the table (where UltraQuant loses)

* **Quality at equal bytes: UltraQuant loses to QServe-KV4.** Same 68 B/vec and about the same speed (K=1 0.36 vs 0.40 ms, K=8 0.55 vs 0.63 ms),
  but ppl delta 0.047 / 0.005 (Mistral / Qwen3) with SmoothAttention + per-token int4 against 0.139 / 0.562 for the UltraQuant FP4 quantiser as shipped.
  UltraQuant is 10-30% faster, QServe-int4 is much more accurate on these two models.
* **NVFP4 beats UltraQuant on quality** (0.056 / 0.081 vs 0.139 / 0.562) for +6% bytes (72 vs 68) and ~14% more time at K=1 (0.41 vs 0.36 ms); at K=8 the gap closes (0.54 vs 0.55 ms).
* **MXFP4 and UltraQuant share the same kernel** (same bytes, same speed by construction); only the scale rule differs. OCP floor-exponent MXFP4 is worse on Mistral
  (0.190 vs 0.139) and better on Qwen3 (0.482 vs 0.562): the ceil rule is not uniformly better.
* **FP8 and KIVI-4 are the accuracy ceilings** (ppl delta <= 0.03) but FP8 is 1.9x and KIVI-4 1.2x the bytes. FP8 is also the FASTEST kernel here (0.29 ms, 47% of HBM copy
  peak) because dequant is a single convert: the FP4 kernels are VALU-bound on dequant (about 20% of peak), so on MI300X 4-bit storage does not buy speed over FP8 at K=1/2.
  UltraQuant's advantage is capacity (3.8x vs 2x), not latency, versus FP8. KIVI-4 (0.54 ms) is the slowest of the 4-5 bit group because of per-channel scale/min traffic and the residual window.
* TurboQuant-3 has no K-query mode in its kernel (K independent calls -> linear in K, 8.7 ms at K=8) and loses on both axes in this table.
* Caveats: UltraQuant / TurboQuant quality rows come from results/quality_raw (earlier runs, same protocol); the ported methods were re-run with an fp16 baseline on this box
  (Mistral 4.872, Qwen3 8.522; quality_raw: 4.873 / 8.522). Kernel "attn cos" is on synthetic LLM-like KV (shared harness make_kv), not real activations.
  FP8 / QServe scales are calibrated online from the first K/V a layer sees. Timings are single runs on a shared GPU (lock-serialised; noise ~5%).
