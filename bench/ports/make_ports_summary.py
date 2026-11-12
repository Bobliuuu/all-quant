"""Build results/ports_summary.md from results/ports_*.json (+ quality_raw for UltraQuant/TurboQuant quality rows)."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
J = {os.path.basename(p)[6:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{R}/ports_*.json"))}
peak = next((j["hbm_copy_peak_GBps"] for j in J.values() if "hbm_copy_peak_GBps" in j), 3909.0)
base = next((j["baselines"] for j in J.values() if j.get("baselines")), {})
def raw(model, m):
    p = f"{R}/quality_raw/{model}__{m}.json"
    return json.load(open(p)) if os.path.exists(p) else None
def qual(model, m):
    r, f = raw(model, m), raw(model, "fp16")
    if not r or not f or "ppl_wikitext2" not in r: return None
    return r["ppl_wikitext2"]["ppl"] - f["ppl_wikitext2"]["ppl"], r["attn"]["mean_cos"]
NOTES = '''
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
'''
rows = []
PORTS = {'fp8','nvfp4','mxfp4','kivi4','qserve_int4'}
NOTE = dict(fp8='per-tensor e4m3fnuz scale; fused K-query', nvfp4='E4M3/16 + tensor scale; fused', mxfp4='OCP floor-exp UE8M0/32; same kernel as UltraQuant', kivi4='per-channel K / per-token V + 128-token bf16 residual; fused', qserve_int4='per-token asym int4 + SmoothAttention; straightforward K-query')
def f2(x): return "-" if x is None else f"{x:.3f}"
def add(name, bpv, ms, cos, dq, note):
    g = lambda K: ms.get(str(K), ms.get(K)) if ms else None
    gb = 8 * 32768 * 8 * 2 * bpv / 1e9 / (g(1) / 1e3) if g(1) else None
    rows.append(f"| {name} | {bpv:g} | " + " | ".join(f2(g(K)) for K in (1, 2, 4, 8)) + f" | {'-' if gb is None else f'{gb:.0f}'} ({'-' if gb is None else f'{100*gb/peak:.0f}'}%) | {f2(cos)} | "
                f"{f2(dq['mistral7b'][0]) if dq.get('mistral7b') else '-'} / {f2(dq['qwen3_8b'][0]) if dq.get('qwen3_8b') else '-'} | "
                f"{f2(dq['mistral7b'][1]) if dq.get('mistral7b') else '-'} / {f2(dq['qwen3_8b'][1]) if dq.get('qwen3_8b') else '-'} | {note} |")
if base:
    b = base.get("fp16_sdpa"); add("fp16 SDPA (reference)", 256, {k: v["ms"] for k, v in b.items()}, 1.0, {}, "torch SDPA, KV re-read not needed (K rows batched)") if b else None
    b = base.get("uq_fp4_fused")
    if b: add("**UltraQuant FP4** (this repo)", 68, {k: v["ms"] for k, v in b.items()}, b["1"]["cos"] if "1" in b else b[1]["cos"],
              {m: qual(m, "uq_fp4") for m in ("mistral7b", "qwen3_8b")}, "real kernel; quality row = repo quantiser (official)")
    b = base.get("tq3_gqa_Kindep")
    if b: add("TurboQuant-3 (amd-experiments)", 52, {k: v["ms"] for k, v in b.items()}, b["1"]["cos"] if "1" in b else b[1]["cos"],
              {m: qual(m, "turbo3") for m in ("mistral7b", "qwen3_8b")}, "no K-query mode: K independent calls")
for name, j in J.items():
    if not j.get("ms_per_call") or "method" not in j: continue
    q = j.get("quality", {})
    dq = {}
    for m in ("mistral7b", "qwen3_8b"):
        d = q.get(m, {})
        k = next((k for k in d if k.startswith("port_")), None)
        if k: dq[m] = (d[k]["dppl"], d[k]["attn_cos"])
    cos = j.get("attn_cos", {}).get("1")
    add(j["method"] + (" (port)" if name in PORTS else ""), j["bytes_per_vec"], j["ms_per_call"], cos, dq, NOTE.get(name, ""))
hdr = ("| method | B/vec(128) | ms K=1 | K=2 | K=4 | K=8 | GB/s K=1 (% of HBM copy peak) | kernel attn cos (synthetic LLM-like KV) | ppl delta wikitext2 mistral / qwen3 | attn cos real KV mistral / qwen3 | note |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|\n")
open(f"{R}/ports_summary.md", "w").write("# Real AMD (Triton/ROCm) ports of the comparison methods, MI300X\n\n"
    f"B8, ctx 32768, Hq64/Hkv8/D128, page 64, cold-cache timing (shared harness bench/ports/common.py). HBM copy peak {peak:.0f} GB/s (results/mi300x_peak.json). "
    "GB/s counts stored KV bytes only (no fp16 residual, no q/out).\n\n" + hdr + "\n".join(rows) + "\n" + NOTES)
print(open(f"{R}/ports_summary.md").read())
