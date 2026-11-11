"""Merge h2h_raw_B*_c*.json chunks -> results/h2h_all.json and results/h2h_summary.md"""
import glob, json, sys, collections
R = sys.argv[1] if len(sys.argv) > 1 else "results"
PEAK = {"AMD Instinct MI300X VF": 3909.16}   # measured HBM copy GB/s (results/mi300x_peak.json)
SPEC = {"AMD Instinct MI300X VF": 5300.0}
rows, acc, dev, bpv = [], [], None, None
for f in sorted(glob.glob(f"{R}/h2h_raw_B*_c*.json")):
    d = json.load(open(f)); rows += d["rows"]; acc += d["accuracy"]; dev = d["device"]; bpv = d["bytes_per_vec"]
json.dump(dict(device=dev, bytes_per_vec=bpv, rows=rows, accuracy=acc), open(f"{R}/h2h_all.json", "w"), indent=1)
T = collections.defaultdict(dict)
for r in rows: T[(r["B"], r["ctx"], r["Hq"], r["K"])][r["method"]] = r
def g(t, m, k="ms"):
    return t[m][k] if m in t else None
f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
L = [f"# Head-to-head microbenchmark: UltraQuant FP4 fused vs amd-experiments TQ3 vs fp16 SDPA ({dev})", "",
     "Synthetic iid-Gaussian KV, Hkv=8, D=128, page 64 (ours), CUDA-event mean of 30 reps (10-ish for the largest), all methods same box/timer.",
     f"Bytes per K-or-V vector per head: fp16 {bpv['fp16']}, TQ3 {bpv['tq3']} (4 B norm + 3x16 B bit-planes), UQ FP4+UE8M0 {bpv['uq_fp4']} (64 B nibbles + 4 B scales).",
     "Ours: best nsplit of {8,16,32,64} per config (pinned commit 1ec6fbe). Theirs: `turboquant_gqa_attention_fwd` (autotuned split-K GQA path), time includes the 128x128 inverse output rotation; q rotation shown separately.",
     "fp16 SDPA: `F.scaled_dot_product_attention` with q reshaped to [B,Hkv,G*K,D] (no KV expansion; non-causal, timing-equivalent). WARNING: torch SDPA is a weak fp16 decode baseline on MI300X at small batch (~100 GB/s at B=1); the 'Honest fp16 baseline' section near the end compares against the best of SDPA / bmm / split-KV and is the fair per-call number.",
     "Apples-to-oranges: TQ3 is 3-bit rotation+Lloyd-Max, K=1 only, no causal-block masking, non-paged contiguous KV; ours is 4-bit MX-style, paged, K<=16 with causal spec masking.", ""]
for Hq in (32, 64):
    L += [f"## Hq/Hkv = {Hq}/8: K=1 decode, ms (lower is better)", "",
          "| B | ctx | fp16 SDPA | TQ3 fused (K=1) | UQ FP4 fused | UQ vs TQ3 | UQ vs fp16 | TQ3 vs fp16 | UQ GB/s | TQ3 GB/s | fp16 GB/s |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (B, c, h, K), t in sorted(T.items()):
        if h != Hq or K != 1: continue
        a, b, u = g(t, "fp16_sdpa_fused"), g(t, "tq3_gqa_K1_incl_outrot"), g(t, "uq_fp4_fused")
        L.append(f"| {B} | {c} | {f(a)} | {f(b)} | {f(u)} | {f(b/u,1)+'x' if b and u else '-'} | {f(a/u,2)+'x' if a and u else '-'} | {f(a/b,2)+'x' if a and b else '-'} | {f(g(t,'uq_fp4_fused','GBps_single_pass'),0)} | {f(g(t,'tq3_gqa_K1_incl_outrot','GBps_single_pass'),0)} | {f(g(t,'fp16_sdpa_fused','GBps_single_pass'),0)} |")
    L.append("")
    L += [f"## Hq/Hkv = {Hq}/8: K independent calls vs fused K (speculative verify), ms", "",
          "| B | ctx | K | fp16 fused | fp16 K-indep | TQ3 K-indep (K x their call) | TQ3 S_q=K one call (no causal, timing only) | UQ fused | UQ K-indep | UQ fused vs TQ3 K-indep | UQ fused vs UQ K-indep |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (B, c, h, K), t in sorted(T.items()):
        if h != Hq or K == 1: continue
        u, ui, ti = g(t, "uq_fp4_fused"), g(t, "uq_fp4_Kindep"), g(t, "tq3_gqa_Kindep")
        L.append(f"| {B} | {c} | {K} | {f(g(t,'fp16_sdpa_fused'))} | {f(g(t,'fp16_sdpa_Kindep'))} | {f(ti)} | {f(g(t,'tq3_gqa_SqK_noncausal'))} | {f(u)} | {f(ui)} | {f(ti/u,1)+'x' if ti and u else '-'} | {f(ui/u,2)+'x' if ui and u else '-'} |")
    L.append("")
L += ["## Accuracy vs fp32 reference on the same synthetic KV (batch element 0, Hq=32)", "",
      "`iid`: Gaussian q (softmax nearly flat, output is a noisy mean of V: hardest for relative error). `peaked`: each query = a random earlier key + noise (concentrated softmax). K=8 uses causal masking over the last 8 tokens (ours only).", "",
      "| B | ctx | regime | K | fp16 SDPA cos | UQ FP4 cos | UQ FP4 rel-L2 | TQ3 cos | TQ3 rel-L2 |", "|--:|--:|--|--:|--:|--:|--:|--:|--:|"]
A = collections.defaultdict(dict)
for r in acc: A[(r["B"], r["ctx"], r["regime"], r["K"])][r["method"]] = r
for (B, c, rg, K), t in sorted(A.items()):
    if B != 1 and c not in (4096, 32768): continue
    L.append(f"| {B} | {c} | {rg} | {K} | {f(g(t,'fp16_sdpa','cos'),4)} | {f(g(t,'uq_fp4','cos'),4)} | {f(g(t,'uq_fp4','rel_l2'),3)} | {f(g(t,'tq3_gqa','cos'),4)} | {f(g(t,'tq3_gqa','rel_l2'),3)} |")
# ---- stronger fp16 baselines ----
import os
if os.path.exists(f"{R}/h2h_fp16_baselines.json"):
    fb = json.load(open(f"{R}/h2h_fp16_baselines.json"))["rows"]
    L += ["", "## Honest fp16 baseline: torch SDPA is NOT the best fp16 decode path on MI300X", "",
          "Torch's SDPA (flash path, no split-KV) reaches only ~100 GB/s at B=1. The `bmm+softmax+bmm` formulation (what amd-experiments' own decode benchmark uses) and a split-KV-via-bmm variant are far faster at small batch. `fp16 best` = min of the three. Hq/Hkv=32/8; K=8 rows are one call with M=G*K rows (non-causal, timing-equivalent).", "",
          "| B | ctx | K | SDPA ms | bmm ms | split-KV bmm ms | fp16 best ms | fp16 best GB/s | UQ fused ms | UQ vs fp16 best | UQ vs SDPA | TQ3 K=1 ms |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in fb:
        t = T.get((r["B"], r["ctx"], 32, r["K"]), {}); u = t.get("uq_fp4_fused", {}).get("ms")
        gbs = r["B"] * r["ctx"] * 8 * 2 * 256 / 1e9 / (r["best_fp16_ms"] / 1e3)
        tq = T.get((r["B"], r["ctx"], 32, 1), {}).get("tq3_gqa_K1_incl_outrot", {}).get("ms") if r["K"] == 1 else None
        L.append(f"| {r['B']} | {r['ctx']} | {r['K']} | {r['sdpa_ms']:.3f} | {r['bmm_ms']:.3f} | {r['splitkv_bmm_ms']:.3f} | {r['best_fp16_ms']:.3f} | {gbs:.0f} | {f(u)} | {f(r['best_fp16_ms']/u,2)+'x' if u else '-'} | {f(r['sdpa_ms']/u,2)+'x' if u else '-'} | {f(tq)} |")
    L += ["", "NOTE: this table is warm-cache (same layer repeated), which favours fp16 at small shapes because KV fits in the 256 MB Infinity Cache; the cold-cache table below is the realistic one (UQ 1.0-1.75x faster than best fp16 there). UQ moves 3.76x fewer KV bytes than fp16 but reaches only 0.3-1.0 TB/s of the 3.9 TB/s measured HBM peak (dequant ALU + tile structure bound, see results/mi300x_tuning_notes.md), so against a bandwidth-efficient fp16 kernel the per-call gain is ~1.0-1.6x, not 3.76x."]
cf = sorted(glob.glob(f"{R}/h2h_cold_raw_B*_c*.json"))
if cf:
    cr = []
    for fn in cf: cr += json.load(open(fn))["rows"]
    json.dump(cr, open(f"{R}/h2h_cold_all.json", "w"), indent=1)
    C = collections.defaultdict(dict)
    for r in cr: C[(r["B"], r["ctx"], r["K"])][r["method"]] = r
    oc = []
    for (B, c, K), t in sorted(C.items()):
        if "uq_fp4_fused" not in t: continue
        u = t["uq_fp4_fused"]; ui = t.get("uq_fp4_Kindep")
        oc.append(dict(device=dev, cache="cold", K=K, fused_ms=u["ms"], base_ms=ui["ms"] if ui else (u["ms"] if K == 1 else None), speedup=(ui["ms"] / u["ms"]) if ui else (1.0 if K == 1 else None),
                       fused_GBps=u["GBps_single_pass"], pct_of_peak=100 * u["GBps_single_pass"] / PEAK.get(dev, float("nan")), pct_of_spec_peak=100 * u["GBps_single_pass"] / SPEC.get(dev, float("nan")),
                       batch=B, ctx=c, hq=32, hkv=8, d=128, page=64, nsplit=u["nsplit"], peak_GBps_measured=PEAK.get(dev)))
    json.dump(oc, open(f"{R}/h2h_uq_rows_cold.json", "w"), indent=1)
    L += ["", "## Cold-cache timings (Infinity Cache flushed before every call), Hq/Hkv=32/8", "",
          "MI300X has a 256 MB Infinity Cache. The warm tables above repeat the same layer's KV every call, so any config whose KV (fp16: B*ctx*0.52 MB/1k tokens; FP4: 3.76x less) fits in 256 MB is partly served from cache, which flatters small-batch numbers (all B=1 rows, and B=8 up to ~16k for fp16; most UQ rows). Here a 1 GB buffer is written before each call and each call is timed with its own events (adds ~few us of event overhead). A real decode loop cycles 32 layers (4+ GB), so cold numbers are the realistic ones.", "",
          "| B | ctx | K | fp16 SDPA | fp16 bmm | fp16 best | TQ3 (K=1) / K-indep | UQ fused | UQ vs fp16 best | UQ GB/s (single-pass) | % of measured HBM peak (3909) | UQ K-indep | UQ fused vs UQ K-indep |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (B, c, K), t in sorted(C.items()):
        a_, b_, u_ = g(t, "fp16_sdpa_fused"), g(t, "fp16_bmm_fused"), g(t, "uq_fp4_fused")
        best16 = min(x for x in (a_, b_) if x)
        tq = g(t, "tq3_gqa_K1_incl_outrot") if K == 1 else g(t, "tq3_gqa_Kindep"); ui = g(t, "uq_fp4_Kindep")
        gb = g(t, "uq_fp4_fused", "GBps_single_pass")
        L.append(f"| {B} | {c} | {K} | {f(a_)} | {f(b_)} | {f(best16)} | {f(tq)} | {f(u_)} | {f(best16/u_,2)+'x' if u_ else '-'} | {f(gb,0)} | {f(100*gb/3909,0) if gb else '-'}% | {f(ui)} | {f(ui/u_,2)+'x' if ui and u_ else '-'} |")
mf = sorted(glob.glob(f"{R}/h2h_methods_c*.json"))
if mf:
    L += ["", "## Other KV-quant methods: attention step at B=8, Hq/Hkv=32/8, K=1", "",
          "Only TQ3 and UQ have fused kernels. fp8/int4 per-tensor (amd-experiments baseline formats), Planar/Iso/Rotor-3 and the TQ3 decompress path are 'dequantise to fp16 then SDPA' (their own bench_all_methods_decode design; fp16 reference here is torch SDPA, 1.4 ms at 32k, bmm would be 0.4 ms). KIVI-style int4 is our own implementation (K per-channel group-32 over tokens, V per-token group-32; asymmetric, fp16 scale+min = 72 B/vec) run through the same unfused path: it shows the *quality* of that format, not a tuned kernel. cos = attention-output cosine vs fp16 on iid Gaussian KV. SemKV/StarKV: no public code found, not run. Their fp8 baseline script scales to 448 although e4m3fnuz max is 240 (NaN outputs: their own results/bench_quality json shows fp8 NaN too); we use 240.", "",
          "| method | bytes/vec | ctx | ms | cos (iid) |", "|--|--:|--:|--:|--:|"]
    for fn in mf:
        for r in json.load(open(fn))["rows"]: L.append(f"| {r['method']} | {r['bytes_per_vec']} | {r['ctx']} | {r['ms']:.3f} | {r['cos']:.4f} |")
L += ["", "Real-KV accuracy (Mistral-7B) is in results/real_kv_summary.md; synthetic accuracy is mostly a property of the data distribution, not the shape."]
open(f"{R}/h2h_summary.md", "w").write("\n".join(L) + "\n")

# ---- UQ rows in the cross-vendor schema (same keys as results/mi300x_b8_c32k_tuned.json + pct_of_peak) ----
out = []
for (B, c, h, K), t in sorted(T.items()):
    if "uq_fp4_fused" not in t: continue
    u = t["uq_fp4_fused"]; ui = t.get("uq_fp4_Kindep")
    out.append(dict(device=dev, K=K, fused_ms=u["ms"], base_ms=ui["ms"] if ui else (u["ms"] if K == 1 else None),
                    speedup=(ui["ms"] / u["ms"]) if ui else (1.0 if K == 1 else None), fused_GBps=u["GBps_single_pass"],
                    pct_of_peak=100 * u["GBps_single_pass"] / PEAK.get(dev, float("nan")), pct_of_spec_peak=100 * u["GBps_single_pass"] / SPEC.get(dev, float("nan")),
                    batch=B, ctx=c, hq=h, hkv=8, d=128, page=64, nsplit=u["nsplit"], peak_GBps_measured=PEAK.get(dev)))
json.dump(out, open(f"{R}/h2h_uq_rows.json", "w"), indent=1)
