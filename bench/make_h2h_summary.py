"""Merge h2h_raw_B*_c*.json chunks -> results/h2h_all.json and results/h2h_summary.md"""
import glob, json, sys, collections
R = sys.argv[1] if len(sys.argv) > 1 else "results"
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
     "fp16: `F.scaled_dot_product_attention` with q reshaped to [B,Hkv,G*K,D] (no KV expansion; non-causal, timing-equivalent).",
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
L += ["", "Real-KV accuracy (Mistral-7B) is in results/real_kv_summary.md; synthetic accuracy is mostly a property of the data distribution, not the shape."]
open(f"{R}/h2h_summary.md", "w").write("\n".join(L) + "\n")

# ---- UQ rows in the cross-vendor schema (same keys as results/mi300x_b8_c32k_tuned.json + pct_of_peak) ----
PEAK = {"AMD Instinct MI300X VF": 3909.16}   # measured HBM copy GB/s (results/mi300x_peak.json)
SPEC = {"AMD Instinct MI300X VF": 5300.0}
out = []
for (B, c, h, K), t in sorted(T.items()):
    if "uq_fp4_fused" not in t: continue
    u = t["uq_fp4_fused"]; ui = t.get("uq_fp4_Kindep")
    out.append(dict(device=dev, K=K, fused_ms=u["ms"], base_ms=ui["ms"] if ui else (u["ms"] if K == 1 else None),
                    speedup=(ui["ms"] / u["ms"]) if ui else (1.0 if K == 1 else None), fused_GBps=u["GBps_single_pass"],
                    pct_of_peak=100 * u["GBps_single_pass"] / PEAK.get(dev, float("nan")), pct_of_spec_peak=100 * u["GBps_single_pass"] / SPEC.get(dev, float("nan")),
                    batch=B, ctx=c, hq=h, hkv=8, d=128, page=64, nsplit=u["nsplit"], peak_GBps_measured=PEAK.get(dev)))
json.dump(out, open(f"{R}/h2h_uq_rows.json", "w"), indent=1)
