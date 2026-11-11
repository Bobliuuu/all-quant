#!/usr/bin/env python3
"""Build results/sycl2_summary.json (+ a markdown table on stdout) from the bench sweeps.
inputs: results/sycl2_bench_{xmx,esimd}.jsonl (bench.exe --json, nsplit sweep), results/sycl_baseline_torchxpu_sdpa.json
Best nsplit per (path, B, ctx, K) is chosen by lowest fused_ms. The % of peak uses the RANDOM-data read peak ('read_gbs')."""
import json, os, sys, collections
R = os.path.join(os.path.dirname(__file__), "..", "..", "results")

def load(fn):
    rows, peak = [], None
    for l in open(os.path.join(R, fn)):
        l = l.strip()
        if not l.startswith("{"):
            continue
        j = json.loads(l)
        if j.get("kind") == "peak":
            peak = j
        else:
            rows.append(j)
    return rows, peak

best = {}
peaks = {}
for path in ("xmx", "esimd"):
    rows, pk = load(f"sycl2_bench_{path}.jsonl")
    peaks[path] = pk
    for r in rows:
        k = (path, r["B"], r["ctx"], r["K"])
        if k not in best or r["fused_ms"] < best[k]["fused_ms"]:
            best[k] = r
torch_ms = {}
try:
    t = json.load(open(os.path.join(R, "sycl_baseline_torchxpu_sdpa.json")))
    for r in (t if isinstance(t, list) else t.get("results", [])):
        torch_ms[(r.get("batch", r.get("B")), r.get("kv_len", r.get("ctx")), r.get("n_q", r.get("K")))] = r
except Exception as e:  # noqa
    print("torch baseline not parsed:", e, file=sys.stderr)
peak = max((p["read_gbs"] for p in peaks.values() if p), default=0)
out = {"device": "Intel Arc B390 (Xe3, 96 XVE)", "read_peak_random_gbs": peak, "shapes": []}
print("| B | ctx | K | XMX barrier path (ms) | ESIMD (ms) | speedup | ESIMD GB/s | % random-data read peak |")
print("|---|---|---|---|---|---|---|---|")
for B in (2, 8):
    for ctx in (4096, 16384, 32768):
        for K in (1, 2, 4, 8):
            a, b = best.get(("xmx", B, ctx, K)), best.get(("esimd", B, ctx, K))
            if not a or not b:
                continue
            sp = a["fused_ms"] / b["fused_ms"]
            out["shapes"].append({"B": B, "ctx": ctx, "K": K, "xmx_ms": a["fused_ms"], "xmx_nsplit": a["nsplit"], "esimd_ms": b["fused_ms"],
                                  "esimd_nsplit": b["nsplit"], "speedup": round(sp, 3), "esimd_gbs": b["fp4kv_gbs"],
                                  "esimd_pct_random_peak": round(100 * b["fp4kv_gbs"] / peak, 1), "xmx_pct_random_peak": round(100 * a["fp4kv_gbs"] / peak, 1)})
            print(f"| {B} | {ctx} | {K} | {a['fused_ms']:.3f} | {b['fused_ms']:.3f} | {sp:.2f}x | {b['fp4kv_gbs']:.1f} | {100*b['fp4kv_gbs']/peak:.0f} |")
json.dump(out, open(os.path.join(R, "sycl2_summary.json"), "w"), indent=1)
