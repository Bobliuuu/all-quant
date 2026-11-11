"""Aggregate results/realworld_*.json -> results/realworld_summary_tables.md + figures/realworld_*.png"""
import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
rows = {}
for f in sorted(glob.glob(f"{R}/realworld_*.json")):
    d = json.load(open(f)); rows[os.path.basename(f)[10:-5]] = d

def g(tag, *k, default=None):
    d = rows.get(tag, {})
    for x in k:
        d = d.get(x, {}) if isinstance(d, dict) else {}
    return d if d != {} else default

def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n if len(a) == len(b) else n

out = []
plat = sys.argv[1] if len(sys.argv) > 1 else "cuda_h100"
def line(*c): out.append("| " + " | ".join(str(x) for x in c) + " |")
def fmt(x, n=3): return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
for kv in ("bf16", "fp8", "uq4"):
    pass
kvs = [k for k in ("bf16", "fp8", "uq4") if any(t.endswith("_" + k) for t in rows)]
out.append(f"## {plat}: capacity / agent / RAG / needle")
line("workload", "kv", "out tok/s", "TTFT p50 s", "TTFT p99 s", "ITL p50 ms", "ITL p99 ms", "KV tokens", "acc")
for t, d in rows.items():
    if not t.startswith(plat) : continue
    r = d.get("result", {}); s = d.get("server", {}); name = t[len(plat) + 1:]
    if any(name.startswith(x) for x in ("agent", "rag", "needle")):
        acc = r.get("F1") if "F1" in r else r.get("acc")
        line(name, d.get("kv"), fmt(r.get("out_tok_per_s"), 0), fmt(r.get("ttft_p50")), fmt(r.get("ttft_p99")),
             fmt((r.get("itl_p50") or 0) * 1e3, 1), fmt((r.get("itl_p99") or 0) * 1e3, 1), s.get("kv_cache_tokens"), fmt(acc))
out.append("\n## spec-decode (n-gram, code edit)")
line("cfg", "kv", "K", "conc", "out tok/s", "ITL p50 ms", "tokens/step", "acc/draft")
for t, d in rows.items():
    if t.startswith(plat) and "_spec_" in t or t.startswith(plat + "_spec"):
        r = d.get("result", {}); line(t[len(plat) + 1:], d.get("kv"), r.get("K"), r.get("conc"), fmt(r.get("out_tok_per_s"), 0),
                                      fmt((r.get("itl_p50") or 0) * 1e3, 2), fmt(r.get("tokens_per_step"), 2), fmt(r.get("mean_accepted_per_draft"), 2))
out.append("\n## drift vs bf16 greedy (first divergence token index, same 24 prompts)")
base = g(f"{plat}_drift_bf16", "result", "token_ids")
for k in kvs:
    if k == "bf16": continue
    ids = g(f"{plat}_drift_{k}", "result", "token_ids")
    if base and ids:
        fd = [first_div(a, b) for a, b in zip(base, ids)]; full = sum(1 for a, b, n in zip(base, ids, fd) if n == len(a) == len(b))
        fd.sort(); out.append(f"- {k}: identical full sequence {full}/{len(fd)}; median first-divergence token {fd[len(fd)//2]}; min {fd[0]}; mean {sum(fd)/len(fd):.0f}")
open(f"{R}/realworld_{plat}_tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
