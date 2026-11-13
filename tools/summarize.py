"""Build results/SUMMARY.md from the raw result files (no numbers typed by hand)."""
import glob
import json
import os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
out = []
P = out.append

# ---- RD
R = json.load(open("results/rd_gaussian.json"))
def rd(method, mode):
    for x in R["rows"]:
        if x["method"] == method and x["mode"] == mode:
            return x
P("## Rate-distortion on iid N(0,1) (wasted bits = R - R*(D), lower is better)\n")
P("Source: `results/rd_gaussian.json` (512k values). Pearl's numbers are read off their Fig. 5.\n")
P("| point | Pearl's Fig. 5 | ours, no side info | ours, free exact per-vector norm | ours, incl. fp16 scale (the codec we ship) |")
P("|---|---|---|---|---|")
for lab, pearl, known, norm, fp in [
    ("PQ 2.75 bpv", 0.39, rd("pq-ball b=22", "known"), rd("pq-ball b=22", "norm"), rd("pq-ball b=21", "fp16")),
    ("PQ 3.75 bpv", 0.42, rd("pq-ball b=30", "known"), rd("pq-ball b=30", "norm"), rd("pq-ball b=29", "fp16")),
]:
    P(f"| {lab} | {pearl:.2f} | {known['waste']:.3f} | {norm['waste']:.3f} | {fp['waste']:.3f} |")
for lab, pearl, row in [("TurboQuant-MSE 3 b", 0.56, rd("turboquant-mse 3b", "norm")), ("TurboQuant-MSE 4 b", 0.60, rd("turboquant-mse 4b", "norm")),
                        ("NVFP4 4.5 bpv", 1.15, rd("nvfp4", "fp16")), ("FP8 e4m3 8 bpv", 2.65, rd("fp8 e4m3 (no scale)", "known"))]:
    P(f"| {lab} | {pearl:.2f} | {row['waste']:.3f} (our re-run) | | |")
hq = [x for x in R["rows"] if x["method"].startswith("hyperquant")]
P("\nHyperQuant E8 + Rice (variable rate, interpolated): " + ", ".join(f"{x['bpv']} bpv -> {x['waste']:.3f}" for x in hq) + ".")
v = rd("pq-voronoi q=6", "fp16"); b = rd("pq-ball b=21", "fp16")
P(f"Voronoi (NestQuant) code at 2.75 bpv incl. scale: {v['waste']:.3f} vs ball {b['waste']:.3f}.\n")

# ---- KL
kl = {}
for f in sorted(glob.glob("results/kl_16k_*.json")):
    r = json.load(open(f))
    for c, v in r["configs"].items():
        kl[c] = (v, r)
if kl:
    any_r = next(iter(kl.values()))[1]
    P(f"## KL divergence from an FP8 KV cache, PG-19, {any_r['model']}, {any_r['nseq']} books x {any_r['ctx']} tokens\n")
    P("Reference: FP8 e4m3 with one fp32 scale per 128-value vector (8.25 bpv). `-kc` = our K mean-centering (softmax-invariant, free in the kernel). Model is not Pearl's.\n")
    P("| config | bits/value (K,V avg) | KL from FP8 (mean) | per book | PPL | Pearl's KL (GLM-5.3) |")
    P("|---|---|---|---|---|---|")
    pearl = {"pq-3.2": 0.057, "pq-2.3": 0.122, "pq-2.1": 0.207}
    order = ["bf16", "fp8-noscale", "nvfp4", "nvfp4-kc", "int4-g32", "tq-mse-3", "tq-mse-3-kc", "tq-mse-2", "tq-mse-2-kc",
             "pq-3.2", "pq-3.2-kc", "pq-3.2-kvc", "pq-3.2-vk", "pq-2.3", "pq-2.3-kc", "pq-2.3-kvc", "pq-2.1", "pq-2.1-kc", "pq-2.1-kvc",
             "vor-3.2", "vor-3.2-kc", "vor-2.3", "vor-2.3-kc", "vor-2.1", "vor-2.1-kc"]
    for c in order + [c for c in kl if c not in order]:
        if c not in kl:
            continue
        v = kl[c][0]
        base = c.replace("-kvc", "").replace("-kc", "")
        pr = pearl.get(base, "")
        P(f"| {c} | {v['bits_per_value']:.3f} | {v['kl_mean']:.4f} | {', '.join(f'{x:.3f}' for x in v['kl_per_seq'])} | {v['ppl']:.2f} | {pr} |")
    P(f"\nFP8 reference PPL {any_r['ref_ppl']:.2f}. Pearl's FP8 noise floor is ~0.04 (nondeterminism); our runs are deterministic, bf16-vs-FP8 shows the FP8 error itself.\n")

# ---- NIAH
nia = {}
for f in sorted(glob.glob("results/niah_8k_*.json")):
    r = json.load(open(f))
    for c, v in r["configs"].items():
        nia[c] = (v, r)
if nia:
    r0 = next(iter(nia.values()))[1]
    P(f"## Needle-in-a-haystack (our RULER-style generator, not RULER), {r0['model']}, {r0['ctx']} tokens, {r0['n_per_task']} prompts per task\n")
    P("| config | single needle | multikey (4 needles) |")
    P("|---|---|---|")
    for c, (v, _) in nia.items():
        P(f"| {c} | {v['acc']['single']:.2f} | {v['acc']['multikey']:.2f} |")
    P("")

# ---- kernels
def load(f):
    return [json.loads(l) for l in open(f) if l.startswith("{")] if os.path.exists(f) else []
P("## Kernel correctness (shared vectors, 8 cases x 3 formats x 4 split counts)\n")
P("| backend | device | passed | worst rel. L2 error (pq) |")
P("|---|---|---|---|")
for f in sorted(glob.glob("results/kernels/*check*.jsonl")):
    rows = load(f)
    if not rows:
        continue
    dev = rows[0].get("device") or {"cuda": "NVIDIA H100", "hip": "AMD MI300X"}.get(rows[0]["backend"], "")
    pq = [x["rel_l2_err"] for x in rows if x["fmt"] == "pq"]
    P(f"| {rows[0]['backend']} | {dev or os.path.basename(f)} | {sum(x['pass'] for x in rows)}/{len(rows)} | {max(pq):.1e} |")
P("")
P("## Kernel timing (us per decode-attention call, synthetic cache, best split count)\n")
P("bytes/token/KV-head: pq 2*bK+2 + 2*bV+2 (102 B at 25/24), bf16 512 B, fp8 264 B.\n")
P("| device | B | T | Hq | Hkv | pq bits K/V | pq | bf16 (same skeleton) | fp8 (same skeleton) | vendor bf16 |")
P("|---|---|---|---|---|---|---|---|---|---|")
sdpa = {(x["B"], x["T"], x["Hq"], x["Hkv"]): x["us"] for x in load("results/kernels/cuda_h100_torch_sdpa.jsonl")}
for f in sorted(glob.glob("results/kernels/*bench*.jsonl")):
    if "tabsmem" in f:
        continue
    rows = load(f)
    by = {}
    for x in rows:
        k = (x["B"], x["T"], x["Hq"], x["Hkv"])
        by.setdefault(k, {})
        if x["fmt"] == "pq":
            by[k].setdefault("pq", []).append((x["bK"], x["bV"], x["us"]))
        else:
            by[k][x["fmt"]] = x["us"]
    dev = (rows[0].get("gpu") or rows[0].get("device", "")) if rows else ""
    dev = f"{dev} ({rows[0]['backend'].upper()})" if rows else dev
    for k, d in by.items():
        for bk, bv, us in d.get("pq", []):
            vend = f"{sdpa[k]:.0f} (torch SDPA)" if "H100" in dev and "CUDA)" in dev and k in sdpa else ""
            P(f"| {dev} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | {bk}/{bv} | {us:.0f} | {d.get('bf16', float('nan')):.0f} | {d.get('fp8', float('nan')):.0f} | {vend} |")
P("")
open("results/SUMMARY.md", "w").write("# Results summary (generated by tools/summarize.py)\n\n" + "\n".join(out) + "\n")
print("\n".join(out))
